import torch
from torch.nn.utils.rnn import pad_sequence


class RankFormer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 tf_dim_feedforward=None,
                 tf_nhead=1,
                 tf_num_layers=2,
                 head_hidden_layers=None,
                 dropout=0.2,
                 fixed_length=False):
        """
        RankFormer: 基于Transformer的排序模型

        Args:
            input_dim: 输入特征维度
            tf_dim_feedforward: Transformer FFN维度，默认为input_dim*4
            tf_nhead: Transformer注意力头数，默认1
            tf_num_layers: Transformer层数
            head_hidden_layers: 打分头的隐藏层,默认[input_dim//2]
            dropout: Dropout比率
            fixed_length: 是否所有列表长度固定(固定则不需要padding和mask)
        """
        super().__init__()

        self.fixed_length = fixed_length

        # 设置默认值
        if tf_dim_feedforward is None:
            tf_dim_feedforward = input_dim * 4  # 标准Transformer配置

        if head_hidden_layers is None:
            head_hidden_layers = [input_dim // 2]  # 默认为输入维度的一半

        # Transformer编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(
            input_dim,
            nhead=tf_nhead,
            dim_feedforward=tf_dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_num_layers,
            norm=None
        )

        # 排序打分头
        self.rank_score_net = MLP(
            input_dim=input_dim,
            hidden_layers=head_hidden_layers,
            output_dim=1,
            dropout=dropout
        )

    def forward(self, feat, length):
        """
        前向传播

        Args:
            feat: Tensor [N, input_dim] - N是所有列表元素的总数
            length: Tensor [B] or int - 每个列表的长度
                    如果fixed_length=True,可以传入int表示统一长度

        Returns:
            rank_score: Tensor [N] - 每个元素的排序分数
        """
        if self.fixed_length:
            # 固定长度模式：直接reshape，无需padding和mask
            if isinstance(length, int):
                list_len = length
                batch_size = feat.shape[0] // list_len
            else:
                list_len = length[0].item()
                batch_size = length.shape[0]

            # 直接reshape成 [B, list_len, input_dim]
            feat = feat.view(batch_size, list_len, -1)

            # Transformer编码（不需要mask）
            tf_embs = self.transformer(feat)

            # 展平回 [N, input_dim]
            tf_embs = tf_embs.view(-1, feat.shape[-1])

        else:
            # 变长模式：需要padding和mask
            feat_per_list = feat.split(length.tolist())

            # Padding并创建mask
            feat = pad_sequence(feat_per_list, batch_first=True, padding_value=0)
            # [B, max_len, input_dim]

            padding_mask = torch.ones(
                (feat.shape[0], feat.shape[1]),
                dtype=torch.bool
            ).to(feat.device)
            for i, list_len in enumerate(length):
                padding_mask[i, :list_len] = False

            # Transformer编码
            tf_embs = self.transformer(feat, src_key_padding_mask=padding_mask)
            # [B, max_len, input_dim]

            # 移除padding,得到所有有效元素
            tf_embs = tf_embs[~padding_mask]  # [N, input_dim]

        # 计算排序分数
        rank_score = self.rank_score_net(tf_embs)  # [N]

        return rank_score


class MLP(torch.nn.Module):
    """简单的多层感知机"""

    def __init__(self, input_dim, hidden_layers=None, output_dim=1, dropout=0.):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32]

        net = []
        for h_dim in hidden_layers:
            net.append(torch.nn.Linear(input_dim, h_dim))
            net.append(torch.nn.ReLU())
            if dropout > 0.:
                net.append(torch.nn.Dropout(dropout))
            input_dim = h_dim
        net.append(torch.nn.Linear(input_dim, output_dim))

        self.net = torch.nn.Sequential(*net)

    def forward(self, feat):
        score = self.net(feat).squeeze(dim=-1)
        return score


import torch.nn as nn
class QueryRefProjection(nn.Module):
    """
    Query和Reference特征的降维投影模块

    功能：
    - Query: 1024维 → 512维
    - Ref: [1024维特征 + 2维坐标] → 512维
    """

    def __init__(self,
                 feat_dim=1024,
                 coord_dim=2,
                 output_dim=512,
                 use_layer_norm=False,
                 use_activation=False,
                 activation='gelu'):
        """
        Args:
            feat_dim: 原始特征维度（query和ref的feat维度）
            coord_dim: 坐标维度（通常是2D: x,y）
            output_dim: 输出的embedding维度
            use_layer_norm: 是否在投影后使用LayerNorm
            use_activation: 是否在投影后使用激活函数（一般不推荐）
            activation: 激活函数类型 ('relu', 'gelu', 'silu')
        """
        super().__init__()

        self.feat_dim = feat_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.use_activation = use_activation

        # Query投影：只有特征
        self.query_proj = nn.Linear(feat_dim, output_dim)

        # Ref投影：特征+坐标
        self.ref_proj = nn.Linear(feat_dim + coord_dim, output_dim)

        # 可选的LayerNorm
        if use_layer_norm:
            self.query_norm = nn.LayerNorm(output_dim)
            self.ref_norm = nn.LayerNorm(output_dim)

        # 可选的激活函数
        if use_activation:
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'silu':
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

    def forward(self, feat_query, feat_ref, coord_ref):
        """
        Args:
            feat_query: Tensor [1, feat_dim] or [B, feat_dim] - Query特征
            feat_ref: Tensor [N, feat_dim] - Reference特征
            coord_ref: Tensor [N, coord_dim] - Reference坐标

        Returns:
            query_emb: Tensor [1, output_dim] or [B, output_dim]
            ref_emb: Tensor [N, output_dim]
        """
        # Query降维
        query_emb = self.query_proj(feat_query)
        if self.use_layer_norm:
            query_emb = self.query_norm(query_emb)
        if self.use_activation:
            query_emb = self.activation(query_emb)

        # Ref: 拼接feat和coord后降维
        ref_input = torch.cat([feat_ref, coord_ref], dim=-1)
        ref_emb = self.ref_proj(ref_input)
        if self.use_layer_norm:
            ref_emb = self.ref_norm(ref_emb)
        if self.use_activation:
            ref_emb = self.activation(ref_emb)

        return query_emb, ref_emb