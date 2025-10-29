import torch
import torch.nn as nn
import torch.nn.functional as F
from .ocn_mlp import ResnetBlockFC

class HashAggregator(nn.Module):
    '''
    一个集成了自适应聚合与特征解码功能的MLP。
    它会根据输入的4D坐标p，动态地学习为多分辨率哈希特征分配权重，
    然后对加权聚合后的特征进行深度处理。
    '''

    def __init__(self,
                 n_levels=16,  # INGP的哈希层级数
                 per_level_feat_dim = 1024,  # INGP每层的特征维度
                 coord_dim=4,  # 4D坐标 (x,y,s,d)
                 output_dim=1024,  # 最终场景特征的维度
                 hidden_dim=512,  # 隐藏层维度
                 n_blocks=2,  # ResNet块的数量
                 norm_type='layernorm'):
        super().__init__()
        self.n_levels = n_levels
        self.per_level_feat_dim = per_level_feat_dim

        # 1. 权重生成网络 (Attention Weight Generator)
        # 输入是4D坐标，输出是每个层级的权重(logits)
        self.weight_generator = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_levels)
        )

        # 2. 特征处理主干 (Feature Processing Backbone)
        # 输入维度是聚合后的特征维度，即 per_level_feat_dim
        # (因为我们是加权求和，而不是拼接)
        self.feature_backbone = nn.ModuleList()
        self.feature_backbone.append(nn.Linear(per_level_feat_dim, hidden_dim))

        for _ in range(n_blocks):
            self.feature_backbone.append(ResnetBlockFC(hidden_dim, norm_type=norm_type))

        self.feature_backbone.append(nn.ReLU())
        self.feature_backbone.append(nn.Linear(hidden_dim, output_dim))

        # 将主干网络的所有层打包成一个Sequential模块，方便调用
        self.feature_backbone = nn.Sequential(*self.feature_backbone)

    def forward(self, multires_feats, p):
        """
        Args:
            multires_feats (torch.Tensor): 来自INGP哈希编码器的原始特征，
                                           Shape: [Batch, L, F] (L=n_levels, F=per_level_feat_dim)
            p (torch.Tensor):              输入的4D坐标，Shape: [Batch, 4]
        Returns:
            torch.Tensor: L2归一化后的最终场景特征
        """
        # --- 自适应聚合过程 ---
        # 1. 根据坐标p生成权重logits
        # Shape: [Batch, L]
        attention_logits = self.weight_generator(p)

        # 2. 通过Softmax函数将logits转换为归一化的权重
        # Shape: [Batch, L]
        attention_weights = F.softmax(attention_logits, dim=-1)

        # 3. 进行加权求和
        # attention_weights.unsqueeze(-1) -> [Batch, L, 1]
        # multires_feats           -> [Batch, L, F]
        # 两者相乘 (broadcasting) -> [Batch, L, F]
        # .sum(dim=1) -> [Batch, F]
        aggregated_feat = (attention_weights.unsqueeze(-1) * multires_feats).sum(dim=1)

        # --- 信号补全/解码过程 ---
        # 将聚合后的特征送入主干网络
        feature = self.feature_backbone(aggregated_feat)

        # L2 Normalization
        feature_normalized = F.normalize(feature, p=2, dim=-1)

        return feature_normalized, attention_weights  # 返回权重用于分析和可视化