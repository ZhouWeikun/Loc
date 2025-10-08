import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S,dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)

    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

class SALAD_FiLM(nn.Module):
    """
    使用FiLM (Feature-wise Linear Modulation) 让global token调制local features
    """

    def __init__(self,
                 input_feat_dim=1024,
                 num_clusters=32,
                 cluster_dim=64,
                 base_dim=1024,  # 最终特征维度
                 dropout=0.3,
                 img_hw=(224, 224),
                 patchsize=14,
                 with_dustbin=True,
                 hidden_layer_dim=512,
                 ):
        super().__init__()

        self.input_feat_dim = input_feat_dim
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.base_dim = base_dim
        self.token_hw = (int(img_hw[0] / patchsize), int(img_hw[1] / patchsize))
        self.with_dustbin = with_dustbin
        self.hidden_layer_dim = hidden_layer_dim

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        # ===== Local Feature Path (基础信号) =====
        # MLP for local features -> 这是被调制的"基础信号"
        self.local_features = nn.Sequential(
            nn.Conv2d(self.input_feat_dim, self.hidden_layer_dim, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(self.hidden_layer_dim, self.cluster_dim, 1)
        )

        # Score matrix for optimal transport
        self.score = nn.Sequential(
            nn.Conv2d(self.input_feat_dim, self.hidden_layer_dim, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(self.hidden_layer_dim, self.num_clusters, 1),
        )

        # Dustbin parameter
        self.dust_bin = nn.Parameter(torch.tensor(0.))

        # 聚合后的local特征维度
        local_agg_dim = self.num_clusters * self.cluster_dim

        # Local特征投影到目标维度
        # self.local_projection = nn.Sequential(
        #     nn.Linear(local_agg_dim, self.hidden_layer_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_layer_dim, self.base_dim)
        # )
        assert local_agg_dim==base_dim

        # ===== Global Token Path (调制器) =====
        # FiLM调制器：global token -> (gamma, beta)
        self.film_modulator = nn.Sequential(
            nn.Linear(self.input_feat_dim, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dim, self.base_dim * 2)  # 输出 gamma 和 beta
        )
        final_layer = self.film_modulator[-1]  # 获取最后一个线性层
        torch.nn.init.constant_(final_layer.weight, 0)
        # 将bias切片，一半初始化为1（对应gamma），一半为0（对应beta）
        torch.nn.init.constant_(final_layer.bias[:self.base_dim], 1)
        torch.nn.init.constant_(final_layer.bias[self.base_dim:], 0)
        

    def forward(self, x):
        """
        x: [B, n_patches+1, input_feat_dim]
        Returns: [B, base_dim] - 归一化后的融合特征
        """
        # 分离global token和local patches
        global_token = x[:, 0]  # [B, input_feat_dim]
        local_patches = x[:, 1:]  # [B, n_patches, input_feat_dim]

        # Reshape local patches to (B, C, H, W)
        B = local_patches.shape[0]
        local_patches = local_patches.reshape(
            B, self.token_hw[0], self.token_hw[1], self.input_feat_dim
        ).permute(0, 3, 1, 2)

        # ===== 步骤1: 提取并聚合local features (基础信号) =====
        f_local = self.local_features(local_patches).flatten(2)  # [B, cluster_dim, n_patches]

        # 计算optimal transport assignment
        scores = self.score(local_patches).flatten(2)  # [B, num_clusters, n_patches]
        assignment = get_matching_probs(scores, self.dust_bin, 3)
        assignment = torch.exp(assignment)

        if self.with_dustbin:
            assignment = assignment[:, :-1, :]  # 移除dustbin列

        # SALAD聚合：每个cluster聚合其对应的features
        # assignment: [B, num_clusters, n_patches]
        # f_local: [B, cluster_dim, n_patches]
        assignment_expanded = assignment.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        # [B, cluster_dim, num_clusters, n_patches]

        f_local_expanded = f_local.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        # [B, cluster_dim, num_clusters, n_patches]

        # 聚合：在n_patches维度上求和
        f_local_agg = (f_local_expanded * assignment_expanded).sum(dim=-1)
        # [B, cluster_dim, num_clusters]

        f_local_agg = f_local_agg.flatten(1)  # [B, cluster_dim * num_clusters]

        # 投影到目标维度
        # f_base = self.local_projection(f_local_agg)  # [B, base_dim]
        f_base = f_local_agg

        # ===== 步骤2: Global token生成调制参数 =====
        modulation_params = self.film_modulator(global_token)  # [B, base_dim * 2]



        # 切分成gamma和beta
        gamma, beta = torch.chunk(modulation_params, 2, dim=-1)
        # gamma: [B, base_dim], beta: [B, base_dim]

        # ===== 步骤3: FiLM调制 =====
        # f_modulated = gamma ⊙ f_base + beta
        f_modulated = gamma * f_base + beta  # [B, base_dim]

        # ===== 步骤4: 最终归一化 =====
        f_final = F.normalize(f_modulated, p=2, dim=-1)

        return f_final
