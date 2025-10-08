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


class SALAD_Residual(nn.Module):
    def __init__(self,
                 input_feat_dim=768,
                 num_clusters=32,
                 cluster_dim=64,
                 base_dim=256,  # 对应hash table输出
                 dropout=0.3,
                 img_hw=(224, 224),
                 patchsize=14,
                 with_dustbin=True,
                 hidden_layer_dim=512,
                 use_residual=True,  # 新增：是否使用残差结构
                 ):
        super().__init__()

        self.input_feat_dim = input_feat_dim
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.hidden_layer_dim = hidden_layer_dim
        self.token_hw = (int(img_hw[0]/patchsize),int(img_hw[1]/patchsize))
        self.with_dustbin = with_dustbin
        self.use_residual = use_residual
        self.base_dim = base_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # 修改：global token变为base feature
        self.base_features = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, base_dim)
        )

        # 修改：local聚合变为residual/modulation
        self.residual_features = nn.Sequential(
            nn.Conv2d(input_feat_dim, hidden_layer_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(hidden_layer_dim, cluster_dim, 1)
        )

        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.input_feat_dim, self.hidden_layer_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.hidden_layer_dim, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(0.)) #todo:改为0初始化看看,org=1.

    def forward(self, x):
        t = x[:, 0]
        x = x[:, 1:]
        x = x.reshape((x.shape[0], self.token_hw[0], self.token_hw[1],
                       self.input_feat_dim)).permute(0, 3, 1, 2)

        # 1. 基础特征（对应hash table输出）
        base = self.base_features(t)  # [B, base_dim]

        # 2. 高频调制特征（对应MLP输出）
        residual = self.residual_features(x).flatten(2)  # [B, cluster_dim, num_tokens]
        p = self.score(x).flatten(2)

        # Sinkhorn
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :] if self.with_dustbin else p

        # 聚合残差
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        residual = residual.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        residual_agg = (residual * p).sum(dim=-1).flatten(1)  # [B, cluster_dim*num_clusters]

        if self.use_residual:
            # 方案1: 残差结构（推荐）
            # base是低频，residual是高频修正
            base_expanded = base.unsqueeze(-1).expand(-1, -1,
                                                      self.num_clusters * self.cluster_dim // self.base_dim)
            base_expanded = base_expanded.reshape(base.shape[0], -1)

            # 确保维度匹配
            if base_expanded.shape[1] != residual_agg.shape[1]:
                # 用一个线性层调整base维度
                if not hasattr(self, 'base_projection'):
                    self.base_projection = nn.Linear(base.shape[1],
                                                     residual_agg.shape[1]).to(base.device)
                base_expanded = self.base_projection(base)

            # feat = base_expanded + residual_agg  # 残差相加
            feat = F.normalize(base_expanded, p=2, dim=-1) + F.normalize(residual_agg,p=2, dim=-1)  # 残差相加
            feat = F.normalize(feat, p=2, dim=-1)

            #debug, analyse the feat
            # part_global = nn.functional.normalize(base_expanded, p=2, dim=-1)
            # part_local = nn.functional.normalize(residual_agg, p=2, dim=-1)
            # global_mean = torch.mean(part_global).item()
            # global_var = torch.var(part_global).item()
            # local_mean = torch.mean(part_local).item()
            # local_var = torch.var(part_local).item()
            # var_per_dim_global = torch.var(part_global, dim=0)
            # var_per_dim_loacl = torch.var(part_local, dim=0)
            # var_of_var_global = torch.var(torch.var(part_global, dim=0), dim=-1)
            # var_of_var_local = torch.var(torch.var(part_local, dim=0), dim=-1)
            # global_scale = torch.norm(base_expanded, p=2, dim=-1).mean()
            # local_scale = torch.norm(residual_agg, p=2, dim=-1).mean()
        else:
            # 方案2: 直接拼接（退化为原始SALAD）
            feat = torch.cat([
                F.normalize(base, p=2, dim=-1),
                F.normalize(residual_agg, p=2, dim=1)
            ], dim=-1)
            feat = F.normalize(feat, p=2, dim=-1)

        return feat

