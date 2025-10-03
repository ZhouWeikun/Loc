import math
import torch
import torch.nn as nn
import numpy as np

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
# todo:needs to finish
def get_matching_probs_modified(S, with_distbin=True,dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    if with_distbin:
        # augment scores matrix
        S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        S_aug[:, m, :] = dustbin_score
        # prepare normalized source and target log-weights
        norm = -torch.tensor(math.log(n + m), device=S.device)
        log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
        log_a[-1] = log_a[-1] + math.log(n - m)
    else:
        # without dustbin
        S_aug = torch.empty(batch_size, m, n, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        # prepare normalized source and target log-weights
        norm = -torch.tensor(math.log(n + m), device=S.device)
        log_a, log_b = norm.expand(m ).contiguous(), norm.expand(n).contiguous()


    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


def log_otp_solver_modified(log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    M = M / reg
    u = torch.zeros_like(log_b)  # 固定u为0
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        # 仅更新v（列归一化）
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs_modified(S, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()
    S_aug = S  # 移除 dustbin 扩展

    # 列约束：log_b 初始化为全0，对应列和为1
    log_b = torch.zeros(n, device=S.device).expand(batch_size, -1)

    log_P = log_otp_solver(
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )

    return log_P


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


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        input_feat_dim (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(self,
                 input_feat_dim=768, #org=1536
                 num_clusters=32, #64
                 cluster_dim=64, #128
                 token_dim=256, #the ouput dim of golbal_token
                 dropout=0.3,
                 img_hw = (224,224),
                 pathchsize = 14,
                 with_dustbin=True,
                 hidden_layer_dim=512,
                 ) -> None:
        super().__init__()

        self.input_feat_dim = input_feat_dim
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.token_hw = (int(img_hw[0]/pathchsize),int(img_hw[1]/pathchsize))
        self.with_dustbin = with_dustbin
        self.hidden_layer_dim = hidden_layer_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        #下面三个networks作用都一样的，即逐像素地将每一个token在channel维度进行非线性压缩
        #token的形状是768*h*w->经过压缩后形状变为?*h*w，但是这个操作的信息交互只在形状为768*1*1的tensor中进行
        #他这个机制其实有点类似qkv：
        # 即输入tokens(一般是256个，维度一般=768)经过self.cluster_features得到的是降维后的tokens(数目不变，维度=cluster_dim)，类比于qkv中的v
        # 而输入tokens经过self.score得到的则是'描述每一个token的聚合权重的attenation_map'，attenation_map有多个(数目=num_clusters，attenation_map展平后的维度=输入token的数目)，对应多个问题下的不同关注点
        # MLP for global scene token g, compress the dim of token
        self.token_features = nn.Sequential(
            nn.Linear(self.input_feat_dim, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dim, self.token_dim)
        )
        # MLP for local features f_i, compress the dim of token
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.input_feat_dim, self.hidden_layer_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.hidden_layer_dim, self.cluster_dim, 1)
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
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        # x, t = x  # Extract features and token
        t = x[:, 0]
        x = x[:, 1:]
        # Reshape to (B, C, H, W)
        x = x.reshape((x.shape[0], self.token_hw[0],  self.token_hw[1], self.input_feat_dim)).permute(0, 3, 1, 2)

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p,self.dust_bin,3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :] if self.with_dustbin else p #这个地方可以可视化热力图？ [n_batch,num_clusters,num_tokens]
        #debug:
        # from pytorch_metric_learning.distances import DotProductSimilarity
        # test_id = 5
        # cossim_computer = DotProductSimilarity(normalize_embeddings=True)
        # simmat = cossim_computer(p[test_id],p[test_id]).detach().cpu().numpy()
        # import numpy as np
        # std = np.sqrt( ((simmat-simmat.mean())**2).mean() )

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)  #aggerate the feats from all img_patches for every cluster
        ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)