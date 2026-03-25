import torch
from torch import nn
import torch.nn.functional as F
import math

class WeightedDirichletEnergyLoss(nn.Module):
    def __init__(self, apply_log=False):
        """
        [Weighted Dirichlet Energy Loss]
        计算加权迪利克雷能量，用于约束流形的平滑性。

        物理意义：
        衡量特征场在正样本（同质）区域的震荡程度。
        L_DE = sum( w_ij * ||f_i - f_j||^2 )
        在这里简化为针对 Anchor 的距离形式：
        L_DE = mean( pos_weight * feat_dist )

        Args:
            apply_log (bool): 是否对能量取 log (防止数值过大，类似 Log-Barrier)。
        """
        super().__init__()
        self.apply_log = apply_log
        self.inf_val = 1e9

    def forward(self, feat_mat, pos_weight, neg_weight=None):
        """
        Args:
            feat_mat: 特征距离矩阵 [B, N] (通常是 dist(anchor, others))
            pos_weight: 正样本权重 [B, N] (表示两点是同质邻居的概率/置信度)
            neg_weight: 负样本权重 (本Loss通常只关注正样本平滑性，这个参数主要为了接口对齐，可忽略)

        Returns:
            loss_de: 标量损失
        """
        # =============================================================
        # 1. 明确目标：只优化正样本的平滑性
        # =============================================================
        # Dirichlet Energy 的定义是针对"相连边"的。
        # 在我们的定义里，pos_weight > neg_weight (或者 pos_weight > threshold) 的点才算"相连"。

        # 为了更稳健，我们可以使用软权重直接计算，或者先做 Mask
        # 方案 A: 全局软加权 (Soft Weighted Energy) - 推荐
        # 能量 = 距离 * 连接强度(pos_weight)
        # 如果 pos_weight 很大（确定是邻居）且 dist 很大（特征不平滑），则能量极高 -> 惩罚。

        # 考虑到 feat_mat 可能是距离，Dirichlet Energy 通常是距离的平方，
        # 但如果是 Metric Learning，直接优化距离(L1/L2)也是平滑约束。
        # 这里假设 feat_mat 已经是距离度量。

        energy_map = feat_mat * torch.clamp(pos_weight-neg_weight, min=0)

        # =============================================================
        # 2. 过滤掉负样本区域
        # =============================================================
        # 我们不希望优化负样本的距离（负样本本来就该远）。
        # 如果不传 neg_weight，我们假设 pos_weight 本身已经包含了结构信息（负样本处趋近0）。
        # 如果传了 neg_weight，我们可以用 mask 显式过滤。

        if neg_weight is not None:
            # 只在 (正权重 > 负权重) 的区域计算平滑性
            # 这是一个"稀疏图"的假设：只平滑我们认为真正相连的边
            is_connected = pos_weight > neg_weight

            # 使用 mask 过滤：不相连的地方能量视为 0 (不惩罚)
            energy_map = energy_map * is_connected.float()

            # 归一化分母：只除以有效边的数量，防止被大量 0 拉低 Loss
            num_edges = is_connected.sum().clamp(min=1.0)
            mean_energy = energy_map.sum() / num_edges
        else:
            # 如果没有负样本信息，就全局平均 (依赖 pos_weight 的稀疏性)
            mean_energy = energy_map.mean()

        # =============================================================
        # 3. 计算最终 Loss
        # =============================================================
        if self.apply_log:
            # log(1 + E) 形式，对异常值不敏感，梯度更平滑
            loss_de = torch.log1p(mean_energy)
        else:
            loss_de = mean_energy

        return loss_de

