import torch
from torch import nn
import torch.nn.functional as F
import math

# 命名规则：
# 1. 前缀：
#    - pairLoss: 显式区分正负项，forward 返回 (loss_pos, loss_neg)。
#    - tripleLoss: 不再区分正负项，forward 只返回一个总 loss。
# 2. 边连接形式：
#    - singleEdge: 矩阵每一行最终只保留一个 hardest positive 和一个 hardest negative。
#    - multiEdge: 矩阵每一行会聚合一组 hard edges。
# 3. 聚合/挖掘方式：
#    - hardest: 直接用最难边。
#    - weightedHardest: hardest 边的筛选或惩罚显式受连续权重调制。
#    - logSum: 对一组 hard edges 做 logsumexp 聚合。
# 4. 可选后缀：
#    - _fm_weight / _fm_mask: 表示监督信号来自 weight 还是 mask 接口。
#
# Loss 分类维度约定：
# 1. 有无权重：指连续权重是否真正进入最终聚合，而不是仅用于生成 pos/neg 区域。
# 2. 是否分 loss_pos 和 loss_neg：分开返回即为 pairLoss，否则为 tripleLoss。
# 3. 是否是 triple 形式：若矩阵每一行最终只保留一个 hardest positive 和一个 hardest negative，
#    则记为 single-hard-pair；否则记为 multi-edge。
#
# 本文件收纳的 loss：
# - tripleLoss_singleEdge_hardest_fm_weight: mask-only / coupled / single-hard-pair
#   说明：虽然输入是 pos_weight, neg_weight，但它们只用于生成 mask，不参与最终聚合加权。
# - HardTripleLoss_fm_mask: mask-only / coupled / single-hard-pair

def pos_inf(dtype):
    return torch.finfo(dtype).max

def neg_inf(dtype):
    return torch.finfo(dtype).min


###############################################################################################
############################ 1. single-hard-pair + mask-only ###########################################
class tripleLoss_singleEdge_hardest_fm_weight(nn.Module):
    # tripleLoss_singleEdge_hardest_fm_weight
    # HardTripleLoss 虽然输入是 pos_weight/neg_weight，但它们只用于生成正负区域，不参与最终 loss 加权，所以仍归为 mask-only。
    def __init__(self, beta=5.0, margin=0.0, learnable_beta=True):
        """
        极简版统一对比损失
        逻辑：
        直接优化最难样本对的相对距离：
        Loss = Softplus( beta * (d_pos_hard - d_neg_hard + margin) )
        Args:
            beta (float): 初始 beta 系数 (inverse temperature)。
                          越大 Loss 越陡峭，对违规越敏感。
            margin (float): 安全间隔 m。
                            目标是 d_pos_hard + m < d_neg_hard
        """
        super().__init__()
        self.margin = margin
        self.inf_val = 1e9

        # 保留 log_alpha / alpha 作为 state_dict key，避免破坏旧 checkpoint 兼容性。
        self.learnable_beta = learnable_beta
        if self.learnable_beta:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(beta)))
        else:
            self.register_buffer('alpha', torch.tensor(beta))

    def forward(self, feat_dist_mat, pos_weight, neg_weight, enable_row_filter=True):
        # 1. 获取 beta
        if self.learnable_beta:
            beta = self.log_alpha.exp().clamp(min=1.0, max=100.0)
        else:
            beta = self.alpha

        # ============================================================
        # 2. 挖掘几何边界 (Hard Mining)
        # ============================================================
        # 仅利用权重生成 Mask
        pos_mask = pos_weight > neg_weight
        neg_mask = ~pos_mask

        pos_mat = feat_dist_mat.clone()
        neg_mat = feat_dist_mat.clone()

        # 找最难的正样本 (Max)
        pos_mat[neg_mask] = -self.inf_val
        d_pos_hard, _ = torch.max(pos_mat, dim=-1)  # [B]

        # 找最难的负样本 (Min)
        neg_mat[pos_mask] = self.inf_val
        d_neg_hard, _ = torch.min(neg_mat, dim=-1)  # [B]

        # ============================================================
        # 3. 基础有效性检查 (Basic Validity)
        # ============================================================
        # 排除掉全是正样本或全是负样本的行
        has_neg = d_neg_hard < self.inf_val
        has_pos = d_pos_hard > -self.inf_val
        valid_rows_mask = has_neg & has_pos

        if valid_rows_mask.sum() == 0:
            # 保持零损失与当前 batch 的计算图相连，避免 AMP 在 step() 时发现
            # optimizer 没有任何梯度可检查而直接报错。
            return feat_dist_mat.sum() * 0.0 + beta * 0.0

        # ============================================================
        # 4. 行过滤控制 (Variable Control)
        # ============================================================

        if enable_row_filter:
            # === 模式 A: 开启过滤 (模拟 WeightedMax) ===
            # 只保留违规的行: (d_pos + m) > d_neg
            is_hard_row = (d_pos_hard + self.margin) > d_neg_hard

            # 最终 Mask = 既要是有效数据行，又要是困难行
            target_mask = valid_rows_mask & is_hard_row

            if target_mask.sum() == 0:
                # 这一批没有违规 hard row 时，返回图内零损失而不是孤立常数。
                return feat_dist_mat.sum() * 0.0 + beta * 0.0

            # 切片取值
            d_pos_final = d_pos_hard[target_mask]
            d_neg_final = d_neg_hard[target_mask]

        else:
            # === 模式 B: 关闭过滤 (模拟 SWTLoss) ===
            # 保留所有包含正负样本的行，无论 Loss 大小
            target_mask = valid_rows_mask

            # 切片取值
            d_pos_final = d_pos_hard[target_mask]
            d_neg_final = d_neg_hard[target_mask]

        # ============================================================
        # 5. 计算 Loss
        # ============================================================
        # diff > 0 代表违规，diff < 0 代表安全
        diff = (d_pos_final + self.margin) - d_neg_final

        # Softplus: log(1 + exp(beta * diff))
        # 注意: 即使 diff < 0 (安全), Softplus 也会产生一个微小的正值 (loss > 0)
        #
        # 关键差异分析:
        # - 如果 enable_row_filter=True: 只计算 diff > 0 的均值。分母小，梯度大。
        # - 如果 enable_row_filter=False: 计算所有均值。
        #   那些 diff < 0 的安全行，虽然 Loss 很小，但不是 0。
        #   它们会产生一个极其微弱的梯度，试图把 d_neg 推得更远 (Global Repulsion)。
        #   同时分母变大 (Batch Size)，导致整体梯度数值变小 (Gradient Dilution)。

        loss = F.softplus(beta * diff).mean()

        return loss


class tripleLoss_singleEdge_hardest_fm_mask(nn.Module):
    # 目前基本是 HardTripleLoss:51 的旧版、简化版、未收口
    # todo:有待实验和完善
    def __init__(self, dtype=torch.float32, slope_pos=5,slope_neg=5,decoupling=False, **kwargs):
        super(tripleLoss_singleEdge_hardest_fm_mask, self).__init__()
        self.register_buffer("pos_ignore", torch.tensor(pos_inf(dtype), dtype=dtype))
        self.register_buffer("neg_ignore", torch.tensor(neg_inf(dtype), dtype=dtype))
        self.alpha = slope_pos
        self.beta = slope_neg
        self.decoupling = decoupling
        self.weight_dist_func = lambda x:1/(1+torch.exp(-8.5*(x-0.15)))

    def forward(self, feat_mat,
                pos_mask,
                metric='dist',
                w_weight = False,
                ):

        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()

        if metric == 'dist':
            # hard minning:
            pos_mat[neg_mask] = self.neg_ignore  # fill the neg with -inf
            neg_mat[pos_mask] = self.pos_ignore  # fill the pos with +inf

            pos_most_hard_per_row = torch.max(pos_mat,dim=-1)  # find the maximum distance of positive value in every row
            neg_most_hard_per_row = torch.min(neg_mat,dim=-1)  # find the minimum distance of negative value in every row

            if not self.decoupling:
                if not w_weight:
                    return torch.log(1 + torch.exp( self.alpha * (pos_most_hard_per_row[0] - neg_most_hard_per_row[0]))).mean()  # (dist_hardp - dist_hardn) large -> loss large
        else:
            raise NotImplementedError("This function has not been implemented yet, baby is looking forward to it!")
