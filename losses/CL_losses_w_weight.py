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
# Loss 分类维度约定：
# 1. 有无权重：指连续权重是否真正进入最终聚合或 hardest 边的打分。
# 2. 是否分 loss_pos 和 loss_neg：分开返回即为 pairLoss，否则为 tripleLoss。
# 3. 是否是 triple 形式：若矩阵每一行最终只保留一个 hardest positive 和一个 hardest negative，
#    则记为 single-hard-pair；否则记为 multi-edge。
# 现在这几类函数的核心不是“某个 pair 本身的绝对相似度/距离好不好”，而是正负 pair 之间的相对差值；用MS语境来说，即不是“self-similarity”而是“relative similarity”


class pairLoss_singleEdge_hardest(nn.Module):
    def __init__(self, beta=10.0, margin=0.0):
        """
        HardTripleLoss的加权版，相较于HardTripleLoss,它会根据权重更优雅地拉近或推开样本，它在优先拉经和推远谁上有辨识度，而HardTripleLoss没有。
        只关注每行最难的一个正样本和最难的一个负样本。
        相当于 Softplus(Max_Diff)。
        特点：
        1. 极快：过滤行后只处理 [B, 1] 的向量。
        2. 锐利：强制拉回最远的正样本，推开最近的负样本。
        3. 无权重：只关注几何距离，忽略 pos/neg_weight。
        """
        super().__init__()
        self.beta = beta
        self.margin = margin
        self.inf_val = 1e9

    def forward(self, feat_mat, pos_weight, neg_weight, metric='dist'):
        """
        逻辑：它不仅用权重生成 Mask，还利用 torch.gather 把那个最难样本对应的具体权重值 (w_pos_hard, w_neg_hard) 提取出来。
        行为：Loss = Softplus(beta * diff * weight).
        效率：内置了 Early Row Filtering,如果一行数据已经满足了安全几何边界，代码会直接跳过该行的后续计算
        """
        if metric != 'dist': raise ValueError("Only 'dist' supported now.")

        # =============================================================
        # 1. Masking & Bounds Calculation
        # =============================================================
        pos_mask = pos_weight > neg_weight
        neg_mask = ~pos_mask

        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()
        pos_mat[neg_mask] = -self.inf_val
        neg_mat[pos_mask] = self.inf_val

        # [修改点 1] 同时获取 Value 和 Indices
        # indices 用于后续去查找对应的权重
        d_pos_hard, ind_pos_hard = torch.max(pos_mat, dim=-1, keepdim=True)  # [B, 1]
        d_neg_hard, ind_neg_hard = torch.min(neg_mat, dim=-1, keepdim=True)  # [B, 1]

        # [修改点 2] 使用 gather 提取对应的权重
        # 我们只关心那个"最难点"的权重，其他的忽略
        w_pos_hard = torch.gather(pos_weight, dim=1, index=ind_pos_hard)  # [B, 1]
        w_neg_hard = torch.gather(neg_weight, dim=1, index=ind_neg_hard)  # [B, 1]

        # [Safety Checks]
        has_neg_context = d_neg_hard < self.inf_val
        has_pos_context = d_pos_hard > -self.inf_val

        d_neg_boundary_safe = torch.where(has_neg_context, d_neg_hard, torch.tensor(2.0, device=feat_mat.device))
        d_pos_boundary_safe = torch.clamp(d_pos_hard, min=0.0)

        # =============================================================
        # 2. Early Row Filtering
        # =============================================================
        is_hard_row = (d_pos_boundary_safe + self.margin) > d_neg_boundary_safe
        target_row_mask = is_hard_row.squeeze(1) & has_pos_context.squeeze(1) & has_neg_context.squeeze(1)

        if target_row_mask.sum() == 0:
            zero = feat_mat.sum() * 0.0
            return zero, zero

        # =============================================================
        # 3. Data Slicing
        # =============================================================
        # 切片距离边界
        d_pos_b_sub = d_pos_boundary_safe[target_row_mask]
        d_neg_b_sub = d_neg_boundary_safe[target_row_mask]

        # 切片对应的权重
        w_pos_b_sub = w_pos_hard[target_row_mask]
        w_neg_b_sub = w_neg_hard[target_row_mask]

        # =============================================================
        # 4. 计算正样本损失 (Weighted Single Hardest Positive)
        # =============================================================
        # Max版本逻辑: Softplus(beta * diff * weight)
        # 权重在非线性函数内部起作用：如果权重很小，Softplus 就会接近 0.69 (log2) 或平缓区域
        # 如果 diff > 0 (违规)，weight 越大惩罚越重
        dist_diff_pos = d_pos_b_sub - (d_neg_b_sub - self.margin)
        loss_pos = F.softplus(self.beta * dist_diff_pos * w_pos_b_sub).mean() / self.beta

        # =============================================================
        # 5. 计算负样本损失 (Weighted Single Hardest Negative)
        # =============================================================
        dist_diff_neg = (d_pos_b_sub + self.margin) - d_neg_b_sub
        loss_neg = F.softplus(self.beta * dist_diff_neg * w_neg_b_sub).mean() / self.beta

        return loss_pos, loss_neg


class pairLoss_singleEdge_weightedHardest(nn.Module):
    def __init__(self, beta=10.0, margin=0.0, learnable_beta=True, **kwargs):
        """
        SoftMultiSimLoss_Max的加强版，同时考虑几何距离和特征空间距离
        Weighted Max Mining:
           Target = argmax( (Dist - Wall) * Weight )
           既不盲目选几何最远(抗噪)，也不盲目选权重最高(抗易样本)。
        """
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")
        self.margin = margin
        self.inf_val = 1e9

        # 保留 log_beta / fixed_beta 作为 state_dict key，避免破坏旧 checkpoint 兼容性。
        self.learnable_beta = learnable_beta
        if self.learnable_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        else:
            self.register_buffer('fixed_beta', torch.tensor(beta))

    def get_current_beta(self):
        if self.learnable_beta:
            beta = self.log_beta.exp().clamp(min=1.0, max=100.0)
        else:
            beta = self.fixed_beta
        return beta

    def forward(self, feat_mat, pos_weight, neg_weight, metric='dist'):
        if metric != 'dist': raise ValueError("Only 'dist' supported now.")
        # 获取当前的 beta 值
        beta = self.get_current_beta()

        # =============================================================
        # 1. Masking & Raw Bounds (确定几何边界墙)
        # =============================================================
        pos_mask = pos_weight > neg_weight
        neg_mask = ~pos_mask

        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()
        pos_mat[neg_mask] = -self.inf_val
        neg_mat[pos_mask] = self.inf_val

        # [关键] 墙（Boundary）依然必须基于"纯几何距离"
        # 为什么？因为如果墙也是加权的，物理意义就乱了。
        # 我们需要知道"客观上"正负样本的边界在哪里。
        d_pos_hard_raw, _ = torch.max(pos_mat, dim=-1, keepdim=True)  # [B, 1]
        d_neg_hard_raw, _ = torch.min(neg_mat, dim=-1, keepdim=True)  # [B, 1]

        has_neg_context = d_neg_hard_raw < self.inf_val
        has_pos_context = d_pos_hard_raw > -self.inf_val

        d_neg_boundary_safe = torch.where(has_neg_context, d_neg_hard_raw, torch.tensor(2.0, device=feat_mat.device))
        d_pos_boundary_safe = torch.clamp(d_pos_hard_raw, min=0.0)

        # =============================================================
        # 2. Early Row Filtering
        # =============================================================
        # 只要客观几何上存在违规，这行就需要处理
        is_hard_row = (d_pos_boundary_safe + self.margin) > d_neg_boundary_safe
        target_row_mask = is_hard_row.squeeze(1) & has_pos_context.squeeze(1) & has_neg_context.squeeze(1)

        if target_row_mask.sum() == 0:
            zero = feat_mat.sum() * 0.0
            return zero, zero

        # =============================================================
        # 3. Data Slicing (切片保留 [K, N] 维度)
        # =============================================================
        # 为了进行行内排序/择优，我们需要保留行内的所有像素信息
        feat_sub = feat_mat[target_row_mask]  # [K, N]
        pos_weight_sub = pos_weight[target_row_mask]  # [K, N]
        neg_weight_sub = neg_weight[target_row_mask]  # [K, N]
        pos_mask_sub = pos_mask[target_row_mask]  # [K, N]
        neg_mask_sub = neg_mask[target_row_mask]  # [K, N]

        d_pos_b_sub = d_pos_boundary_safe[target_row_mask]  # [K, 1]
        d_neg_b_sub = d_neg_boundary_safe[target_row_mask]  # [K, 1]

        # =============================================================
        # 4. 正样本挖掘 (Weighted Max Mining)
        # =============================================================
        # 逻辑：找出 (diff * weight) 最大的那个样本
        # A. 计算所有像素相对于"几何墙"的违规程度
        # [K, N] - [K, 1] -> [K, N]
        raw_diff_pos = feat_sub - (d_neg_b_sub - self.margin)
        # B. 加权违规分
        # score > 0 代表违规且有权重。如果 diff < 0 (安全)，score 也会是负的，自然会被 max 忽略
        weighted_score_pos = raw_diff_pos * pos_weight_sub
        # C. Masking
        # 把非正样本区域的分数设为 -inf，确保它们不会被选中
        weighted_score_pos[~pos_mask_sub] = -self.inf_val
        # D. 核心：行内择优 (Row-wise Max)
        # 这一步就是在做"预计算所有困难正样本的loss"并取最大
        max_weighted_diff_pos, _ = torch.max(weighted_score_pos, dim=-1)  # [K]
        # E. 计算 Loss
        # 这里直接对 max 出来的分数做 Softplus
        loss_pos = F.softplus(beta * max_weighted_diff_pos).mean() / beta

        # =============================================================
        # 5. 负样本挖掘 (Weighted Max Mining)
        # =============================================================
        # 逻辑同上

        # A. 计算违规程度
        raw_diff_neg = (d_pos_b_sub + self.margin) - feat_sub
        # B. 加权
        weighted_score_neg = raw_diff_neg * neg_weight_sub
        # C. Masking
        weighted_score_neg[~neg_mask_sub] = -self.inf_val
        # D. 行内择优
        max_weighted_diff_neg, _ = torch.max(weighted_score_neg, dim=-1)  # [K]
        # E. 计算 Loss
        loss_neg = F.softplus(beta * max_weighted_diff_neg).mean() / beta

        return loss_pos, loss_neg


class pairLoss_multiEdge_logSum(nn.Module):
    def __init__(self, beta=10.0, margin=0.1, learnable_beta=True,
                 base_pull_strength=0.1, pull_mode='logsum', **kwargs):
        """
        [Advanced 4D Neural Field Metric Loss]

        结合了 Multi-Similarity Loss 的加权挖掘机制与 Dirichlet Energy (DE) 的场平滑思想。
        专为无人机 4D 视觉定位 (x, y, scale, direction) 设计，旨在解决无限精度场拟合中的
        “边界撕裂”与“梯度消失”问题。

        Args:
            beta (float):
                Sigmoid/Softplus 的逆温度系数 (Inverse Temperature)。
                β 越大，Loss 对困难样本 (Hard Examples) 越敏感，近似于 Max/Min 操作；
                β 越小，Loss 越平滑，关注整体分布。

            margin (float):
                几何安全边界 m。我们期望：d_neg > d_pos + m。
                在 4D 空间中，这相当于在正确位姿周围建立一个半径为 m 的“绝对安全球”。

            learnable_beta (bool):
                是否将 β 设为可学习参数。
                - 建议 True: 让网络在训练初期（低 β）关注全局收敛，后期（高 β）关注边界精修。

            base_pull_strength (float):
                [DE 能量项核心参数] 基础拉力系数 (0.0 ~ 1.0)。
                即使正样本已经处于“安全区”（误差很小），是否仍保留一定的梯度拉力？
                - 作用：模拟狄利克雷能量 (Dirichlet Energy)，维持特征场的“张力”，防止场在安全区内
                  变得松弛或平坦（Zero Gradient），从而保证亚像素优化的连续性。

            pull_mode (str): 正样本拉近策略
                - 'logsum': [能量最小化视角]
                  挖掘所有违规的正样本像素，利用 LogSumExp 压制整体违规能量。
                  适合：连续场拟合，使整个正样本区域的特征分布更均匀平滑。
                - 'hardest': [几何约束视角]
                  仅锚定最远（最差）的正样本点。
                  适合：早期训练，强制将离群点拉回，建立 UDF 的基本骨架。
        """
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")
        self.margin = margin
        self.base_pull_strength = base_pull_strength
        self.pull_mode = pull_mode
        self.inf_val = 1e9

        # 保留 log_beta / fixed_beta 作为 state_dict key，避免破坏旧 checkpoint 兼容性。
        self.learnable_beta = learnable_beta
        if self.learnable_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        else:
            self.register_buffer('fixed_beta', torch.tensor(beta))

    def get_beta(self):
        """获取当前 beta 系数，限制在 [1.0, 100.0] 防止数值溢出"""
        if self.learnable_beta:
            return self.log_beta.exp().clamp(min=1.0, max=100.0)
        return self.fixed_beta

    def forward(self, feat_mat, pos_weight, neg_weight, metric='dist'):
        """
        Args:
            feat_mat: [B, N] 特征距离矩阵 (通常是 L2 Distance 或 decoded SDF value)。
            pos_weight: [B, N] 正样本权重 (Soft Mask，指示正样本区域)。
            neg_weight: [B, N] 负样本权重 (Soft Mask，指示负样本区域)。
            metric: 'dist' (距离模式) 或 'sim' (相似度模式)。
        """

        # =====================================================================
        # 0. Metric Compatibility Check (接口预留)
        # =====================================================================
        # 虽然目前只用 distance，但保留接口以便未来切换到 Cosine Similarity
        if metric == 'sim':
            # 如果是相似度，逻辑需反转：sim_pos > sim_neg + margin
            # 且 LogSumExp 的符号也需要调整。目前暂未实现。
            raise NotImplementedError(
                "Metric 'sim' (Similarity) is not implemented yet. "
                "Current 4D Neural Field logic relies on UDF-like distance minimization. "
                "Please use metric='dist'."
            )
        elif metric != 'dist':
            raise ValueError(f"Unknown metric: {metric}. Use 'dist'.")

        beta = self.get_beta()

        # =====================================================================
        # 1. Geometry Mining & Bounds (几何边界挖掘)
        # =====================================================================
        # 即使是 Soft Loss，我们也需要知道当前的“硬边界”在哪里，以便进行过滤
        pos_mask = pos_weight > neg_weight
        neg_mask = ~pos_mask

        # 填充 Inf 以便计算 min/max
        d_pos_raw = feat_mat.clone().masked_fill_(neg_mask, -self.inf_val)
        d_neg_raw = feat_mat.clone().masked_fill_(pos_mask, self.inf_val)

        # d_pos_hard: 正样本中最差的点（离得最远，最需要拉回来）
        # d_neg_hard: 负样本中最险的点（离得最近，最需要推出去）
        d_pos_hard, _ = torch.max(d_pos_raw, dim=-1, keepdim=True)
        d_neg_hard, _ = torch.min(d_neg_raw, dim=-1, keepdim=True)

        # =====================================================================
        # 2. Early Row Filtering (计算加速与噪声过滤)
        # =====================================================================
        # 核心逻辑：只有当 (最远正样本 + margin) > 最近负样本 时，该样本对才存在风险。
        # 那些已经完美分离（Safe）的样本行，不应参与梯度计算（避免过拟合简单样本）。

        has_context = (d_pos_hard > -self.inf_val) & (d_neg_hard < self.inf_val)
        is_unsafe = (d_pos_hard + self.margin) > d_neg_hard

        # 最终有效的行掩码
        row_mask = (has_context & is_unsafe).squeeze(1)

        # [Early Exit] 如果所有数据都非常安全，直接返回 0 Loss
        if not row_mask.any():
            zero = feat_mat.sum() * 0.0
            return zero, zero

        # [Data Slicing] 只取出这就需要优化的子集
        feat_sub = feat_mat[row_mask]
        pw_sub = pos_weight[row_mask];
        nw_sub = neg_weight[row_mask]
        pm_sub = pos_mask[row_mask];
        nm_sub = neg_mask[row_mask]
        d_ph_sub = d_pos_hard[row_mask];
        d_nh_sub = d_neg_hard[row_mask]

        # =====================================================================
        # 3. Positive Pulling with DE Philosophy (正样本拉近)
        # =====================================================================
        # 目标：最小化正样本距离。
        # 区别在于我们是想拉动“最远的一个点”还是“所有偏离的点”。

        if self.pull_mode == 'hardest':
            # --- Mode A: Hardest Anchor (几何强约束) ---
            # 思想：只要把最远的点拉回来，中间的点自然也就回来了。
            # 这类似于把一张布的四个角钉死。

            # 计算违规程度：正样本是否跑到了负样本的“警戒线”外？
            diff_pos = d_ph_sub - (d_nh_sub - self.margin)

            # 动态权重：违规越严重，拉力越大
            dynamic_weight = torch.sigmoid(beta * diff_pos)

            # [DE 思想体现]: base_pull_strength
            # 即使 dynamic_weight 接近 0 (样本安全)，我们依然保留 base_pull_strength 的拉力。
            # 这就像狄利克雷能量中的“弹性势能”，保证场始终有收缩趋势，不会松弛。
            scale = self.base_pull_strength + (1.0 - self.base_pull_strength) * dynamic_weight
            loss_pos = (d_ph_sub * scale).mean()

        else:
            # --- Mode B: LogSumExp Energy (场平滑优化) ---
            # 思想：正样本区域的每一个像素都不应该偏离。
            # 这类似于最小化整个膜的表面张力。

            # 1. 计算所有像素相对于“负样本边界”的偏离量
            diff_pos = feat_sub - (d_nh_sub - self.margin)

            # 2. 掩码：只计算那些真正违规（diff > 0）的正像素
            v_mask_pos = pm_sub & (diff_pos > 0)

            # 3. LogSumExp 聚合所有微小的违规能量
            loss_pos = self._logsumexp_loss(diff_pos * pw_sub, v_mask_pos, beta)

        # =====================================================================
        # 4. Negative Pushing (负样本推离)
        # =====================================================================
        # 目标：最大化负样本距离（使其 > d_pos_hard + margin）。
        # 这里统一使用 LogSumExp，因为任何一个入侵的负样本都是危险的。

        # 计算入侵量：负样本是否进入了正样本的“安全圈”？
        diff_neg = (d_ph_sub + self.margin) - feat_sub

        # 掩码：只关注那些确实入侵了（diff > 0）的负像素
        v_mask_neg = nm_sub & (diff_neg > 0)

        loss_neg = self._logsumexp_loss(diff_neg * nw_sub, v_mask_neg, beta)

        return loss_pos, loss_neg

    def _logsumexp_loss(self, weighted_diff, mask, beta):
        """
        LogSumExp 核心算子 - 实现 Softplus 风格的能量最小化
        Formula: L = (1/beta) * log( sum( exp(beta * diff) ) + 1 )
        为什么要 +1 (Zero Padding)?
        1. 物理含义：如果没有违规样本 (全 -inf)，log(0 + 1) = 0，Loss 完美归零。
        2. 梯度特性：当 diff 很大时，梯度接近 1 (线性惩罚)；当 diff 很小时，梯度平滑衰减。
        """
        logits = beta * weighted_diff
        # 将非掩码区域设为 -inf，使其在 exp 后为 0，不影响 sum
        logits_masked = logits.masked_fill(~mask, -self.inf_val)

        # 拼接 0，对应公式中的 "+ 1"
        zeros = torch.zeros((logits.size(0), 1), device=logits.device)
        concat = torch.cat([logits_masked, zeros], dim=1)

        return (1.0 / beta) * torch.logsumexp(concat, dim=1).mean()


class SoftWeightedRelativeMSLoss(nn.Module):
    def __init__(self, beta=10.0, margin=0., mining_mode='all', metric='dist'):
        """ todo:待检验，作为baseline Loss作为对比使用
  它是个混合体：
  - mining_mode='all' 时属于 mined-set + logsumexp + weighted
  - mining_mode='max' 时属于 single-hard-pair + weighted

        Soft Weighted Relative Multi-Similarity Loss
        Args:
            beta (float): LogSumExp 的缩放因子 (仅在 mode='all' 时生效)。值越大，越接近 max。
            margin (float): 相对边界值。
                            - metric='dist': d_neg 必须比 d_pos 远 margin (d_neg > d_pos + m)
                            - metric='sim':  s_neg 必须比 s_pos 低 margin (s_neg < s_pos - m)
            mining_mode (str):
                - 'all': 使用 LogSumExp 聚合所有困难样本 (平滑，优化 DE 友好)。
                - 'max': 每行只取加权违规最大的一个样本 (锐利，收敛快)。
            metric (str): 'dist' (距离, 越小越好) 或 'sim' (相似度, 越大越好)。
        """
        super().__init__()
        self.beta = beta
        self.margin = margin
        self.mining_mode = mining_mode
        self.metric = metric

        # 极小值/极大值，用于 masked_fill
        self.inf_val = 1e9

    def forward(self, feat_mat, pos_weight, neg_weight):
        """
        Args:
            feat_mat: (B, N) 特征矩阵 (根据 metric 可能是距离矩阵，也可能是相似度矩阵)
            pos_weight: (B, N) 正样本几何权重 (用于确定动态阈值)
            neg_weight: (B, N) 负样本几何权重 (用于加权惩罚)
        """
        # 定义正rv负样本区域 (基于几何权重)
        # pos_weight > neg_weight 视为正样本区 (Red Cue > Blue Curve)
        is_pos_region = pos_weight > neg_weight
        is_neg_region = neg_weight > pos_weight  # 或者 ~is_pos_region

        # =============================================================
        # 困难样本挖掘+惩罚权重计算
        # =============================================================
        if self.metric == 'dist':
            # =============================================================
            # 分支 A: 基于距离 (Distance) - 值越小越相似
            # =============================================================

            # 1. 确定动态阈值 (Find Hardest Positive)
            # ---------------------------------------------------
            # 我们要找"最远"的正样本作为基准 (Max d_pos)
            pos_dists_masked = feat_mat.clone()
            # 将非正样本区域填为 -inf，使其在 max 中被忽略
            pos_dists_masked[~is_pos_region] = -self.inf_val

            # d_pos_hard: (B, 1)
            d_pos_hard, _ = torch.max(pos_dists_masked, dim=1, keepdim=True)

            # [Safety] 如果某行全是负样本(d_pos_hard=-inf)，将其置0防止NaN (虽不应发生)
            d_pos_hard = torch.clamp(d_pos_hard, min=0.0)

            # 2. 计算违规程度 (Calculate Violation)
            # ---------------------------------------------------
            # 约束: d_neg > d_pos + margin
            # 违规: raw_violation = (d_pos + margin) - d_neg > 0
            raw_violation = (d_pos_hard + self.margin) - feat_mat  #raw_violation>0意味着困难

            # 3. 施加几何软权重 (Apply Soft Weight)
            # ---------------------------------------------------
            # 即使违规了，如果 neg_weight 很小 (物理上很近)，我们轻判
            # weighted_violation = raw_violation * torch.clamp(neg_weight-pos_weight, min=0.0)
            weighted_violation = raw_violation * neg_weight

            # 4. 生成有效掩码
            # 必须是负样本区域 且 确实发生了违规
            valid_violation_mask = is_neg_region & (raw_violation > 0)

        elif self.metric == 'sim':
            # =============================================================
            # 分支 B: 基于相似度 (Similarity) - 值越大越相似 (TODO)
            # =============================================================
            # 预留接口，逻辑与 dist 相反：
            # 1. 找"最低"的正样本相似度 (Min s_pos)
            # 2. 约束: s_neg < s_pos - margin
            # 3. 违规: raw_violation = s_neg - (s_pos - margin) > 0
            raise NotImplementedError("Metric 'sim' is not implemented yet. Baby step first!")

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # =============================================================
        # 聚合 Loss (Aggregation based on Mining Mode)
        # =============================================================

        # 如果没有任何有效的违规样本，直接返回 0
        if valid_violation_mask.sum() == 0:
            return torch.tensor(0.0, device=feat_mat.device, requires_grad=True)

        if self.mining_mode == 'all':
            # --- Mode: LogSumExp (Multi-Similarity Style) ---
            # 这种模式下梯度更平滑，所有违规样本都提供推力

            # 1. 放大梯度
            logits = self.beta * weighted_violation

            # 2. Masking: 将非违规项设为 -inf (exp(-inf)=0)
            logits_masked = logits.masked_fill(~valid_violation_mask, -self.inf_val)

            # 3. Padding with 0 for Softplus effect: log(1 + sum(exp(x)))
            zeros = torch.zeros((logits.size(0), 1), device=logits.device)
            logits_concat = torch.cat([logits_masked, zeros], dim=1)

            # 4. Compute Loss
            loss = (1.0 / self.beta) * torch.logsumexp(logits_concat, dim=1).mean()

        elif self.mining_mode == 'max':
            # --- Mode: Max (Triplet Style) ---
            # 这种模式下每行只惩罚最严重的那个违规

            # 1. Masking: 将非违规项设为 -inf (以便 max 选中最大的 valid violation)
            # 注意: Max 不需要 beta 缩放
            val_masked = weighted_violation.masked_fill(~valid_violation_mask, -self.inf_val)

            # 2. Find Max Violation per row
            max_val, _ = torch.max(val_masked, dim=1)

            # 3. ReLU & Mean
            # 过滤掉那些本来就没有违规样本的行 (即 max 出来是 -inf 的行)
            loss = torch.relu(max_val).mean()

        else:
            raise ValueError(f"Unknown mining_mode: {self.mining_mode}")

        return loss
