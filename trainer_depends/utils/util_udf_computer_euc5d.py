import torch

class UDFComputer(object):
    def __init__(self, norm_processor):
        """
        Args:
            norm_processor: CoordsNormProcessor 的实例
        """
        self.processor = norm_processor

        # 我们可以保留权重，但建议尽量接近 1.0
        # 如果你确实觉得旋转很难学，可以稍微调高旋转的权重，而不是根据距离动态调整
        self.weights = torch.tensor([0.6, 0.6, 0.5, 0.5, 0.3], device=norm_processor.device) #0.72:0.36:0.09->0.6,0.43,0.3
        # 对应: [NR, NC, Cos, Sin, LogScale]

    def compute_udf_fm_4d(self, q_coords_raw_4d, ref_coords_raw_4d):
        """
        计算 Ground Truth UDF 值
        输入是原始物理坐标 [N, 4]
        """
        # 1. 统一转换到 5D 归一化空间
        # [N, 5] -> (nr, nc, cos, sin, log_scale)
        q_5d = self.processor.raw_to_norm(q_coords_raw_4d)
        ref_5d = self.processor.raw_to_norm(ref_coords_raw_4d)

        # 2. 计算加权欧式距离 (Weighted Euclidean Distance)
        diff = q_5d - ref_5d

        # 应用权重 (Broadcasting)
        diff_weighted = diff * self.weights

        # 3. 计算 L2 范数 (即最终距离)
        # dim=-1 保证输出形状为 [N]
        udf_val = torch.norm(diff_weighted, p=2, dim=-1)

        return udf_val

    def compute_udf_fm_normed_5d(self, q_5d, ref_5d):
        """
        [入口 B] 直接从归一化 5D 坐标计算 UDF
        输入: [N, 5] 或 [B, N, 5] -> (nr, nc, cos, sin, log_scale)
        范围: 通常在 [-1, 1] 之间
        """
        # 1. 维度对齐检查 (Broadcasting Support)
        # 如果维度不一致（例如 q是[1, 5], ref是[N, 5]），PyTorch会自动处理，这里无需额外操作

        # 2. 计算加权差值
        diff = q_5d - ref_5d

        # 3. 应用权重
        # self.weights 会自动广播到 [..., 5]
        diff_weighted = diff * self.weights

        # 4. 计算 L2 范数 (欧式距离)
        # dim=-1 保证最后输出形状为 [N] 或 [B, N]
        udf_val = torch.norm(diff_weighted, p=2, dim=-1)

        return udf_val

    def compute_udf_matrix_fm_normed_5d(self, q_5d, ref_5d):
        """
        计算成对距离矩阵
        输入:
            q_5d: [M, 5] 查询坐标（归一化5D）
            ref_5d: [N, 5] 参考坐标（归一化5D）
        输出:
            distance_matrix: [M, N] 距离矩阵
        """
        # 扩展维度以便计算成对距离
        # q_5d: [M, 5] -> [M, 1, 5]
        # ref_5d: [N, 5] -> [1, N, 5]
        q_expanded = q_5d.unsqueeze(1)  # [M, 1, 5]
        ref_expanded = ref_5d.unsqueeze(0)  # [1, N, 5]

        # 计算差值 [M, N, 5]
        diff = q_expanded - ref_expanded

        # 应用权重
        diff_weighted = diff * self.weights  # [M, N, 5]

        # 计算 L2 范数 -> [M, N]
        distance_matrix = torch.norm(diff_weighted, p=2, dim=-1)

        return distance_matrix

    def compute_weight_matrix_fm_4d(self, q_4d, ref_4d, gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale=None):
        q_expanded = q_4d.unsqueeze(1)  # [M, 1, 4]
        ref_expanded = ref_4d.unsqueeze(0)  # [1, N, 4]

        # 计算差值 [M, N, 4]
        diff = q_expanded - ref_expanded

        dist_nrc = torch.norm(diff[...,:2],p=2, dim=-1)
        weight_nrc = torch.exp(-(dist_nrc ** 2) / (2 * gs_sigma_nrc ** 2))

        dist_rot = torch.abs(diff[...,2:3])
        dist_rot = torch.min(dist_rot, 2 * torch.pi - dist_rot)
        weight_radrot = torch.exp(-(dist_rot ** 2) / (2 * gs_sigma_radrot ** 2))

        # 合并权重 (核心步骤)
        combined_weight = weight_nrc * weight_radrot
        return combined_weight



