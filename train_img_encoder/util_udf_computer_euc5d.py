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
        self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=norm_processor.device)
        # 对应: [NR, NC, Cos, Sin, LogScale]

    def compute_udf(self, q_coords_raw_4d, ref_coords_raw_4d):
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