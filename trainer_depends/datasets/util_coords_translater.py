from codecs import ignore_errors

import torch
import numpy as np


class CoordsNormProcessor:
    """
    坐标空间转换枢纽
    Path: Raw(4D) <-> Linear(4D) -> Net(5D)
    """

    def __init__(self, sat_dataset_instance):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === 1. 位置范围 (NRC) ===
        self.nrc_min = torch.tensor([sat_dataset_instance.nr2sample_min, sat_dataset_instance.nc2sample_min],
                                    device=self.device, dtype=torch.float32)
        self.nrc_max = torch.tensor([sat_dataset_instance.nr2sample_max, sat_dataset_instance.nc2sample_max],
                                    device=self.device, dtype=torch.float32)
        self.nrc_diff = self.nrc_max - self.nrc_min

        # === 2. 尺度范围 (Scale) - 预计算 Log 边界 ===
        s_bnd = sat_dataset_instance.satimgsize_scale_to_refm_boundary
        self.scale_log_min = torch.tensor([np.log(max(s_bnd[0], 1e-6))], device=self.device, dtype=torch.float32)
        self.scale_log_max = torch.tensor([np.log(s_bnd[1])], device=self.device, dtype=torch.float32)
        self.scale_log_diff = self.scale_log_max - self.scale_log_min

    # ============================================================
    # 核心转换逻辑
    # ============================================================

    def raw_to_linear(self, coords_raw):
        """
        [物理层 -> 计算层]
        Input:  [..., 4] (nr, nc, theta_rad, scale_raw)
        Output: [..., 4] (nr_n, nc_n, theta_lin, scale_lin) 都在 [-1, 1]
        """
        if coords_raw.device != self.device:
            coords_raw = coords_raw.to(self.device)

        # 1. NRC: Linear map
        nrc = coords_raw[..., 0:2]
        nrc_n = 2.0 * (nrc - self.nrc_min) / self.nrc_diff - 1.0

        # 2. Theta: Rad [-pi, pi] -> Linear [-1, 1]
        theta = coords_raw[..., 2:3]
        theta_n = theta / torch.pi  # 假设 theta 已经是 -pi~pi

        # 3. Scale: Log map -> Linear [-1, 1]
        scale = coords_raw[..., 3:4]
        # Clamp protect
        scale = torch.clamp(scale, min=torch.exp(self.scale_log_min), max=torch.exp(self.scale_log_max))
        scale_log = torch.log(scale)
        scale_n = 2.0 * (scale_log - self.scale_log_min) / self.scale_log_diff - 1.0

        return torch.cat([nrc_n, theta_n, scale_n], dim=-1)

    def linear_to_net(self, coords_linear):
        """
        [计算层 -> 网络层]
        Input:  [..., 4] (nr_n, nc_n, theta_lin, scale_lin)
        Output: [..., 5] (nr_n, nc_n, cos, sin, scale_lin)
        """
        # 前2维 (NRC) 和 第4维 (Scale) 保持不变
        nrc_n = coords_linear[..., 0:2]
        scale_n = coords_linear[..., 3:4]

        # 第3维 (Theta) 转 Cos/Sin
        theta_lin = coords_linear[..., 2:3]
        theta_rad = theta_lin * torch.pi
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)

        return torch.cat([nrc_n, cos_t, sin_t, scale_n], dim=-1)

    def linear_to_raw(self, coords_linear):
        """
        [计算层 -> 物理层] (通常用于可视化)
        """
        # 1. NRC
        nrc_n = coords_linear[..., 0:2]
        nrc_raw = (nrc_n + 1.0) / 2.0 * self.nrc_diff + self.nrc_min

        # 2. Theta
        theta_n = coords_linear[..., 2:3]
        theta_raw = theta_n * torch.pi

        # 3. Scale
        scale_n = coords_linear[..., 3:4]
        scale_log = (scale_n + 1.0) / 2.0 * self.scale_log_diff + self.scale_log_min
        scale_raw = torch.exp(scale_log)

        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=-1)

    def raw_to_net(self, coords_raw_4d, append_linear_rot=False):
        """
        Input:  [..., 4] -> (nr, nc, theta_rad, scale_ratio)
        Args:
            append_linear_rot (bool): 是否在最后追加线性归一化的旋转 theta/pi [-1, 1]
                                      默认为 False，输出 5D。
                                      如果为 True，输出 6D，用于 HashGrid Z轴映射。
        Output:
            [..., 5] or [..., 6]
        """
        if coords_raw_4d.device != self.device:
            coords_raw_4d = coords_raw_4d.to(self.device)

        # 1. 位置归一化 (Linear) -> [-1, 1]
        nrc_raw = coords_raw_4d[..., 0:2]
        nrc_norm = 2.0 * (nrc_raw - self.nrc_min) / self.nrc_diff - 1.0

        # 2. 旋转嵌入 (Cos, Sin) -> [-1, 1]
        theta = coords_raw_4d[..., 2:3]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 3. 尺度归一化 (Log -> Linear) -> [-1, 1]
        scale_raw = coords_raw_4d[..., 3:4]
        scale_raw_clamped = torch.clamp(scale_raw,
                                        min=torch.exp(self.scale_log_min),
                                        max=torch.exp(self.scale_log_max))
        scale_log = torch.log(scale_raw_clamped)
        scale_norm = 2.0 * (scale_log - self.scale_log_min) / self.scale_log_diff - 1.0

        # 基础 5D 输出: [nr, nc, cos, sin, log_s]
        norm_output = torch.cat([nrc_norm, cos_t, sin_t, scale_norm], dim=-1)

        # 4. (动态可选) 追加线性旋转 [-1, 1]
        if append_linear_rot:
            # Theta 范围 [-pi, pi] -> 归一化到 [-1, 1]
            theta_linear = theta / torch.pi
            # 拼接: [..., 6]
            norm_output = torch.cat([norm_output, theta_linear], dim=-1)

        return norm_output


    def net_to_raw(self, coords_norm):
        """
        Input:  [..., 5] or [..., 6]
        Output: [..., 4] -> (nr, nc, theta, scale)
        """
        if coords_norm.device != self.device:
            coords_norm = coords_norm.to(self.device)

        # 无论输入是 5D 还是 6D，始终只取前 5 维
        coords_basic = coords_norm[..., :5]

        # 1. 还原位置
        nrc_norm = coords_basic[..., 0:2]
        nrc_raw = (nrc_norm + 1.0) / 2.0 * self.nrc_diff + self.nrc_min

        # 2. 还原旋转 (Atan2)
        cos_t = coords_basic[..., 2]
        sin_t = coords_basic[..., 3]
        theta_raw = torch.atan2(sin_t, cos_t).unsqueeze(-1)

        # 3. 还原尺度
        scale_norm = coords_basic[..., 4:5]
        scale_log = (scale_norm + 1.0) / 2.0 * self.scale_log_diff + self.scale_log_min
        scale_raw = torch.exp(scale_log)

        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=-1)

    # ============================================================
    # 辅助工具：Sigma 转换 & 权重计算 (基于 Linear 空间)
    # ============================================================

    def get_linear_sigmas(self, gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale):
        """
        将物理 Sigma 转换为 Linear 空间 Sigma (用于 Sampler)
        """
        # NRC
        avg_nrc_span = (self.nrc_diff[0] + self.nrc_diff[1]) / 2.0
        sigma_lin_nrc = gs_sigma_nrc * (2.0 / (avg_nrc_span + 1e-6))

        # Theta (Rad -> Linear [-1, 1])
        # Range = 2*pi -> 2.0
        sigma_lin_theta = gs_sigma_radrot * (2.0 / (2 * torch.pi))

        # Scale (Log -> Linear [-1, 1])
        sigma_lin_scale = gs_sigma_scale * (2.0 / (self.scale_log_diff + 1e-6))

        # 返回列表顺序：[r, c, theta, s]
        sigma_lin_nrc_val = float(sigma_lin_nrc)
        sigma_lin_theta_val = float(sigma_lin_theta)
        sigma_lin_scale_val = float(sigma_lin_scale)
        return [sigma_lin_nrc_val, sigma_lin_nrc_val, sigma_lin_theta_val, sigma_lin_scale_val]

    def compute_weight_matrix_linear(self, q_lin, ref_lin, norm_sigmas,ignore_dim=None):
        """
        在 4D Linear 空间计算权重 (极快)

        Args:
            q_lin: [B, M, 4]
            ref_lin: [B, N, 4]
            norm_sigmas: list [sigma_r, sigma_c, sigma_theta, sigma_s] (归一化后的)
            ignore_dim: int or list[int] or None. 需要忽略的维度索引。
                        例如: 2 表示忽略旋转, [2, 3] 表示忽略旋转和尺度。
        """
        # 1. 维度处理与广播
        if q_lin.dim() == 2: q_lin = q_lin.unsqueeze(0)
        if ref_lin.dim() == 2: ref_lin = ref_lin.unsqueeze(0)

        q = q_lin.unsqueeze(2)  # [B, M, 1, 4]
        ref = ref_lin.unsqueeze(1)  # [B, 1, N, 4]

        # 2. Sigma 准备
        sigmas = torch.as_tensor(norm_sigmas, device=q.device, dtype=q.dtype).view(1, 1, 1, 4)

        # 3. 计算基础差值 (Linear Difference)
        delta = q - ref  # [B, M, N, 4]

        # 4. 计算距离平方 (Square Distance)
        # 先对所有维度直接求平方 (此时 Theta 维度的平方是错的，因为没考虑周期性)
        dist_sq = delta ** 2

        # 5. 修正 Theta (Index 2)
        # d_cyclic = min(|d|, 2 - |d|)
        d_theta_raw = torch.abs(delta[..., 2])
        d_theta_cyclic = torch.min(d_theta_raw, 2.0 - d_theta_raw)
        # 直接覆盖 Index 2 的值
        dist_sq[..., 2] = d_theta_cyclic ** 2

        # =========================================================
        # 维度屏蔽 (Masking)
        # =========================================================
        if ignore_dim is not None:
            # 1. 统一转为列表
            if isinstance(ignore_dim, int):
                dims_to_ignore = [ignore_dim]
            else:
                dims_to_ignore = ignore_dim

            # 2. 创建掩码 [1, 1, 1, 1] -> [1, 1, 0, 1]
            # 默认全是 1 (保留)
            mask = torch.ones(4, device=q.device, dtype=q.dtype)

            # 将忽略的维度设为 0
            mask[dims_to_ignore] = 0.0

            # 3. 广播并应用掩码
            # [4] -> [1, 1, 1, 4]
            mask = mask.view(1, 1, 1, 4)

            # 强制将忽略维度的距离置为 0
            # 0 / sigma^2 = 0，所以在 sum 时就没有贡献了
            dist_sq = dist_sq * mask

        # 6. 加权求和
        # E = sum( d^2 / (2 * sigma^2) )
        energy = torch.sum(dist_sq / (2 * sigmas ** 2), dim=-1)  # [B, M, N]

        return torch.exp(-energy)
