import torch
import numpy as np


class CoordsNormProcessor:
    """
    EBM 定位专用坐标处理器 (Hardcoded for Log-Scale & 5D Rotation)
    """

    def __init__(self, sat_dataset_instance):
        """
        Args:
            sat_dataset_instance: 数据集实例，包含物理边界
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === 1. 位置 (NRC) [Row, Col] ===
        min_nr = sat_dataset_instance.nr2sample_min
        min_nc = sat_dataset_instance.nc2sample_min
        max_nr = sat_dataset_instance.nr2sample_max
        max_nc = sat_dataset_instance.nc2sample_max

        self.nrc_min = torch.tensor([min_nr, min_nc], device=self.device, dtype=torch.float32)
        self.nrc_max = torch.tensor([max_nr, max_nc], device=self.device, dtype=torch.float32)
        self.nrc_diff = self.nrc_max - self.nrc_min

        # === 3. 尺度 (Scale) ===
        scale_bnd = sat_dataset_instance.satimgsize_scale_to_refm_boundary
        s_min_val = scale_bnd[0]
        s_max_val = scale_bnd[1]

        self.scale_log_min = torch.tensor([np.log(max(s_min_val, 1e-6))], device=self.device, dtype=torch.float32)
        self.scale_log_max = torch.tensor([np.log(s_max_val)], device=self.device, dtype=torch.float32)
        self.scale_log_diff = self.scale_log_max - self.scale_log_min

    def raw_to_norm(self, coords_raw_4d, append_linear_rot=False):
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

    def norm_to_raw(self, coords_norm):
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

    def get_normalized_sigmas(self, gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale=None):
        """
        [工具方法] 将物理 Sigma 转换为 归一化空间 Sigma

        用途：
        1. 传给 NormalizedGaussianSampler 用于采样
        2. 传给 compute_weight_matrix_from_norm 用于计算 Pos/Scale 的能量

        Args:
            gs_sigma_nrc: 物理位置 Sigma (米/像素)
            gs_sigma_radrot: 物理角度 Sigma (弧度)
            gs_sigma_scale: 物理尺度 Sigma (Log unit)

        Returns:
            dict: 包含转换后的 Sigma，键名为 'nrc', 'rot', 'scale'
        """
        # 1. 位置 Sigma (Physical -> [-1, 1])
        # Scale Factor = 2.0 / Physical_Range
        avg_nrc_span = (self.nrc_diff[0] + self.nrc_diff[1]) / 2.0
        sigma_n_nrc = gs_sigma_nrc * (2.0 / (avg_nrc_span + 1e-6))

        # 2. 尺度 Sigma (Log Physical -> [-1, 1])
        # 这里的 scale_log_diff 已经是 Log 范围了
        sigma_n_scale = None
        if gs_sigma_scale is not None:
            sigma_n_scale = gs_sigma_scale * (2.0 / (self.scale_log_diff + 1e-6))

        # 3. 旋转 Sigma (Linear Angle -> [-1, 1])
        # 注意：这个 sigma_n_rot 仅用于 NormalizedGaussianSampler 对 theta_linear 进行采样
        # 在计算权重矩阵(compute_weight)时，因为用的是 acos(dot)，还是得用物理弧度 sigma
        sigma_n_rot = gs_sigma_radrot * (2.0 / (2 * torch.pi))

        return {
            'nrc': sigma_n_nrc,
            'scale': sigma_n_scale,
            'rot_linear': sigma_n_rot,  # 用于采样 [-1, 1] 的 theta
            'rot_rad': gs_sigma_radrot  # 原始物理弧度，用于权重计算
        }

    def compute_weight_matrix_from_norm(self, q_norm, ref_norm,
                                        gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale=None,
                                        use_exact_rot_dist=True):
        """
        智能权重计算 (支持 5D 和 6D 输入)

        自动检测输入维度:
        - 5D input: [nr, nc, cos, sin, log_s_norm] -> 使用 acos 或 chord 近似计算角度差
        - 6D input: [..., log_s_norm, theta_linear] -> 直接使用 theta_linear 计算角度差 (更快更稳)
        """
        # 1. 获取归一化 Sigma
        sigmas = self.get_normalized_sigmas(gs_sigma_nrc, gs_sigma_radrot, gs_sigma_scale)

        sigma_n_nrc = sigmas['nrc']
        sigma_n_scale = sigmas['scale']
        sigma_rot_phys = sigmas['rot_rad']  # 物理弧度 Sigma

        # 2. 维度检查与广播准备
        is_batched = q_norm.dim() == 3
        if not is_batched:
            q_norm = q_norm.unsqueeze(0)
            ref_norm = ref_norm.unsqueeze(0)

        q = q_norm.unsqueeze(2)  # [B, M, 1, D]
        ref = ref_norm.unsqueeze(1)  # [B, 1, N, D]

        input_dim = q_norm.shape[-1]  # 检测是 5 还是 6

        # 3. 能量计算

        # --- A. 位置 (Dim 0, 1) ---
        diff_nrc = q[..., 0:2] - ref[..., 0:2]
        energy_nrc = torch.sum(diff_nrc ** 2, dim=-1) / (2 * sigma_n_nrc ** 2)

        # --- B. 尺度 (Dim 4) ---
        energy_scale = 0.0
        if sigma_n_scale is not None:
            # Scale 始终在 index 4
            diff_scale = q[..., 4:5] - ref[..., 4:5]
            energy_scale = torch.sum(diff_scale ** 2, dim=-1) / (2 * sigma_n_scale ** 2)

        # --- C. 旋转 (智能分支) ---

        if input_dim == 6:
            # === 分支 1: 极速模式 (直接使用第6维 theta) ===
            # 你的 theta_linear 是 [-1, 1]，对应 [-pi, pi]
            # 我们先还原回物理弧度，方便和 sigma_rot_phys 配合

            theta_q = q[..., 5] * torch.pi  # [-pi, pi]
            theta_ref = ref[..., 5] * torch.pi

            # 计算周期性距离
            diff = torch.abs(theta_q - theta_ref)
            diff = torch.min(diff, 2 * torch.pi - diff)  # Wrap-around!

            energy_rot = (diff ** 2) / (2 * sigma_rot_phys ** 2)

        else:
            # === 分支 2: 兼容模式 (从 cos, sin 恢复) ===
            cos_sin_q = q[..., 2:4]
            cos_sin_ref = ref[..., 2:4]

            if use_exact_rot_dist:
                # Arccos 方式
                dot = torch.sum(cos_sin_q * cos_sin_ref, dim=-1)
                dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
                dist_rot = torch.acos(dot)
                energy_rot = (dist_rot ** 2) / (2 * sigma_rot_phys ** 2)
            else:
                # 弦长近似
                diff_rot = cos_sin_q - cos_sin_ref
                dist_sq_rot = torch.sum(diff_rot ** 2, dim=-1)
                energy_rot = dist_sq_rot / (2 * sigma_rot_phys ** 2)

        # 4. 合并
        total_energy = energy_nrc + energy_rot + energy_scale
        weights = torch.exp(-total_energy)

        if not is_batched:
            weights = weights.squeeze(0)

        return weights

