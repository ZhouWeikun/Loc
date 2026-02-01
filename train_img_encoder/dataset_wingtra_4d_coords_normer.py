import torch
import numpy as np

class CoordsNormProcessor:
    """
    EBM 定位专用坐标处理器 (Hardcoded for Log-Scale & 5D Rotation)

    Mapping:
        Raw [nr, nc, theta, scale] -> Norm [nr, nc, cos, sin, log_scale]

    Ranges:
        Raw: Physical ranges -> Norm: All in [-1, 1]
    """

    def __init__(self, sat_dataset_instance):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === 1. 位置 (NRC) [Row, Col] ===
        # 显式提取，确保顺序不错位
        min_nr = sat_dataset_instance.nr2sample_min
        min_nc = sat_dataset_instance.nc2sample_min
        max_nr = sat_dataset_instance.nr2sample_max
        max_nc = sat_dataset_instance.nc2sample_max

        self.nrc_min = torch.tensor([min_nr, min_nc], device=self.device, dtype=torch.float32)
        self.nrc_max = torch.tensor([max_nr, max_nc], device=self.device, dtype=torch.float32)
        self.nrc_diff = self.nrc_max - self.nrc_min

        # === 2. 旋转 (Rotation) ===
        # 旋转将被映射为 (cos, sin)，天然在 [-1, 1]，无需存储 Min/Max

        # === 3. 尺度 (Scale) - 强制使用 Log 空间 ===
        # 获取物理/比例尺度的边界
        scale_bnd = sat_dataset_instance.satimgsize_scale_to_refm_boundary
        s_min_val = scale_bnd[0]
        s_max_val = scale_bnd[1]

        # 直接存储 Log 域的边界
        # epsilon 1e-6 防止 log(0)
        self.scale_log_min = torch.tensor([np.log(max(s_min_val, 1e-6))], device=self.device,dtype=torch.float32)
        self.scale_log_max = torch.tensor([np.log(s_max_val)], device=self.device,dtype=torch.float32)
        self.scale_log_diff = self.scale_log_max - self.scale_log_min

    def raw_to_norm(self, coords_raw_4d):
        """
        Input:  [N, 4] -> (nr, nc, theta_rad, scale_ratio)
        Output: [N, 5] -> (nr_norm, nc_norm, cos_theta, sin_theta, scale_log_norm)
        范围都在 [-1, 1] 之间
        """
        if coords_raw_4d.device != self.device:
            coords_raw_4d = coords_raw_4d.to(self.device)

        # 1. 位置归一化 (Linear)
        nrc_raw = coords_raw_4d[..., 0:2]
        nrc_norm = 2.0 * (nrc_raw - self.nrc_min) / self.nrc_diff - 1.0

        # 2. 旋转嵌入 (Cos, Sin)
        theta = coords_raw_4d[..., 2:3]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 3. 尺度归一化 (Log -> Linear)
        scale_raw = coords_raw_4d[..., 3:4]
        # 先取 Log
        scale_log = torch.log(torch.clamp(scale_raw, min=1e-6))
        # 再归一化 Log 值到 [-1, 1]
        scale_norm = 2.0 * (scale_log - self.scale_log_min) / self.scale_log_diff - 1.0

        # 4. 拼接 [N, 5]
        return torch.cat([nrc_norm, cos_t, sin_t, scale_norm], dim=1)

    def norm_to_raw(self, coords_norm_5d):
        """
        Input:  [N, 5] -> (nr_norm, nc_norm, cos, sin, scale_log_norm)
        Output: [N, 4] -> (nr, nc, theta, scale)
        """
        if coords_norm_5d.device != self.device:
            coords_norm_5d = coords_norm_5d.to(self.device)

        # 1. 还原位置
        nrc_norm = coords_norm_5d[..., 0:2]
        nrc_raw = (nrc_norm + 1.0) / 2.0 * self.nrc_diff + self.nrc_min

        # 2. 还原旋转 (Atan2)
        cos_t = coords_norm_5d[..., 2]
        sin_t = coords_norm_5d[..., 3]
        theta_raw = torch.atan2(sin_t, cos_t).unsqueeze(1)  # [-pi, pi]

        # 3. 还原尺度 (Linear -> Exp)
        scale_norm = coords_norm_5d[..., 4:5]
        # 还原回 Log 域数值
        scale_log = (scale_norm + 1.0) / 2.0 * self.scale_log_diff + self.scale_log_min
        # Exp 回物理数值
        scale_raw = torch.exp(scale_log)

        # 4. 拼接 [N, 4]
        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=1)

    def sanitize_5d_coords(self, coords_norm_5d):
        """
        【郎之万采样专用】
        修正流形约束：强制 (cos, sin) 回到单位圆上。
        应在每次梯度更新或加噪声后调用。
        """
        # Clone 保证不产生 In-place 操作副作用（视情况而定，一般安全起见）
        coords_clean = coords_norm_5d.clone()

        # 提取旋转向量 [cos, sin]
        vec_rot = coords_clean[..., 2:4]

        # 归一化: v / ||v||
        norm_val = torch.norm(vec_rot, dim=1, keepdim=True) + 1e-8
        coords_clean[..., 2:4] = vec_rot / norm_val

        return coords_clean