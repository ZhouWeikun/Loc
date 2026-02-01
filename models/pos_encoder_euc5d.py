import torch
import torch.nn as nn
from .pos_encoder import PositionalEncoder


class Coords5DEncoder(nn.Module):
    """
    智能 5D 坐标编码器
    逻辑：
    1. RC & Scale: [-1, 1] -> 乘以 pi -> PE (线性高频映射)
    2. Rotation: [cos, sin] -> atan2 -> theta [-pi, pi] -> PE (旋转谐波映射)
    """

    def __init__(self, multires_rc=8, multires_rot=6, multires_scale=4,
                 include_input_rc_scale=True, # RC和Scale是否包含原始输入
                 log_sampling=True):
        super(Coords5DEncoder, self).__init__()

        # 1. RC 编码器 (2D)
        self.rc_encoder = PositionalEncoder(
            input_dims=2,
            multires=multires_rc,
            include_input=include_input_rc_scale,
            log_sampling=log_sampling
        )

        # 2. 旋转编码器 (1D: theta)
        # [关键] input_dims=1 (标量角度)
        # [关键] include_input=False (避免 -pi/pi 断裂问题)
        self.rot_encoder = PositionalEncoder(
            input_dims=1,
            multires=multires_rot,
            include_input=False,
            log_sampling=log_sampling
        )

        # 3. 尺度编码器 (1D)
        self.scale_encoder = PositionalEncoder(
            input_dims=1,
            multires=multires_scale,
            include_input=include_input_rc_scale,
            log_sampling=log_sampling
        )

        # 计算总输出维度
        self.out_dim = (self.rc_encoder.out_dim +
                        self.rot_encoder.out_dim +
                        self.scale_encoder.out_dim)

    def forward(self, coords_5d):
        """
        Args:
            coords_5d: [..., 5] -> [nr, nc, cos, sin, log_s] (都在 -1 到 1 之间)
        """
        original_shape = coords_5d.shape
        coords_flat = coords_5d.reshape(-1, 5)

        # === A. 处理 RC (线性) ===
        # 取前两维 [-1, 1]
        rc_norm = coords_flat[..., :2]
        # [建议] 乘以 PI，将范围扩展到 [-pi, pi]，让 PE 的第0级频率能覆盖一个完整波长
        rc_encoded = self.rc_encoder(rc_norm * torch.pi)

        # === B. 处理 Rotation (非线性恢复) ===
        # 取 cos, sin
        cos_t = coords_flat[..., 2]
        sin_t = coords_flat[..., 3]
        # [关键] 恢复弧度 theta [-pi, pi]
        # atan2 是这一步的灵魂，它把 5D 里的几何向量变回了物理角度
        theta_rad = torch.atan2(sin_t, cos_t).unsqueeze(-1) # [N, 1]
        # 喂给 PE (此时 PE 会计算 sin(1*theta), sin(2*theta)... 完美的谐波)
        rot_encoded = self.rot_encoder(theta_rad)

        # === C. 处理 Scale (线性) ===
        scale_norm = coords_flat[..., 4:5]
        # [建议] 同样乘以 PI
        scale_encoded = self.scale_encoder(scale_norm * torch.pi)

        # === D. 拼接 ===
        coords_encoded = torch.cat([rc_encoded, rot_encoded, scale_encoded], dim=-1)

        # 恢复形状
        output_shape = list(original_shape[:-1]) + [self.out_dim]
        return coords_encoded.reshape(output_shape)
