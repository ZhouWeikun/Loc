import math
from typing import Any, Optional
import numpy as np
import torch


def neg_mask_visloc(
    coords_candidates: Any,
    coords_query: Any,
    radius_nrc: Optional[float] = None,
    radius_rot_rad: Optional[float] = None,
    radius_scale_log: Optional[float] = None,
) -> torch.Tensor:
    """
    Visloc negative mask (stub).

    Args:
        coords_candidates: [B, K, D] or [N, D]
        coords_query: [B, D] or [D]
        radius_nrc: distance threshold on [row, col]
        radius_rot_rad: distance threshold on rotation (radians)
        radius_scale_log: distance threshold on scale (log space)

    Returns:
        mask: boolean tensor, True means negative.
    """
    a = _to_tensor(coords_candidates)
    b = _to_tensor(coords_query)
    a, b = _align(a, b)

    pos_mask = torch.ones(a.shape[:-1], dtype=torch.bool, device=a.device)
    used = False

    if radius_nrc is not None:
        dist_rc = torch.linalg.norm(a[..., :2] - b[..., :2], ord=2, dim=-1)
        pos_mask = pos_mask & (dist_rc <= float(radius_nrc))
        used = True

    if radius_rot_rad is not None:
        rot_diff = _rot_diff(a[..., 2], b[..., 2])
        pos_mask = pos_mask & (rot_diff <= float(radius_rot_rad))
        used = True

    if radius_scale_log is not None:
        scale_a = a[..., 3].clamp(min=1e-6)
        scale_b = b[..., 3].clamp(min=1e-6)
        log_diff = torch.abs(torch.log(scale_a) - torch.log(scale_b))
        pos_mask = pos_mask & (log_diff <= float(radius_scale_log))
        used = True

    if not used:
        return torch.zeros(a.shape[:-1], dtype=torch.bool, device=a.device)

    return ~pos_mask


class SceneNegMasker:
    """
    Per-scene neg-mask wrapper holding thresholds as attributes.
    """

    def __init__(
        self,
        radius_nrc: Optional[float] = None,
        radius_rot_rad: Optional[float] = None,
        radius_scale_log: Optional[float] = None,
    ) -> None:
        self.radius_nrc = radius_nrc
        self.radius_rot_rad = radius_rot_rad
        self.radius_scale_log = radius_scale_log

    def neg_mask(
        self,
        coords_candidates: Any,
        coords_query: Any,
        #下面的参数允许函数调用时动态改变radius
        radius_nrc: Optional[float] = None,
        radius_rot_rad: Optional[float] = None,
        radius_scale_log: Optional[float] = None,
    ) -> torch.Tensor:
        return neg_mask_visloc(
            coords_candidates,
            coords_query,
            radius_nrc=self.radius_nrc if radius_nrc is None else radius_nrc,
            radius_rot_rad=self.radius_rot_rad if radius_rot_rad is None else radius_rot_rad,
            radius_scale_log=self.radius_scale_log if radius_scale_log is None else radius_scale_log,
        )


# Optional helper (keep minimal and local to this file).
def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.as_tensor(x)

def _align(coords_a: torch.Tensor, coords_b: torch.Tensor):
    a = coords_a
    b = coords_b
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if a.ndim == 3 and b.ndim == 2:
        b = b[:, None, :]
    return a, b

def _rot_diff(a_rot: torch.Tensor, b_rot: torch.Tensor) -> torch.Tensor:
    diff = a_rot - b_rot
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff.abs()
