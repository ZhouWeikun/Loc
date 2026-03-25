import torch
from typing import Dict, Tuple, Optional
eps = 1e-12

@torch.no_grad()
def analyse_feature_frequency(
    feats_F: torch.Tensor,
    feats_Z: torch.Tensor,
    wo_DC: bool = True,
    norm: str = "ortho",
    eps: float = 1e-12,
    cdf_tau: float = 0.95,
    hf_frac: float = 0.33,
    channel_norm: bool = True,
    return_radial: bool = True,
) -> Dict[str, object]:
    """
    对两个特征场做频谱分析。

    Args:
        feats_F: [H, W, C] INGP 特征场（或任意 baseline 特征场）
        feats_Z: [H, W, d] projector 特征场
        norm: torch.fft.fft2 的 norm 参数（推荐 "ortho"）
        cdf_tau: roll-off 的累计能量阈值（默认 0.95）
        hf_frac: 高频阈值 f0 所占 Nyquist 半径比例（默认 0.33）
        channel_norm: True 时按“每通道功率谱归一化后再平均”（更稳）；False 时直接平均通道功率谱
        return_radial: 是否返回径向谱 E_F/E_Z

    Returns:
        {
          "P_F": [H,W] 归一化功率谱(步骤6)，未 shift
          "P_Z": [H,W] 归一化功率谱(步骤6)，未 shift
          "metrics_F": {"fc":..., "f95":..., "hf_ratio":..., "f0_bin":...}
          "metrics_Z": {...}
          "delta": {"fc":..., "f95":..., "hf_ratio_Z_over_F":...}
          "E_F": [R+1] (可选) 径向谱
          "E_Z": [R+1] (可选) 径向谱
        }
    """
    assert feats_F.dim() == 3 and feats_Z.dim() == 3, "Input must be [H, W, C]"
    assert feats_F.shape[:2] == feats_Z.shape[:2], "H,W must match"

    # [C,H,W]
    F = feats_F.permute(2, 0, 1).contiguous()
    Z = feats_Z.permute(2, 0, 1).contiguous()

    # 去 DC：每通道减均值
    if wo_DC:
        F = F - F.mean(dim=(1, 2), keepdim=True)
        Z = Z - Z.mean(dim=(1, 2), keepdim=True)

    # FFT
    fft_F = torch.fft.fft2(F, norm=norm)
    fft_Z = torch.fft.fft2(Z, norm=norm)

    # 功率谱 per-channel
    PF_c = fft_F.abs().pow(2)  # [C,H,W]
    PZ_k = fft_Z.abs().pow(2)  # [d,H,W]

    if channel_norm:
        # 每通道归一化到总功率=1，再跨通道平均
        PF_c = PF_c / (PF_c.sum(dim=(1, 2), keepdim=True) + eps)
        PZ_k = PZ_k / (PZ_k.sum(dim=(1, 2), keepdim=True) + eps)
        P_F = PF_c.mean(dim=0)  # [H,W]
        P_Z = PZ_k.mean(dim=0)
    else:
        # 直接跨通道平均
        P_F = PF_c.mean(dim=0)
        P_Z = PZ_k.mean(dim=0)

    # 步骤6：整体归一化（返回给你用于可视化/后续处理）
    P_F = P_F / (P_F.sum() + eps)
    P_Z = P_Z / (P_Z.sum() + eps)

    # fftshift 后做径向谱与指标
    P_Fs = torch.fft.fftshift(P_F, dim=(-2, -1))
    P_Zs = torch.fft.fftshift(P_Z, dim=(-2, -1))

    E_F = radial_profile(P_Fs)
    E_Z = radial_profile(P_Zs)

    fc_F, f95_F, hf_F, f0_F = spectrum_metrics(E_F, cdf_tau=cdf_tau, hf_frac=hf_frac, eps=eps)
    fc_Z, f95_Z, hf_Z, f0_Z = spectrum_metrics(E_Z, cdf_tau=cdf_tau, hf_frac=hf_frac, eps=eps)

    out = {
        "P_F": P_F,
        "P_Z": P_Z,
        "metrics_F": {"fc": fc_F, "f95": f95_F, "hf_ratio": hf_F, "f0_bin": f0_F},
        "metrics_Z": {"fc": fc_Z, "f95": f95_Z, "hf_ratio": hf_Z, "f0_bin": f0_Z},
        "delta": {
            "fc": fc_Z - fc_F,
            "f95": f95_Z - f95_F,
            "hf_ratio_Z_over_F": hf_Z / (hf_F + eps),
        }
    }

    if return_radial:
        out["E_F"] = E_F
        out["E_Z"] = E_Z

    return out

def radial_profile(P_hw: torch.Tensor) -> torch.Tensor:
    """
    P_hw: [H, W] 实数功率谱，建议已 fftshift（低频在中心）
    return: E_r [R+1] 径向平均谱
    """
    H, W = P_hw.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=P_hw.device),
        torch.arange(W, device=P_hw.device),
        indexing='ij'
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_bin = torch.round(r).to(torch.long).flatten()
    P_flat = P_hw.flatten()

    # 【改进点1】限制最大半径为短边的一半 (内切圆)
    # 这样保证每个 bin 都是完整的圆环，统计才具有各向同性的代表性
    valid_max_r = min(H, W) // 2

    # 创建 Mask，只保留内切圆内的数据
    mask = r_bin <= valid_max_r

    r_bin = r_bin[mask]
    P_flat = P_flat[mask]

    prof = torch.zeros(valid_max_r + 1, device=P_hw.device)
    cnt = torch.zeros(valid_max_r + 1, device=P_hw.device)

    prof.scatter_add_(0, r_bin, P_flat)
    cnt.scatter_add_(0, r_bin, torch.ones_like(P_flat))

    # 防止除零 (虽然 masked 后一般不会，但为了健壮性)
    prof = prof / cnt.clamp_min(1)

    return prof

def spectrum_metrics(E_r: torch.Tensor,cdf_tau:float = 0.95, hf_frac: float = 0.33, eps: float = 1e-12):
    """
    E_r: [R+1] 径向能量谱（非负）
    hf_frac: 高频阈值位置 = hf_frac * Nyquist_radius (按 bin)
    returns: (fc, f95, hf_ratio)
    """
    E = E_r / (E_r.sum() + eps)
    r = torch.arange(E.numel(), device=E.device).float()

    # spectral centroid
    fc = (r * E).sum()

    # 95% roll-off
    cdf = torch.cumsum(E, dim=0)
    f95 = int((cdf >= cdf_tau).nonzero(as_tuple=False)[0].item())

    # high-frequency energy ratio
    f0 = int(hf_frac * (E.numel() - 1))
    hf = E[f0 + 1:].sum()

    return float(fc.item()), float(f95), float(hf.item()), int(f0)