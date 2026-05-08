#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image


DEFAULT_DETAILS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "gen_fm_exps"
    / "analysis"
    / "stage1_crtl_ckpts2exps"
    / "mathing_refine"
    / "selavpr_ctrl_dkm_top5"
)


def _iter_detail_paths(inputs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.glob("**/gim_refine_details.pt")))
        else:
            raise FileNotFoundError(path)
    return sorted(set(paths))


def _wrap_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _rot_err_deg(pred: float, gt: float) -> float:
    return abs(_wrap_pi(float(pred) - float(gt))) * 180.0 / math.pi


def _scale_ratio(pred: float, gt: float) -> float:
    pred = max(float(pred), 1e-9)
    gt = max(float(gt), 1e-9)
    return max(pred / gt, gt / pred)


def _affine_theta_scale(affine_2x3) -> Tuple[float, float]:
    a00, a01, _ = affine_2x3[0]
    a10, a11, _ = affine_2x3[1]
    theta = math.atan2(float(a10), float(a00))
    sx = math.hypot(float(a00), float(a10))
    sy = math.hypot(float(a01), float(a11))
    return theta, 0.5 * (sx + sy)


def _query_size(query_path: str) -> Tuple[float, float]:
    with Image.open(query_path) as img:
        w, h = img.size
    return float(w), float(h)


def _nominal_query_to_patch_scale(query_path: str, patch_size: float, mode: str = "height") -> float:
    w, h = _query_size(query_path)
    if mode == "height":
        query_extent = h
    elif mode == "width":
        query_extent = w
    elif mode == "sqrt":
        query_extent = math.sqrt(w * h)
    elif mode == "none":
        query_extent = patch_size
    else:
        raise ValueError(f"Unknown scale normalization mode: {mode}")
    return float(patch_size) / max(float(query_extent), 1e-9)


def _variant_coord(base_coord, theta: float, scale_corr: float, rot_sign: int, scale_mode: str):
    out = [float(v) for v in base_coord]
    out[2] = _wrap_pi(out[2] + float(rot_sign) * float(theta))
    if scale_mode == "mul":
        out[3] = out[3] * float(scale_corr)
    elif scale_mode == "div":
        out[3] = out[3] / max(float(scale_corr), 1e-9)
    elif scale_mode != "same":
        raise ValueError(scale_mode)
    return out


def _summarize(rows: List[Tuple[List[float], List[float], float, float]]) -> Dict[str, Dict[str, float]]:
    variants = [
        ("base", 0, "same"),
        ("rot_plus", +1, "same"),
        ("rot_minus", -1, "same"),
        ("scale_mul", 0, "mul"),
        ("scale_div", 0, "div"),
        ("plus_mul", +1, "mul"),
        ("plus_div", +1, "div"),
        ("minus_mul", -1, "mul"),
        ("minus_div", -1, "div"),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for name, rot_sign, scale_mode in variants:
        rot_errs = []
        scale_ratios = []
        both_5_12 = []
        for gt, base, theta, scale_corr in rows:
            pred = _variant_coord(base, theta, scale_corr, rot_sign, scale_mode)
            re = _rot_err_deg(pred[2], gt[2])
            sr = _scale_ratio(pred[3], gt[3])
            rot_errs.append(re)
            scale_ratios.append(sr)
            both_5_12.append(re <= 5.0 and sr <= 1.2)
        rot_t = torch.tensor(rot_errs, dtype=torch.float32)
        scale_t = torch.tensor(scale_ratios, dtype=torch.float32)
        out[name] = {
            "rot_median": float(rot_t.median().item()),
            "rot_mean": float(rot_t.mean().item()),
            "rot_at_5": float((rot_t <= 5.0).float().mean().item() * 100.0),
            "scale_median": float(scale_t.median().item()),
            "scale_mean": float(scale_t.mean().item()),
            "scale_at_1p2": float((scale_t <= 1.2).float().mean().item() * 100.0),
            "rot5_scale1p2": float(torch.tensor(both_5_12, dtype=torch.float32).mean().item() * 100.0),
        }
    pred_corr = torch.tensor([float(row[3]) for row in rows], dtype=torch.float32).clamp(min=1e-9)
    ideal_corr = torch.tensor([float(row[0][3]) / max(float(row[1][3]), 1e-9) for row in rows], dtype=torch.float32).clamp(min=1e-9)
    log_pred = torch.log(pred_corr)
    log_ideal = torch.log(ideal_corr)
    pred_centered = log_pred - log_pred.mean()
    ideal_centered = log_ideal - log_ideal.mean()
    denom = pred_centered.norm() * ideal_centered.norm()
    corr = float((pred_centered @ ideal_centered / denom).item()) if float(denom.item()) > 1e-12 else 0.0
    out["_scale_corr"] = {
        "pred_median": float(pred_corr.median().item()),
        "ideal_median": float(ideal_corr.median().item()),
        "log_corr": corr,
        "pred_log_mae": float(torch.abs(log_pred - log_ideal).mean().item()),
    }
    return out


def diagnose_details(path: Path, min_inliers: int, scale_norm: str = "height"):
    details = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for record in details["query_details"]:
        order = record.get("rerank_order_prefix") or []
        if not order:
            continue
        candidate = record["candidates"][int(order[0])]
        match_result = candidate.get("match_result", {})
        affine = match_result.get("affine_2x3")
        if affine is None or int(match_result.get("inlier_count", 0)) < int(min_inliers):
            continue
        theta, aff_scale = _affine_theta_scale(affine)
        patch_size = float(candidate["patch_meta"]["patch_size"])
        nominal = _nominal_query_to_patch_scale(record["query_path"], patch_size, mode=scale_norm)
        if not math.isfinite(aff_scale) or not math.isfinite(nominal) or aff_scale <= 0 or nominal <= 0:
            continue
        scale_corr = aff_scale / nominal
        rows.append((record["gt_coord_4d"], record["top1_coord_after_rerank"], theta, scale_corr))
    return details, rows, _summarize(rows) if rows else {}


def main():
    parser = argparse.ArgumentParser(description="Diagnose affine-derived rot/scale correction variants.")
    parser.add_argument("--details-root", action="append", type=Path, default=[DEFAULT_DETAILS_ROOT])
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--scale-norm", choices=("height", "width", "sqrt", "none", "all"), default="height")
    args = parser.parse_args()

    scale_norms = ("height", "width", "sqrt", "none") if args.scale_norm == "all" else (args.scale_norm,)
    for path in _iter_detail_paths(args.details_root):
        details = None
        for scale_norm in scale_norms:
            details, rows, summary = diagnose_details(path, min_inliers=args.min_inliers, scale_norm=scale_norm)
            print(f"\n{path.parent.name} scale_norm={scale_norm} valid={len(rows)}/{len(details['query_details'])}")
            for name, metrics in summary.items():
                if name == "_scale_corr":
                    print(
                        f"  {'scale_corr':10s} "
                        f"pred_med={metrics['pred_median']:6.3f} ideal_med={metrics['ideal_median']:6.3f} "
                        f"log_corr={metrics['log_corr']:6.3f} log_mae={metrics['pred_log_mae']:6.3f}"
                    )
                    continue
                print(
                    f"  {name:10s} "
                    f"rot_med={metrics['rot_median']:6.2f} rot_mean={metrics['rot_mean']:6.2f} "
                    f"rot@5={metrics['rot_at_5']:6.2f} "
                    f"scale_med={metrics['scale_median']:6.3f} scale_mean={metrics['scale_mean']:6.3f} "
                    f"scale@1.2={metrics['scale_at_1p2']:6.2f} "
                    f"both={metrics['rot5_scale1p2']:6.2f}"
                )
        if len(scale_norms) > 1:
            continue


if __name__ == "__main__":
    main()
