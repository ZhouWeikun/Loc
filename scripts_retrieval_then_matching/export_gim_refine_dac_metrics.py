#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords


def _json_cell(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_detail_paths(root: Path) -> List[Path]:
    return sorted(root.glob("**/gim_refine_details.pt"))


def _parse_split(experiment_dir: str) -> Tuple[str, str]:
    match = re.search(r"_(interval|segment)(\d+|same_scene_top\d+)?", str(experiment_dir))
    if not match:
        return "", ""
    mode, tag = match.groups()
    return mode, tag or ""


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _derive_nrc2meter(thresholds: Dict) -> float:
    if "nrc2meter" in thresholds:
        return float(thresholds["nrc2meter"])
    if "dist_meter" in thresholds and "norm_dist" in thresholds:
        return float(thresholds["dist_meter"]) / max(float(thresholds["norm_dist"]), 1e-8)
    raise KeyError(f"Cannot derive nrc2meter from thresholds: {thresholds}")


def _normalize_refined_topk_order(details: Dict) -> None:
    refined = details.get("coords_topk_refined")
    reranked = details.get("coords_topk_reranked")
    if not (torch.is_tensor(refined) and torch.is_tensor(reranked)):
        return
    if tuple(refined.shape) != tuple(reranked.shape) or refined.ndim != 3 or refined.shape[-1] != 4:
        return
    fixed = reranked.clone()
    fixed[:, 0] = refined[:, 0].to(dtype=fixed.dtype, device=fixed.device)
    details["coords_topk_refined"] = fixed


def _progressive_eval(coords_topk, coords_gt, k_values, dist_th_nrc, rot_th_deg, scale_ratio_th):
    acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
        coords_topk.to(torch.float32),
        coords_gt.to(torch.float32),
        dist_th=float(dist_th_nrc),
        rot_th_deg=float(rot_th_deg),
        scale_ratio_th=float(scale_ratio_th),
        k_values=tuple(int(v) for v in k_values),
    )
    progressive = dict(acc_metrics_raw["progressive_acc_metrics"])
    return progressive, {str(k): float(v) for k, v in err_stats.items()}


def _median_or_empty(values: torch.Tensor):
    if values.numel() == 0:
        return ""
    return float(values.median().item())


def _conditional_top1_medians(coords_topk, coords_gt, dist_th_nrc, rot_th_deg, scale_ratio_th, nrc2meter):
    coords_topk = coords_topk.to(torch.float32)
    coords_gt = coords_gt.to(coords_topk.device, dtype=torch.float32)
    pred = coords_topk[:, 0]
    gt = coords_gt

    dist_err = torch.norm(pred[:, :2] - gt[:, :2], p=2, dim=-1)
    rot_diff = torch.abs(pred[:, 2] - gt[:, 2])
    rot_err = torch.rad2deg(torch.minimum(rot_diff, 2 * torch.pi - rot_diff))
    pred_scale = pred[:, 3].clamp(min=1e-6)
    gt_scale = gt[:, 3].clamp(min=1e-6)
    scale_ratio = torch.maximum(pred_scale / gt_scale, gt_scale / pred_scale)

    d_mask = dist_err <= float(dist_th_nrc)
    dr_mask = d_mask & (rot_err <= float(rot_th_deg))
    drs_mask = dr_mask & (scale_ratio <= float(scale_ratio_th))

    dist_med_nrc = _median_or_empty(dist_err[d_mask])
    dist_med_meter = "" if dist_med_nrc == "" else float(dist_med_nrc) * float(nrc2meter)
    return {
        "n_success_dist": int(d_mask.sum().item()),
        "n_success_dist_rot": int(dr_mask.sum().item()),
        "n_success_dist_rot_scale": int(drs_mask.sum().item()),
        "dist_median_given_dist_success_nrc": dist_med_nrc,
        "dist_median_given_dist_success_m": dist_med_meter,
        "rot_median_given_dist_rot_success_deg": _median_or_empty(rot_err[dr_mask]),
        "scale_median_given_dist_rot_scale_success": _median_or_empty(scale_ratio[drs_mask]),
    }


def _threshold_configs(default_thresholds: Dict, nrc2meter: float) -> List[Dict[str, float]]:
    default_dist_nrc = float(default_thresholds["norm_dist"]) * 0.5
    default_dist_meter = default_dist_nrc * float(nrc2meter)
    default_rot = float(default_thresholds.get("rot", 11.0)) * 0.5
    return [
        {
            "config_id": "cfg_default_half",
            "label": "default_half_radius",
            "dist_th_nrc": default_dist_nrc,
            "dist_th_meter": default_dist_meter,
            "rot_th_deg": default_rot,
            "scale_ratio_th": 1.15,
        },
        {
            "config_id": "cfg_100m",
            "label": "100m_rot10_scale1.2",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": 1.2,
        },
        {
            "config_id": "cfg_50m",
            "label": "50m_rot10_scale1.2",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": 1.2,
        },
        {
            "config_id": "cfg_25m",
            "label": "25m_rot10_scale1.2",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": 1.2,
        },
    ]


def _stage_rows(details_path: Path, details: Dict, report: Dict) -> List[Dict[str, object]]:
    spec = dict(details.get("spec", {}))
    scene = str(report.get("scene_name", spec.get("scene_name", "")))
    dataset = str(report.get("dataset_name", spec.get("dataset_name", "")))
    experiment_dir = str(report.get("experiment_dir", spec.get("experiment_dir", details_path.parent.name)))
    split_mode, split_tag = _parse_split(experiment_dir)
    thresholds = dict(details.get("thresholds", report.get("thresholds", {})))
    nrc2meter = _derive_nrc2meter(thresholds)

    coords_gt = details["coords_gt"].to(torch.float32)
    k_values = list(report.get("k_values", [1, 5]))
    if not k_values:
        k_values = [1, 5]

    stage_defs = [
        ("baseline", "coords_topk_baseline"),
        ("rerank", "coords_topk_reranked"),
        ("refine", "coords_topk_refined"),
    ]
    cfgs = _threshold_configs(thresholds, nrc2meter=nrc2meter)
    rows = []
    for stage_name, coords_key in stage_defs:
        coords_topk = details[coords_key].to(torch.float32)
        for cfg in cfgs:
            progressive, err_stats = _progressive_eval(
                coords_topk=coords_topk,
                coords_gt=coords_gt,
                k_values=k_values,
                dist_th_nrc=cfg["dist_th_nrc"],
                rot_th_deg=cfg["rot_th_deg"],
                scale_ratio_th=cfg["scale_ratio_th"],
            )
            conditional = _conditional_top1_medians(
                coords_topk=coords_topk,
                coords_gt=coords_gt,
                dist_th_nrc=cfg["dist_th_nrc"],
                rot_th_deg=cfg["rot_th_deg"],
                scale_ratio_th=cfg["scale_ratio_th"],
                nrc2meter=nrc2meter,
            )

            dist_mean_nrc = float(err_stats["mean_dist_err_top1"])
            dist_median_nrc = float(err_stats["median_dist_err_top1"])
            row = {
                "scene": scene,
                "dataset": dataset,
                "experiment_dir": experiment_dir,
                "split_mode": split_mode,
                "split_tag": split_tag,
                "stage": stage_name,
                "config_id": cfg["config_id"],
                "config_label": cfg["label"],
                "n_queries": int(coords_gt.shape[0]),
                "k_values": _json_cell(k_values),
                "details_path": str(details_path),
                "report_path": str(details_path.with_name("gim_refine_report.json")),
                "bundle_path": str(spec.get("bundle_path", report.get("bundle_path", ""))),
                "refine_target_retrieval_dir": str(
                    details.get(
                        "refine_target_retrieval_dir",
                        report.get("refine_target_retrieval_dir", ""),
                    )
                ),
                "matcher_model": str(report.get("matcher_model", "")),
                "topn_match": int(report.get("topn_match", 0) or 0),
                "min_inliers_for_refine": int(report.get("min_inliers_for_refine", 0) or 0),
                "nrc2meter": float(nrc2meter),
                "dist_th_nrc": float(cfg["dist_th_nrc"]),
                "dist_th_meter": float(cfg["dist_th_meter"]),
                "rot_th_deg": float(cfg["rot_th_deg"]),
                "scale_ratio_th": float(cfg["scale_ratio_th"]),
                "dist_recall": _json_cell(progressive["dist_recall"]),
                "dist_rot_recall": _json_cell(progressive["dist_rot_recall"]),
                "dist_rot_scale_recall": _json_cell(progressive["dist_rot_scale_recall"]),
                "dist_recall_top1": float(progressive["dist_recall"].get("top1_acc", 0.0)),
                "dist_recall_top5": float(progressive["dist_recall"].get("top5_acc", 0.0)),
                "dist_rot_recall_top1": float(progressive["dist_rot_recall"].get("top1_acc", 0.0)),
                "dist_rot_recall_top5": float(progressive["dist_rot_recall"].get("top5_acc", 0.0)),
                "dist_rot_scale_recall_top1": float(progressive["dist_rot_scale_recall"].get("top1_acc", 0.0)),
                "dist_rot_scale_recall_top5": float(progressive["dist_rot_scale_recall"].get("top5_acc", 0.0)),
                "dist_mean_nrc": dist_mean_nrc,
                "dist_mean_m": dist_mean_nrc * float(nrc2meter),
                "dist_median_nrc": dist_median_nrc,
                "dist_median_m": dist_median_nrc * float(nrc2meter),
                "rot_mean_deg": float(err_stats["mean_rot_err_top1"]),
                "rot_median_deg": float(err_stats["median_rot_err_top1"]),
                "scale_mean": float(err_stats["mean_scale_ratio_top1"]),
                "scale_median": float(err_stats["median_scale_ratio_top1"]),
                **conditional,
            }
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export baseline/refine progressive recall and error metrics for DAC GIM refine runs."
    )
    parser.add_argument("--details-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    details_root = args.details_root.resolve()
    detail_paths = _iter_detail_paths(details_root)
    if not detail_paths:
        raise FileNotFoundError(f"No gim_refine_details.pt found under {details_root}")

    rows = []
    for details_path in detail_paths:
        details = _load_torch(details_path)
        _normalize_refined_topk_order(details)
        report = _load_json(details_path.with_name("gim_refine_report.json"))
        rows.extend(_stage_rows(details_path=details_path, details=details, report=report))

    rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            str(r["scene"]),
            str(r["split_mode"]),
            str(r["stage"]),
            str(r["config_id"]),
        )
    )
    _write_csv(args.output_csv, rows)
    print(f"[GIMRefineDACMetrics] wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
