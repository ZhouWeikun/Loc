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
import yaml
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MATCHING_REFINE_ROOT = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "stage1_crtl_ckpts2exps"
    / "mathing_refine"
)
DEFAULT_RECALL_CFG = MATCHING_REFINE_ROOT / "matching_refine_recall_cfg.yaml"
DEFAULT_OUTPUT_CSV = MATCHING_REFINE_ROOT / "matching_refine_recall_multi_cfg.csv"
DEFAULT_LONG_OUTPUT_CSV = MATCHING_REFINE_ROOT / "matching_refine_recall_multi_cfg_long.csv"

from trainers.util_core_eval import compute_progressive_topk_acc_from_coords


def _json_cell(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th) -> str:
    return (
        f"dist={float(dist_th_nrc):.6f} nrc / {float(dist_th_meter):.3f} m; "
        f"rot={float(rot_th_deg):.1f} deg; "
        f"scale={float(scale_ratio_th):.3f}x"
    )


def _load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Recall cfg must be a mapping: {path}")
    for key in ("k_values", "stages", "configs"):
        if key not in cfg:
            raise KeyError(f"Recall cfg missing required key {key!r}: {path}")
    return cfg


def _iter_detail_paths(inputs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.name != "gim_refine_details.pt":
                raise FileNotFoundError(f"Expected gim_refine_details.pt file, got: {path}")
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.glob("**/gim_refine_details.pt")))
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    deduped = []
    seen = set()
    for path in sorted(paths):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _parse_split(experiment_dir: str) -> Tuple[str, str]:
    match = re.search(r"_(interval|segment|segmnet)(\d+)_", str(experiment_dir))
    if not match:
        return "", ""
    mode, tag = match.groups()
    if mode == "segmnet":
        mode = "segment"
    return mode, tag


def _safe_get_top(metric_dict: Dict, key: str) -> float:
    try:
        return float(metric_dict.get(key, 0.0))
    except Exception:
        return 0.0


def _config_applies(spec: Dict, meta: Dict[str, object]) -> bool:
    scene = str(meta.get("scene", ""))
    dataset = str(meta.get("dataset", ""))
    split_mode = str(meta.get("split_mode", ""))
    for key, value in (
        ("scene", scene),
        ("dataset", dataset),
        ("split_mode", split_mode),
    ):
        allowed = spec.get(key)
        if allowed is None:
            continue
        if isinstance(allowed, str):
            allowed_values = {allowed}
        else:
            allowed_values = {str(v) for v in allowed}
        if value not in allowed_values:
            return False
    scene_pattern = spec.get("scene_pattern")
    if scene_pattern is not None and re.search(str(scene_pattern), scene) is None:
        return False
    dataset_pattern = spec.get("dataset_pattern")
    if dataset_pattern is not None and re.search(str(dataset_pattern), dataset) is None:
        return False
    return True


def _derive_nrc2meter(thresholds: Dict) -> float:
    if "nrc2meter" in thresholds:
        return float(thresholds["nrc2meter"])
    if "dist_meter" in thresholds and "norm_dist" in thresholds:
        return float(thresholds["dist_meter"]) / max(float(thresholds["norm_dist"]), 1e-8)
    raise KeyError(f"Cannot derive nrc2meter from thresholds: {thresholds}")


def _wrap_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _affine_theta_rad(affine_2x3) -> float:
    return float(math.atan2(float(affine_2x3[1][0]), float(affine_2x3[0][0])))


def _affine_theta_and_scale(affine_2x3) -> Tuple[float, float]:
    theta = _affine_theta_rad(affine_2x3)
    a00, a01 = float(affine_2x3[0][0]), float(affine_2x3[0][1])
    a10, a11 = float(affine_2x3[1][0]), float(affine_2x3[1][1])
    sx = math.hypot(a00, a10)
    sy = math.hypot(a01, a11)
    return theta, 0.5 * (sx + sy)


def _query_image_height(query_path: str) -> float:
    with Image.open(query_path) as img:
        return float(img.size[1])


def _scale_correction_height(record: Dict, candidate: Dict, affine_2x3) -> float:
    _, affine_scale = _affine_theta_and_scale(affine_2x3)
    patch_meta = dict(candidate.get("patch_meta", {}))
    patch_size = float(patch_meta.get("patch_size", 224.0))
    query_height = _query_image_height(str(record["query_path"]))
    nominal_query_to_patch = patch_size / max(query_height, 1e-8)
    return float(affine_scale) / max(nominal_query_to_patch, 1e-8)


def _normalize_refined_topk_order(details: Dict) -> None:
    """Keep refined top-1 on top of reranked candidates, and backfill affine rot for old details."""
    refined = details.get("coords_topk_refined")
    reranked = details.get("coords_topk_reranked")
    if not (torch.is_tensor(refined) and torch.is_tensor(reranked)):
        return
    if tuple(refined.shape) != tuple(reranked.shape) or refined.ndim != 3 or refined.shape[-1] != 4:
        return
    fixed = reranked.clone()
    fixed[:, 0] = refined[:, 0].to(dtype=fixed.dtype, device=fixed.device)
    refine_pose = dict(details.get("refine_pose", {}))
    needs_rot_backfill = not bool(refine_pose.get("rot_from_affine", False))
    needs_scale_backfill = not bool(refine_pose.get("scale_from_affine", False))
    if needs_rot_backfill or needs_scale_backfill:
        for query_idx, record in enumerate(details.get("query_details", [])):
            if query_idx >= int(fixed.shape[0]):
                break
            refine_result = dict(record.get("refine_result", {}))
            if str(refine_result.get("status", "")) != "refined":
                continue
            rerank_order = record.get("rerank_order_prefix") or []
            if not rerank_order:
                continue
            try:
                candidate = record["candidates"][int(rerank_order[0])]
                affine = candidate["match_result"].get("affine_2x3")
                if affine is None:
                    continue
            except Exception:
                continue
            if needs_rot_backfill:
                theta = _affine_theta_rad(affine)
                fixed[query_idx, 0, 2] = _wrap_angle_rad(float(fixed[query_idx, 0, 2].item()) - theta)
            if needs_scale_backfill:
                try:
                    scale_corr = _scale_correction_height(record=record, candidate=candidate, affine_2x3=affine)
                except Exception:
                    continue
                fixed[query_idx, 0, 3] = float(fixed[query_idx, 0, 3].item()) * float(scale_corr)
        details["refine_pose"] = {
            "rot_from_affine": True,
            "rot_formula": "rot_new=wrap_pi(rot_old-atan2(affine[1,0],affine[0,0]))",
            "scale_from_affine": True,
            "scale_formula": "scale_new=scale_old*(affine_iso_scale/(patch_size/query_height))",
            "scale_norm": "height",
            "backfilled_by_recall_export": True,
        }
    details["coords_topk_refined"] = fixed


def _progressive_eval(coords_topk, coords_gt, k_values, dist_th_nrc, rot_th_deg, scale_ratio_th):
    acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
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
        "n_top1_D": int(d_mask.sum().item()),
        "n_top1_DR": int(dr_mask.sum().item()),
        "n_top1_DRS": int(drs_mask.sum().item()),
        "median_dist_err_top1_given_D": dist_med_nrc,
        "median_dist_err_meter_top1_given_D": dist_med_meter,
        "median_rot_err_top1_given_DR": _median_or_empty(rot_err[dr_mask]),
        "median_scale_ratio_top1_given_DRS": _median_or_empty(scale_ratio[drs_mask]),
    }


def _metadata(details_path: Path, details: Dict, report: Dict) -> Dict[str, object]:
    spec = dict(details.get("spec", {}))
    scene = str(report.get("scene_name", spec.get("scene_name", "")))
    dataset = str(report.get("dataset_name", spec.get("dataset_name", "")))
    aggregator = str(report.get("aggregator", spec.get("aggregator", "")))
    experiment_dir = str(report.get("experiment_dir", spec.get("experiment_dir", details_path.parent.name)))
    split_mode, split_tag = _parse_split(experiment_dir)
    refine_target_retrieval_dir = str(
        details.get(
            "refine_target_retrieval_dir",
            report.get("refine_target_retrieval_dir", Path(str(spec.get("bundle_path", ""))).parent),
        )
    )
    return {
        "scene": scene,
        "dataset": dataset,
        "aggregator": aggregator,
        "experiment_dir": experiment_dir,
        "split_mode": split_mode,
        "split_tag": split_tag,
        "matcher_model": str(report.get("matcher_model", "")),
        "topn_match": report.get("topn_match", ""),
        "min_inliers_for_refine": report.get("min_inliers_for_refine", ""),
        "n_queries": int(details["coords_gt"].shape[0]),
        "export_dir": str(report.get("export_dir", details_path.parent)),
        "details_path": str(details_path),
        "report_path": str(details_path.with_name("gim_refine_report.json")),
        "refine_target_retrieval_dir": refine_target_retrieval_dir,
        "bundle_path": str(spec.get("bundle_path", report.get("bundle_path", ""))),
        "stage1_ckpt": str(spec.get("stage1_ckpt", "")),
        "query_subset_rule": str(spec.get("query_subset_rule", report.get("query_subset_rule", ""))),
    }


def _build_rows(details_path: Path, cfg: Dict):
    details = _load_torch(details_path)
    _normalize_refined_topk_order(details)
    report = _load_json(details_path.with_name("gim_refine_report.json"))
    meta = _metadata(details_path, details, report)
    thresholds = dict(details.get("thresholds", report.get("thresholds", {})))
    nrc2meter = _derive_nrc2meter(thresholds)
    coords_gt = details["coords_gt"].to(torch.float32)
    max_k = int(details["coords_topk_baseline"].shape[1])
    k_values = [int(k) for k in cfg["k_values"] if int(k) <= max_k]
    if not k_values:
        raise ValueError(f"No valid k_values <= {max_k} for {details_path}")

    wide_row = dict(meta)
    wide_row.update(
        {
            "nrc2meter": float(nrc2meter),
            "k_values": _json_cell(k_values),
            "source_default_thresholds": _json_cell(thresholds),
        }
    )
    long_rows = []

    for spec in cfg["configs"]:
        if not _config_applies(spec, meta):
            continue
        config_id = str(spec["id"])
        label = str(spec["label"])
        dist_th_meter = float(spec["dist_th_meter"])
        rot_th_deg = float(spec["rot_th_deg"])
        scale_ratio_th = float(spec["scale_ratio_th"])
        dist_th_nrc = dist_th_meter / max(float(nrc2meter), 1e-8)
        threshold = _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th)

        wide_row[f"{config_id}_label"] = label
        wide_row[f"{config_id}_threshold"] = threshold

        for stage in cfg["stages"]:
            stage_name = str(stage["name"])
            coords_key = str(stage["coords_key"])
            coords_topk = details[coords_key].to(torch.float32)
            progressive, err_stats = _progressive_eval(
                coords_topk=coords_topk,
                coords_gt=coords_gt,
                k_values=k_values,
                dist_th_nrc=dist_th_nrc,
                rot_th_deg=rot_th_deg,
                scale_ratio_th=scale_ratio_th,
            )
            conditional_medians = _conditional_top1_medians(
                coords_topk=coords_topk,
                coords_gt=coords_gt,
                dist_th_nrc=dist_th_nrc,
                rot_th_deg=rot_th_deg,
                scale_ratio_th=scale_ratio_th,
                nrc2meter=nrc2meter,
            )

            for group_name in ("dist_recall", "dist_rot_recall", "dist_rot_scale_recall"):
                group_metrics = progressive[group_name]
                wide_row[f"{config_id}_{stage_name}_{group_name}"] = _json_cell(group_metrics)
                wide_row[f"{config_id}_{stage_name}_{group_name}_top1"] = _safe_get_top(group_metrics, "top1_acc")
                wide_row[f"{config_id}_{stage_name}_{group_name}_top5"] = _safe_get_top(group_metrics, "top5_acc")

            wide_row[f"{config_id}_{stage_name}_mean_dist_err_top1"] = err_stats["mean_dist_err_top1"]
            wide_row[f"{config_id}_{stage_name}_median_dist_err_top1"] = err_stats["median_dist_err_top1"]
            wide_row[f"{config_id}_{stage_name}_mean_rot_err_top1"] = err_stats["mean_rot_err_top1"]
            wide_row[f"{config_id}_{stage_name}_median_rot_err_top1"] = err_stats["median_rot_err_top1"]
            wide_row[f"{config_id}_{stage_name}_mean_scale_ratio_top1"] = err_stats["mean_scale_ratio_top1"]
            wide_row[f"{config_id}_{stage_name}_median_scale_ratio_top1"] = err_stats["median_scale_ratio_top1"]
            for key, value in conditional_medians.items():
                wide_row[f"{config_id}_{stage_name}_{key}"] = value

            long_row = dict(meta)
            long_row.update(
                {
                    "config_id": config_id,
                    "label": label,
                    "stage": stage_name,
                    "coords_key": coords_key,
                    "dist_th_meter": dist_th_meter,
                    "dist_th_nrc": dist_th_nrc,
                    "rot_th_deg": rot_th_deg,
                    "scale_ratio_th": scale_ratio_th,
                    "threshold": threshold,
                    "nrc2meter": float(nrc2meter),
                    "k_values": _json_cell(k_values),
                    "dist_recall": _json_cell(progressive["dist_recall"]),
                    "dist_rot_recall": _json_cell(progressive["dist_rot_recall"]),
                    "dist_rot_scale_recall": _json_cell(progressive["dist_rot_scale_recall"]),
                    "dist_recall_top1": _safe_get_top(progressive["dist_recall"], "top1_acc"),
                    "dist_rot_recall_top1": _safe_get_top(progressive["dist_rot_recall"], "top1_acc"),
                    "dist_rot_scale_recall_top1": _safe_get_top(progressive["dist_rot_scale_recall"], "top1_acc"),
                    "dist_recall_top5": _safe_get_top(progressive["dist_recall"], "top5_acc"),
                    "dist_rot_recall_top5": _safe_get_top(progressive["dist_rot_recall"], "top5_acc"),
                    "dist_rot_scale_recall_top5": _safe_get_top(progressive["dist_rot_scale_recall"], "top5_acc"),
                    **err_stats,
                    **conditional_medians,
                }
            )
            long_rows.append(long_row)

    return wide_row, long_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
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


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Recompute recall metrics from saved GIM matching/refine details."
    )
    parser.add_argument(
        "--details-root",
        type=Path,
        action="append",
        default=None,
        help="Directory containing gim_refine_details.pt files, or a single details file. Can be repeated.",
    )
    parser.add_argument("--recall-cfg", type=Path, default=DEFAULT_RECALL_CFG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--long-output-csv", type=Path, default=DEFAULT_LONG_OUTPUT_CSV)
    return parser


def main():
    args = build_argparser().parse_args()
    cfg = _load_cfg(args.recall_cfg)
    detail_inputs = args.details_root or [MATCHING_REFINE_ROOT]
    detail_paths = _iter_detail_paths(detail_inputs)
    if not detail_paths:
        raise FileNotFoundError(f"No gim_refine_details.pt files found under: {detail_inputs}")

    wide_rows = []
    long_rows = []
    for details_path in detail_paths:
        wide_row, detail_long_rows = _build_rows(details_path, cfg)
        wide_rows.append(wide_row)
        long_rows.extend(detail_long_rows)

    wide_rows.sort(key=lambda r: (str(r.get("split_mode", "")), str(r.get("dataset", "")), str(r.get("scene", "")), str(r.get("aggregator", "")), str(r.get("experiment_dir", ""))))
    long_rows.sort(key=lambda r: (str(r.get("split_mode", "")), str(r.get("dataset", "")), str(r.get("scene", "")), str(r.get("aggregator", "")), str(r.get("experiment_dir", "")), str(r.get("config_id", "")), str(r.get("stage", ""))))
    _write_csv(args.output_csv, wide_rows)
    _write_csv(args.long_output_csv, long_rows)
    print(f"[MatchingRefineRecall] wrote {len(wide_rows)} wide rows to {args.output_csv}")
    print(f"[MatchingRefineRecall] wrote {len(long_rows)} long rows to {args.long_output_csv}")


if __name__ == "__main__":
    main()
