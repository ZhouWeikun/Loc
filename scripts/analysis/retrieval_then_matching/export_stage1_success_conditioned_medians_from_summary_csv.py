#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.util_core_eval import compute_progressive_topk_acc_from_coords


def _json_cell(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th) -> str:
    return (
        f"dist={float(dist_th_nrc):.6f} nrc / {float(dist_th_meter):.3f} m; "
        f"rot={float(rot_th_deg):.1f} deg; "
        f"scale={float(scale_ratio_th):.3f}x"
    )


def _load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid cfg: {path}")
    for key in ("configs",):
        if key not in cfg:
            raise KeyError(f"Missing key {key!r} in cfg {path}")
    return cfg


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config_applies(spec: Dict, meta: Dict[str, str]) -> bool:
    scene = str(meta.get("scene", ""))
    dataset = str(meta.get("dataset", ""))
    split_mode = str(meta.get("split_mode", ""))
    for key, value in (("scene", scene), ("dataset", dataset), ("split_mode", split_mode)):
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


def _parse_split(experiment_dir: str) -> str:
    match = re.search(r"_(interval|segment|segmnet)\d+_", str(experiment_dir))
    if not match:
        return ""
    mode = match.group(1)
    return "segment" if mode == "segmnet" else mode


def _safe_get_top(metric_dict: Dict, key: str) -> float:
    try:
        return float(metric_dict.get(key, 0.0))
    except Exception:
        return 0.0


def _median_or_empty(values: torch.Tensor):
    if values.numel() == 0:
        return ""
    return float(values.median().item())


def _progressive_eval(coords_topk, coords_gt, dist_th_nrc, rot_th_deg, scale_ratio_th, k_values):
    acc_metrics_raw, _ = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
        dist_th=float(dist_th_nrc),
        rot_th_deg=float(rot_th_deg),
        scale_ratio_th=float(scale_ratio_th),
        k_values=tuple(int(v) for v in k_values),
    )
    return dict(acc_metrics_raw["progressive_acc_metrics"])


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


def _bundle_path_from_row(row: Dict[str, str]) -> Path:
    bundle_path = Path(str(row["bundle_path"]))
    if not bundle_path.is_absolute():
        bundle_path = (REPO_ROOT / bundle_path).resolve()
    return bundle_path


def _build_rows(summary_row: Dict[str, str], recall_cfg: Dict, med_cfg: Dict):
    bundle_path = _bundle_path_from_row(summary_row)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    coords_topk = payload["coords_topk"].to(torch.float32)
    coords_gt = payload["coords_gt"].to(torch.float32)
    meta = dict(summary_row)
    meta["split_mode"] = _parse_split(str(summary_row.get("experiment_dir", "")))

    try:
        nrc2meter = float(summary_row["nrc2meter"])
    except Exception:
        thresholds = dict(payload["report"]["thresholds"])
        nrc2meter = float(thresholds["nrc2meter"])

    k_values = tuple(int(v) for v in recall_cfg.get("k_values", [1]))

    wide_row = dict(summary_row)
    wide_row["success_median_source_bundle_path"] = str(bundle_path)
    wide_row["success_median_recall_cfg"] = ""
    wide_row["success_median_med_cfg"] = ""
    long_rows = []

    for spec in med_cfg["configs"]:
        if not _config_applies(spec, meta):
            continue
        label = str(spec["label"])
        config_id = str(spec["id"])
        dist_th_meter = float(spec["dist_th_meter"])
        rot_th_deg = float(spec["rot_th_deg"])
        scale_ratio_th = float(spec["scale_ratio_th"])
        dist_th_nrc = dist_th_meter / max(float(nrc2meter), 1e-8)
        threshold = _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th)

        progressive = _progressive_eval(
            coords_topk=coords_topk,
            coords_gt=coords_gt,
            dist_th_nrc=dist_th_nrc,
            rot_th_deg=rot_th_deg,
            scale_ratio_th=scale_ratio_th,
            k_values=k_values,
        )
        conditional = _conditional_top1_medians(
            coords_topk=coords_topk,
            coords_gt=coords_gt,
            dist_th_nrc=dist_th_nrc,
            rot_th_deg=rot_th_deg,
            scale_ratio_th=scale_ratio_th,
            nrc2meter=nrc2meter,
        )

        wide_row[f"{config_id}_label"] = label
        wide_row[f"{config_id}_threshold"] = threshold
        wide_row[f"{config_id}_dist_recall"] = _json_cell(progressive["dist_recall"])
        wide_row[f"{config_id}_dist_rot_recall"] = _json_cell(progressive["dist_rot_recall"])
        wide_row[f"{config_id}_dist_rot_scale_recall"] = _json_cell(progressive["dist_rot_scale_recall"])
        wide_row[f"{config_id}_dist_recall_top1"] = _safe_get_top(progressive["dist_recall"], "top1_acc")
        wide_row[f"{config_id}_dist_rot_recall_top1"] = _safe_get_top(progressive["dist_rot_recall"], "top1_acc")
        wide_row[f"{config_id}_dist_rot_scale_recall_top1"] = _safe_get_top(progressive["dist_rot_scale_recall"], "top1_acc")
        for key, value in conditional.items():
            wide_row[f"{config_id}_{key}"] = value

        long_row = dict(summary_row)
        long_row.update(
            {
                "split_mode": meta["split_mode"],
                "config_id": config_id,
                "label": label,
                "dist_th_meter": dist_th_meter,
                "dist_th_nrc": dist_th_nrc,
                "rot_th_deg": rot_th_deg,
                "scale_ratio_th": scale_ratio_th,
                "threshold": threshold,
                "dist_recall": _json_cell(progressive["dist_recall"]),
                "dist_rot_recall": _json_cell(progressive["dist_rot_recall"]),
                "dist_rot_scale_recall": _json_cell(progressive["dist_rot_scale_recall"]),
                "dist_recall_top1": _safe_get_top(progressive["dist_recall"], "top1_acc"),
                "dist_rot_recall_top1": _safe_get_top(progressive["dist_rot_recall"], "top1_acc"),
                "dist_rot_scale_recall_top1": _safe_get_top(progressive["dist_rot_scale_recall"], "top1_acc"),
                **conditional,
            }
        )
        long_rows.append(long_row)

    return wide_row, long_rows


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Append success-conditioned median metrics to a Stage1 summary CSV using bundle_path."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--recall-cfg", type=Path, required=True)
    parser.add_argument("--med-recall-cfg", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--long-output-csv", type=Path, required=True)
    return parser


def main():
    args = build_argparser().parse_args()
    summary_rows = _load_rows(args.summary_csv)
    recall_cfg = _load_cfg(args.recall_cfg)
    med_cfg = _load_cfg(args.med_recall_cfg)

    wide_rows = []
    long_rows = []
    for row in summary_rows:
        wide_row, one_long_rows = _build_rows(row, recall_cfg=recall_cfg, med_cfg=med_cfg)
        wide_row["success_median_recall_cfg"] = str(args.recall_cfg)
        wide_row["success_median_med_cfg"] = str(args.med_recall_cfg)
        for item in one_long_rows:
            item["success_median_recall_cfg"] = str(args.recall_cfg)
            item["success_median_med_cfg"] = str(args.med_recall_cfg)
        wide_rows.append(wide_row)
        long_rows.extend(one_long_rows)

    _write_csv(args.output_csv, wide_rows)
    _write_csv(args.long_output_csv, long_rows)
    print(f"[Stage1SuccessMedian] wrote {len(wide_rows)} wide rows to {args.output_csv}")
    print(f"[Stage1SuccessMedian] wrote {len(long_rows)} long rows to {args.long_output_csv}")


if __name__ == "__main__":
    main()
