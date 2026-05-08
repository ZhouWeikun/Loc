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

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords


DEFAULT_GALLERY_ROOT = REPO_ROOT / "gen_fm_exps" / "gallery_bank_stage2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "stage2_progressive_metrics_zurich_interval"
DEFAULT_INCLUDE_RE = r"zurich_interval"
DEFAULT_GALLERY_DIR_LIST = None
BUNDLE_FILENAME = "stage2_retrieval_eval_bundle.pt"

# Edit this block when changing Stage-2 progressive recall / median thresholds.
# dist can be specified either as:
# - {"dist_lambda": ...}: multiplier on the scene half-image radius in NRC units.
# - {"dist_th_meter": ...}: meter threshold converted to NRC via nrc2meter.
RECALL_CONFIGS = [
    {
        "id": "cfg01",
        "label": "default_distlambda0p55_rot5p5_scale1p2",
        "dist_lambda": 0.55,
        "rot_th_deg": 5.5,
        "scale_ratio_th": 1.2,
    },
    {
        "id": "cfg02",
        "label": "100m_rot10_scale1p2",
        "dist_th_meter": 100.0,
        "rot_th_deg": 10.0,
        "scale_ratio_th": 1.2,
    },
    {
        "id": "cfg04",
        "label": "50m_rot10_scale1p2",
        "dist_th_meter": 50.0,
        "rot_th_deg": 10.0,
        "scale_ratio_th": 1.2,
    },
    {
        "id": "cfg06",
        "label": "25m_rot10_scale1p2",
        "dist_th_meter": 25.0,
        "rot_th_deg": 10.0,
        "scale_ratio_th": 1.2,
    },
]


def _json_cell(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


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


def _float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_ratio(num, den):
    num = _float_or_none(num)
    den = _float_or_none(den)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return num / den


def _derive_nrc2meter(report: Dict) -> float:
    thresholds = dict(report.get("thresholds", {}) or {})
    nrc2meter = _float_or_none(thresholds.get("nrc2meter", None))
    if nrc2meter is not None and nrc2meter > 0:
        return nrc2meter

    candidates = [
        _safe_ratio(report.get("error_rc_meter"), report.get("error_rc_norm")),
        _safe_ratio(report.get("error_rc_meter_median"), report.get("error_rc_norm_median")),
    ]
    candidates = [v for v in candidates if v is not None and v > 0]
    if not candidates:
        raise ValueError("Unable to derive nrc2meter from report.")
    return float(candidates[0])


def _derive_halfimg_radius_nrc(report: Dict, config: Dict) -> float:
    thresholds = dict(report.get("thresholds", {}) or {})
    eval_cfg = dict(config.get("retrieval_eval_cfg", {}) or {})
    norm_dist = _float_or_none(thresholds.get("norm_dist", None))
    if norm_dist is None or norm_dist <= 0:
        raise ValueError("Missing positive thresholds.norm_dist; cannot derive default dist threshold.")
    dist_lambda = _float_or_none(eval_cfg.get("dist_lambda", None))
    if dist_lambda is None:
        dist_lambda = 1.0
    if dist_lambda <= 0:
        raise ValueError(f"Invalid dist_lambda={dist_lambda}; cannot derive default dist threshold.")
    return float(norm_dist) / float(dist_lambda)


def _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th) -> str:
    return (
        f"dist={float(dist_th_nrc):.6f} nrc / {float(dist_th_meter):.3f} m; "
        f"rot={float(rot_th_deg):.1f} deg; "
        f"scale={float(scale_ratio_th):.3f}x"
    )


def _recall_config_specs(halfimg_radius_nrc: float, nrc2meter: float) -> List[Dict[str, object]]:
    specs = []
    for item in RECALL_CONFIGS:
        spec = dict(item)
        if "dist_lambda" in spec:
            dist_th_nrc = float(halfimg_radius_nrc) * float(spec["dist_lambda"])
            dist_th_meter = dist_th_nrc * float(nrc2meter)
        elif "dist_th_meter" in spec:
            dist_th_meter = float(spec["dist_th_meter"])
            dist_th_nrc = dist_th_meter / max(float(nrc2meter), 1e-8)
        else:
            raise ValueError(f"Recall config must define dist_lambda or dist_th_meter: {item}")
        spec["dist_th_nrc"] = float(dist_th_nrc)
        spec["dist_th_meter"] = float(dist_th_meter)
        spec["rot_th_deg"] = float(spec["rot_th_deg"])
        spec["scale_ratio_th"] = float(spec["scale_ratio_th"])
        specs.append(spec)
    return specs


def _iter_bundle_paths(gallery_root: Path, include_re: str) -> Iterable[Path]:
    pattern = re.compile(include_re) if include_re else None
    for path in sorted(gallery_root.glob(f"**/{BUNDLE_FILENAME}")):
        text = str(path)
        if pattern is not None and pattern.search(text) is None:
            continue
        yield path


def _read_gallery_dir_list(path: Path) -> List[Path]:
    paths = []
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            gallery_dir = Path(line)
            if not gallery_dir.is_absolute():
                gallery_dir = (base_dir / gallery_dir).resolve()
            if not gallery_dir.is_dir():
                raise FileNotFoundError(f"Gallery directory from {path}:{lineno} does not exist: {gallery_dir}")
            bundle_path = gallery_dir / BUNDLE_FILENAME
            if not bundle_path.is_file():
                raise FileNotFoundError(f"Missing {BUNDLE_FILENAME} for {path}:{lineno}: {bundle_path}")
            paths.append(bundle_path)
    return paths


def _resolve_bundle_paths(gallery_root: Path, include_re: str, gallery_dir_list: Path = None) -> List[Path]:
    if gallery_dir_list is not None:
        return _read_gallery_dir_list(gallery_dir_list)
    return list(_iter_bundle_paths(gallery_root, include_re))


def _parse_overlap_from_path(path: Path):
    match = re.search(r"overlap(\d+)", str(path))
    if not match:
        return ""
    return int(match.group(1)) / 100.0


def _safe_get_top(metric_dict: Dict, key: str):
    try:
        return float(metric_dict.get(key, 0.0))
    except Exception:
        return 0.0


def _meters_from_nrc(value, nrc2meter):
    value = _float_or_none(value)
    if value is None:
        return ""
    return float(value) * float(nrc2meter)


def _flatten_error_metrics(prefix: str, progressive_errors: Dict, nrc2meter: float) -> Dict[str, object]:
    group_specs = [
        ("D", "dist_recall"),
        ("DR", "dist_rot_recall"),
        ("DRS", "dist_rot_scale_recall"),
    ]
    row = {}
    for short_name, group_key in group_specs:
        metrics = dict(progressive_errors.get(group_key, {}) or {})
        group_prefix = f"{prefix}_{short_name}"
        row[f"{group_prefix}_n_success_top1"] = metrics.get("n_success_top1", "")
        row[f"{group_prefix}_top1_success_rate"] = metrics.get("top1_success_rate", "")
        dist_med = metrics.get("median_dist_err_top1_given_success", None)
        row[f"{group_prefix}_median_dist_err_top1_given_success_nrc"] = "" if dist_med is None else dist_med
        row[f"{group_prefix}_median_dist_err_top1_given_success_m"] = _meters_from_nrc(dist_med, nrc2meter)
        row[f"{group_prefix}_median_rot_err_top1_given_success_deg"] = (
            "" if metrics.get("median_rot_err_top1_given_success", None) is None
            else metrics.get("median_rot_err_top1_given_success")
        )
        row[f"{group_prefix}_median_scale_ratio_top1_given_success"] = (
            "" if metrics.get("median_scale_ratio_top1_given_success", None) is None
            else metrics.get("median_scale_ratio_top1_given_success")
        )
    return row


def _build_rows(bundle_path: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    coords_topk = payload["coords_topk"].to(torch.float32)
    coords_gt = payload["coords_gt"].to(torch.float32)
    report = dict(payload["report"])
    config = dict(payload["config"])
    gallery_summary = dict(report.get("runtime_gallery_summary", {}) or config.get("gallery_summary", {}) or {})
    layout_cfg = dict(config.get("layout_cfg", {}) or gallery_summary.get("layout_cfg", {}) or {})
    k_values = tuple(int(v) for v in report.get("k_values", [1, 5, 10, 20, 50, 256, 512, 1024]))
    nrc2meter = _derive_nrc2meter(report)
    halfimg_radius_nrc = _derive_halfimg_radius_nrc(report, config)
    specs = _recall_config_specs(halfimg_radius_nrc=halfimg_radius_nrc, nrc2meter=nrc2meter)

    shared_errors = dict(report.get("shared_errors", {}) or {})
    base_row = {
        "scene": str(report.get("scene_name", "")),
        "gallery_dir": str(bundle_path.parent),
        "bundle_path": str(bundle_path),
        "overlap": layout_cfg.get("overlap", _parse_overlap_from_path(bundle_path)),
        "layout_mode": layout_cfg.get("mode", ""),
        "delta_rot_deg": layout_cfg.get("delta_rot_deg", ""),
        "n_scales": layout_cfg.get("n_scales", ""),
        "scale_mode": layout_cfg.get("scale_mode", ""),
        "n_queries": int(report.get("n_queries", coords_gt.shape[0])),
        "k_values": _json_cell(list(k_values)),
        "nrc2meter": nrc2meter,
        "halfimg_radius_nrc": halfimg_radius_nrc,
        "global_dist_mean_nrc": shared_errors.get("mean_dist_err_top1", report.get("error_rc_norm", "")),
        "global_dist_median_nrc": shared_errors.get("median_dist_err_top1", report.get("error_rc_norm_median", "")),
        "global_dist_mean_m": report.get("error_rc_meter", ""),
        "global_dist_median_m": report.get("error_rc_meter_median", ""),
        "global_rot_mean_deg": shared_errors.get("mean_rot_err_top1", report.get("error_rot_deg", "")),
        "global_rot_median_deg": shared_errors.get("median_rot_err_top1", report.get("error_rot_deg_median", "")),
        "global_scale_ratio_mean": shared_errors.get("mean_scale_ratio_top1", report.get("error_scale_ratio", "")),
        "global_scale_ratio_median": shared_errors.get("median_scale_ratio_top1", report.get("error_scale_ratio_median", "")),
        "stage2_ckpt": str(config.get("stage2_ckpt", "")),
    }

    wide_row = dict(base_row)
    long_rows = []
    for spec in specs:
        acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
            coords_topk,
            coords_gt,
            dist_th=float(spec["dist_th_nrc"]),
            rot_th_deg=float(spec["rot_th_deg"]),
            scale_ratio_th=float(spec["scale_ratio_th"]),
            k_values=k_values,
        )
        progressive = dict(acc_metrics_raw["progressive_acc_metrics"])
        progressive_errors = dict(acc_metrics_raw["progressive_error_metrics"])
        prefix = str(spec["id"])
        threshold = _threshold_cell(
            dist_th_nrc=spec["dist_th_nrc"],
            dist_th_meter=spec["dist_th_meter"],
            rot_th_deg=spec["rot_th_deg"],
            scale_ratio_th=spec["scale_ratio_th"],
        )

        wide_row[f"{prefix}_label"] = spec["label"]
        wide_row[f"{prefix}_threshold"] = threshold
        wide_row[f"{prefix}_dist_recall"] = _json_cell(progressive["dist_recall"])
        wide_row[f"{prefix}_dist_rot_recall"] = _json_cell(progressive["dist_rot_recall"])
        wide_row[f"{prefix}_dist_rot_scale_recall"] = _json_cell(progressive["dist_rot_scale_recall"])
        wide_row[f"{prefix}_progressive_error_metrics"] = _json_cell(progressive_errors)
        wide_row[f"{prefix}_dist_recall_top1"] = _safe_get_top(progressive["dist_recall"], "top1_acc")
        wide_row[f"{prefix}_dist_rot_recall_top1"] = _safe_get_top(progressive["dist_rot_recall"], "top1_acc")
        wide_row[f"{prefix}_dist_rot_scale_recall_top1"] = _safe_get_top(progressive["dist_rot_scale_recall"], "top1_acc")
        wide_row.update(_flatten_error_metrics(prefix, progressive_errors, nrc2meter))

        long_row = dict(base_row)
        long_row.update({
            "config_id": spec["id"],
            "label": spec["label"],
            "dist_th_nrc": spec["dist_th_nrc"],
            "dist_th_meter": spec["dist_th_meter"],
            "rot_th_deg": spec["rot_th_deg"],
            "scale_ratio_th": spec["scale_ratio_th"],
            "threshold": threshold,
            "dist_recall": _json_cell(progressive["dist_recall"]),
            "dist_rot_recall": _json_cell(progressive["dist_rot_recall"]),
            "dist_rot_scale_recall": _json_cell(progressive["dist_rot_scale_recall"]),
            "progressive_error_metrics": _json_cell(progressive_errors),
            "dist_recall_top1": _safe_get_top(progressive["dist_recall"], "top1_acc"),
            "dist_rot_recall_top1": _safe_get_top(progressive["dist_rot_recall"], "top1_acc"),
            "dist_rot_scale_recall_top1": _safe_get_top(progressive["dist_rot_scale_recall"], "top1_acc"),
            "err_stats": _json_cell(err_stats),
        })
        long_row.update(_flatten_error_metrics(str(spec["id"]), progressive_errors, nrc2meter))
        long_rows.append(long_row)

    return wide_row, long_rows


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Export Stage2 progressive recall and success-conditioned median metrics from eval bundles."
    )
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--include-re", default=DEFAULT_INCLUDE_RE)
    parser.add_argument(
        "--gallery-dir-list",
        type=Path,
        default=DEFAULT_GALLERY_DIR_LIST,
        help=f"Text file with one gallery output directory per line. Each directory must contain {BUNDLE_FILENAME}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="visloc03_interval82_stage2_progressive")
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    bundle_paths = _resolve_bundle_paths(
        gallery_root=args.gallery_root,
        include_re=args.include_re,
        gallery_dir_list=args.gallery_dir_list,
    )
    if not bundle_paths and not args.allow_empty:
        if args.gallery_dir_list is not None:
            raise FileNotFoundError(f"No stage2 eval bundles listed in {args.gallery_dir_list}")
        raise FileNotFoundError(f"No stage2 eval bundles found under {args.gallery_root} matching {args.include_re!r}")

    wide_rows = []
    long_rows = []
    for bundle_path in bundle_paths:
        wide_row, one_long_rows = _build_rows(bundle_path)
        wide_rows.append(wide_row)
        long_rows.extend(one_long_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = args.output_dir / f"{args.output_prefix}_wide.csv"
    long_csv = args.output_dir / f"{args.output_prefix}_long.csv"
    _write_csv(wide_csv, wide_rows)
    _write_csv(long_csv, long_rows)
    print(f"[Stage2Progressive] bundles={len(bundle_paths)}")
    print(f"[Stage2Progressive] wrote wide: {wide_csv}")
    print(f"[Stage2Progressive] wrote long: {long_csv}")


if __name__ == "__main__":
    main()
