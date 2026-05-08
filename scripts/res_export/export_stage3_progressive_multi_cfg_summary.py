#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords


def _json_cell(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th):
    return (
        f"dist={float(dist_th_nrc):.6f} nrc / {float(dist_th_meter):.3f} m; "
        f"rot={float(rot_th_deg):.1f} deg; "
        f"scale={float(scale_ratio_th):.3f}x"
    )


def _parse_experiment_name(experiment_name: str):
    match = re.match(r"^stage2_(.+?)_(interval|segment|segmnet)(\d+)_(.+)$", experiment_name)
    if not match:
        return {
            "scene_token": experiment_name,
            "scene": experiment_name,
            "dataset": None,
            "mode": None,
            "window_size": None,
            "experiment_suffix": None,
        }

    scene_token, mode, window_size, suffix = match.groups()
    if mode == "segmnet":
        mode = "segment"
    if scene_token.startswith("visloc") and scene_token[6:].isdigit():
        scene = f"visloc_{scene_token[6:]}"
        dataset = "visloc"
    else:
        scene = scene_token
        dataset = "wingtra"
    return {
        "scene_token": scene_token,
        "scene": scene,
        "dataset": dataset,
        "mode": mode,
        "window_size": int(window_size),
        "experiment_suffix": suffix,
    }


def _recall_config_specs(default_dist_th_nrc, default_dist_th_meter, nrc2meter):
    scale_ratio_th = 1.2
    return [
        {
            "id": "cfg01",
            "label": "default_dist_rot5p5_scale1p2",
            "dist_th_nrc": float(default_dist_th_nrc),
            "dist_th_meter": float(default_dist_th_meter),
            "rot_th_deg": 11.0 * 0.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg02",
            "label": "100m_rot10_scale1p2",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },

        {
            "id": "cfg05",
            "label": "50m_rot10_scale1p2",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg08",
            "label": "25m_rot10_scale1p2",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
    ]


def _resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name == "stage3_retrieval_bundle.pt":
            return path.parent
        raise FileNotFoundError(f"Unsupported file input: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    if (path / "stage3_retrieval_bundle.pt").is_file():
        return path
    run_dirs = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_dir()
            and child.name.startswith("stage3_triplets_")
            and (child / "stage3_retrieval_bundle.pt").is_file()
        ],
        key=lambda p: p.name,
    )
    if run_dirs:
        return run_dirs[-1]
    raise FileNotFoundError(f"No stage3 run dir found under: {path}")


def _load_manifest(run_dir: Path):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bundle(run_dir: Path):
    bundle_path = run_dir / "stage3_retrieval_bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing bundle: {bundle_path}")
    return torch.load(bundle_path, map_location="cpu")


def _topk_values_from_report(seed_mode_reports: dict, coords_topk: torch.Tensor):
    keys = []
    final_section = seed_mode_reports.get("seed_mode_final", {})
    acc_metrics = final_section.get("acc_metrics", {}) if isinstance(final_section, dict) else {}
    for key in acc_metrics.keys():
        match = re.match(r"top(\d+)_acc$", str(key))
        if match:
            keys.append(int(match.group(1)))
    if keys:
        return tuple(sorted(set(keys)))
    max_k = int(coords_topk.shape[1])
    fallback = [1, 5, 10, 16, 32, 64, 128, 256, 512, 1024]
    return tuple(k for k in fallback if k <= max_k)


def _progressive_recall(coords_topk, coords_gt, k_values, dist_th_nrc, rot_th_deg, scale_ratio_th):
    acc_metrics_raw, _ = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
        dist_th=float(dist_th_nrc),
        rot_th_deg=float(rot_th_deg),
        scale_ratio_th=float(scale_ratio_th),
        k_values=tuple(int(k) for k in k_values),
    )
    return dict(acc_metrics_raw["progressive_acc_metrics"])


def _build_row(run_dir: Path):
    bundle = _load_bundle(run_dir)
    manifest = _load_manifest(run_dir)
    experiment_name = run_dir.parents[1].name
    epoch_dir_name = run_dir.parent.name
    parsed = _parse_experiment_name(experiment_name)

    coords_topk = bundle["coords_evo"].to(torch.float32)
    coords_gt = bundle["coords_gt"].to(torch.float32)
    seed_mode_reports = dict(bundle.get("seed_mode_reports", {}))
    seed_mode_eval_config = dict(bundle.get("seed_mode_eval_config", {}))
    final_eval_cfg = dict(seed_mode_eval_config.get("final_eval_cfg_effective", {}))
    dataset_metrics = bundle.get("dataset_metrics", None)
    if dataset_metrics is None:
        dataset_metrics = manifest.get("dataset_metrics", None)
    if isinstance(dataset_metrics, dict):
        halfimg_radius_nrc = float(dataset_metrics["halfimg_radius_nrc"])
        halfimg_radius_meter = float(dataset_metrics["halfimg_radius_meter"])
        nrc2meter = float(dataset_metrics["nrc2meter_factor"])
    elif "dist_th" in final_eval_cfg and "dist_th_meter" in final_eval_cfg:
        default_dist_th_nrc = float(final_eval_cfg["dist_th"])
        default_dist_th_meter = float(final_eval_cfg["dist_th_meter"])
        nrc2meter = default_dist_th_meter / max(default_dist_th_nrc, 1e-8)
        dist_lambda = float(final_eval_cfg.get("dist_lambda", 1.1 * 0.5))
        halfimg_radius_nrc = default_dist_th_nrc / max(dist_lambda, 1e-8)
        halfimg_radius_meter = halfimg_radius_nrc * nrc2meter
    else:
        raise KeyError(
            f"dataset_metrics missing from run and cannot be derived from final_eval_cfg_effective: {run_dir}"
        )

    # cfg01 follows gen_fm_exps/测试指标要求.md:
    # dist_th = sat_dataset.halfimg_radius_nrc * 1.1 * 0.5
    default_dist_lambda = 1.1 * 0.5
    default_dist_th_nrc = default_dist_lambda * halfimg_radius_nrc
    default_dist_th_meter = default_dist_th_nrc * nrc2meter

    k_values = _topk_values_from_report(seed_mode_reports, coords_topk)
    recall_specs = _recall_config_specs(
        default_dist_th_nrc=default_dist_th_nrc,
        default_dist_th_meter=default_dist_th_meter,
        nrc2meter=nrc2meter,
    )

    final_report = dict(seed_mode_reports.get("seed_mode_final", {}))
    final_err_stats = dict(final_report.get("err_stats", {}))
    final_acc_metrics = dict(final_report.get("acc_metrics", {}))

    row = {
        "dataset": parsed["dataset"],
        "scene": parsed["scene"],
        "mode": parsed["mode"],
        "window_size": parsed["window_size"],
        "experiment_name": experiment_name,
        "experiment_suffix": parsed["experiment_suffix"],
        "epoch_dir_name": epoch_dir_name,
        "run_dir_name": run_dir.name,
        "n_queries": int(coords_gt.shape[0]),
        "dist_median": float(final_err_stats["median_dist_err_top1"]) * nrc2meter,
        "dist_mean": float(final_err_stats["mean_dist_err_top1"]) * nrc2meter,
        "dist_median_nrc": float(final_err_stats["median_dist_err_top1"]),
        "dist_mean_nrc": float(final_err_stats["mean_dist_err_top1"]),
        "rot_median": float(final_err_stats["median_rot_err_top1"]),
        "rot_mean": float(final_err_stats["mean_rot_err_top1"]),
        "scale_median": float(final_err_stats["median_scale_ratio_top1"]),
        "scale_mean": float(final_err_stats["mean_scale_ratio_top1"]),
        "halfimg_radius_nrc": halfimg_radius_nrc,
        "halfimg_radius_meter": halfimg_radius_meter,
        "nrc2meter": nrc2meter,
        "default_seed_final_top1_acc": float(final_acc_metrics.get("top1_acc", 0.0)),
        "default_seed_final_top5_acc": float(final_acc_metrics.get("top5_acc", 0.0)),
        "default_seed_final_top10_acc": float(final_acc_metrics.get("top10_acc", 0.0)),
        "report_path": str((run_dir / "seed_mode_reports.json").resolve()),
        "bundle_path": str((run_dir / "stage3_retrieval_bundle.pt").resolve()),
        "run_dir": str(run_dir.resolve()),
    }

    for spec in recall_specs:
        progressive = _progressive_recall(
            coords_topk=coords_topk,
            coords_gt=coords_gt,
            k_values=k_values,
            dist_th_nrc=spec["dist_th_nrc"],
            rot_th_deg=spec["rot_th_deg"],
            scale_ratio_th=spec["scale_ratio_th"],
        )
        prefix = spec["id"]
        row[f"{prefix}_label"] = spec["label"]
        row[f"{prefix}_threshold"] = _threshold_cell(
            dist_th_nrc=spec["dist_th_nrc"],
            dist_th_meter=spec["dist_th_meter"],
            rot_th_deg=spec["rot_th_deg"],
            scale_ratio_th=spec["scale_ratio_th"],
        )
        row[f"{prefix}_dist_recall"] = _json_cell(progressive["dist_recall"])
        row[f"{prefix}_dist_rot_recall"] = _json_cell(progressive["dist_rot_recall"])
        row[f"{prefix}_dist_rot_scale_recall"] = _json_cell(progressive["dist_rot_scale_recall"])
    return row


def main():
    parser = argparse.ArgumentParser(description="Export Stage3 progressive multi-config summary CSV.")
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="Run dir, run root (containing stage3_triplets_*), or stage3_retrieval_bundle.pt path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "gen_fm_exps" / "analysis" / "stage3_best_epoch_summary_progressive_multi_cfg.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    run_dirs = [_resolve_run_dir(path) for path in args.input_paths]
    rows = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        row = _build_row(run_dir)
        rows.append(row)
        print(f"[Stage3Summary] processed {idx}/{len(run_dirs)} -> {row['scene']} / {row['experiment_name']}")

    rows.sort(key=lambda row: (row["dataset"] or "", row["scene"] or "", row["mode"] or "", row["experiment_name"]))

    fieldnames = [
        "dataset",
        "scene",
        "mode",
        "window_size",
        "experiment_name",
        "experiment_suffix",
        "epoch_dir_name",
        "run_dir_name",
        "n_queries",
        "dist_median",
        "dist_mean",
        "dist_median_nrc",
        "dist_mean_nrc",
        "rot_median",
        "rot_mean",
        "scale_median",
        "scale_mean",
        "halfimg_radius_nrc",
        "halfimg_radius_meter",
        "nrc2meter",
        "default_seed_final_top1_acc",
        "default_seed_final_top5_acc",
        "default_seed_final_top10_acc",
        "report_path",
        "bundle_path",
        "run_dir",
    ]
    for idx in range(1, 11):
        prefix = f"cfg{idx:02d}"
        fieldnames.extend([
            f"{prefix}_label",
            f"{prefix}_threshold",
            f"{prefix}_dist_recall",
            f"{prefix}_dist_rot_recall",
            f"{prefix}_dist_rot_scale_recall",
        ])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Stage3Summary] wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
