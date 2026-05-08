#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords


def _bundle_paths(gallery_root: Path):
    patterns = [
        "zurich_overlap050_bins63x45x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_g2m_epoch049/stage1_retrieval_eval_bundle.pt",
        "zurich_overlap050_bins63x45x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt",
        "zurich_overlap050_bins63x45x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_netvlad_epoch049/stage1_retrieval_eval_bundle.pt",
        "zuchwil_overlap050_bins48x67x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_g2m_epoch049/stage1_retrieval_eval_bundle.pt",
        "zuchwil_overlap050_bins48x67x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt",
        "zuchwil_overlap050_bins48x67x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_netvlad_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_03_overlap050_bins46x57x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_g2m_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_03_overlap050_bins46x57x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_03_overlap050_bins46x57x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_03_overlap050_bins46x57x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad_epoch060/stage1_retrieval_eval_bundle.pt",
        "visloc_04_overlap050_bins56x21x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_g2m_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_04_overlap050_bins56x21x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_04_overlap050_bins56x21x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad_epoch049/stage1_retrieval_eval_bundle.pt",
        "visloc_04_overlap050_bins56x21x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad_epoch060/stage1_retrieval_eval_bundle.pt",
    ]
    paths = [gallery_root / rel for rel in patterns]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected bundle files:\n" + "\n".join(missing))
    return paths


def _json_cell(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _threshold_cell(dist_th_nrc, dist_th_meter, rot_th_deg, scale_ratio_th):
    return (
        f"dist={float(dist_th_nrc):.6f} nrc / {float(dist_th_meter):.3f} m; "
        f"rot={float(rot_th_deg):.1f} deg; "
        f"scale={float(scale_ratio_th):.3f}x"
    )


def _recall_config_specs(default_dist_th_nrc, default_dist_th_meter, nrc2meter):
    scale_ratio_th = 1.15
    return [
        {
            "id": "cfg01",
            "label": "default_dist_rot5p5_scale1p15",
            "dist_th_nrc": float(default_dist_th_nrc),
            "dist_th_meter": float(default_dist_th_meter),
            "rot_th_deg": 11.0 * 0.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg02",
            "label": "100m_rot10_scale1p15",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg03",
            "label": "100m_rot7p5_scale1p15",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 7.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg04",
            "label": "100m_rot5_scale1p15",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 5.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg05",
            "label": "50m_rot10_scale1p15",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg06",
            "label": "50m_rot7p5_scale1p15",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 7.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg07",
            "label": "50m_rot5_scale1p15",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 5.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg08",
            "label": "25m_rot10_scale1p15",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg09",
            "label": "25m_rot7p5_scale1p15",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 7.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg10",
            "label": "25m_rot5_scale1p15",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 5.0,
            "scale_ratio_th": scale_ratio_th,
        },
    ]


def _progressive_recall(coords_topk, coords_gt, k_values, dist_th_nrc, rot_th_deg, scale_ratio_th):
    acc_metrics_raw, _ = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
        dist_th=dist_th_nrc,
        rot_th_deg=rot_th_deg,
        scale_ratio_th=scale_ratio_th,
        k_values=tuple(int(k) for k in k_values),
    )
    return dict(acc_metrics_raw["progressive_acc_metrics"])


def _load_bundle(path: Path):
    return torch.load(path, map_location="cpu")


def _build_row(bundle_path: Path):
    payload = _load_bundle(bundle_path)
    report = dict(payload["report"])
    config = dict(payload["config"])
    thresholds = dict(report["thresholds"])
    err_stats = dict(report["err_stats"])
    coords_topk = payload["coords_topk"].to(torch.float32)
    coords_gt = payload["coords_gt"].to(torch.float32)

    scene_name = str(report["scene_name"])
    experiment_dir = bundle_path.parent.name
    dataset_name = "visloc" if scene_name.startswith("visloc_") else "wingtra"
    aggregator = experiment_dir.split("_dinov2B2_")[-1].rsplit("_epoch", 1)[0]
    epoch_tag = experiment_dir.rsplit("_epoch", 1)[-1]
    nrc2meter = float(thresholds["nrc2meter"])
    k_values = tuple(int(k) for k in report["k_values"])

    default_dist_th_nrc = float(thresholds["norm_dist"])
    default_dist_th_meter = float(thresholds["dist_meter"])
    recall_specs = _recall_config_specs(
        default_dist_th_nrc=default_dist_th_nrc,
        default_dist_th_meter=default_dist_th_meter,
        nrc2meter=nrc2meter,
    )

    dist_mean_nrc = float(err_stats["mean_dist_err_top1"])
    dist_median_nrc = float(err_stats["median_dist_err_top1"])
    dist_mean_m = dist_mean_nrc * nrc2meter
    dist_median_m = dist_median_nrc * nrc2meter

    row = {
        "dataset": dataset_name,
        "scene": scene_name,
        "aggregator": aggregator,
        "experiment_dir": experiment_dir,
        "epoch": epoch_tag,
        "n_queries": int(report["n_queries"]),
        "bundle_path": str(bundle_path),
        "report_path": str(bundle_path.with_name("stage1_retrieval_eval_report.json")),
        "nrc2meter": nrc2meter,
        "dist_median": dist_median_m,
        "dist_mean": dist_mean_m,
        "dist_median_nrc": dist_median_nrc,
        "dist_mean_nrc": dist_mean_nrc,
        "rot_median": float(err_stats["median_rot_err_top1"]),
        "rot_mean": float(err_stats["mean_rot_err_top1"]),
        "scale_median": float(err_stats["median_scale_ratio_top1"]),
        "scale_mean": float(err_stats["mean_scale_ratio_top1"]),
        "stage1_ckpt": str(config.get("stage1_ckpt", "")),
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
    parser = argparse.ArgumentParser(description="Export Stage1 interval82 gallery summary CSV.")
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=REPO_ROOT / "gen_fm_exps" / "gallery_bank_stage1",
        help="Root directory containing Stage1 gallery-bank outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "gen_fm_exps" / "analysis" / "stage1_interval82_gallery_summary_progressive_multi_cfg.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    bundle_paths = _bundle_paths(args.gallery_root)
    rows = []
    for idx, path in enumerate(bundle_paths, start=1):
        row = _build_row(bundle_path=path)
        rows.append(row)
        print(f"[Stage1Summary] processed {idx}/{len(bundle_paths)} -> {row['scene']} / {row['aggregator']}")
    rows.sort(key=lambda row: (row["dataset"], row["scene"], row["aggregator"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "scene",
        "aggregator",
        "experiment_dir",
        "epoch",
        "n_queries",
        "dist_median",
        "dist_mean",
        "dist_median_nrc",
        "dist_mean_nrc",
        "rot_median",
        "rot_mean",
        "scale_median",
        "scale_mean",
        "nrc2meter",
        "stage1_ckpt",
        "report_path",
        "bundle_path",
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
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Stage1Summary] wrote {len(rows)} rows to {args.output_csv}")
    print("[Stage1Summary] recall configs: cfg01 default + cfg02-cfg10 manual meter thresholds")


if __name__ == "__main__":
    main()
