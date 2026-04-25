#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

CFG_SPECS = [
    ("cfg04", "100m_rot5_scale1p15"),
    ("cfg07", "50m_rot5_scale1p15"),
    ("cfg10", "25m_rot5_scale1p15"),
]
def main():
    parser = argparse.ArgumentParser(
        description="Export compact Stage1 overlap050 recall summary with raw progressive recall JSON for 100/50/25m @ rot=5, scale=1.15."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Source stage1_ctrl_gallery_overlap050.csv")
    parser.add_argument("--output-csv", type=Path, required=True, help="Target compact CSV path")
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        rows_in = list(csv.DictReader(f))

    fieldnames = [
        "dataset",
        "scene",
        "aggregator",
        "experiment_dir",
        "epoch",
        "best_ckpt_path",
        "stage1_ckpt",
        "gallery_root_dir",
        "gallery_mode",
        "gallery_overlap",
        "gallery_n_rot",
        "gallery_n_scale",
        "gallery_scale_mode",
        "gallery_save_dir",
        "eval_artifact_dir",
        "bundle_path",
        "report_path",
        "dist_median_m",
        "rot_median_deg",
        "scale_median_ratio",
    ]
    for _, label in CFG_SPECS:
        fieldnames.append(f"{label}_threshold")
        fieldnames.append(f"{label}_progressive_recall")

    rows_out = []
    for row_in in rows_in:
        row_out = {
            "dataset": row_in.get("dataset", ""),
            "scene": row_in.get("scene", ""),
            "aggregator": row_in.get("aggregator", ""),
            "experiment_dir": row_in.get("experiment_dir", ""),
            "epoch": row_in.get("epoch", ""),
            "best_ckpt_path": row_in.get("best_ckpt_path", ""),
            "stage1_ckpt": row_in.get("stage1_ckpt", ""),
            "gallery_root_dir": row_in.get("gallery_root_dir", ""),
            "gallery_mode": row_in.get("gallery_mode", ""),
            "gallery_overlap": row_in.get("gallery_overlap", ""),
            "gallery_n_rot": row_in.get("gallery_n_rot", ""),
            "gallery_n_scale": row_in.get("gallery_n_scale", ""),
            "gallery_scale_mode": row_in.get("gallery_scale_mode", ""),
            "gallery_save_dir": row_in.get("gallery_save_dir", ""),
            "eval_artifact_dir": row_in.get("eval_artifact_dir", ""),
            "bundle_path": row_in.get("bundle_path", ""),
            "report_path": row_in.get("report_path", ""),
            "dist_median_m": row_in.get("dist_median", ""),
            "rot_median_deg": row_in.get("rot_median", ""),
            "scale_median_ratio": row_in.get("scale_median", ""),
        }
        for cfg_id, label in CFG_SPECS:
            row_out[f"{label}_threshold"] = row_in.get(f"{cfg_id}_threshold", "")
            row_out[f"{label}_progressive_recall"] = row_in.get(f"{cfg_id}_dist_rot_scale_recall", "")
        rows_out.append(row_out)

    rows_out.sort(key=lambda x: (x["dataset"], x["scene"], x["aggregator"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[Stage1CompactRecall] wrote {len(rows_out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
