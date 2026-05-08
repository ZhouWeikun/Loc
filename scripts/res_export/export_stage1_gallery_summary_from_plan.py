#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

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


def _recall_config_specs(default_dist_th_nrc, default_dist_th_meter, nrc2meter, scale_ratio_th=1.15):
    scale_tag = str(scale_ratio_th).replace(".", "p")
    return [
        {
            "id": "cfg01",
            "label": f"default_dist_rot5p5_scale{scale_tag}",
            "dist_th_nrc": float(default_dist_th_nrc),
            "dist_th_meter": float(default_dist_th_meter),
            "rot_th_deg": 11.0 * 0.5,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg02",
            "label": f"100m_rot10_scale{scale_tag}",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg03",
            "label": f"100m_rot5_scale{scale_tag}",
            "dist_th_nrc": 100.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 100.0,
            "rot_th_deg": 5.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg04",
            "label": f"50m_rot10_scale{scale_tag}",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg05",
            "label": f"50m_rot5_scale{scale_tag}",
            "dist_th_nrc": 50.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 50.0,
            "rot_th_deg": 5.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg06",
            "label": f"25m_rot10_scale{scale_tag}",
            "dist_th_nrc": 25.0 / max(float(nrc2meter), 1e-8),
            "dist_th_meter": 25.0,
            "rot_th_deg": 10.0,
            "scale_ratio_th": scale_ratio_th,
        },
        {
            "id": "cfg07",
            "label": f"25m_rot5_scale{scale_tag}",
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


def _expected_artifact_paths(save_dir: Path):
    return {
        "gallery_save_dir": str(save_dir),
        "eval_artifact_dir": str(save_dir),
        "gallery_coords_pt": str(save_dir / "coords_gallery.pt"),
        "gallery_feats_pt": str(save_dir / "feats_gallery.pt"),
        "gallery_meta_json": str(save_dir / "gallery_meta.json"),
        "manifest_path": str(save_dir / "stage1_retrieval_eval_manifest.json"),
        "config_json_path": str(save_dir / "stage1_retrieval_eval_config.json"),
        "report_path": str(save_dir / "stage1_retrieval_eval_report.json"),
        "bundle_path": str(save_dir / "stage1_retrieval_eval_bundle.pt"),
    }


def _bundle_glob_suffix(exp):
    ckpt_stem = Path(str(exp["selected_ckpt"])).stem
    return f"{exp['experiment_dir']}_{ckpt_stem}/stage1_retrieval_eval_bundle.pt"


def _resolve_bundle_path(plan_payload, exp, job):
    planned = str(job.get("planned_gallery_save_dir", "") or "").strip()
    expected = None if not planned else Path(planned) / "stage1_retrieval_eval_bundle.pt"
    if expected is not None and expected.is_file():
        return expected

    gallery_root = Path(plan_payload["shared"]["output_gallery_root_dir"])
    suffix = _bundle_glob_suffix(exp)
    matches = sorted(gallery_root.glob(f"**/{suffix}"))
    scene_token = f"/{job['scene_name']}_"
    scene_matches = [m for m in matches if scene_token in str(m)]
    if len(scene_matches) == 1:
        return scene_matches[0]
    if len(scene_matches) > 1:
        raise RuntimeError(
            f"Multiple bundle matches found for {exp['experiment_dir']} / {job['scene_name']}: "
            + ", ".join(str(m) for m in scene_matches)
        )
    if expected is not None:
        return expected
    return gallery_root / "__missing__" / suffix


def _iter_plan_jobs(plan_payload):
    for exp in plan_payload.get("experiments", []):
        for job in exp.get("jobs", []):
            yield exp, job


def _build_row(exp, job, bundle_path: Path, plan_yaml: Path, gallery_root_dir: str, gallery_cfg, scale_ratio_th=1.15):
    payload = _load_bundle(bundle_path)
    report = dict(payload["report"])
    config = dict(payload["config"])
    thresholds = dict(report["thresholds"])
    err_stats = dict(report["err_stats"])
    coords_topk = payload["coords_topk"].to(torch.float32)
    coords_gt = payload["coords_gt"].to(torch.float32)

    scene_name = str(report["scene_name"])
    experiment_dir = str(exp["experiment_dir"])
    dataset_name = str(exp["dataset"])
    aggregator = str(exp.get("method_base", exp.get("method", "")))
    epoch_tag = str(exp["best_epoch_from_log"])
    nrc2meter = float(thresholds["nrc2meter"])
    k_values = tuple(int(k) for k in report["k_values"])

    default_dist_th_nrc = float(thresholds["norm_dist"])
    default_dist_th_meter = float(thresholds["dist_meter"])
    recall_specs = _recall_config_specs(
        default_dist_th_nrc=default_dist_th_nrc,
        default_dist_th_meter=default_dist_th_meter,
        nrc2meter=nrc2meter,
        scale_ratio_th=scale_ratio_th,
    )

    dist_mean_nrc = float(err_stats["mean_dist_err_top1"])
    dist_median_nrc = float(err_stats["median_dist_err_top1"])
    dist_mean_m = dist_mean_nrc * nrc2meter
    dist_median_m = dist_median_nrc * nrc2meter

    save_dir = bundle_path.parent
    row = {
        "dataset": dataset_name,
        "scene": scene_name,
        "aggregator": aggregator,
        "experiment_dir": experiment_dir,
        "epoch": epoch_tag,
        "n_queries": int(report["n_queries"]),
        "dist_median": dist_median_m,
        "dist_mean": dist_mean_m,
        "dist_median_nrc": dist_median_nrc,
        "dist_mean_nrc": dist_mean_nrc,
        "rot_median": float(err_stats["median_rot_err_top1"]),
        "rot_mean": float(err_stats["mean_rot_err_top1"]),
        "scale_median": float(err_stats["median_scale_ratio_top1"]),
        "scale_mean": float(err_stats["mean_scale_ratio_top1"]),
        "nrc2meter": nrc2meter,
        "stage1_ckpt": str(config.get("stage1_ckpt", exp.get("selected_ckpt", ""))),
        "best_ckpt_path": str(exp.get("selected_ckpt", "")),
        "experiment_base_dir": str(exp.get("ckpt_dir", "")),
        "gallery_root_dir": str(gallery_root_dir),
        "gallery_mode": str(gallery_cfg.get("mode", "")),
        "gallery_overlap": float(gallery_cfg.get("overlap", 0.0)),
        "gallery_n_rot": int(gallery_cfg.get("n_rot", 0)),
        "gallery_n_scale": int(gallery_cfg.get("n_scale", 0)),
        "gallery_scale_mode": str(gallery_cfg.get("scale_mode", "")),
        "plan_yaml": str(plan_yaml),
        "query_split_policy_ref": str(exp.get("query_split_policy_ref", "")),
    }
    row.update(_expected_artifact_paths(save_dir))
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
    parser = argparse.ArgumentParser(description="Export Stage1 gallery summary CSV from a plan YAML.")
    parser.add_argument("--plan-yaml", type=Path, required=True, help="Plan YAML file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing bundle paths instead of raising.",
    )
    parser.add_argument(
        "--scale-ratio-th",
        type=float,
        default=1.15,
        help="Scale ratio threshold for all recall configs (default: 1.15).",
    )
    args = parser.parse_args()

    plan_payload = yaml.safe_load(args.plan_yaml.read_text(encoding="utf-8"))
    gallery_root_dir = str(plan_payload.get("shared", {}).get("output_gallery_root_dir", ""))
    gallery_cfg = dict(plan_payload.get("shared", {}).get("gallery", {}))
    rows = []
    missing = []
    jobs = list(_iter_plan_jobs(plan_payload))
    for idx, (exp, job) in enumerate(jobs, start=1):
        bundle_path = _resolve_bundle_path(plan_payload=plan_payload, exp=exp, job=job)
        if not bundle_path.is_file():
            if args.allow_missing:
                missing.append(str(bundle_path))
                continue
            raise FileNotFoundError(f"Missing bundle file: {bundle_path}")
        row = _build_row(
            exp=exp,
            job=job,
            bundle_path=bundle_path,
            plan_yaml=args.plan_yaml,
            gallery_root_dir=gallery_root_dir,
            gallery_cfg=gallery_cfg,
            scale_ratio_th=args.scale_ratio_th,
        )
        rows.append(row)
        print(f"[Stage1PlanSummary] processed {idx}/{len(jobs)} -> {row['scene']} / {row['aggregator']}")

    rows.sort(key=lambda row: (row["dataset"], row["scene"], row["aggregator"]))
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
        "best_ckpt_path",
        "experiment_base_dir",
        "gallery_root_dir",
        "gallery_mode",
        "gallery_overlap",
        "gallery_n_rot",
        "gallery_n_scale",
        "gallery_scale_mode",
        "gallery_save_dir",
        "eval_artifact_dir",
        "gallery_coords_pt",
        "gallery_feats_pt",
        "gallery_meta_json",
        "manifest_path",
        "config_json_path",
        "report_path",
        "bundle_path",
        "plan_yaml",
        "query_split_policy_ref",
    ]
    for idx in range(1, 8):
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

    print(f"[Stage1PlanSummary] wrote {len(rows)} rows to {args.output_csv}")
    if missing:
        print(f"[Stage1PlanSummary] skipped {len(missing)} missing bundle paths")


if __name__ == "__main__":
    main()
