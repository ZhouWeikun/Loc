#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "ours_ckpt_best"
    / "stage2_exps_res_tripleloss_wRS_salad.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "ours_ckpt_best"
    / "per_stage_recall_recompute"
)
DEFAULT_PLAN_CFG = DEFAULT_OUTPUT_DIR / "plan_cfg.yaml"

DEFAULT_RECALL_CONFIGS = {
    "cfg01": {
        "label": "default_dist_rot5p5_scale1p2",
        "dist_lambda": 0.55,
        "rot_th": 5.5,
        "scale_ratio_th": 1.2,
    },
    "cfg02": {
        "label": "100m_rot10_scale1p2",
        "dist_th_meter": 100,
        "rot_th": 10,
        "scale_ratio_th": 1.2,
    },
    "cfg03": {
        "label": "50m_rot10_scale1p2",
        "dist_th_meter": 50,
        "rot_th": 10,
        "scale_ratio_th": 1.2,
    },
    "cfg04": {
        "label": "25m_rot10_scale1p2",
        "dist_th_meter": 25,
        "rot_th": 10,
        "scale_ratio_th": 1.2,
    },
}
DEFAULT_STAGES = [
    {
        "name": "coarse_retrieval",
        "label": "Coarse-Retrieval",
        "coords_key": "coords_grid",
        "scores_key": "scores_grid",
    },
    {
        "name": "seed_mode_init",
        "label": "Seed-Mode-Init",
        "coords_key": "coords_mode",
        "scores_key": "scores_mode",
    },
    {
        "name": "seed_mode_final",
        "label": "Seed-Mode-Final",
        "coords_key": "coords_evo",
        "scores_key": "scores_evo",
    },
]
DEFAULT_K_VALUES = [1, 5, 10, 16, 32, 64, 128, 256, 512, 1024]
CRITERIA = ("dist_recall", "dist_rot_recall", "dist_rot_scale_recall")
TOPK_COLUMNS = [f"top{k}_acc" for k in DEFAULT_K_VALUES]


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a dict: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_int(value: Any) -> int | str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return int(text)
    except ValueError:
        return text


def _experiment_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    ckpt_path = row.get("ckpt_path", "")
    experiment_dir = str(Path(ckpt_path).parent) if ckpt_path else ""
    output_root = row.get("output_root", "")
    stage3_stdout_log = str(Path(output_root) / "stage3_test_stdout.log") if output_root else ""
    return {
        "id": f"exp{int(row.get('spec_idx') or len(row)):02d}",
        "spec_idx": _safe_int(row.get("spec_idx")),
        "dataset": row.get("dataset", ""),
        "scene": row.get("scene", ""),
        "scene_token": row.get("scene_token", ""),
        "mode": row.get("mode", ""),
        "window_size": _safe_int(row.get("window_size")),
        "experiment_name": row.get("experiment_name", ""),
        "experiment_dir": experiment_dir,
        "ckpt_path": ckpt_path,
        "ckpt_name": row.get("ckpt_name", ""),
        "ckpt_epoch": _safe_int(row.get("ckpt_epoch")),
        "opts_path": row.get("opts_path", ""),
        "output_root": output_root,
        "run_dir": row.get("run_dir", ""),
        "bundle_path": row.get("bundle_path", ""),
        "stage3_stdout_log": stage3_stdout_log,
        "recall_cfgs": ["cfg01", "cfg02", "cfg03", "cfg04"],
    }


def init_plan(args: argparse.Namespace) -> None:
    source_csv = _as_path(args.input_csv)
    plan_cfg = _as_path(args.plan_cfg)
    output_dir = _as_path(args.output_dir)
    rows = _read_csv_rows(source_csv)
    experiments = [_experiment_from_csv_row(row) for row in rows]

    plan = {
        "version": 1,
        "description": "Recompute Stage3 per-stage progressive recall from saved stage3_retrieval_bundle.pt files.",
        "source_csv": str(source_csv),
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "k_values": "from_reports",
        "fallback_k_values": DEFAULT_K_VALUES,
        "stages": DEFAULT_STAGES,
        "recall_configs": DEFAULT_RECALL_CONFIGS,
        "experiments": experiments,
    }
    _write_yaml(plan_cfg, plan)
    print(f"[Stage3PerStageRecall] wrote plan with {len(experiments)} experiments -> {plan_cfg}")


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    import torch

    return torch.load(bundle_path, map_location="cpu")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _dataset_metrics(bundle: dict[str, Any], run_dir: Path) -> dict[str, float]:
    metrics = bundle.get("dataset_metrics", None)
    if not isinstance(metrics, dict):
        metrics = _load_manifest(run_dir).get("dataset_metrics", None)
    if not isinstance(metrics, dict):
        raise KeyError(f"dataset_metrics missing from bundle/manifest: {run_dir}")
    return {
        "halfimg_radius_nrc": float(metrics["halfimg_radius_nrc"]),
        "halfimg_radius_meter": float(metrics["halfimg_radius_meter"]),
        "nrc2meter_factor": float(metrics["nrc2meter_factor"]),
    }


def _resolve_threshold(raw_cfg: dict[str, Any], dataset_metrics: dict[str, float]) -> dict[str, float | None]:
    halfimg_radius_nrc = float(dataset_metrics["halfimg_radius_nrc"])
    nrc2meter = float(dataset_metrics["nrc2meter_factor"])

    if raw_cfg.get("dist_th") is not None:
        dist_th_nrc = float(raw_cfg["dist_th"])
    elif raw_cfg.get("dist_th_meter") is not None:
        dist_th_nrc = float(raw_cfg["dist_th_meter"]) / max(nrc2meter, 1e-8)
    else:
        dist_lambda = float(raw_cfg.get("dist_lambda", 0.55))
        dist_th_nrc = halfimg_radius_nrc * dist_lambda

    return {
        "dist_th_nrc": float(dist_th_nrc),
        "dist_th_meter": float(dist_th_nrc * nrc2meter),
        "rot_th": None if raw_cfg.get("rot_th", None) is None else float(raw_cfg["rot_th"]),
        "scale_ratio_th": (
            None
            if raw_cfg.get("scale_ratio_th", raw_cfg.get("scale_th", None)) is None
            else float(raw_cfg.get("scale_ratio_th", raw_cfg.get("scale_th")))
        ),
    }


def _topk_from_report(bundle: dict[str, Any], stage_name: str, max_k: int) -> list[int]:
    reports = bundle.get("seed_mode_reports", {})
    if isinstance(reports, dict):
        stage_report = reports.get(stage_name, {})
        if isinstance(stage_report, dict):
            progressive = stage_report.get("progressive_acc_metrics", {})
            keys: set[int] = set()
            if isinstance(progressive, dict):
                for metrics in progressive.values():
                    if not isinstance(metrics, dict):
                        continue
                    for key in metrics.keys():
                        match = re.match(r"top(\d+)_acc$", str(key))
                        if match:
                            keys.add(int(match.group(1)))
            if keys:
                return [k for k in sorted(keys) if k <= max_k]
    return []


def _k_values_for_stage(plan: dict[str, Any], bundle: dict[str, Any], stage_name: str, max_k: int) -> list[int]:
    plan_k = plan.get("k_values", "from_reports")
    if str(plan_k).strip().lower() == "from_reports":
        k_values = _topk_from_report(bundle, stage_name, max_k=max_k)
    elif isinstance(plan_k, list):
        k_values = [int(k) for k in plan_k if int(k) <= max_k]
    else:
        k_values = []
    if not k_values:
        fallback = plan.get("fallback_k_values", DEFAULT_K_VALUES)
        k_values = [int(k) for k in fallback if int(k) <= max_k]
    return k_values or [1]


def _metric_at(metrics: dict[str, float], k: int) -> float | None:
    key = f"top{k}_acc"
    return None if key not in metrics else float(metrics[key])


def _row_base(exp: dict[str, Any], cfg_name: str, cfg: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": exp.get("id", ""),
        "spec_idx": exp.get("spec_idx", ""),
        "dataset": exp.get("dataset", ""),
        "scene": exp.get("scene", ""),
        "scene_token": exp.get("scene_token", ""),
        "mode": exp.get("mode", ""),
        "window_size": exp.get("window_size", ""),
        "experiment_name": exp.get("experiment_name", ""),
        "ckpt_name": exp.get("ckpt_name", ""),
        "ckpt_epoch": exp.get("ckpt_epoch", ""),
        "recall_cfg": cfg_name,
        "recall_label": cfg.get("label", cfg_name),
        "dist_th_nrc": threshold["dist_th_nrc"],
        "dist_th_meter": threshold["dist_th_meter"],
        "rot_th": threshold["rot_th"],
        "scale_ratio_th": threshold["scale_ratio_th"],
    }


def _format_report_section(stage_label: str, criterion_label: str, metrics: dict[str, float]) -> list[str]:
    lines = [
        f"{stage_label} | {criterion_label}",
        "-" * 72,
    ]
    for key in sorted(metrics.keys(), key=lambda item: int(str(item).replace("top", "").replace("_acc", ""))):
        lines.append(f"{key:<15} | {float(metrics[key]):<15.2f}")
    return lines


def _format_error_stats(stage_label: str, err_stats: dict[str, float], nrc2meter: float) -> list[str]:
    mean_dist_nrc = float(err_stats["mean_dist_err_top1"])
    median_dist_nrc = float(err_stats["median_dist_err_top1"])
    mean_dist_meter = mean_dist_nrc * nrc2meter
    median_dist_meter = median_dist_nrc * nrc2meter
    return [
        f"{stage_label} | Top-1 Error Stats",
        "-" * 72,
        (
            "Dist  Error: "
            f"Mean={mean_dist_meter:.3f}m / {mean_dist_nrc:.6f} nrc, "
            f"Median={median_dist_meter:.3f}m / {median_dist_nrc:.6f} nrc"
        ),
        (
            "Rot   Error: "
            f"Mean={float(err_stats['mean_rot_err_top1']):.3f}deg, "
            f"Median={float(err_stats['median_rot_err_top1']):.3f}deg"
        ),
        (
            "Scale Error: "
            f"Mean={float(err_stats['mean_scale_ratio_top1']):.6f}x, "
            f"Median={float(err_stats['median_scale_ratio_top1']):.6f}x"
        ),
        "",
    ]


def run_plan(args: argparse.Namespace) -> None:
    # Delay torch-dependent imports so init-plan can run in a lightweight Python.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords

    plan_cfg = _as_path(args.plan_cfg)
    plan = _load_yaml(plan_cfg)
    output_dir = _as_path(args.output_dir or plan.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    recall_configs = plan.get("recall_configs", {})
    if not isinstance(recall_configs, dict) or not recall_configs:
        raise KeyError(f"plan has no recall_configs: {plan_cfg}")
    stages = plan.get("stages", DEFAULT_STAGES)
    experiments = plan.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise KeyError(f"plan has no experiments: {plan_cfg}")

    rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    nested_results: list[dict[str, Any]] = []
    report_lines: list[str] = []

    for exp_idx, exp in enumerate(experiments, start=1):
        bundle_path = _as_path(exp["bundle_path"])
        run_dir = _as_path(exp.get("run_dir") or bundle_path.parent)
        bundle = _load_bundle(bundle_path)
        coords_gt = bundle["coords_gt"].to(dtype=__import__("torch").float32)
        metrics_dataset = _dataset_metrics(bundle, run_dir)
        nrc2meter = float(metrics_dataset["nrc2meter_factor"])

        exp_result = {
            "experiment": dict(exp),
            "dataset_metrics": metrics_dataset,
            "recall_configs": {},
        }
        report_lines.extend(
            [
                "",
                "=" * 96,
                f"[{exp_idx}/{len(experiments)}] {exp.get('scene')} | {exp.get('experiment_name')} | {exp.get('ckpt_name')}",
                f"bundle: {bundle_path}",
                "=" * 96,
            ]
        )

        for cfg_name in exp.get("recall_cfgs", list(recall_configs.keys())):
            if cfg_name not in recall_configs:
                raise KeyError(f"Experiment {exp.get('id')} references missing recall cfg: {cfg_name}")
            raw_cfg = recall_configs[cfg_name]
            threshold = _resolve_threshold(raw_cfg, metrics_dataset)
            cfg_result = {
                "label": raw_cfg.get("label", cfg_name),
                "threshold": threshold,
                "stages": {},
            }
            report_lines.extend(
                [
                    "",
                    f"Recall Config: {cfg_name} ({raw_cfg.get('label', cfg_name)})",
                    (
                        f"threshold: dist={threshold['dist_th_nrc']:.6f} nrc / "
                        f"{threshold['dist_th_meter']:.3f} m, "
                        f"rot={threshold['rot_th']}, scale={threshold['scale_ratio_th']}"
                    ),
                ]
            )

            for stage in stages:
                stage_name = stage["name"]
                stage_label = stage.get("label", stage_name)
                coords_key = stage["coords_key"]
                if coords_key not in bundle or bundle[coords_key] is None:
                    continue
                coords_pred = bundle[coords_key].to(dtype=__import__("torch").float32)
                k_values = _k_values_for_stage(plan, bundle, stage_name=stage_name, max_k=int(coords_pred.shape[1]))
                acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
                    coords_pred=coords_pred,
                    coords_gt=coords_gt,
                    dist_th=float(threshold["dist_th_nrc"]),
                    rot_th_deg=threshold["rot_th"],
                    scale_ratio_th=threshold["scale_ratio_th"],
                    k_values=tuple(k_values),
                )
                progressive = acc_metrics_raw["progressive_acc_metrics"]
                stage_result = {
                    "stage_label": stage_label,
                    "coords_key": coords_key,
                    "k_values": k_values,
                    "progressive_acc_metrics": progressive,
                    "err_stats": err_stats,
                    "err_stats_meter": {
                        "mean_dist_err_top1_meter": float(err_stats["mean_dist_err_top1"]) * nrc2meter,
                        "median_dist_err_top1_meter": float(err_stats["median_dist_err_top1"]) * nrc2meter,
                    },
                }
                cfg_result["stages"][stage_name] = stage_result
                error_rows.append(
                    {
                        **_row_base(exp, cfg_name, raw_cfg, threshold),
                        "stage": stage_name,
                        "stage_label": stage_label,
                        "coords_key": coords_key,
                        "n_queries": int(coords_gt.shape[0]),
                        "k_values": _json_cell(k_values),
                        "dist_median_nrc": float(err_stats["median_dist_err_top1"]),
                        "dist_mean_nrc": float(err_stats["mean_dist_err_top1"]),
                        "dist_median_meter": float(err_stats["median_dist_err_top1"]) * nrc2meter,
                        "dist_mean_meter": float(err_stats["mean_dist_err_top1"]) * nrc2meter,
                        "rot_median": float(err_stats["median_rot_err_top1"]),
                        "rot_mean": float(err_stats["mean_rot_err_top1"]),
                        "scale_median": float(err_stats["median_scale_ratio_top1"]),
                        "scale_mean": float(err_stats["mean_scale_ratio_top1"]),
                        "experiment_dir": exp.get("experiment_dir", ""),
                        "output_root": exp.get("output_root", ""),
                        "run_dir": exp.get("run_dir", ""),
                        "bundle_path": exp.get("bundle_path", ""),
                        "ckpt_path": exp.get("ckpt_path", ""),
                        "opts_path": exp.get("opts_path", ""),
                        "stage3_stdout_log": exp.get("stage3_stdout_log", ""),
                    }
                )
                report_lines.extend(_format_error_stats(stage_label, err_stats, nrc2meter))

                criterion_labels = {
                    "dist_recall": "Dist Recall",
                    "dist_rot_recall": "Dist+Rot Recall",
                    "dist_rot_scale_recall": "Dist+Rot+Scale Recall",
                }
                for criterion in CRITERIA:
                    metrics = progressive[criterion]
                    row = {
                        **_row_base(exp, cfg_name, raw_cfg, threshold),
                        "stage": stage_name,
                        "stage_label": stage_label,
                        "coords_key": coords_key,
                        "criterion": criterion,
                        "n_queries": int(coords_gt.shape[0]),
                        "k_values": _json_cell(k_values),
                        "dist_median_nrc": float(err_stats["median_dist_err_top1"]),
                        "dist_mean_nrc": float(err_stats["mean_dist_err_top1"]),
                        "dist_median_meter": float(err_stats["median_dist_err_top1"]) * nrc2meter,
                        "dist_mean_meter": float(err_stats["mean_dist_err_top1"]) * nrc2meter,
                        "rot_median": float(err_stats["median_rot_err_top1"]),
                        "rot_mean": float(err_stats["mean_rot_err_top1"]),
                        "scale_median": float(err_stats["median_scale_ratio_top1"]),
                        "scale_mean": float(err_stats["mean_scale_ratio_top1"]),
                        "experiment_dir": exp.get("experiment_dir", ""),
                        "output_root": exp.get("output_root", ""),
                        "run_dir": exp.get("run_dir", ""),
                        "bundle_path": exp.get("bundle_path", ""),
                        "ckpt_path": exp.get("ckpt_path", ""),
                        "opts_path": exp.get("opts_path", ""),
                        "stage3_stdout_log": exp.get("stage3_stdout_log", ""),
                    }
                    for key in TOPK_COLUMNS:
                        k = int(key[3:-4])
                        value = _metric_at(metrics, k)
                        row[key] = "" if value is None else value
                    rows.append(row)

                    report_lines.extend(_format_report_section(stage_label, criterion_labels[criterion], metrics))
                    report_lines.append("")

            exp_result["recall_configs"][cfg_name] = cfg_result
        nested_results.append(exp_result)
        print(f"[Stage3PerStageRecall] processed {exp_idx}/{len(experiments)} -> {exp.get('id')} {exp.get('scene')}")

    summary_csv = output_dir / "per_stage_recall_summary.csv"
    error_csv = output_dir / "per_stage_error_summary.csv"
    summary_json = output_dir / "per_stage_recall_summary.json"
    report_txt = output_dir / "per_stage_recall_report.txt"

    fieldnames = [
        "experiment_id",
        "spec_idx",
        "dataset",
        "scene",
        "scene_token",
        "mode",
        "window_size",
        "experiment_name",
        "ckpt_name",
        "ckpt_epoch",
        "recall_cfg",
        "recall_label",
        "dist_th_nrc",
        "dist_th_meter",
        "rot_th",
        "scale_ratio_th",
        "stage",
        "stage_label",
        "coords_key",
        "criterion",
        "n_queries",
        "k_values",
        *TOPK_COLUMNS,
        "dist_median_nrc",
        "dist_mean_nrc",
        "dist_median_meter",
        "dist_mean_meter",
        "rot_median",
        "rot_mean",
        "scale_median",
        "scale_mean",
        "experiment_dir",
        "output_root",
        "run_dir",
        "bundle_path",
        "ckpt_path",
        "opts_path",
        "stage3_stdout_log",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    error_fieldnames = [
        "experiment_id",
        "spec_idx",
        "dataset",
        "scene",
        "scene_token",
        "mode",
        "window_size",
        "experiment_name",
        "ckpt_name",
        "ckpt_epoch",
        "recall_cfg",
        "recall_label",
        "dist_th_nrc",
        "dist_th_meter",
        "rot_th",
        "scale_ratio_th",
        "stage",
        "stage_label",
        "coords_key",
        "n_queries",
        "k_values",
        "dist_median_nrc",
        "dist_mean_nrc",
        "dist_median_meter",
        "dist_mean_meter",
        "rot_median",
        "rot_mean",
        "scale_median",
        "scale_mean",
        "experiment_dir",
        "output_root",
        "run_dir",
        "bundle_path",
        "ckpt_path",
        "opts_path",
        "stage3_stdout_log",
    ]
    with error_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=error_fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(nested_results, f, ensure_ascii=False, indent=2)
    report_txt.write_text("\n".join(report_lines).lstrip() + "\n", encoding="utf-8")

    print(f"[Stage3PerStageRecall] wrote {len(rows)} rows -> {summary_csv}")
    print(f"[Stage3PerStageRecall] wrote {len(error_rows)} error rows -> {error_csv}")
    print(f"[Stage3PerStageRecall] wrote nested json -> {summary_json}")
    print(f"[Stage3PerStageRecall] wrote log-style report -> {report_txt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Stage3 per-stage progressive recall from saved bundles.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    init_parser = subparsers.add_parser("init-plan", help="Build a YAML plan from a Stage3 experiment CSV.")
    init_parser.add_argument("--input-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    init_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    init_parser.add_argument("--plan-cfg", type=Path, default=DEFAULT_PLAN_CFG)
    init_parser.set_defaults(func=init_plan)

    run_parser = subparsers.add_parser("run", help="Run recomputation from a YAML plan.")
    run_parser.add_argument("--plan-cfg", type=Path, default=DEFAULT_PLAN_CFG)
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir in plan.")
    run_parser.set_defaults(func=run_plan)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
