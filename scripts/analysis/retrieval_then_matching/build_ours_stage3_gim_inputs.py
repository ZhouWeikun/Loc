#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "ours_ckpt_best"
    / "per_stage_recall_recompute"
    / "per_stage_recall_table_threshold_filtered_no_gem_netvlad.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "ours_ckpt_best"
    / "ours_stage3_gim_inputs_salad_tripletloss_wRS"
)
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_ROOT / "ours_stage3_gim_input_summary.csv"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return _to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _to_pt_bundleable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_pt_bundleable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_pt_bundleable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_source_bundle_path(row: dict[str, str]) -> tuple[Path, Path]:
    bundle_path = Path(row["bundle_path"]).expanduser().resolve()
    run_dir = Path(row.get("run_dir") or bundle_path.parent).expanduser().resolve()
    if bundle_path.is_file():
        return bundle_path, run_dir

    output_root = Path(row.get("output_root") or run_dir.parent).expanduser().resolve()
    candidates = sorted(
        output_root.glob("stage3_triplets_test_*/stage3_retrieval_bundle.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            output_root.glob("**/stage3_retrieval_bundle.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No stage3_retrieval_bundle.pt found for missing bundle: {bundle_path}")
    resolved = candidates[0].resolve()
    return resolved, resolved.parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _scene_name(row: dict[str, str]) -> str:
    return row.get("scene", "") or {
        "visloc03": "visloc_03",
        "visloc04": "visloc_04",
    }.get(row.get("scene_token", ""), row.get("scene_token", ""))


def _dataset_name(scene: str) -> str:
    return "visloc" if str(scene).startswith("visloc_") else "wingtra"


def _slug(value: str, max_len: int = 180) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return value[:max_len].rstrip("_")


def _stage3_report(bundle: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    report = dict(bundle.get("seed_mode_reports", {}))
    if report:
        return report
    return _load_json(run_dir / "seed_mode_reports.json")


def _stage3_eval_config(bundle: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cfg = dict(bundle.get("seed_mode_eval_config", {}))
    if cfg:
        return cfg
    return _load_json(run_dir / "seed_mode_eval_config.json")


def _dataset_metrics(bundle: dict[str, Any], run_dir: Path) -> dict[str, float]:
    metrics = bundle.get("dataset_metrics")
    if not isinstance(metrics, dict):
        manifest = _load_json(run_dir / "manifest.json")
        metrics = manifest.get("dataset_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "nrc2meter": _safe_float(metrics.get("nrc2meter_factor"), 1.0),
        "halfimg_radius_nrc": _safe_float(metrics.get("halfimg_radius_nrc"), 0.0),
        "halfimg_radius_meter": _safe_float(metrics.get("halfimg_radius_meter"), 0.0),
    }


def _thresholds(row: dict[str, str], bundle: dict[str, Any], run_dir: Path) -> dict[str, float]:
    metrics = _dataset_metrics(bundle, run_dir)
    nrc2meter = metrics["nrc2meter"]
    dist_meter = _safe_float(row.get("dist_th_meter"), metrics["halfimg_radius_meter"])
    if dist_meter <= 0 and metrics["halfimg_radius_nrc"] > 0:
        dist_meter = metrics["halfimg_radius_nrc"] * nrc2meter
    return {
        "norm_dist": dist_meter / max(nrc2meter, 1e-8),
        "dist_meter": dist_meter,
        "nrc2meter": nrc2meter,
        "rot": _safe_float(row.get("rot_th"), 10.0),
        "scale_ratio": _safe_float(row.get("scale_ratio_th"), 1.2),
    }


def _pick_rows(rows: list[dict[str, str]], coords_key: str) -> list[dict[str, str]]:
    selected = []
    seen = set()
    for row in rows:
        text = f"{row.get('experiment_name', '')} {row.get('experiment_dir', '')}".lower()
        if "gem" in text or "netvlad" in text:
            continue
        if row.get("stage") and row.get("stage") != "seed_mode_final":
            continue
        if row.get("coords_key") and row.get("coords_key") != coords_key:
            continue
        if row.get("criterion") and row.get("criterion") != "dist_recall":
            continue
        bundle_path = row.get("bundle_path", "")
        if not bundle_path:
            continue
        key = (bundle_path, row.get("scene_token", ""), row.get("mode", ""))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def build_one(row: dict[str, str], output_root: Path, coords_key: str, scores_key: str) -> dict[str, Any]:
    source_bundle_path, run_dir = _resolve_source_bundle_path(row)
    bundle = _load_torch(source_bundle_path)
    if coords_key not in bundle:
        raise KeyError(f"{coords_key!r} missing from {source_bundle_path}")
    coords_topk = bundle[coords_key].to(torch.float32)
    coords_gt = bundle["coords_gt"].to(torch.float32)
    if coords_topk.ndim != 3 or coords_topk.shape[-1] != 4:
        raise ValueError(f"{coords_key} must have shape [N,K,4], got {tuple(coords_topk.shape)}")
    if int(coords_topk.shape[0]) != int(coords_gt.shape[0]):
        raise ValueError(f"N mismatch for {source_bundle_path}: {coords_topk.shape} vs {coords_gt.shape}")

    scene = _scene_name(row)
    dataset = _dataset_name(scene)
    ckpt_path = str(Path(row["ckpt_path"]).expanduser().resolve())
    experiment_name = row.get("experiment_name", source_bundle_path.parent.name)
    ckpt_stem = Path(row.get("ckpt_name") or ckpt_path).stem
    export_name = _slug(f"{experiment_name}__{ckpt_stem}__{coords_key}")
    out_dir = output_root / f"{export_name}__{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _thresholds(row, bundle, run_dir)
    k_values = [k for k in (1, 5, 10, 16, 32, 64, 128, 256, 512, 1024) if k <= int(coords_topk.shape[1])]
    if not k_values:
        k_values = [1]

    stage3_eval_cfg = _stage3_eval_config(bundle, run_dir)
    report = {
        "scene_name": scene,
        "dataset_name": dataset,
        "aggregator": "ours_stage3_salad_tripletloss_wRS",
        "experiment_dir": export_name,
        "source_experiment_name": experiment_name,
        "source_ckpt_name": row.get("ckpt_name", ""),
        "source_stage3_bundle": str(source_bundle_path),
        "source_coords_key": coords_key,
        "source_scores_key": scores_key if scores_key in bundle else "",
        "thresholds": thresholds,
        "k_values": k_values,
        "n_queries": int(coords_gt.shape[0]),
        "stage3_reports": _stage3_report(bundle, run_dir),
    }
    config = {
        "stage1_ckpt": ckpt_path,
        "stage3_ckpt": ckpt_path,
        "opts_path": row.get("opts_path", ""),
        "source_stage3_bundle": str(source_bundle_path),
        "source_coords_key": coords_key,
        "retrieval_eval_cfg": stage3_eval_cfg,
    }
    if scores_key in bundle:
        config["source_scores_key"] = scores_key

    payload = {
        "coords_topk": coords_topk,
        "coords_gt": coords_gt,
        "report": report,
        "config": config,
        "source_stage3_bundle": str(source_bundle_path),
        "source_bundle_schema_version": bundle.get("schema_version", ""),
    }
    if scores_key in bundle:
        payload["scores_topk"] = bundle[scores_key].detach().cpu()

    bundle_out = out_dir / "stage1_retrieval_eval_bundle.pt"
    report_out = out_dir / "stage1_retrieval_eval_report.json"
    torch.save(_to_pt_bundleable(payload), bundle_out)
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)

    return {
        "dataset": dataset,
        "scene": scene,
        "scene_token": row.get("scene_token", ""),
        "mode": row.get("mode", ""),
        "aggregator": report["aggregator"],
        "experiment_dir": export_name,
        "source_experiment_name": experiment_name,
        "ckpt_name": row.get("ckpt_name", ""),
        "stage1_ckpt": ckpt_path,
        "source_stage3_bundle": str(source_bundle_path),
        "source_coords_key": coords_key,
        "n_queries": int(coords_gt.shape[0]),
        "k_values": json.dumps(k_values),
        "dist_th_meter": thresholds["dist_meter"],
        "rot_th": thresholds["rot"],
        "scale_ratio_th": thresholds["scale_ratio"],
        "report_path": str(report_out),
        "bundle_path": str(bundle_out),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GIM-refine-compatible bundles from ours Stage3 outputs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--coords-key", type=str, default="coords_evo")
    parser.add_argument("--scores-key", type=str, default="scores_evo")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = _pick_rows(_read_csv(args.input_csv), coords_key=args.coords_key)
    if not rows:
        raise ValueError(f"No matching Stage3 rows found in {args.input_csv}")
    summary_rows = [build_one(row, args.output_root, args.coords_key, args.scores_key) for row in rows]
    summary_rows.sort(key=lambda r: (str(r["scene_token"]), str(r["mode"]), str(r["experiment_dir"])))
    _write_csv(args.summary_csv, summary_rows)
    print(f"[OursStage3GIMInputs] wrote {len(summary_rows)} bundles under {args.output_root}")
    print(f"[OursStage3GIMInputs] summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
