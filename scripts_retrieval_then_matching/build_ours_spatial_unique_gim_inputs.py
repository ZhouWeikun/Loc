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


REPO_ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "ours_ckpt_best"
    / "ours_stage3_gim_inputs_salad_tripletloss_wRS_spatial_unique_top5_summary.csv"
)


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


def _slug(value: str, max_len: int = 180) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return value[:max_len].rstrip("_")


def _scene_name(row: dict[str, str]) -> str:
    return row.get("scene", "") or {
        "visloc03": "visloc_03",
        "visloc04": "visloc_04",
    }.get(row.get("scene_token", ""), row.get("scene_token", ""))


def _dataset_name(scene: str) -> str:
    return "visloc" if str(scene).startswith("visloc_") else "wingtra"


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


def _ensure_tensor(bundle: dict[str, Any], key: str) -> torch.Tensor:
    if key not in bundle:
        raise KeyError(f"{key!r} missing from source bundle")
    value = bundle[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError(f"{key} must have shape [N,K,4], got {tuple(value.shape)}")
    return value.to(torch.float32).cpu()


def _ensure_score_tensor(bundle: dict[str, Any], key: str, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    value = bundle.get(key)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if tuple(value.shape[:2]) != expected_shape:
        return None
    return value.to(torch.float32).cpu()


def _candidate_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a[:2] - b[:2]).item())


def build_spatial_unique_topk(
    bundle: dict[str, Any],
    final_coords_key: str,
    mode_coords_key: str,
    grid_coords_key: str,
    final_scores_key: str,
    mode_scores_key: str,
    grid_scores_key: str,
    distance_threshold: float,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    final_coords = _ensure_tensor(bundle, final_coords_key)
    coords_gt = bundle["coords_gt"].to(torch.float32).cpu()
    if int(final_coords.shape[0]) != int(coords_gt.shape[0]):
        raise ValueError(f"N mismatch: {final_coords.shape} vs {coords_gt.shape}")

    candidate_sources: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
    final_scores = _ensure_score_tensor(bundle, final_scores_key, tuple(final_coords.shape[:2]))
    candidate_sources.append((final_coords_key, final_coords, final_scores))

    if mode_coords_key in bundle:
        mode_coords = _ensure_tensor(bundle, mode_coords_key)
        mode_scores = _ensure_score_tensor(bundle, mode_scores_key, tuple(mode_coords.shape[:2]))
        candidate_sources.append((mode_coords_key, mode_coords, mode_scores))

    if grid_coords_key in bundle:
        grid_coords = _ensure_tensor(bundle, grid_coords_key)
        grid_scores = _ensure_score_tensor(bundle, grid_scores_key, tuple(grid_coords.shape[:2]))
        candidate_sources.append((grid_coords_key, grid_coords, grid_scores))

    n_queries = int(final_coords.shape[0])
    selected_coords = torch.empty((n_queries, int(topk), 4), dtype=torch.float32)
    selected_scores = torch.empty((n_queries, int(topk)), dtype=torch.float32)
    has_any_scores = any(scores is not None for _, _, scores in candidate_sources)
    source_counts: dict[str, int] = {source_name: 0 for source_name, _, _ in candidate_sources}
    selected_counts = []
    duplicate_skips = 0
    fallback_queries = 0

    for query_idx in range(n_queries):
        chosen: list[torch.Tensor] = []
        chosen_scores: list[float] = []
        chosen_sources: list[str] = []

        for source_name, coords, scores in candidate_sources:
            for rank_idx in range(int(coords.shape[1])):
                cand = coords[query_idx, rank_idx]
                if chosen and any(_candidate_distance(cand, prev) <= float(distance_threshold) for prev in chosen):
                    duplicate_skips += 1
                    continue
                chosen.append(cand.clone())
                if scores is None:
                    chosen_scores.append(float("nan"))
                else:
                    chosen_scores.append(float(scores[query_idx, rank_idx].item()))
                chosen_sources.append(source_name)
                if len(chosen) == int(topk):
                    break
            if len(chosen) == int(topk):
                break

        if len(chosen) < int(topk):
            raise RuntimeError(
                f"Query {query_idx} only selected {len(chosen)} spatial-unique candidates; "
                f"need {topk}. Check source candidate depth or distance threshold."
            )

        selected_coords[query_idx] = torch.stack(chosen, dim=0)
        selected_scores[query_idx] = torch.tensor(chosen_scores, dtype=torch.float32)
        selected_counts.append(len(chosen))
        if any(source != final_coords_key for source in chosen_sources):
            fallback_queries += 1
        for source in chosen_sources:
            source_counts[source] = source_counts.get(source, 0) + 1

    stats = {
        "distance_threshold_nrc": float(distance_threshold),
        "topk": int(topk),
        "source_counts": source_counts,
        "n_queries": n_queries,
        "selected_count_min": int(min(selected_counts)),
        "selected_count_max": int(max(selected_counts)),
        "fallback_queries": int(fallback_queries),
        "duplicate_skips": int(duplicate_skips),
        "candidate_sources": [source_name for source_name, _, _ in candidate_sources],
    }
    return selected_coords, selected_scores if has_any_scores else None, stats


def build_one(
    row: dict[str, str],
    output_subdir: str,
    final_coords_key: str,
    mode_coords_key: str,
    grid_coords_key: str,
    final_scores_key: str,
    mode_scores_key: str,
    grid_scores_key: str,
    topk: int,
) -> dict[str, Any]:
    source_bundle_path, run_dir = _resolve_source_bundle_path(row)
    bundle = _load_torch(source_bundle_path)
    coords_gt = bundle["coords_gt"].to(torch.float32).cpu()

    metrics = _dataset_metrics(bundle, run_dir)
    distance_threshold = metrics["halfimg_radius_nrc"]
    if distance_threshold <= 0:
        raise ValueError(f"Invalid halfimg_radius_nrc in {source_bundle_path}: {distance_threshold}")

    coords_topk, scores_topk, selection_stats = build_spatial_unique_topk(
        bundle=bundle,
        final_coords_key=final_coords_key,
        mode_coords_key=mode_coords_key,
        grid_coords_key=grid_coords_key,
        final_scores_key=final_scores_key,
        mode_scores_key=mode_scores_key,
        grid_scores_key=grid_scores_key,
        distance_threshold=distance_threshold,
        topk=topk,
    )

    scene = _scene_name(row)
    dataset = _dataset_name(scene)
    ckpt_path = str(Path(row["ckpt_path"]).expanduser().resolve())
    experiment_name = row.get("experiment_name", source_bundle_path.parent.name)
    ckpt_stem = Path(row.get("ckpt_name") or ckpt_path).stem
    export_name = _slug(f"{experiment_name}__{ckpt_stem}__spatial_unique_top{topk}")
    out_dir = source_bundle_path.parent / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _thresholds(row, bundle, run_dir)
    k_values = [1, int(topk)]
    stage3_eval_cfg = _stage3_eval_config(bundle, run_dir)
    report = {
        "scene_name": scene,
        "dataset_name": dataset,
        "aggregator": f"ours_stage3_salad_tripletloss_wRS_spatial_unique_top{topk}",
        "experiment_dir": export_name,
        "source_experiment_name": experiment_name,
        "source_ckpt_name": row.get("ckpt_name", ""),
        "source_stage3_bundle": str(source_bundle_path),
        "source_coords_key": final_coords_key,
        "source_scores_key": final_scores_key if final_scores_key in bundle else "",
        "thresholds": thresholds,
        "k_values": k_values,
        "n_queries": int(coords_gt.shape[0]),
        "stage3_reports": _stage3_report(bundle, run_dir),
        "spatial_unique_selection": selection_stats,
    }
    config = {
        "stage1_ckpt": ckpt_path,
        "stage3_ckpt": ckpt_path,
        "opts_path": row.get("opts_path", ""),
        "source_stage3_bundle": str(source_bundle_path),
        "source_coords_key": final_coords_key,
        "source_fallback_coords_keys": [mode_coords_key, grid_coords_key],
        "retrieval_eval_cfg": stage3_eval_cfg,
        "spatial_unique_selection": selection_stats,
    }
    if scores_topk is not None:
        config["source_scores_key"] = final_scores_key

    payload = {
        "coords_topk": coords_topk,
        "coords_gt": coords_gt,
        "report": report,
        "config": config,
        "source_stage3_bundle": str(source_bundle_path),
        "source_bundle_schema_version": bundle.get("schema_version", ""),
        "spatial_unique_selection": selection_stats,
    }
    if scores_topk is not None:
        payload["scores_topk"] = scores_topk

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
        "source_coords_key": final_coords_key,
        "fallback_coords_keys": json.dumps([mode_coords_key, grid_coords_key]),
        "spatial_unique_threshold_nrc": selection_stats["distance_threshold_nrc"],
        "spatial_unique_threshold_meter": selection_stats["distance_threshold_nrc"] * metrics["nrc2meter"],
        "fallback_queries": selection_stats["fallback_queries"],
        "source_counts": json.dumps(selection_stats["source_counts"], ensure_ascii=False),
        "duplicate_skips": selection_stats["duplicate_skips"],
        "n_queries": int(coords_gt.shape[0]),
        "k_values": json.dumps(k_values),
        "dist_th_meter": thresholds["dist_meter"],
        "rot_th": thresholds["rot"],
        "scale_ratio_th": thresholds["scale_ratio"],
        "report_path": str(report_out),
        "bundle_path": str(bundle_out),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build spatial-unique top-k GIM input bundles from ours Stage3 outputs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-subdir", type=str, default="gim_input_spatial_unique_top5_halfimg")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--final-coords-key", type=str, default="coords_evo")
    parser.add_argument("--mode-coords-key", type=str, default="coords_mode")
    parser.add_argument("--grid-coords-key", type=str, default="coords_grid")
    parser.add_argument("--final-scores-key", type=str, default="scores_evo")
    parser.add_argument("--mode-scores-key", type=str, default="scores_mode")
    parser.add_argument("--grid-scores-key", type=str, default="scores_grid")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = _pick_rows(_read_csv(args.input_csv), coords_key=args.final_coords_key)
    if not rows:
        raise ValueError(f"No matching Stage3 rows found in {args.input_csv}")
    summary_rows = [
        build_one(
            row=row,
            output_subdir=args.output_subdir,
            final_coords_key=args.final_coords_key,
            mode_coords_key=args.mode_coords_key,
            grid_coords_key=args.grid_coords_key,
            final_scores_key=args.final_scores_key,
            mode_scores_key=args.mode_scores_key,
            grid_scores_key=args.grid_scores_key,
            topk=args.topk,
        )
        for row in rows
    ]
    summary_rows.sort(key=lambda r: (str(r["scene_token"]), str(r["mode"]), str(r["experiment_dir"])))
    _write_csv(args.summary_csv, summary_rows)
    print(f"[OursSpatialUniqueGIMInputs] wrote {len(summary_rows)} bundles next to source Stage3 outputs")
    print(f"[OursSpatialUniqueGIMInputs] summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
