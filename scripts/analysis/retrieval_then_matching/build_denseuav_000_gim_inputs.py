#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_dac_same_scene_gim_inputs import (  # noqa: E402
    _build_acc_report,
    _build_thresholds,
    _bundle_paths,
    _k_values_for_topk,
    _opts_yaml_from_ckpt,
    _pick_stage1_ckpt,
    _resolve_dataset_pair,
)


DEFAULT_DENSEUAV_ROOT = REPO_ROOT / "gen_fm_exps" / "analysis" / "denseuav_pred"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gen_fm_exps" / "analysis" / "denseuav_000_gim_inputs"


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


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_torch(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_scene_name(value: str) -> Optional[str]:
    scene = str(value).strip().lower().replace("-", "_")
    scene = scene.replace("visloc03", "visloc_03").replace("visloc04", "visloc_04")
    if scene in {"03", "visloc_03"}:
        return "visloc_03"
    if scene in {"04", "visloc_04"}:
        return "visloc_04"
    if scene == "zuchwil":
        return "zuchwil"
    if scene == "zurich":
        return "zurich"
    return None


def _scene_from_topdir(bundle_path: Path, root: Path) -> Optional[str]:
    try:
        top = bundle_path.relative_to(root).parts[0]
    except Exception:
        top = bundle_path.parts[-4] if len(bundle_path.parts) >= 4 else ""
    return _normalize_scene_name(top)


def _split_from_path(bundle_path: Path) -> Tuple[str, str]:
    text = "/".join(bundle_path.parts)
    match = re.search(r"(interval|segment)(\d+)", text)
    if not match:
        raise ValueError(f"Cannot infer split mode from {bundle_path}")
    return match.group(1), f"{match.group(1)}{match.group(2)}"


def _scene_key(scene_name: str) -> str:
    return {
        "visloc_03": "visloc03",
        "visloc_04": "visloc04",
        "zuchwil": "zuchwil",
        "zurich": "zurich",
    }[scene_name]


def _dataset_name(scene_name: str) -> str:
    return "visloc" if scene_name.startswith("visloc_") else "wingtra"


def _source_query_preview(config: Dict[str, Any]) -> Tuple[str, str]:
    query_paths = config.get("query_paths", [])
    if not isinstance(query_paths, list) or not query_paths:
        return "", ""
    return Path(str(query_paths[0])).name, Path(str(query_paths[-1])).name


def _validate_query_paths(config: Dict[str, Any], n_queries: int, source_bundle: Path) -> List[str]:
    query_paths = config.get("query_paths", None)
    if not isinstance(query_paths, list) or not query_paths:
        raise ValueError(f"{source_bundle} does not contain config.query_paths; basename alignment is required.")
    if len(query_paths) != int(n_queries):
        raise ValueError(
            f"Query path count mismatch for {source_bundle}: query_paths={len(query_paths)} coords_gt={int(n_queries)}"
        )
    return [str(path) for path in query_paths]


def build_one_bundle(source_bundle: Path, denseuav_root: Path, output_root: Path) -> Optional[Dict[str, object]]:
    payload = _load_torch(source_bundle)
    report_src = dict(payload.get("report", {}))
    config_src = dict(payload.get("config", {}))

    scene_from_report = _normalize_scene_name(str(report_src.get("scene_name", "")))
    scene_from_dir = _scene_from_topdir(source_bundle, denseuav_root)
    if scene_from_report is None:
        raise ValueError(f"Cannot normalize report.scene_name={report_src.get('scene_name')!r} in {source_bundle}")
    if scene_from_dir is None:
        raise ValueError(f"Cannot infer scene from top-level directory in {source_bundle}")
    if scene_from_report != scene_from_dir:
        return {
            "skipped": True,
            "skip_reason": "scene_mismatch",
            "source_bundle": str(source_bundle),
            "report_scene": str(report_src.get("scene_name", "")),
            "normalized_report_scene": scene_from_report,
            "directory_scene": scene_from_dir,
        }

    scene_name = scene_from_report
    split_mode, split_tag = _split_from_path(source_bundle)
    dataset_name = _dataset_name(scene_name)
    stage1_ckpt = _pick_stage1_ckpt(dataset_name=dataset_name, split_mode=split_mode)
    opts_yaml = _opts_yaml_from_ckpt(stage1_ckpt)
    _, sat_dataset, _ = _resolve_dataset_pair(opts_yaml=opts_yaml, scene_name=scene_name)

    coords_topk = payload["coords_topk"].detach().cpu().to(torch.float32)
    coords_gt = payload["coords_gt"].detach().cpu().to(torch.float32)
    if coords_topk.ndim != 3 or coords_topk.shape[-1] != 4:
        raise ValueError(f"coords_topk should be [N,K,4], got {tuple(coords_topk.shape)} in {source_bundle}")
    if coords_gt.ndim != 2 or coords_gt.shape[-1] != 4:
        raise ValueError(f"coords_gt should be [N,4], got {tuple(coords_gt.shape)} in {source_bundle}")
    if int(coords_topk.shape[0]) != int(coords_gt.shape[0]):
        raise ValueError(f"Query count mismatch between coords_topk and coords_gt in {source_bundle}")

    query_paths = _validate_query_paths(config_src, n_queries=int(coords_gt.shape[0]), source_bundle=source_bundle)
    topk = int(coords_topk.shape[1])
    k_values_src = report_src.get("k_values", None)
    if isinstance(k_values_src, list) and k_values_src:
        k_values = tuple(int(v) for v in k_values_src if int(v) <= topk)
    else:
        k_values = _k_values_for_topk(topk)
    if 1 not in k_values:
        k_values = (1, *k_values)
    if topk >= 5 and 5 not in k_values:
        k_values = (*k_values, 5)
    k_values = tuple(sorted(set(int(v) for v in k_values)))

    thresholds = _build_thresholds(sat_dataset)
    acc_metrics_raw, err_stats, progressive_acc_metrics, acc_metrics = _build_acc_report(
        coords_topk=coords_topk,
        coords_gt=coords_gt,
        thresholds=thresholds,
        k_values=k_values,
    )

    experiment_dir = f"denseuav_{_scene_key(scene_name)}_{split_tag}_000__gim_input"
    save_dir = output_root / experiment_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    first_query, last_query = _source_query_preview(config_src)
    config_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "stage1_ckpt": str(stage1_ckpt),
        "load2test": "",
        "gallery_save_dir": str(save_dir),
        "query_paths": query_paths,
        "gallery_paths": list(config_src.get("gallery_paths", [])) if isinstance(config_src.get("gallery_paths", []), list) else [],
        "layout_cfg": {
            "mode": "denseuav_000_top20",
            "source_bundle": str(source_bundle),
            "source_report_scene": str(report_src.get("scene_name", "")),
            "split_mode": split_mode,
            "split_tag": split_tag,
            "topk": topk,
        },
        "feature_cfg": None,
        "retrieval_eval_cfg": {
            "use_train_uav": False,
            "batch_size": 32,
            "num_workers": 0,
            "query_rot2uniform": False,
            "query_scale2uniform": False,
            "k_values": list(k_values),
            "dist_th": float(thresholds["norm_dist"]),
            "rot_th_deg": float(thresholds["rot"]),
            "scale_ratio_th": float(thresholds["scale_ratio"]),
            "max_queries": None,
        },
        "gallery_summary": {
            "source": "denseuav_000",
            "n_points_effective_per_query": topk,
            "query_alignment": "config.query_paths_basename_to_uav_dataset_test",
        },
        "gallery_meta": {
            "scene_key": _scene_key(scene_name),
            "scene_name": scene_name,
            "dataset_name": dataset_name,
            "source_bundle": str(source_bundle),
            "source_mat_path": str(config_src.get("mat_path", "")),
            "source_drone_csv": str(config_src.get("drone_csv", "")),
            "source_sat_csv": str(config_src.get("sat_csv", "")),
        },
    }

    report_meta = {
        "integrate_scale": True,
        "scale_select_mode": None,
        "legacy_acc_metrics_source": str(acc_metrics_raw.get("legacy_acc_metrics_source", "")),
        "progressive_acc_metric_sources": _to_jsonable(acc_metrics_raw.get("progressive_acc_metric_sources", {})),
        "query_subset": {
            "enabled": False,
            "source_split": "test",
            "n_before": int(coords_gt.shape[0]),
            "n_after": int(coords_gt.shape[0]),
            "alignment": "config.query_paths_basename_to_uav_dataset_test",
        },
        "progressive_recall_policy": {
            "dist_recall": "dist<=dist_th",
            "dist_rot_recall": "dist<=dist_th and rot<=rot_th",
            "dist_rot_scale_recall": "dist<=dist_th and rot<=rot_th and scale<=scale_ratio_th",
            "rot_fallback_to_dist": False,
            "scale_fallback_to_dist_rot": False,
        },
        "source_bundle": str(source_bundle),
        "source_report_scene": str(report_src.get("scene_name", "")),
        "source_report": _to_jsonable(report_src),
    }
    report_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "report_title": f"DenseUAV _000 Retrieval Eval [{scene_name}]",
        "n_queries": int(coords_gt.shape[0]),
        "n_eval": int(coords_gt.shape[0]),
        "k_values": list(k_values),
        "thresholds": thresholds,
        "report_meta": report_meta,
        "acc_metrics": acc_metrics,
        "progressive_acc_metrics": _to_jsonable(progressive_acc_metrics),
        "err_stats": _to_jsonable(err_stats),
        "runtime_gallery_summary": {
            "source": "denseuav_000",
            "topk": topk,
            "scene_key": _scene_key(scene_name),
            "split_tag": split_tag,
        },
    }
    manifest_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "gallery_save_dir": str(save_dir),
        "files": {
            "config_json": "stage1_retrieval_eval_config.json",
            "report_json": "stage1_retrieval_eval_report.json",
            "bundle_pt": "stage1_retrieval_eval_bundle.pt",
        },
    }
    bundle_payload = {
        "schema_version": 1,
        "config": config_payload,
        "report": report_payload,
        "coords_topk": coords_topk,
        "coords_gt": coords_gt,
    }

    paths = _bundle_paths(save_dir)
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(manifest_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    with paths["config_json"].open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    with paths["report_json"].open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    torch.save(_to_pt_bundleable(bundle_payload), paths["bundle_pt"])

    return {
        "skipped": False,
        "experiment_dir": experiment_dir,
        "scene": scene_name,
        "scene_name": scene_name,
        "dataset": dataset_name,
        "aggregator": "denseuav",
        "stage1_ckpt": str(stage1_ckpt),
        "bundle_path": str(paths["bundle_pt"]),
        "report_path": str(paths["report_json"]),
        "config_path": str(paths["config_json"]),
        "manifest_path": str(paths["manifest"]),
        "source_bundle": str(source_bundle),
        "source_report_scene": str(report_src.get("scene_name", "")),
        "split_mode": split_mode,
        "split_tag": split_tag,
        "n_queries": int(coords_gt.shape[0]),
        "topk": topk,
        "k_values": json.dumps(list(k_values), ensure_ascii=False, separators=(",", ":")),
        "query_first": first_query,
        "query_last": last_query,
        "top1_acc": float(acc_metrics.get("top1_acc", 0.0)),
        "top5_acc": float(acc_metrics.get("top5_acc", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GIM-refine-compatible inputs from DenseUAV *_000.pt bundles.")
    parser.add_argument("--denseuav-root", type=Path, default=DEFAULT_DENSEUAV_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-skipped-csv", action="store_true")
    args = parser.parse_args()

    denseuav_root = args.denseuav_root.resolve()
    output_root = args.output_root.resolve()
    source_bundles = sorted(denseuav_root.glob("**/*_000.pt"))
    if not source_bundles:
        raise FileNotFoundError(f"No *_000.pt files found under {denseuav_root}")

    built_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    for source_bundle in source_bundles:
        row = build_one_bundle(source_bundle=source_bundle.resolve(), denseuav_root=denseuav_root, output_root=output_root)
        if row is None:
            continue
        if row.get("skipped"):
            skipped_rows.append(row)
            print(f"[skip] {source_bundle}: {row.get('skip_reason')}")
        else:
            built_rows.append(row)
            print(f"[build] {row['experiment_dir']} n={row['n_queries']} topk={row['topk']}")

    summary_csv = output_root / "denseuav_000_gim_input_summary.csv"
    _write_csv(summary_csv, built_rows)
    if skipped_rows or args.include_skipped_csv:
        _write_csv(output_root / "denseuav_000_gim_input_skipped.csv", skipped_rows)

    print(f"Built {len(built_rows)} bundle(s); skipped {len(skipped_rows)}.")
    print(f"Saved summary csv to {summary_csv}")


if __name__ == "__main__":
    main()
