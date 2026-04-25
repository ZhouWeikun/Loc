#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "scripts" / "analysis"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from link_dac_predictions import (  # noqa: E402
    DEFAULT_DATASET_SETTING_DIR,
    _candidate_filename_from_row,
    _load_scene_attrs,
    _normalize_header,
    _parse_prediction_filename,
    _read_csv_rows,
    _scene_name_from_key,
    _write_csv,
)
from trainer_depends.base.trainer_base import BaseTrainer  # noqa: E402
from trainer_depends.config.parser import get_parse  # noqa: E402
from trainers.util_core_eval import compute_progressive_topk_acc_from_coords  # noqa: E402


DEFAULT_FILTERED_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "DAC_pred_same_scene_top5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gen_fm_exps" / "analysis" / "DAC_same_scene_top5_gim_inputs"
DEFAULT_GALLERY_GEOINFO_ATTR = "2d_gallery_geoinfo_overlap000"

PREFERRED_CKPT_DIRS = {
    ("wingtra", "interval"): REPO_ROOT
    / "gen_fm_exps"
    / "ckpts"
    / "stage1_wingtra_interval91_wRejectSampling_msLoss_dinov2_adF4_salad_2",
    ("wingtra", "segment"): REPO_ROOT
    / "gen_fm_exps"
    / "ckpts"
    / "stage1_wingtra_segment91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad",
    ("visloc", "interval"): REPO_ROOT
    / "gen_fm_exps"
    / "ckpts"
    / "stage1_visloc_interval82_wRejectSampling_msLoss_dinov2_adF4_salad_1",
    ("visloc", "segment"): REPO_ROOT
    / "gen_fm_exps"
    / "ckpts"
    / "stage1_visloc_segment82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad",
}


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


def _read_links_rows(path: Path) -> List[Dict[str, str]]:
    rows = _read_csv_rows(path)
    out = []
    for row in rows:
        out.append({_normalize_header(k): v for k, v in row.items()})
    return out


def _parse_epoch_num(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _pick_stage1_ckpt(dataset_name: str, split_mode: str) -> Path:
    key = (str(dataset_name), str(split_mode))
    ckpt_dir = PREFERRED_CKPT_DIRS.get(key)
    if ckpt_dir is None or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing preferred ckpt dir for {key}: {ckpt_dir}")
    candidates = sorted(ckpt_dir.glob("epoch*.pth"), key=lambda p: (_parse_epoch_num(p), p.name))
    if not candidates:
        raise FileNotFoundError(f"No epoch*.pth found under {ckpt_dir}")
    return candidates[-1]


def _opts_yaml_from_ckpt(ckpt_path: Path) -> Path:
    opts_yaml = ckpt_path.parent / "opts.yaml"
    if not opts_yaml.is_file():
        raise FileNotFoundError(f"Missing opts.yaml for {ckpt_path}")
    return opts_yaml


def _load_opt(opts_yaml: Path, selected_scene_name: str):
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "build_dac_same_scene_gim_inputs.py",
            "--p_yaml",
            str(opts_yaml),
            "--selected_scene_name",
            str(selected_scene_name),
        ]
        opt = get_parse(print_summary=False)
    finally:
        sys.argv = argv_backup
    return opt


def _force_single_scene(opt, selected_scene_name: str):
    scenes_setting = getattr(opt, "scenes_setting", None)
    if not isinstance(scenes_setting, dict):
        return opt
    scenes = scenes_setting.get("scenes", None)
    if not isinstance(scenes, list) or len(scenes) <= 1:
        return opt

    filtered_scenes = [
        copy.deepcopy(scene) for scene in scenes if str(scene.get("name", "")) == str(selected_scene_name)
    ]
    if not filtered_scenes:
        available = ", ".join(str(scene.get("name", "")) for scene in scenes)
        raise KeyError(
            f"selected_scene_name {selected_scene_name!r} not found in opts.yaml scenes list. "
            f"Available scenes: {available}"
        )
    scenes_setting = copy.deepcopy(scenes_setting)
    scenes_setting["selected_scene_name"] = str(selected_scene_name)
    scenes_setting["scenes"] = filtered_scenes
    opt.scenes_setting = scenes_setting
    return opt


def _resolve_dataset_pair(opts_yaml: Path, scene_name: str):
    opt = _load_opt(opts_yaml=opts_yaml, selected_scene_name=scene_name)
    opt = _force_single_scene(opt, selected_scene_name=scene_name)
    opt.satmaps_on_cpu = True
    opt.num_worker = 0
    trainer = BaseTrainer(opt)
    trainer._init_datasets(create_train_loader=False)
    return trainer, trainer.sat_datasets[scene_name], trainer.uav_datasets_test[scene_name]


def _build_same_scene_gallery_coord_index(
    dataset_setting_dir: Path,
    gallery_geoinfo_attr: str = DEFAULT_GALLERY_GEOINFO_ATTR,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    scene_attrs = _load_scene_attrs(dataset_setting_dir)
    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scene_key, attrs in scene_attrs.items():
        if gallery_geoinfo_attr not in attrs:
            continue
        rows = _read_csv_rows(Path(str(attrs[gallery_geoinfo_attr])))
        scene_index: Dict[str, Dict[str, float]] = {}
        for row in rows:
            key = _candidate_filename_from_row(row)
            record = {
                "row": float(row["row"]),
                "col": float(row["col"]),
                "crop_size_px": float(row.get("crop_size_px", row.get("crop_size_int", ""))),
                "x_geo": float(row["x_geo"]),
                "y_geo": float(row["y_geo"]),
            }
            if key in scene_index:
                prev = scene_index[key]
                if (
                    abs(prev["row"] - record["row"]) > 1e-4
                    or abs(prev["col"] - record["col"]) > 1e-4
                    or abs(prev["crop_size_px"] - record["crop_size_px"]) > 1e-4
                ):
                    raise ValueError(f"Inconsistent duplicated gallery coord for {scene_key}/{key}")
                continue
            scene_index[key] = record
        index[scene_key] = scene_index
    return index


def _candidate_coord_4d(
    sat_dataset,
    gallery_record: Dict[str, float],
) -> List[float]:
    nr = float(gallery_record["row"]) / float(sat_dataset.satmap_hw_max) + float(sat_dataset.nr_tiftop)
    nc = float(gallery_record["col"]) / float(sat_dataset.satmap_hw_max) + float(sat_dataset.nc_tifleft)
    scale = float(gallery_record["crop_size_px"]) * float(sat_dataset.geo_res_m) / max(float(sat_dataset.scale_ref_m), 1e-8)
    return [nr, nc, 0.0, scale]


def _k_values_for_topk(topk: int) -> Tuple[int, ...]:
    values = [1]
    if topk >= 5:
        values.append(5)
    else:
        values.append(int(topk))
    dedup = []
    for value in values:
        if value not in dedup:
            dedup.append(value)
    return tuple(dedup)


def _build_thresholds(sat_dataset) -> Dict[str, float]:
    return {
        "norm_dist": float(sat_dataset.halfimg_radius_nrc) * 1.1,
        "dist_meter": float(sat_dataset.halfimg_radius_meter) * 1.1,
        "nrc2meter": float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8),
        "rot": 11.0,
        "scale_ratio": 1.15,
    }


def _build_acc_report(
    coords_topk: torch.Tensor,
    coords_gt: torch.Tensor,
    thresholds: Dict[str, float],
    k_values: Sequence[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
        dist_th=float(thresholds["norm_dist"]),
        rot_th_deg=float(thresholds["rot"]),
        scale_ratio_th=float(thresholds["scale_ratio"]),
        k_values=tuple(int(v) for v in k_values),
    )
    progressive_acc_metrics = (
        dict(acc_metrics_raw.get("progressive_acc_metrics", {}))
        if isinstance(acc_metrics_raw.get("progressive_acc_metrics", {}), dict)
        else {}
    )
    acc_metrics = {
        str(key): float(value)
        for key, value in acc_metrics_raw.items()
        if str(key).startswith("top") and str(key).endswith("_acc")
    }
    return acc_metrics_raw, err_stats, progressive_acc_metrics, acc_metrics


def _bundle_paths(save_dir: Path) -> Dict[str, Path]:
    return {
        "manifest": save_dir / "stage1_retrieval_eval_manifest.json",
        "config_json": save_dir / "stage1_retrieval_eval_config.json",
        "report_json": save_dir / "stage1_retrieval_eval_report.json",
        "bundle_pt": save_dir / "stage1_retrieval_eval_bundle.pt",
    }


def _summary_row(
    experiment_dir: str,
    scene_name: str,
    dataset_name: str,
    stage1_ckpt: Path,
    bundle_path: Path,
    report_path: Path,
    source_links_csv: Path,
    pred_file: str,
    n_queries: int,
    topk: int,
    acc_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "experiment_dir": experiment_dir,
        "scene": scene_name,
        "scene_name": scene_name,
        "dataset": dataset_name,
        "aggregator": "dac_same_scene_top5",
        "stage1_ckpt": str(stage1_ckpt),
        "bundle_path": str(bundle_path),
        "report_path": str(report_path),
        "source_links_csv": str(source_links_csv),
        "source_pred_file": pred_file,
        "n_queries": int(n_queries),
        "topk": int(topk),
        "top1_acc": float(acc_metrics.get("top1_acc", 0.0)),
        "top5_acc": float(acc_metrics.get("top5_acc", acc_metrics.get(f"top{topk}_acc", 0.0))),
    }


def _write_summary_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    _write_csv(path, fieldnames, rows)


def build_one_bundle(
    links_csv: Path,
    dataset_setting_dir: Path,
    output_root: Path,
    gallery_coord_index: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    rows = _read_links_rows(links_csv)
    if not rows:
        raise ValueError(f"No rows found in {links_csv}")

    first = rows[0]
    pred_file = str(first["pred_file"])
    pred_meta = _parse_prediction_filename(Path(pred_file))
    scene_key = str(pred_meta["scene_key"])
    scene_name = str(pred_meta["scene_name"])
    split_mode = str(pred_meta["split_mode"])
    dataset_name = "visloc" if scene_key.startswith("visloc") else "wingtra"

    stage1_ckpt = _pick_stage1_ckpt(dataset_name=dataset_name, split_mode=split_mode)
    opts_yaml = _opts_yaml_from_ckpt(stage1_ckpt)
    _, sat_dataset, uav_dataset = _resolve_dataset_pair(opts_yaml=opts_yaml, scene_name=scene_name)

    rows_by_source_filename: Dict[str, Dict[str, str]] = {}
    for row in rows:
        source_filename = str(row["query_source_filename"])
        if source_filename in rows_by_source_filename:
            raise ValueError(f"Duplicate query_source_filename {source_filename} in {links_csv}")
        rows_by_source_filename[source_filename] = row

    dataset_filenames = [Path(path).name for path in uav_dataset.uavimg_paths_test]
    if len(dataset_filenames) != len(rows):
        raise ValueError(
            f"Query count mismatch for {links_csv.name}: dataset has {len(dataset_filenames)} test queries, "
            f"csv has {len(rows)} rows."
        )

    ordered_rows: List[Dict[str, str]] = []
    missing = []
    for source_filename in dataset_filenames:
        row = rows_by_source_filename.get(source_filename)
        if row is None:
            missing.append(source_filename)
            continue
        ordered_rows.append(row)
    if missing:
        raise KeyError(f"Missing {len(missing)} query rows when aligning {links_csv.name}: {missing[:10]}")

    topk_cols = sorted(
        [key for key in first.keys() if key.startswith("top") and key.endswith("_pred_filename")],
        key=lambda key: int(key[3:].split("_", 1)[0]),
    )
    topk = len(topk_cols)
    if topk <= 0:
        raise ValueError(f"No topk columns found in {links_csv}")

    coords_gt = uav_dataset.uav_coords_4d_torch_test.detach().cpu().to(torch.float32)
    coords_topk_list = []
    for row in ordered_rows:
        query_scene_key = str(row["query_scene_key"])
        scene_gallery = gallery_coord_index[query_scene_key]
        candidate_coords = []
        for rank in range(1, topk + 1):
            candidate_filename = str(row[f"top{rank}_pred_filename"])
            gallery_record = scene_gallery.get(candidate_filename)
            if gallery_record is None:
                raise KeyError(f"Candidate {candidate_filename} not found in gallery coord index for {query_scene_key}")
            candidate_coords.append(_candidate_coord_4d(sat_dataset=sat_dataset, gallery_record=gallery_record))
        coords_topk_list.append(candidate_coords)
    coords_topk = torch.tensor(coords_topk_list, dtype=torch.float32)

    thresholds = _build_thresholds(sat_dataset)
    k_values = _k_values_for_topk(topk)
    acc_metrics_raw, err_stats, progressive_acc_metrics, acc_metrics = _build_acc_report(
        coords_topk=coords_topk,
        coords_gt=coords_gt,
        thresholds=thresholds,
        k_values=k_values,
    )

    experiment_dir = links_csv.stem.replace("_links", "__gim_input")
    save_dir = output_root / experiment_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "stage1_ckpt": str(stage1_ckpt),
        "load2test": "",
        "gallery_save_dir": str(save_dir),
        "layout_cfg": {
            "mode": "dac_same_scene_top5",
            "source_pred_file": pred_file,
            "source_links_csv": str(links_csv),
            "gallery_overlap": "000",
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
            "source": "dac_same_scene_top5",
            "n_points_effective_per_query": int(topk),
            "query_alignment": "uav_dataset_test_by_source_filename",
        },
        "gallery_meta": {
            "scene_key": scene_key,
            "scene_name": scene_name,
            "dataset_name": dataset_name,
            "source_pred_file": pred_file,
            "source_links_csv": str(links_csv),
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
            "alignment": "uav_dataset_test_by_source_filename",
        },
        "progressive_recall_policy": {
            "dist_recall": "dist<=dist_th",
            "dist_rot_recall": "dist<=dist_th and rot<=rot_th",
            "dist_rot_scale_recall": "dist<=dist_th and rot<=rot_th and scale<=scale_ratio_th",
            "rot_fallback_to_dist": False,
            "scale_fallback_to_dist_rot": False,
        },
        "source_pred_file": pred_file,
        "source_links_csv": str(links_csv),
    }
    report_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "report_title": f"DAC Same-Scene Retrieval Eval [{scene_name}]",
        "n_queries": int(coords_gt.shape[0]),
        "n_eval": int(coords_gt.shape[0]),
        "k_values": list(k_values),
        "thresholds": thresholds,
        "report_meta": report_meta,
        "acc_metrics": acc_metrics,
        "progressive_acc_metrics": _to_jsonable(progressive_acc_metrics),
        "err_stats": _to_jsonable(err_stats),
        "runtime_gallery_summary": {
            "source": "dac_same_scene_top5",
            "topk": int(topk),
            "scene_key": scene_key,
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

    result = {
        "experiment_dir": experiment_dir,
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "bundle_path": str(paths["bundle_pt"]),
        "report_path": str(paths["report_json"]),
        "config_path": str(paths["config_json"]),
        "manifest_path": str(paths["manifest"]),
        "stage1_ckpt": str(stage1_ckpt),
        "source_links_csv": str(links_csv),
        "source_pred_file": pred_file,
        "n_queries": int(coords_gt.shape[0]),
        "topk": int(topk),
        "top1_acc": float(acc_metrics.get("top1_acc", 0.0)),
        "top5_acc": float(acc_metrics.get("top5_acc", acc_metrics.get(f"top{topk}_acc", 0.0))),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stage1_retrieval_eval_bundle.pt inputs from DAC same-scene top5 CSVs."
    )
    parser.add_argument("--filtered-dir", type=Path, default=DEFAULT_FILTERED_DIR)
    parser.add_argument("--dataset-setting-dir", type=Path, default=DEFAULT_DATASET_SETTING_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--links-csv", type=Path, action="append", default=None)
    parser.add_argument(
        "--gallery-geoinfo-attr",
        type=str,
        default=DEFAULT_GALLERY_GEOINFO_ATTR,
        help="Scene attrs key that points to the gallery geoinfo CSV used by the linked DAC predictions.",
    )
    args = parser.parse_args()

    filtered_dir = args.filtered_dir.resolve()
    dataset_setting_dir = args.dataset_setting_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    links_csvs = [Path(p).resolve() for p in (args.links_csv or [])]
    if not links_csvs:
        links_csvs = sorted(filtered_dir.glob("*_same_scene_top5_links.csv"))
    if not links_csvs:
        raise FileNotFoundError(f"No *_same_scene_top5_links.csv found under {filtered_dir}")

    gallery_coord_index = _build_same_scene_gallery_coord_index(
        dataset_setting_dir=dataset_setting_dir,
        gallery_geoinfo_attr=args.gallery_geoinfo_attr,
    )

    results = []
    for links_csv in links_csvs:
        results.append(
            build_one_bundle(
                links_csv=links_csv,
                dataset_setting_dir=dataset_setting_dir,
                output_root=output_root,
                gallery_coord_index=gallery_coord_index,
            )
        )

    summary_rows = [
        _summary_row(
            experiment_dir=str(row["experiment_dir"]),
            scene_name=str(row["scene_name"]),
            dataset_name=str(row["dataset_name"]),
            stage1_ckpt=Path(str(row["stage1_ckpt"])),
            bundle_path=Path(str(row["bundle_path"])),
            report_path=Path(str(row["report_path"])),
            source_links_csv=Path(str(row["source_links_csv"])),
            pred_file=str(row["source_pred_file"]),
            n_queries=int(row["n_queries"]),
            topk=int(row["topk"]),
            acc_metrics={"top1_acc": float(row["top1_acc"]), "top5_acc": float(row["top5_acc"])},
        )
        for row in results
    ]
    summary_csv = output_root / "dac_same_scene_gim_input_summary.csv"
    _write_summary_csv(summary_csv, summary_rows)

    summary_json = output_root / "dac_same_scene_gim_input_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_root": str(output_root),
                "summary_csv": str(summary_csv),
                "files": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved bundles to {output_root}")
    print(f"Saved summary csv to {summary_csv}")
    print(f"Saved summary json to {summary_json}")


if __name__ == "__main__":
    main()
