#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.res_export.link_dac_predictions import (  # noqa: E402
    DEFAULT_DATASET_SETTING_DIR,
    _candidate_filename_from_row,
    _load_scene_attrs,
    _normalize_header,
    _read_csv_rows,
    _scene_name_from_key,
    _write_csv,
)
from scripts_retrieval_then_matching.build_dac_same_scene_gim_inputs import (  # noqa: E402
    _build_acc_report,
    _build_same_scene_gallery_coord_index,
    _build_thresholds,
    _candidate_coord_4d,
    _k_values_for_topk,
    _opts_yaml_from_ckpt,
    _pick_stage1_ckpt,
    _resolve_dataset_pair,
)


DEFAULT_PRED_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "qdfl_pred_topN_wingtra"
DEFAULT_FILTERED_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "qdfl_pred_topN_wingtra_same_scene_top5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gen_fm_exps" / "analysis" / "qdfl_pred_topN_wingtra_same_scene_top5_gim_inputs"
DEFAULT_GALLERY_GEOINFO_ATTR = "2d_gallery_geoinfo_overlap000"

SCENE_TAG_TO_KEY = {
    "Zurich": "zurich",
    "Zuchwil": "zuchwil",
}

QD_FILENAME_RE = re.compile(r"^qdfl_(Zurich|Zuchwil)_top\d+_(segment|interval)$")


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


def _parse_qdfl_prediction_filename(path: Path) -> Dict[str, str]:
    match = QD_FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unsupported QDFL prediction filename: {path.name}")
    scene_tag, split_mode = match.groups()
    scene_key = SCENE_TAG_TO_KEY[scene_tag]
    return {
        "pred_file": path.name,
        "scene_tag": scene_tag,
        "scene_key": scene_key,
        "scene_name": _scene_name_from_key(scene_key),
        "split_mode": split_mode,
        "config_key": f"{split_mode}91",
    }


def _sorted_top_filename_columns(row: Dict[str, str]) -> List[str]:
    cols = []
    for key in row.keys():
        normalized = _normalize_header(key)
        if normalized.startswith("top") and normalized.endswith("_filename"):
            rank = int(normalized[3:].split("_", 1)[0])
            cols.append((rank, normalized))
    cols.sort()
    return [name for _, name in cols]


def _ll_key_from_filename(filename: str) -> str:
    stem = Path(str(filename)).stem
    lat, lon = stem.split("_", 1)
    return f"{float(lat):.8f}_{float(lon):.9f}"


def _load_all_query_meta(dataset_setting_dir: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    scene_attrs = _load_scene_attrs(dataset_setting_dir)
    for scene_key in ("zurich", "zuchwil"):
        for split_cfg in ("segment91", "interval91"):
            for split_name in ("train", "test"):
                csv_path = dataset_setting_dir / f"{scene_key}_{split_cfg}_{split_name}.csv"
                for row in _read_csv_rows(csv_path):
                    key = (scene_key, f"{float(row['latitude']):.8f}_{float(row['longitude']):.9f}")
                    out.setdefault(
                        key,
                        {
                            "query_source_csv_row_index": row.get("source_csv_row_index", ""),
                            "query_source_filename": row.get("filename", ""),
                            "query_uavimg_path": row.get("uavimg_path", ""),
                            "query_scene_key": row.get("scene_key", scene_key),
                            "query_scene_name": row.get("scene_name", _scene_name_from_key(scene_key)),
                            "query_dataset_name": row.get("dataset_name", "wingtra"),
                            "query_split_config_first_seen": row.get("split_config", ""),
                            "query_split_name_first_seen": row.get("split_name", ""),
                            "query_key_lat": row.get("latitude", ""),
                            "query_key_lon": row.get("longitude", ""),
                            "query_latitude": row.get("latitude", ""),
                            "query_longitude": row.get("longitude", ""),
                            "query_geo_row": row.get("geo_row_proj2056", ""),
                            "query_geo_col": row.get("geo_col_proj2056", ""),
                            "query_h_cover_m": row.get("h_cover_m", row.get("h_cvoer_m", "")),
                            "query_rotdeg_fm_north_anticlock": row.get("rotdeg_fm_north_anticlock", ""),
                            "query_aff2d_corrected": row.get("aff2d_corrected", ""),
                        },
                    )
        raw_csv = Path(str(scene_attrs[scene_key]["p_uav_geocsv"]))
        for row in _read_csv_rows(raw_csv):
            key = (scene_key, f"{float(row['latitude']):.8f}_{float(row['longitude']):.9f}")
            out.setdefault(
                key,
                {
                    "query_source_csv_row_index": row.get("source_csv_row_index", ""),
                    "query_source_filename": row.get("filename", ""),
                    "query_uavimg_path": str(Path(str(scene_attrs[scene_key]["p_uavinfo_json"])).parent.parent / "uavimgs_h384" / row.get("filename", "")),
                    "query_scene_key": scene_key,
                    "query_scene_name": _scene_name_from_key(scene_key),
                    "query_dataset_name": "wingtra",
                    "query_split_config_first_seen": "raw_geocsv",
                    "query_split_name_first_seen": "raw_geocsv",
                    "query_key_lat": row.get("latitude", ""),
                    "query_key_lon": row.get("longitude", ""),
                    "query_latitude": row.get("latitude", ""),
                    "query_longitude": row.get("longitude", ""),
                    "query_geo_row": row.get("geo_row_proj2056", ""),
                    "query_geo_col": row.get("geo_col_proj2056", ""),
                    "query_h_cover_m": row.get("h_cover_m", row.get("h_cvoer_m", "")),
                    "query_rotdeg_fm_north_anticlock": row.get("rotdeg_fm_north_anticlock", ""),
                    "query_aff2d_corrected": row.get("aff2d_corrected", ""),
                },
            )
    return out


def _build_uav_filename_index(uav_dataset) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    coords_all = uav_dataset.uav_coords_4d_torch.detach().cpu().to(torch.float32)
    for idx, path in enumerate(uav_dataset.uavimg_paths):
        index[Path(path).name] = {
            "path": str(path),
            "coords_4d": coords_all[int(idx)],
        }
    return index


def _filter_and_link_one(
    pred_path: Path,
    filtered_dir: Path,
    query_meta_index: Dict[Tuple[str, str], Dict[str, object]],
    scene_attrs: Dict[str, Dict[str, object]],
    topk: int,
    gallery_geoinfo_attr: str,
) -> Tuple[Path, Dict[str, object]]:
    pred_meta = _parse_qdfl_prediction_filename(pred_path)
    scene_key = pred_meta["scene_key"]
    gallery_index: Dict[str, List[Dict[str, object]]] = {}
    for key, attrs in scene_attrs.items():
        rows = _read_csv_rows(Path(str(attrs[gallery_geoinfo_attr])))
        for row in rows:
            candidate_filename = _candidate_filename_from_row(row)
            gallery_index.setdefault(candidate_filename, []).append(
                {
                    "gallery_scene_key": key,
                    "gallery_scene_name": _scene_name_from_key(key),
                    "gallery_name": row.get("name", ""),
                    "gallery_tile_path": row.get("tile_path", ""),
                    "gallery_source_tif": row.get("source_tif", ""),
                    "gallery_source_tif_stem": row.get("source_tif_stem", ""),
                    "gallery_geoinfo_csv": str(attrs[gallery_geoinfo_attr]),
                    "gallery_center_lat": row.get("center_lat_wgs84", ""),
                    "gallery_center_lon": row.get("center_lon_wgs84", ""),
                }
            )

    pred_rows = _read_csv_rows(pred_path)
    top_cols = _sorted_top_filename_columns(pred_rows[0])
    raw_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    same_scene_counts: List[int] = []

    for query_order, pred_row in enumerate(pred_rows):
        query_filename = pred_row["query_filename"]
        query_key = (scene_key, _ll_key_from_filename(query_filename))
        if query_key not in query_meta_index:
            raise KeyError(f"Query {query_filename} from {pred_path.name} not found in dataset_setting train/test rows.")
        query_info = dict(query_meta_index[query_key])
        query_info.update(
            {
                "query_filename": query_filename,
                "query_match_source": "dataset_setting_train_test_by_latlon",
                "query_split_config": pred_meta["config_key"],
                "query_split_name": "qdfl_pred_file",
            }
        )

        same_scene_candidates = []
        for top_col in top_cols:
            candidate_filename = pred_row[top_col]
            matches = gallery_index.get(candidate_filename, [])
            same_matches = [m for m in matches if str(m["gallery_scene_key"]) == scene_key]
            if not same_matches:
                continue
            primary = same_matches[0]
            same_scene_candidates.append(
                {
                    "candidate_filename": candidate_filename,
                    "original_rank": int(top_col[3:].split("_", 1)[0]),
                    "candidate_match_count_total": len(matches),
                    "candidate_match_count_same_scene": len(same_matches),
                    "primary": primary,
                    "same_scene_tile_paths": [m["gallery_tile_path"] for m in same_matches],
                    "same_scene_source_tif_stems": [m["gallery_source_tif_stem"] for m in same_matches],
                }
            )

        selected = same_scene_candidates[:topk]
        same_scene_counts.append(len(same_scene_candidates))

        raw_row: Dict[str, object] = {"query_filename": query_filename}
        wide_row: Dict[str, object] = {
            "pred_file": pred_meta["pred_file"],
            "scene_key": pred_meta["scene_key"],
            "scene_name": pred_meta["scene_name"],
            "split_mode": pred_meta["split_mode"],
            "config_key": pred_meta["config_key"],
            "query_order": query_order,
            "n_same_scene_candidates_in_top128": len(same_scene_candidates),
            "has_enough_same_scene_topk": int(len(selected) >= topk),
        }
        wide_row.update(query_info)

        for rank in range(1, topk + 1):
            key = f"top{rank}"
            if rank <= len(selected):
                candidate = selected[rank - 1]
                primary = candidate["primary"]
                raw_row[key] = candidate["candidate_filename"]
                wide_row[f"{key}_pred_filename"] = candidate["candidate_filename"]
                wide_row[f"{key}_original_rank"] = candidate["original_rank"]
                wide_row[f"{key}_match_count_total"] = candidate["candidate_match_count_total"]
                wide_row[f"{key}_match_count_same_scene"] = candidate["candidate_match_count_same_scene"]
                wide_row[f"{key}_primary_tile_path"] = primary["gallery_tile_path"]
                wide_row[f"{key}_primary_name"] = primary["gallery_name"]
                wide_row[f"{key}_primary_source_tif_stem"] = primary["gallery_source_tif_stem"]
                wide_row[f"{key}_primary_scene_key"] = primary["gallery_scene_key"]
                wide_row[f"{key}_primary_scene_name"] = primary["gallery_scene_name"]
                wide_row[f"{key}_all_same_scene_tile_paths"] = "|".join(candidate["same_scene_tile_paths"])

                long_row = {
                    "pred_file": pred_meta["pred_file"],
                    "scene_key": pred_meta["scene_key"],
                    "scene_name": pred_meta["scene_name"],
                    "split_mode": pred_meta["split_mode"],
                    "config_key": pred_meta["config_key"],
                    "query_order": query_order,
                    "filtered_topk_rank": rank,
                    "original_rank": candidate["original_rank"],
                    "candidate_filename": candidate["candidate_filename"],
                    "candidate_match_count_total": candidate["candidate_match_count_total"],
                    "candidate_match_count_same_scene": candidate["candidate_match_count_same_scene"],
                    "candidate_primary_tile_path": primary["gallery_tile_path"],
                    "candidate_primary_name": primary["gallery_name"],
                    "candidate_primary_source_tif": primary["gallery_source_tif"],
                    "candidate_primary_source_tif_stem": primary["gallery_source_tif_stem"],
                    "candidate_primary_scene_key": primary["gallery_scene_key"],
                    "candidate_primary_scene_name": primary["gallery_scene_name"],
                    "candidate_center_lat": primary["gallery_center_lat"],
                    "candidate_center_lon": primary["gallery_center_lon"],
                    "candidate_all_same_scene_tile_paths": "|".join(candidate["same_scene_tile_paths"]),
                    "candidate_all_same_scene_source_tif_stems": "|".join(candidate["same_scene_source_tif_stems"]),
                    "candidate_geoinfo_csv": primary["gallery_geoinfo_csv"],
                }
                long_row.update(query_info)
                long_rows.append(long_row)
            else:
                raw_row[key] = ""

        raw_rows.append(raw_row)
        wide_rows.append(wide_row)

    stem = pred_path.stem.replace("top128", f"same_scene_top{topk}")
    raw_path = filtered_dir / f"{stem}.csv"
    links_path = filtered_dir / f"{stem}_links.csv"
    long_path = filtered_dir / f"{stem}_links_long.csv"
    _write_csv(raw_path, ["query_filename"] + [f"top{i}" for i in range(1, topk + 1)], raw_rows)
    _write_csv(links_path, list(dict.fromkeys(k for row in wide_rows for k in row.keys())), wide_rows)
    _write_csv(long_path, list(dict.fromkeys(k for row in long_rows for k in row.keys())), long_rows)

    summary = {
        **pred_meta,
        "n_queries": len(pred_rows),
        "filtered_topk": topk,
        "n_queries_with_enough_same_scene_topk": sum(1 for v in same_scene_counts if v >= topk),
        "n_queries_insufficient_same_scene_topk": sum(1 for v in same_scene_counts if v < topk),
        "min_same_scene_candidates_in_top128": min(same_scene_counts),
        "max_same_scene_candidates_in_top128": max(same_scene_counts),
        "avg_same_scene_candidates_in_top128": sum(same_scene_counts) / max(len(same_scene_counts), 1),
        "raw_csv": str(raw_path),
        "links_csv": str(links_path),
        "long_csv": str(long_path),
    }
    return links_path, summary


def _bundle_paths(save_dir: Path) -> Dict[str, Path]:
    return {
        "manifest": save_dir / "stage1_retrieval_eval_manifest.json",
        "config_json": save_dir / "stage1_retrieval_eval_config.json",
        "report_json": save_dir / "stage1_retrieval_eval_report.json",
        "bundle_pt": save_dir / "stage1_retrieval_eval_bundle.pt",
    }


def _build_one_bundle(
    links_csv: Path,
    output_root: Path,
    gallery_coord_index: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    rows = _read_csv_rows(links_csv)
    pred_file = str(rows[0]["pred_file"])
    pred_meta = _parse_qdfl_prediction_filename(Path(pred_file))
    scene_key = pred_meta["scene_key"]
    scene_name = pred_meta["scene_name"]
    split_mode = pred_meta["split_mode"]
    dataset_name = "wingtra"

    stage1_ckpt = _pick_stage1_ckpt(dataset_name=dataset_name, split_mode=split_mode)
    opts_yaml = _opts_yaml_from_ckpt(stage1_ckpt)
    _, sat_dataset, uav_dataset = _resolve_dataset_pair(opts_yaml=opts_yaml, scene_name=scene_name)
    uav_index = _build_uav_filename_index(uav_dataset)

    topk_cols = sorted(
        [key for key in rows[0].keys() if key.startswith("top") and key.endswith("_pred_filename")],
        key=lambda key: int(key[3:].split("_", 1)[0]),
    )
    topk = len(topk_cols)
    if topk <= 0:
        raise ValueError(f"No topk columns found in {links_csv}")

    coords_gt = []
    coords_topk = []
    query_paths = []
    missing_queries = []
    for row in rows:
        source_filename = str(row["query_source_filename"])
        uav_record = uav_index.get(source_filename)
        if uav_record is not None:
            query_paths.append(str(uav_record["path"]))
            coords_gt.append(uav_record["coords_4d"])
        else:
            query_path = str(row.get("query_uavimg_path", ""))
            if not query_path or not Path(query_path).is_file():
                missing_queries.append(source_filename)
                continue
            georc = np.asarray([[float(row["query_geo_row"]), float(row["query_geo_col"])]], dtype=np.float32)
            nrc = sat_dataset.transfrom_georc_to_nrc(georc, dtype=np.float32, source_epsg_code=2056)[0]
            rot = np.deg2rad(float(row["query_rotdeg_fm_north_anticlock"]))
            scale = float(row["query_h_cover_m"]) / float(sat_dataset.scale_ref_m)
            query_paths.append(query_path)
            coords_gt.append(torch.tensor([float(nrc[0]), float(nrc[1]), float(rot), float(scale)], dtype=torch.float32))

        candidate_coords = []
        scene_gallery = gallery_coord_index[scene_key]
        for rank in range(1, topk + 1):
            candidate_filename = str(row[f"top{rank}_pred_filename"])
            if not candidate_filename:
                raise ValueError(f"Missing top{rank}_pred_filename for {source_filename} in {links_csv}")
            gallery_record = scene_gallery.get(candidate_filename)
            if gallery_record is None:
                raise KeyError(f"Candidate {candidate_filename} not found in gallery coord index for {scene_key}")
            candidate_coords.append(_candidate_coord_4d(sat_dataset=sat_dataset, gallery_record=gallery_record))
        coords_topk.append(candidate_coords)

    if missing_queries:
        raise KeyError(f"Missing {len(missing_queries)} query source filenames in loaded UAV dataset: {missing_queries[:10]}")

    coords_gt_t = torch.stack(coords_gt).to(torch.float32)
    coords_topk_t = torch.tensor(coords_topk, dtype=torch.float32)

    thresholds = _build_thresholds(sat_dataset)
    k_values = _k_values_for_topk(topk)
    acc_metrics_raw, err_stats, progressive_acc_metrics, acc_metrics = _build_acc_report(
        coords_topk=coords_topk_t,
        coords_gt=coords_gt_t,
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
        "query_paths": query_paths,
        "layout_cfg": {
            "mode": "qdfl_same_scene_top5",
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
            "source": "qdfl_same_scene_top5",
            "n_points_effective_per_query": int(topk),
            "query_alignment": "query_paths_from_dataset_setting_train_test",
        },
        "gallery_meta": {
            "scene_key": scene_key,
            "scene_name": scene_name,
            "dataset_name": dataset_name,
            "source_pred_file": pred_file,
            "source_links_csv": str(links_csv),
        },
    }
    report_payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "report_title": f"QDFL Same-Scene Retrieval Eval [{scene_name}]",
        "n_queries": int(coords_gt_t.shape[0]),
        "n_eval": int(coords_gt_t.shape[0]),
        "k_values": list(k_values),
        "thresholds": thresholds,
        "report_meta": {
            "integrate_scale": True,
            "legacy_acc_metrics_source": str(acc_metrics_raw.get("legacy_acc_metrics_source", "")),
            "progressive_acc_metric_sources": _to_jsonable(acc_metrics_raw.get("progressive_acc_metric_sources", {})),
            "query_subset": {
                "enabled": True,
                "source_split": "dataset_setting_train_test",
                "n_after": int(coords_gt_t.shape[0]),
                "alignment": "query_paths",
            },
            "source_pred_file": pred_file,
            "source_links_csv": str(links_csv),
        },
        "acc_metrics": acc_metrics,
        "progressive_acc_metrics": _to_jsonable(progressive_acc_metrics),
        "err_stats": _to_jsonable(err_stats),
        "runtime_gallery_summary": {
            "source": "qdfl_same_scene_top5",
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
        "coords_topk": coords_topk_t,
        "coords_gt": coords_gt_t,
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
        "experiment_dir": experiment_dir,
        "scene": scene_name,
        "scene_name": scene_name,
        "dataset": dataset_name,
        "aggregator": "qdfl_same_scene_top5",
        "stage1_ckpt": str(stage1_ckpt),
        "bundle_path": str(paths["bundle_pt"]),
        "report_path": str(paths["report_json"]),
        "source_links_csv": str(links_csv),
        "source_pred_file": pred_file,
        "n_queries": int(coords_gt_t.shape[0]),
        "topk": int(topk),
        "top1_acc": float(acc_metrics.get("top1_acc", 0.0)),
        "top5_acc": float(acc_metrics.get("top5_acc", acc_metrics.get(f"top{topk}_acc", 0.0))),
    }


def _write_summary_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    _write_csv(path, fieldnames, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GIM Top5 inputs from Wingtra QDFL top128 CSVs.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--dataset-setting-dir", type=Path, default=DEFAULT_DATASET_SETTING_DIR)
    parser.add_argument("--filtered-dir", type=Path, default=DEFAULT_FILTERED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gallery-geoinfo-attr", type=str, default=DEFAULT_GALLERY_GEOINFO_ATTR)
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    dataset_setting_dir = args.dataset_setting_dir.resolve()
    filtered_dir = args.filtered_dir.resolve()
    output_root = args.output_root.resolve()
    filtered_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    query_meta_index = _load_all_query_meta(dataset_setting_dir)
    scene_attrs = _load_scene_attrs(dataset_setting_dir)
    gallery_coord_index = _build_same_scene_gallery_coord_index(
        dataset_setting_dir=dataset_setting_dir,
        gallery_geoinfo_attr=args.gallery_geoinfo_attr,
    )

    links_csvs = []
    link_summaries = []
    for pred_path in sorted(pred_dir.glob("*.csv")):
        links_csv, summary = _filter_and_link_one(
            pred_path=pred_path,
            filtered_dir=filtered_dir,
            query_meta_index=query_meta_index,
            scene_attrs=scene_attrs,
            topk=int(args.topk),
            gallery_geoinfo_attr=args.gallery_geoinfo_attr,
        )
        links_csvs.append(links_csv)
        link_summaries.append(summary)

    results = [_build_one_bundle(links_csv=links_csv, output_root=output_root, gallery_coord_index=gallery_coord_index) for links_csv in links_csvs]
    summary_csv = output_root / "qdfl_same_scene_gim_input_summary.csv"
    _write_summary_csv(summary_csv, results)
    summary_json = output_root / "qdfl_same_scene_gim_input_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pred_dir": str(pred_dir),
                "filtered_dir": str(filtered_dir),
                "output_root": str(output_root),
                "summary_csv": str(summary_csv),
                "link_summaries": link_summaries,
                "files": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved filtered Top{int(args.topk)} links to {filtered_dir}")
    print(f"Saved bundles to {output_root}")
    print(f"Saved summary csv to {summary_csv}")
    print(f"Saved summary json to {summary_json}")


if __name__ == "__main__":
    main()
