#!/usr/bin/env python3
import argparse
import csv
import json
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRED_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "DAC_pred_top128"
DEFAULT_DATASET_SETTING_DIR = REPO_ROOT / "gen_fm_exps" / "dataset_setting"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "DAC_pred_top128_linked"

SCENE_TAG_TO_KEY = {
    "03": "visloc03",
    "04": "visloc04",
    "Zuchwil": "zuchwil",
    "Zurich": "zurich",
}

SCENE_KEY_TO_SCENE_ATTRS = {
    "visloc03": "visloc03_scene_attrs.json",
    "visloc04": "visloc04_scene_attrs.json",
    "zuchwil": "zuchwil_scene_attrs.json",
    "zurich": "zurich_scene_attrs.json",
}


def _normalize_header(header: str) -> str:
    return str(header).lstrip("\ufeff")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({_normalize_header(k): (v if v is not None else "") for k, v in row.items()})
    return rows


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def _query_filename_from_row(row: Dict[str, str], lat_field: str, lon_field: str) -> str:
    return f"{row[lat_field]}_{row[lon_field]}.JPG"


def _candidate_filename_from_row(row: Dict[str, str]) -> str:
    return f"{row['center_lat_wgs84']}_{row['center_lon_wgs84']}.tif"


def _get_valid_range_mad(values: List[float], thresh: float = 3.5) -> Tuple[float, float]:
    values_sorted = sorted(float(v) for v in values)
    n = len(values_sorted)
    if n == 0:
        raise ValueError("MAD filter received no values.")
    median = values_sorted[n // 2] if n % 2 == 1 else 0.5 * (values_sorted[n // 2 - 1] + values_sorted[n // 2])
    abs_dev = [abs(v - median) for v in values_sorted]
    abs_dev_sorted = sorted(abs_dev)
    mad = abs_dev_sorted[n // 2] if n % 2 == 1 else 0.5 * (abs_dev_sorted[n // 2 - 1] + abs_dev_sorted[n // 2])
    if mad == 0:
        return min(values_sorted), max(values_sorted)
    valid = []
    for value in values_sorted:
        modified_z_score = 0.6745 * (value - median) / mad
        if abs(modified_z_score) <= thresh:
            valid.append(value)
    if not valid:
        return min(values_sorted), max(values_sorted)
    return min(valid), max(valid)


def _compute_split_indices(n_samples: int, train_ratio: float, split_mode: str) -> Tuple[List[int], List[int]]:
    indices = list(range(int(n_samples)))
    split_mode = str(split_mode).strip().lower()
    if split_mode == "segment":
        n_train = int(n_samples * train_ratio)
        n_train = min(max(n_train, 1), max(n_samples - 1, 1))
        return indices[:n_train], indices[n_train:]
    if split_mode == "interval":
        ratio_frac = Fraction(str(train_ratio)).limit_denominator(1000)
        period = int(ratio_frac.denominator)
        train_per_period = int(ratio_frac.numerator)
        train_indices = [i for i in indices if (i % period) < train_per_period]
        test_indices = [i for i in indices if (i % period) >= train_per_period]
        if not train_indices or not test_indices:
            n_train = int(n_samples * train_ratio)
            n_train = min(max(n_train, 1), max(n_samples - 1, 1))
            return indices[:n_train], indices[n_train:]
        return train_indices, test_indices
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _invert_geotransform(x_geo: float, y_geo: float, geotransform: List[float]) -> Tuple[float, float]:
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in geotransform]
    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-12:
        raise ValueError(f"Invalid geotransform with near-zero determinant: {geotransform}")
    dx = float(x_geo) - gt0
    dy = float(y_geo) - gt3
    col = (gt5 * dx - gt2 * dy) / det
    row = (-gt4 * dx + gt1 * dy) / det
    return row, col


def _scene_name_from_key(scene_key: str) -> str:
    return {
        "visloc03": "visloc_03",
        "visloc04": "visloc_04",
        "zuchwil": "zuchwil",
        "zurich": "zurich",
    }[scene_key]


def _config_key(scene_key: str, split_mode: str) -> str:
    suffix = "82" if scene_key.startswith("visloc") else "91"
    return f"{split_mode}{suffix}"


def _parse_prediction_filename(path: Path) -> Dict[str, str]:
    stem = path.stem
    scene_tag, split_mode, _ = stem.split("_", 2)
    scene_key = SCENE_TAG_TO_KEY[scene_tag]
    return {
        "pred_file": path.name,
        "scene_key": scene_key,
        "scene_name": _scene_name_from_key(scene_key),
        "split_mode": split_mode,
        "config_key": _config_key(scene_key, split_mode),
    }


def _load_manifest(dataset_setting_dir: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    manifest_path = dataset_setting_dir / "dataset_split_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        (row["scene_key"], row["config_key"]): row
        for row in manifest
    }


def _load_scene_attrs(dataset_setting_dir: Path) -> Dict[str, Dict[str, object]]:
    out = {}
    for scene_key, filename in SCENE_KEY_TO_SCENE_ATTRS.items():
        with (dataset_setting_dir / filename).open("r", encoding="utf-8") as f:
            out[scene_key] = json.load(f)
    return out


def _load_visloc_query_index(manifest_row: Dict[str, object]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    test_csv_path = Path(str(manifest_row["test_csv"]))
    rows = _read_csv_rows(test_csv_path)
    index: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = _query_filename_from_row(row, "lat", "lon")
        if key in index:
            raise ValueError(f"Duplicate visloc query key {key} in {test_csv_path}")
        index[key] = {
            "query_filename": key,
            "query_source_csv_row_index": row.get("source_csv_row_index", ""),
            "query_source_filename": row.get("filename", ""),
            "query_uavimg_path": row.get("uavimg_path", ""),
            "query_scene_key": row.get("scene_key", ""),
            "query_scene_name": row.get("scene_name", ""),
            "query_dataset_name": row.get("dataset_name", ""),
            "query_split_config": row.get("split_config", ""),
            "query_split_name": row.get("split_name", ""),
            "query_match_source": "dataset_setting_test_csv",
            "query_key_lat": row.get("lat", ""),
            "query_key_lon": row.get("lon", ""),
            "query_latitude": row.get("latitude", ""),
            "query_longitude": row.get("longitude", ""),
            "query_geo_row": row.get("geo_row_proj32650", ""),
            "query_geo_col": row.get("geo_col_proj32650", ""),
            "query_h_cover_m": row.get("h_cover_m", ""),
            "query_rotdeg_fm_north_anticlock": row.get("rotdeg_fm_north_anticlock", ""),
            "query_aff2d_corrected": row.get("aff2d_corrected", ""),
        }
    meta = {
        "query_match_source": "dataset_setting_test_csv",
        "query_source_csv": str(test_csv_path),
        "n_queries_indexed": len(index),
    }
    return index, meta


def _load_neuloc_query_index(
    scene_key: str,
    split_mode: str,
    manifest_row: Dict[str, object],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    geocsv_path = Path(str(manifest_row["p_uav_geocsv"]))
    satinfo_path = Path(str(manifest_row["p_satinfo_json"]))
    uavinfo_path = Path(str(manifest_row["p_uavinfo_json"]))
    rows = _read_csv_rows(geocsv_path)
    with satinfo_path.open("r", encoding="utf-8") as f:
        satinfo = json.load(f)
    with uavinfo_path.open("r", encoding="utf-8") as f:
        uavinfo = json.load(f)

    geo_res_m = 0.5 * (abs(float(satinfo["x_resolution_m"])) + abs(float(satinfo["y_resolution_m"])))
    aff_mask = [_parse_bool(row["aff2d_corrected"]) for row in rows]
    h_cover_all = [float(row["h_cover_m"]) for row in rows]
    h_cover_corrected = [value for value, keep in zip(h_cover_all, aff_mask) if keep]
    scale_ref_m = (math.floor(sum(h_cover_corrected) / len(h_cover_corrected) / 10.0) + 1.0) * 10.0
    scale_corrected = [value / scale_ref_m for value in h_cover_corrected]
    lower_bound, upper_bound = _get_valid_range_mad(scale_corrected)

    scale_filtered_rows: List[Tuple[int, Dict[str, str]]] = []
    for source_idx, row in enumerate(rows):
        if not _parse_bool(row["aff2d_corrected"]):
            continue
        scale_value = float(row["h_cover_m"]) / scale_ref_m
        if not (lower_bound < scale_value < upper_bound):
            continue
        scale_filtered_rows.append((source_idx, row))

    first_sat_path = Path(str(satinfo["filepaths"][0]))
    sat_image = Image.open(first_sat_path)
    satmap_h, satmap_w = sat_image.height, sat_image.width
    satmap_hw_max = max(satmap_h, satmap_w)
    crop_sizes = [float(row["h_cover_m"]) / geo_res_m for _, row in scale_filtered_rows]
    max_crop = max(crop_sizes)
    min_crop = min(crop_sizes)
    satmap_edge_pixs = max_crop + 224.0
    nr_min = satmap_edge_pixs / satmap_hw_max
    nc_min = nr_min
    nr_max = (satmap_h - satmap_edge_pixs) / satmap_hw_max
    nc_max = (satmap_w - satmap_edge_pixs) / satmap_hw_max
    s_min = min_crop * geo_res_m / scale_ref_m
    s_max = max_crop * geo_res_m / scale_ref_m

    filtered_rows: List[Tuple[int, Dict[str, str]]] = []
    for source_idx, row in scale_filtered_rows:
        geo_row = float(row["geo_row_proj2056"])
        geo_col = float(row["geo_col_proj2056"])
        raster_row, raster_col = _invert_geotransform(geo_col, geo_row, satinfo["geo_transform"])
        nr = raster_row / satmap_hw_max
        nc = raster_col / satmap_hw_max
        scale_value = float(row["h_cover_m"]) / scale_ref_m
        if not (nr_min - 1e-6 <= nr <= nr_max + 1e-6):
            continue
        if not (nc_min - 1e-6 <= nc <= nc_max + 1e-6):
            continue
        if not (s_min - 1e-6 <= scale_value <= s_max + 1e-6):
            continue
        filtered_rows.append((source_idx, row))

    _, test_indices = _compute_split_indices(
        n_samples=len(filtered_rows),
        train_ratio=float(manifest_row["split_train_ratio"]),
        split_mode=split_mode,
    )
    test_rows = [filtered_rows[idx] for idx in test_indices]

    uavimgs_dir = Path(str(uavinfo["uavimgs_dir"]))
    index: Dict[str, Dict[str, object]] = {}
    for source_idx, row in test_rows:
        key = _query_filename_from_row(row, "latitude", "longitude")
        if key in index:
            raise ValueError(f"Duplicate neuloc query key {key} in reconstructed split for {scene_key}")
        index[key] = {
            "query_filename": key,
            "query_source_csv_row_index": source_idx,
            "query_source_filename": row.get("filename", ""),
            "query_uavimg_path": str(uavimgs_dir / row.get("filename", "")),
            "query_scene_key": scene_key,
            "query_scene_name": _scene_name_from_key(scene_key),
            "query_dataset_name": manifest_row.get("dataset_name", ""),
            "query_split_config": _config_key(scene_key, split_mode),
            "query_split_name": "test",
            "query_match_source": "dataset_neuloc_4d_recomputed_test_split",
            "query_key_lat": row.get("latitude", ""),
            "query_key_lon": row.get("longitude", ""),
            "query_latitude": row.get("latitude", ""),
            "query_longitude": row.get("longitude", ""),
            "query_geo_row": row.get("geo_row_proj2056", ""),
            "query_geo_col": row.get("geo_col_proj2056", ""),
            "query_h_cover_m": row.get("h_cover_m", ""),
            "query_rotdeg_fm_north_anticlock": row.get("rotdeg_fm_north_anticlock", ""),
            "query_aff2d_corrected": row.get("aff2d_corrected", ""),
        }
    meta = {
        "query_match_source": "dataset_neuloc_4d_recomputed_test_split",
        "query_source_csv": str(geocsv_path),
        "sat_source_json": str(satinfo_path),
        "uav_source_json": str(uavinfo_path),
        "scale_ref_m": scale_ref_m,
        "scale_lower_bound": lower_bound,
        "scale_upper_bound": upper_bound,
        "n_after_scale_filter": len(scale_filtered_rows),
        "n_after_sat_range_filter": len(filtered_rows),
        "n_queries_indexed": len(index),
    }
    return index, meta


def _load_query_index(
    scene_key: str,
    split_mode: str,
    manifest_row: Dict[str, object],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    if scene_key.startswith("visloc"):
        return _load_visloc_query_index(manifest_row)
    return _load_neuloc_query_index(scene_key, split_mode, manifest_row)


def _build_global_gallery_index(scene_attrs: Dict[str, Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    index: Dict[str, List[Dict[str, object]]] = {}
    for scene_key, attrs in scene_attrs.items():
        geoinfo_path = Path(str(attrs["2d_gallery_geoinfo_overlap000"]))
        rows = _read_csv_rows(geoinfo_path)
        for row in rows:
            key = _candidate_filename_from_row(row)
            record = {
                "gallery_scene_key": scene_key,
                "gallery_scene_name": _scene_name_from_key(scene_key),
                "gallery_overlap": "000",
                "gallery_geoinfo_csv": str(geoinfo_path),
                "gallery_name": row.get("name", ""),
                "gallery_tile_path": row.get("tile_path", ""),
                "gallery_source_tif": row.get("source_tif", ""),
                "gallery_source_tif_stem": row.get("source_tif_stem", ""),
                "gallery_center_lat": row.get("center_lat_wgs84", ""),
                "gallery_center_lon": row.get("center_lon_wgs84", ""),
            }
            index.setdefault(key, []).append(record)
    return index


def _link_prediction_file(
    pred_path: Path,
    query_index: Dict[str, Dict[str, object]],
    gallery_index: Dict[str, List[Dict[str, object]]],
    topk: int,
    pred_meta: Dict[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    pred_rows = _read_csv_rows(pred_path)
    if not pred_rows:
        raise ValueError(f"No rows found in {pred_path}")

    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for query_order, pred_row in enumerate(pred_rows):
        query_filename = pred_row["query_filename"]
        if query_filename not in query_index:
            raise KeyError(f"Query {query_filename} from {pred_path.name} not found in query index.")
        query_info = query_index[query_filename]

        wide_row: Dict[str, object] = {
            "pred_file": pred_meta["pred_file"],
            "scene_key": pred_meta["scene_key"],
            "scene_name": pred_meta["scene_name"],
            "split_mode": pred_meta["split_mode"],
            "config_key": pred_meta["config_key"],
            "query_order": query_order,
        }
        wide_row.update(query_info)

        for rank in range(1, topk + 1):
            topk_col = f"top{rank}"
            candidate_filename = pred_row[topk_col]
            candidate_matches = gallery_index.get(candidate_filename)
            if not candidate_matches:
                raise KeyError(f"Candidate {candidate_filename} from {pred_path.name} not found in gallery index.")

            primary = candidate_matches[0]
            all_tile_paths = "|".join(match["gallery_tile_path"] for match in candidate_matches)
            all_source_tif_stems = "|".join(match["gallery_source_tif_stem"] for match in candidate_matches)

            long_row: Dict[str, object] = {
                "pred_file": pred_meta["pred_file"],
                "scene_key": pred_meta["scene_key"],
                "scene_name": pred_meta["scene_name"],
                "split_mode": pred_meta["split_mode"],
                "config_key": pred_meta["config_key"],
                "query_order": query_order,
                "topk_rank": rank,
                "candidate_filename": candidate_filename,
                "candidate_match_count": len(candidate_matches),
                "candidate_primary_tile_path": primary["gallery_tile_path"],
                "candidate_primary_name": primary["gallery_name"],
                "candidate_primary_source_tif": primary["gallery_source_tif"],
                "candidate_primary_source_tif_stem": primary["gallery_source_tif_stem"],
                "candidate_primary_scene_key": primary["gallery_scene_key"],
                "candidate_primary_scene_name": primary["gallery_scene_name"],
                "candidate_primary_overlap": primary["gallery_overlap"],
                "candidate_center_lat": primary["gallery_center_lat"],
                "candidate_center_lon": primary["gallery_center_lon"],
                "candidate_all_tile_paths": all_tile_paths,
                "candidate_all_source_tif_stems": all_source_tif_stems,
                "candidate_geoinfo_csv": primary["gallery_geoinfo_csv"],
            }
            long_row.update(query_info)
            long_rows.append(long_row)

            wide_row[f"top{rank}_pred_filename"] = candidate_filename
            wide_row[f"top{rank}_match_count"] = len(candidate_matches)
            wide_row[f"top{rank}_primary_tile_path"] = primary["gallery_tile_path"]
            wide_row[f"top{rank}_primary_name"] = primary["gallery_name"]
            wide_row[f"top{rank}_primary_source_tif_stem"] = primary["gallery_source_tif_stem"]
            wide_row[f"top{rank}_primary_scene_key"] = primary["gallery_scene_key"]
            wide_row[f"top{rank}_primary_scene_name"] = primary["gallery_scene_name"]
            wide_row[f"top{rank}_all_tile_paths"] = all_tile_paths

        wide_rows.append(wide_row)

    summary = {
        "pred_file": pred_meta["pred_file"],
        "scene_key": pred_meta["scene_key"],
        "scene_name": pred_meta["scene_name"],
        "split_mode": pred_meta["split_mode"],
        "config_key": pred_meta["config_key"],
        "n_queries": len(wide_rows),
        "n_topk_links": len(long_rows),
        "topk_used": topk,
    }
    return wide_rows, long_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Link DAC top-k prediction CSVs back to query and gallery metadata.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--dataset-setting-dir", type=Path, default=DEFAULT_DATASET_SETTING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    dataset_setting_dir = args.dataset_setting_dir.resolve()
    output_dir = args.output_dir.resolve()
    topk = int(args.topk)

    manifest = _load_manifest(dataset_setting_dir)
    scene_attrs = _load_scene_attrs(dataset_setting_dir)
    gallery_index = _build_global_gallery_index(scene_attrs)

    summaries: List[Dict[str, object]] = []
    for pred_path in sorted(pred_dir.glob("*.csv")):
        pred_meta = _parse_prediction_filename(pred_path)
        manifest_row = manifest[(pred_meta["scene_key"], pred_meta["config_key"])]
        query_index, query_meta = _load_query_index(
            scene_key=pred_meta["scene_key"],
            split_mode=pred_meta["split_mode"],
            manifest_row=manifest_row,
        )
        wide_rows, long_rows, summary = _link_prediction_file(
            pred_path=pred_path,
            query_index=query_index,
            gallery_index=gallery_index,
            topk=topk,
            pred_meta=pred_meta,
        )
        summary.update(query_meta)

        wide_path = output_dir / f"{pred_path.stem}_query_top{topk}_links.csv"
        long_path = output_dir / f"{pred_path.stem}_top{topk}_links_long.csv"

        wide_fields = list(dict.fromkeys(key for row in wide_rows for key in row.keys()))
        long_fields = list(dict.fromkeys(key for row in long_rows for key in row.keys()))
        _write_csv(wide_path, wide_fields, wide_rows)
        _write_csv(long_path, long_fields, long_rows)

        summary["wide_csv"] = str(wide_path)
        summary["long_csv"] = str(long_path)
        summaries.append(summary)

    summary_path = output_dir / "link_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pred_dir": str(pred_dir),
                "dataset_setting_dir": str(dataset_setting_dir),
                "output_dir": str(output_dir),
                "topk_used": topk,
                "gallery_overlap": "000",
                "files": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved linked outputs to {output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
