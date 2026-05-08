#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from link_dac_predictions import (
    DEFAULT_DATASET_SETTING_DIR,
    DEFAULT_PRED_DIR,
    REPO_ROOT,
    _candidate_filename_from_row,
    _load_manifest,
    _load_query_index,
    _load_scene_attrs,
    _normalize_header,
    _parse_prediction_filename,
    _read_csv_rows,
    _scene_name_from_key,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "DAC_pred_same_scene_top5"
DEFAULT_GALLERY_GEOINFO_ATTR = "2d_gallery_geoinfo_overlap000"


def _sorted_top_columns(row: Dict[str, str]) -> List[str]:
    cols = []
    for key in row.keys():
        normalized = _normalize_header(key)
        if normalized.startswith("top") and normalized[3:].isdigit():
            cols.append((int(normalized[3:]), normalized))
        elif normalized.startswith("top") and normalized.endswith("_filename"):
            rank_text = normalized[3:].split("_", 1)[0]
            if rank_text.isdigit():
                cols.append((int(rank_text), normalized))
    cols.sort()
    return [name for _, name in cols]


def _top_column_rank(top_col: str) -> int:
    suffix = _normalize_header(top_col)[3:]
    return int(suffix.split("_", 1)[0])


def _pick_same_scene_primary(
    candidate_matches: List[Dict[str, object]],
    query_scene_key: str,
) -> Dict[str, object]:
    for match in candidate_matches:
        if match["gallery_scene_key"] == query_scene_key:
            return match
    raise KeyError(f"No same-scene candidate match found for query scene {query_scene_key}")


def _pipe_join(values: Iterable[str]) -> str:
    return "|".join(str(v) for v in values)


def _build_gallery_index(
    scene_attrs: Dict[str, Dict[str, object]],
    gallery_geoinfo_attr: str,
) -> Dict[str, List[Dict[str, object]]]:
    index: Dict[str, List[Dict[str, object]]] = {}
    for scene_key, attrs in scene_attrs.items():
        if gallery_geoinfo_attr not in attrs:
            continue
        geoinfo_path = Path(str(attrs[gallery_geoinfo_attr]))
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


def _filter_prediction_file(
    pred_path: Path,
    query_index: Dict[str, Dict[str, object]],
    gallery_index: Dict[str, List[Dict[str, object]]],
    pred_meta: Dict[str, str],
    topk: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    pred_rows = _read_csv_rows(pred_path)
    if not pred_rows:
        raise ValueError(f"No rows found in {pred_path}")

    top_columns = _sorted_top_columns(pred_rows[0])
    if len(top_columns) < topk:
        raise ValueError(f"{pred_path} has only {len(top_columns)} top columns, requested topk={topk}")

    raw_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    same_scene_counts: List[int] = []
    insufficient_queries: List[str] = []

    for query_order, pred_row in enumerate(pred_rows):
        query_filename = pred_row["query_filename"]
        if query_filename not in query_index:
            raise KeyError(f"Query {query_filename} from {pred_path.name} not found in query index.")
        query_info = query_index[query_filename]
        query_scene_key = str(query_info["query_scene_key"])

        same_scene_candidates: List[Dict[str, object]] = []
        for top_col in top_columns:
            candidate_filename = pred_row[top_col]
            candidate_matches = gallery_index.get(candidate_filename)
            if not candidate_matches:
                raise KeyError(f"Candidate {candidate_filename} from {pred_path.name} not found in gallery index.")

            scene_keys = sorted({str(match["gallery_scene_key"]) for match in candidate_matches})
            if query_scene_key not in scene_keys:
                continue

            primary = _pick_same_scene_primary(candidate_matches, query_scene_key)
            same_scene_matches = [
                match for match in candidate_matches if str(match["gallery_scene_key"]) == query_scene_key
            ]
            original_rank = _top_column_rank(top_col)
            same_scene_candidates.append(
                {
                    "candidate_filename": candidate_filename,
                    "original_rank": original_rank,
                    "candidate_match_count_total": len(candidate_matches),
                    "candidate_match_count_same_scene": len(same_scene_matches),
                    "primary": primary,
                    "same_scene_tile_paths": [match["gallery_tile_path"] for match in same_scene_matches],
                    "same_scene_source_tif_stems": [
                        match["gallery_source_tif_stem"] for match in same_scene_matches
                    ],
                }
            )

        same_scene_counts.append(len(same_scene_candidates))
        if len(same_scene_candidates) < topk:
            insufficient_queries.append(query_filename)

        selected = same_scene_candidates[:topk]

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

        for filtered_rank in range(1, topk + 1):
            key = f"top{filtered_rank}"
            if filtered_rank <= len(selected):
                candidate = selected[filtered_rank - 1]
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
                wide_row[f"{key}_all_same_scene_tile_paths"] = _pipe_join(candidate["same_scene_tile_paths"])

                long_row: Dict[str, object] = {
                    "pred_file": pred_meta["pred_file"],
                    "scene_key": pred_meta["scene_key"],
                    "scene_name": pred_meta["scene_name"],
                    "split_mode": pred_meta["split_mode"],
                    "config_key": pred_meta["config_key"],
                    "query_order": query_order,
                    "filtered_topk_rank": filtered_rank,
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
                    "candidate_primary_overlap": primary["gallery_overlap"],
                    "candidate_center_lat": primary["gallery_center_lat"],
                    "candidate_center_lon": primary["gallery_center_lon"],
                    "candidate_all_same_scene_tile_paths": _pipe_join(candidate["same_scene_tile_paths"]),
                    "candidate_all_same_scene_source_tif_stems": _pipe_join(
                        candidate["same_scene_source_tif_stems"]
                    ),
                    "candidate_geoinfo_csv": primary["gallery_geoinfo_csv"],
                }
                long_row.update(query_info)
                long_rows.append(long_row)
            else:
                raw_row[key] = ""

        raw_rows.append(raw_row)
        wide_rows.append(wide_row)

    enough_count = sum(1 for count in same_scene_counts if count >= topk)
    summary = {
        "pred_file": pred_meta["pred_file"],
        "scene_key": pred_meta["scene_key"],
        "scene_name": pred_meta["scene_name"],
        "split_mode": pred_meta["split_mode"],
        "config_key": pred_meta["config_key"],
        "n_queries": len(pred_rows),
        "filtered_topk": topk,
        "n_queries_with_enough_same_scene_topk": enough_count,
        "n_queries_insufficient_same_scene_topk": len(pred_rows) - enough_count,
        "min_same_scene_candidates_in_top128": min(same_scene_counts),
        "max_same_scene_candidates_in_top128": max(same_scene_counts),
        "avg_same_scene_candidates_in_top128": sum(same_scene_counts) / max(len(same_scene_counts), 1),
        "insufficient_queries_head": insufficient_queries[:20],
    }
    return raw_rows, wide_rows, long_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter DAC top128 predictions to same-scene candidates and regenerate top-k CSVs."
    )
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--dataset-setting-dir", type=Path, default=DEFAULT_DATASET_SETTING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--gallery-geoinfo-attr",
        type=str,
        default=DEFAULT_GALLERY_GEOINFO_ATTR,
        help="Scene attrs key that points to the gallery geoinfo CSV used by the DAC predictions.",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    dataset_setting_dir = args.dataset_setting_dir.resolve()
    output_dir = args.output_dir.resolve()
    topk = int(args.topk)

    manifest = _load_manifest(dataset_setting_dir)
    scene_attrs = _load_scene_attrs(dataset_setting_dir)
    gallery_index = _build_gallery_index(scene_attrs, args.gallery_geoinfo_attr)

    summaries: List[Dict[str, object]] = []
    for pred_path in sorted(pred_dir.glob("*.csv")):
        pred_meta = _parse_prediction_filename(pred_path)
        manifest_row = manifest[(pred_meta["scene_key"], pred_meta["config_key"])]
        query_index, query_meta = _load_query_index(
            scene_key=pred_meta["scene_key"],
            split_mode=pred_meta["split_mode"],
            manifest_row=manifest_row,
        )
        raw_rows, wide_rows, long_rows, summary = _filter_prediction_file(
            pred_path=pred_path,
            query_index=query_index,
            gallery_index=gallery_index,
            pred_meta=pred_meta,
            topk=topk,
        )
        summary.update(query_meta)

        stem = pred_path.stem
        stem = stem.replace("top_128", f"same_scene_top{topk}")
        stem = stem.replace("top128", f"same_scene_top{topk}")
        stem = stem.removesuffix("_oldtest")
        raw_path = output_dir / f"{stem}.csv"
        wide_path = output_dir / f"{stem}_links.csv"
        long_path = output_dir / f"{stem}_links_long.csv"

        raw_fields = ["query_filename"] + [f"top{i}" for i in range(1, topk + 1)]
        wide_fields = list(dict.fromkeys(key for row in wide_rows for key in row.keys()))
        long_fields = list(dict.fromkeys(key for row in long_rows for key in row.keys()))

        _write_csv(raw_path, raw_fields, raw_rows)
        _write_csv(wide_path, wide_fields, wide_rows)
        _write_csv(long_path, long_fields, long_rows)

        summary["filtered_csv"] = str(raw_path)
        summary["wide_csv"] = str(wide_path)
        summary["long_csv"] = str(long_path)
        summaries.append(summary)

    summary_path = output_dir / "same_scene_topk_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pred_dir": str(pred_dir),
                "dataset_setting_dir": str(dataset_setting_dir),
                "output_dir": str(output_dir),
                "filtered_topk": topk,
                "scene_filter": "candidate_primary_scene_key must match query_scene_key",
                "gallery_geoinfo_attr": args.gallery_geoinfo_attr,
                "files": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved same-scene filtered outputs to {output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
