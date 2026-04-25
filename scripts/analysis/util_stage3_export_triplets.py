from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from PIL import Image


def _as_coords_tensor(coords: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    if torch.is_tensor(coords):
        return coords.detach().to(dtype=torch.float32)
    return torch.as_tensor(coords, dtype=torch.float32)


def _denormalize_with_stats(img_tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    if img_tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image tensor, got shape={tuple(img_tensor.shape)}")
    if img_tensor.device.type != "cpu":
        img_tensor = img_tensor.cpu()
    mean_t = torch.as_tensor(mean, dtype=img_tensor.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=img_tensor.dtype).view(-1, 1, 1)
    img_np = (img_tensor * std_t + mean_t).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return img_np


def _save_image_uint8(img_uint8: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(save_path)


def _wrap_angle_error_deg(pred_rad: float, gt_rad: float) -> float:
    delta = (pred_rad - gt_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def _compute_top1_error_stat_block(values: Sequence[float]) -> dict:
    values_np = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if values_np.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "best95_mean": None,
            "best90_mean": None,
            "worst5_mean": None,
            "worst5_max": None,
        }

    values_sorted = np.sort(values_np)
    n_total = int(values_sorted.size)
    n_best95 = max(1, int(np.floor(n_total * 0.95)))
    n_best90 = max(1, int(np.floor(n_total * 0.90)))
    n_worst5 = max(1, int(np.ceil(n_total * 0.05)))
    worst5 = values_sorted[-n_worst5:]

    return {
        "count": n_total,
        "mean": float(values_sorted.mean()),
        "median": float(np.median(values_sorted)),
        "best95_mean": float(values_sorted[:n_best95].mean()),
        "best90_mean": float(values_sorted[:n_best90].mean()),
        "worst5_mean": float(worst5.mean()),
        "worst5_max": float(worst5.max()),
    }


def _build_top1_error_statistics(summary_rows: Sequence[dict]) -> dict:
    dist_values = [float(row["dist_2d_nrc"]) for row in summary_rows]
    rot_values = [float(row["rot_error_deg"]) for row in summary_rows]
    scale_values = [float(row["scale_ratio"]) for row in summary_rows]
    return {
        "dist_2d_nrc": _compute_top1_error_stat_block(dist_values),
        "rot_error_deg": _compute_top1_error_stat_block(rot_values),
        "scale_ratio": _compute_top1_error_stat_block(scale_values),
    }


def _make_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return _make_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _make_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_pt_bundleable(value):
    if isinstance(value, dict):
        return {str(k): _make_pt_bundleable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_pt_bundleable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_make_pt_bundleable(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_stage_top1_coords(
    coords: torch.Tensor,
    coord_source: str,
) -> Tuple[torch.Tensor, List[str]]:
    coords = _as_coords_tensor(coords)
    if coords.ndim == 3:
        if int(coords.shape[1]) == 0:
            raise ValueError("coords has zero top-k dimension.")
        coords_top1 = coords[:, 0, :]
    elif coords.ndim == 2:
        coords_top1 = coords
    else:
        raise ValueError(f"coords must be [B,K,4] or [B,4], got shape={tuple(coords.shape)}")

    n_query = int(coords_top1.shape[0])
    return coords_top1.clone(), [str(coord_source)] * n_query


def _as_top1_coords(coords: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    coords_t = _as_coords_tensor(coords)
    if coords_t.ndim == 3:
        if int(coords_t.shape[1]) == 0:
            raise ValueError("coords tensor has zero top-k dimension.")
        return coords_t[:, 0, :]
    if coords_t.ndim == 2:
        return coords_t
    raise ValueError(f"coords must be [B,K,4] or [B,4], got shape={tuple(coords_t.shape)}")


def _as_top1_scores(scores: Optional[torch.Tensor | np.ndarray | Sequence[float]], n_query: int) -> Optional[torch.Tensor]:
    if scores is None:
        return None
    scores_t = _as_coords_tensor(scores)
    if scores_t.ndim == 2:
        if int(scores_t.shape[1]) == 0:
            raise ValueError("scores tensor has zero top-k dimension.")
        scores_t = scores_t[:, 0]
    elif scores_t.ndim != 1:
        raise ValueError(f"scores must be [B,K] or [B], got shape={tuple(scores_t.shape)}")
    if int(scores_t.shape[0]) != int(n_query):
        raise ValueError(f"scores size mismatch: expected {n_query}, got {tuple(scores_t.shape)}")
    return scores_t.to(dtype=torch.float32)


def _compute_stage_error_metrics(pred_coord: torch.Tensor, gt_coord: torch.Tensor) -> dict:
    pred_coord = _as_coords_tensor(pred_coord).reshape(4)
    gt_coord = _as_coords_tensor(gt_coord).reshape(4)
    dist_2d = float(torch.norm(pred_coord[:2] - gt_coord[:2], dim=0).item())
    rot_error_deg = _wrap_angle_error_deg(float(pred_coord[2].item()), float(gt_coord[2].item()))
    pred_scale = max(float(pred_coord[3].item()), 1e-6)
    gt_scale = max(float(gt_coord[3].item()), 1e-6)
    scale_ratio = max(pred_scale / gt_scale, gt_scale / pred_scale)
    return {
        "dist_2d_nrc": dist_2d,
        "rot_error_deg": rot_error_deg,
        "scale_ratio": float(scale_ratio),
    }


def _build_stage_predictions_from_results(
    results: dict,
) -> List[dict]:
    if not isinstance(results, dict):
        return []

    stage_specs = [
        ("grid", ("coords_grid",), ("scores_grid",)),
        ("mode", ("coords_mode",), ("scores_mode",)),
        ("evo", ("coords_evo",), ("scores_evo",)),
    ]

    stage_predictions = []
    for stage_name, coord_keys, score_keys in stage_specs:
        coord_key = next((key for key in coord_keys if key in results and results[key] is not None), None)
        if coord_key is None:
            continue
        score_key = next((key for key in score_keys if key in results and results[key] is not None), None)
        coords_top1 = _as_top1_coords(results[coord_key])
        scores_top1 = _as_top1_scores(results.get(score_key, None), n_query=int(coords_top1.shape[0]))
        stage_predictions.append(
            {
                "stage_name": stage_name,
                "coords_top1": coords_top1,
                "scores_top1": scores_top1,
                "coord_sources": [coord_key] * int(coords_top1.shape[0]),
                "coords_key": coord_key,
                "score_key": score_key,
            }
        )

    if len(stage_predictions) == 0 and "coords_evo" in results:
        coords_top1, coord_sources = _resolve_stage_top1_coords(
            coords=results["coords_evo"],
            coord_source="coords_evo",
        )
        stage_predictions.append(
            {
                "stage_name": "evo",
                "coords_top1": coords_top1,
                "scores_top1": _as_top1_scores(results.get("scores_evo", None), n_query=int(coords_top1.shape[0])),
                "coord_sources": coord_sources,
                "coords_key": "coords_evo",
                "score_key": "scores_evo" if "scores_evo" in results else None,
            }
        )

    return stage_predictions


def _build_retrieval_bundle_from_results(results: dict) -> dict:
    required_keys = (
        "scores_grid",
        "coords_grid",
        "scores_mode",
        "coords_mode",
        "scores_evo",
        "coords_evo",
        "coords_gt",
    )
    missing = [key for key in required_keys if key not in results or results[key] is None]
    if missing:
        raise KeyError(f"results missing retrieval bundle keys: {missing}")

    bundle = {
        "schema_version": 2,
        "stage_order": ["grid", "mode", "evo"],
        "reserved_future_keys": ["scores_gd", "coords_gd"],
    }
    for key in required_keys:
        bundle[key] = results[key]

    if results.get("seed_mode_eval_config", None) is not None:
        bundle["seed_mode_eval_config"] = _make_jsonable(results["seed_mode_eval_config"])
    if results.get("seed_mode_reports", None) is not None:
        bundle["seed_mode_reports"] = _make_jsonable(results["seed_mode_reports"])

    return _make_pt_bundleable(bundle)


def export_stage3_retrieval_triplets(
    trainer,
    coords_evo: torch.Tensor,
    coords_gt: torch.Tensor,
    output_root: str,
    use_train_uav: bool = False,
    query_results: Optional[Sequence[object]] = None,
    apply_rotation: bool = True,
    export_batch_size: int = 32,
    stage_predictions: Optional[Sequence[dict]] = None,
    run_config: Optional[dict] = None,
    report_bundle: Optional[dict] = None,
    retrieval_bundle: Optional[dict] = None,
) -> str:
    """
    Export per-query Stage3 analysis assets:
    - query image
    - predicted satellite crop for each exported stage top1
    - GT satellite crop(s), one per satmap
    """
    if output_root is None or str(output_root).strip() == "":
        raise ValueError("output_root must be a non-empty path.")

    sat_dataset = getattr(trainer, "sat_dataset", None)
    uav_dataset = getattr(trainer, "uav_dataset_train" if use_train_uav else "uav_dataset_test", None)
    if sat_dataset is None:
        raise ValueError("trainer.sat_dataset is required for Stage3 triplet export.")
    if uav_dataset is None:
        raise ValueError("trainer UAV dataset is required for Stage3 triplet export.")

    coords_gt = _as_coords_tensor(coords_gt).reshape(-1, 4)
    n_query = int(coords_gt.shape[0])

    if stage_predictions is None:
        coords_stage3_top1, coord_sources = _resolve_stage_top1_coords(
            coords=coords_evo,
            coord_source="coords_evo",
        )
        stage_predictions = [
            {
                "stage_name": "evo",
                "coords_top1": coords_stage3_top1.reshape(-1, 4),
                "scores_top1": None,
                "coord_sources": coord_sources,
                "coords_key": "coords_evo",
                "score_key": None,
            }
        ]

    normalized_stage_predictions = []
    for stage_pred in stage_predictions:
        stage_name = str(stage_pred.get("stage_name", "stage")).strip().lower()
        coords_top1 = _as_top1_coords(stage_pred["coords_top1"]).reshape(-1, 4)
        if int(coords_top1.shape[0]) != n_query:
            raise ValueError(
                f"Stage '{stage_name}' size mismatch: expected {n_query}, got {tuple(coords_top1.shape)}"
            )
        scores_top1 = _as_top1_scores(stage_pred.get("scores_top1", None), n_query=n_query)
        coord_sources = stage_pred.get("coord_sources", None)
        if coord_sources is None or len(coord_sources) != n_query:
            coord_source_default = str(stage_pred.get("coords_key", stage_name))
            coord_sources = [coord_source_default] * n_query
        normalized_stage_predictions.append(
            {
                "stage_name": stage_name,
                "coords_top1": coords_top1,
                "scores_top1": scores_top1,
                "coord_sources": list(coord_sources),
                "coords_key": stage_pred.get("coords_key", stage_name),
                "score_key": stage_pred.get("score_key", None),
            }
        )

    if len(normalized_stage_predictions) == 0:
        raise ValueError("No stage_predictions available for export.")

    primary_stage_name = "evo"
    if all(stage_pred["stage_name"] != primary_stage_name for stage_pred in normalized_stage_predictions):
        primary_stage_name = normalized_stage_predictions[-1]["stage_name"]

    split_name = "train" if use_train_uav else "test"
    query_paths = list(uav_dataset.uavimg_paths_train if use_train_uav else uav_dataset.uavimg_paths_test)
    source_indices = np.asarray(uav_dataset.train_indices if use_train_uav else uav_dataset.test_indices, dtype=np.int64)
    if len(query_paths) < n_query or int(source_indices.shape[0]) < n_query:
        raise ValueError(
            f"Dataset split length is smaller than exported queries: "
            f"len(paths)={len(query_paths)}, len(indices)={int(source_indices.shape[0])}, n_query={n_query}"
        )

    dataset_metrics = None
    if hasattr(sat_dataset, "halfimg_radius_nrc") and hasattr(sat_dataset, "halfimg_radius_meter"):
        halfimg_radius_nrc = float(sat_dataset.halfimg_radius_nrc)
        halfimg_radius_meter = float(sat_dataset.halfimg_radius_meter)
        dataset_metrics = {
            "halfimg_radius_nrc": halfimg_radius_nrc,
            "halfimg_radius_meter": halfimg_radius_meter,
            "nrc2meter_factor": halfimg_radius_meter / max(halfimg_radius_nrc, 1e-8),
        }

    export_batch_size = max(1, int(export_batch_size))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).expanduser().resolve() / f"stage3_triplets_{split_name}_n{n_query}_{timestamp}"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "created_at_utc": timestamp,
        "split": split_name,
        "n_query": n_query,
        "apply_rotation": bool(apply_rotation),
        "n_satmaps": int(sat_dataset.n_satmaps),
        "primary_stage": primary_stage_name,
        "stage_predictions": [
            {
                "stage_name": stage_pred["stage_name"],
                "coords_key": stage_pred["coords_key"],
                "score_key": stage_pred["score_key"],
                "image_filename": f"pred_{stage_pred['stage_name']}_top1.png",
            }
            for stage_pred in normalized_stage_predictions
        ],
        "output_dir": str(run_dir),
        "summary_csv": "stage3_triplets_summary.csv",
        "summary_json": "stage3_triplets_summary.json",
        "top1_error_stats_json": "stage3_top1_error_stats.json",
        "top1_error_stats_csv": "stage3_top1_error_stats.csv",
        "reserved_future_keys": ["scores_gd", "coords_gd"],
    }
    if dataset_metrics is not None:
        manifest["dataset_metrics"] = dict(dataset_metrics)
    if run_config is not None:
        manifest["run_config_json"] = "seed_mode_eval_config.json"
    if report_bundle is not None:
        manifest["seed_mode_reports_json"] = "seed_mode_reports.json"
        manifest["seed_mode_reports_contains_progressive_recall"] = True
    if retrieval_bundle is not None:
        if dataset_metrics is not None and isinstance(retrieval_bundle, dict):
            retrieval_bundle = dict(retrieval_bundle)
            retrieval_bundle["dataset_metrics"] = dict(dataset_metrics)
        manifest["retrieval_bundle_pt"] = "stage3_retrieval_bundle.pt"
        manifest["retrieval_bundle_schema_version"] = 2
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    if run_config is not None:
        run_config_payload = _make_jsonable(run_config)
        with open(run_dir / "seed_mode_eval_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config_payload, f, ensure_ascii=True, indent=2)
    if report_bundle is not None:
        report_bundle_payload = _make_jsonable(report_bundle)
        with open(run_dir / "seed_mode_reports.json", "w", encoding="utf-8") as f:
            json.dump(report_bundle_payload, f, ensure_ascii=True, indent=2)
    if retrieval_bundle is not None:
        torch.save(_make_pt_bundleable(retrieval_bundle), run_dir / "stage3_retrieval_bundle.pt")

    pred_mean = sat_dataset.satinfo_dict["means_normalized"][0]
    pred_std = sat_dataset.satinfo_dict["stds_normalized"][0]
    summary_rows = []

    total_batches = (n_query + export_batch_size - 1) // export_batch_size
    progress = tqdm.tqdm(
        range(0, n_query, export_batch_size),
        total=total_batches,
        desc="Export Stage3 triplets",
        leave=True,
    )
    with torch.no_grad():
        for start in progress:
            end = min(start + export_batch_size, n_query)
            stage_crop_batches = {}
            for stage_pred in normalized_stage_predictions:
                stage_crop_batches[stage_pred["stage_name"]] = sat_dataset.crop_satimg_by_4d_coords_fast(
                    stage_pred["coords_top1"][start:end],
                    apply_rotation=bool(apply_rotation),
                    chunk_size=export_batch_size,
                    id_satmap2sample=0,
                )
            gt_crops_batches = []
            for satmap_id in range(int(sat_dataset.n_satmaps)):
                gt_crops_batches.append(
                    sat_dataset.crop_satimg_by_4d_coords_fast(
                        coords_gt[start:end],
                        apply_rotation=bool(apply_rotation),
                        chunk_size=export_batch_size,
                        id_satmap2sample=satmap_id,
                    )
                )

            for local_idx, q_idx in enumerate(range(start, end)):
                sample_dir = samples_dir / f"{q_idx:05d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                query_path = query_paths[q_idx]
                with Image.open(query_path) as img_query:
                    query_tensor = uav_dataset.uav_transform_test(img_query.convert("RGB"))
                query_img = uav_dataset.denormalize_img(query_tensor)
                _save_image_uint8(query_img, sample_dir / "query.png")

                gt_paths = []
                for satmap_id, gt_crops_batch in enumerate(gt_crops_batches):
                    gt_mean = sat_dataset.satinfo_dict["means_normalized"][satmap_id]
                    gt_std = sat_dataset.satinfo_dict["stds_normalized"][satmap_id]
                    gt_img = _denormalize_with_stats(gt_crops_batch[local_idx], gt_mean, gt_std)
                    gt_filename = f"gt_satmap{satmap_id:02d}.png"
                    _save_image_uint8(gt_img, sample_dir / gt_filename)
                    gt_paths.append(gt_filename)

                gt_coord = coords_gt[q_idx].detach().cpu()
                saved_files = ["query.png"]
                stage_meta = {}
                primary_stage_meta = None
                for stage_pred in normalized_stage_predictions:
                    stage_name = stage_pred["stage_name"]
                    pred_coord = stage_pred["coords_top1"][q_idx].detach().cpu()
                    stage_errors = _compute_stage_error_metrics(pred_coord, gt_coord)
                    pred_img = _denormalize_with_stats(stage_crop_batches[stage_name][local_idx], pred_mean, pred_std)
                    pred_filename = f"pred_{stage_name}_top1.png"
                    _save_image_uint8(pred_img, sample_dir / pred_filename)
                    saved_files.append(pred_filename)
                    score_top1 = None
                    if stage_pred["scores_top1"] is not None:
                        score_top1 = float(stage_pred["scores_top1"][q_idx].item())
                    stage_entry = {
                        "coord_source": stage_pred["coord_sources"][q_idx],
                        "coord_top1": [float(x) for x in pred_coord.tolist()],
                        "score_top1": score_top1,
                        "pred_filename": pred_filename,
                        **stage_errors,
                    }
                    stage_meta[stage_name] = stage_entry
                    if stage_name == primary_stage_name:
                        primary_stage_meta = stage_entry

                saved_files.extend(gt_paths)
                if primary_stage_meta is None:
                    primary_stage_meta = next(iter(stage_meta.values()))
                pred_coord_best = primary_stage_meta["coord_top1"]
                dist_2d = float(primary_stage_meta["dist_2d_nrc"])
                rot_error_deg = float(primary_stage_meta["rot_error_deg"])
                scale_ratio = float(primary_stage_meta["scale_ratio"])

                sample_meta = {
                    "query_id": int(q_idx),
                    "source_index": int(source_indices[q_idx]),
                    "query_filename": Path(query_path).name,
                    "query_path": str(Path(query_path).resolve()),
                    "sample_dir": str(sample_dir.relative_to(run_dir)),
                    "primary_stage": primary_stage_name,
                    "pred_coord_source": primary_stage_meta["coord_source"],
                    "pred_satmap_id": 0,
                    "pred_coord_best": pred_coord_best,
                    "gt_coord": [float(x) for x in gt_coord.tolist()],
                    "dist_2d_nrc": dist_2d,
                    "rot_error_deg": rot_error_deg,
                    "scale_ratio": float(scale_ratio),
                    "stage_predictions": stage_meta,
                    "saved_files": saved_files,
                }
                with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(sample_meta, f, ensure_ascii=True, indent=2)

                pred_coord_primary = torch.as_tensor(pred_coord_best, dtype=torch.float32)
                summary_row = {
                    "query_id": int(q_idx),
                    "source_index": int(source_indices[q_idx]),
                    "query_filename": Path(query_path).name,
                    "query_path": str(Path(query_path).resolve()),
                    "sample_dir": str(sample_dir.relative_to(run_dir)),
                    "primary_stage": primary_stage_name,
                    "pred_coord_source": primary_stage_meta["coord_source"],
                    "pred_satmap_id": 0,
                    "pred_nr": float(pred_coord_primary[0].item()),
                    "pred_nc": float(pred_coord_primary[1].item()),
                    "pred_rot_rad": float(pred_coord_primary[2].item()),
                    "pred_rot_deg": float(math.degrees(float(pred_coord_primary[2].item()))),
                    "pred_scale": float(pred_coord_primary[3].item()),
                    "gt_nr": float(gt_coord[0].item()),
                    "gt_nc": float(gt_coord[1].item()),
                    "gt_rot_rad": float(gt_coord[2].item()),
                    "gt_rot_deg": float(math.degrees(float(gt_coord[2].item()))),
                    "gt_scale": float(gt_coord[3].item()),
                    "delta_nr": float((pred_coord_primary[0] - gt_coord[0]).item()),
                    "delta_nc": float((pred_coord_primary[1] - gt_coord[1]).item()),
                    "dist_2d_nrc": dist_2d,
                    "rot_error_deg": rot_error_deg,
                    "scale_ratio": float(scale_ratio),
                }
                for stage_name, stage_entry in stage_meta.items():
                    stage_coord = stage_entry["coord_top1"]
                    summary_row[f"{stage_name}_coord_source"] = stage_entry["coord_source"]
                    summary_row[f"{stage_name}_score_top1"] = stage_entry["score_top1"]
                    summary_row[f"{stage_name}_pred_nr"] = float(stage_coord[0])
                    summary_row[f"{stage_name}_pred_nc"] = float(stage_coord[1])
                    summary_row[f"{stage_name}_pred_rot_rad"] = float(stage_coord[2])
                    summary_row[f"{stage_name}_pred_rot_deg"] = float(math.degrees(float(stage_coord[2])))
                    summary_row[f"{stage_name}_pred_scale"] = float(stage_coord[3])
                    summary_row[f"{stage_name}_dist_2d_nrc"] = float(stage_entry["dist_2d_nrc"])
                    summary_row[f"{stage_name}_rot_error_deg"] = float(stage_entry["rot_error_deg"])
                    summary_row[f"{stage_name}_scale_ratio"] = float(stage_entry["scale_ratio"])
                summary_rows.append(summary_row)

    summary_json_path = run_dir / "stage3_triplets_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=True, indent=2)

    summary_csv_path = run_dir / "stage3_triplets_summary.csv"
    if len(summary_rows) > 0:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query_id",
                    "source_index",
                    "query_filename",
                    "query_path",
                    "sample_dir",
                    "primary_stage",
                    "pred_coord_source",
                    "pred_satmap_id",
                    "pred_nr",
                    "pred_nc",
                    "pred_rot_rad",
                    "pred_rot_deg",
                    "pred_scale",
                    "gt_nr",
                    "gt_nc",
                    "gt_rot_rad",
                    "gt_rot_deg",
                    "gt_scale",
                    "delta_nr",
                    "delta_nc",
                    "dist_2d_nrc",
                    "rot_error_deg",
                    "scale_ratio",
                ]
            )

    top1_error_stats = {
        "created_at_utc": timestamp,
        "split": split_name,
        "n_query": n_query,
        "metrics": _build_top1_error_statistics(summary_rows),
    }

    top1_error_stats_json_path = run_dir / "stage3_top1_error_stats.json"
    with open(top1_error_stats_json_path, "w", encoding="utf-8") as f:
        json.dump(top1_error_stats, f, ensure_ascii=True, indent=2)

    top1_error_stats_csv_path = run_dir / "stage3_top1_error_stats.csv"
    with open(top1_error_stats_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "metric",
            "count",
            "mean",
            "median",
            "best95_mean",
            "best90_mean",
            "worst5_mean",
            "worst5_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, metric_stats in top1_error_stats["metrics"].items():
            writer.writerow(
                {
                    "metric": metric_name,
                    **metric_stats,
                }
            )

    return str(run_dir)


def export_stage3_retrieval_triplets_from_results(
    trainer,
    results: dict,
    output_root: str,
    use_train_uav: bool = False,
    apply_rotation: bool = True,
    export_batch_size: int = 32,
) -> str:
    if not isinstance(results, dict):
        raise TypeError(f"results must be a dict, got {type(results)!r}")
    required_keys = (
        "scores_grid",
        "coords_grid",
        "scores_mode",
        "coords_mode",
        "scores_evo",
        "coords_evo",
        "coords_gt",
    )
    missing = [key for key in required_keys if key not in results or results[key] is None]
    if missing:
        raise KeyError(f"results missing required keys: {missing}")

    return export_stage3_retrieval_triplets(
        trainer=trainer,
        coords_evo=results["coords_evo"],
        coords_gt=results["coords_gt"],
        output_root=output_root,
        use_train_uav=use_train_uav,
        query_results=results.get("query_results", None),
        apply_rotation=bool(apply_rotation),
        export_batch_size=int(export_batch_size),
        stage_predictions=_build_stage_predictions_from_results(
            results=results,
        ),
        run_config=results.get("seed_mode_eval_config", None),
        report_bundle=results.get("seed_mode_reports", None),
        retrieval_bundle=_build_retrieval_bundle_from_results(results),
    )
