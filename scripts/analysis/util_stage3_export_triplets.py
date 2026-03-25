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
    mean_t = torch.tensor(mean, dtype=img_tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img_tensor.dtype).view(-1, 1, 1)
    img_np = (img_tensor * std_t + mean_t).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return img_np


def _save_image_uint8(img_uint8: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(save_path)


def _wrap_angle_error_deg(pred_rad: float, gt_rad: float) -> float:
    delta = (pred_rad - gt_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def _resolve_stage3_top1_coords(
    coords_pred: torch.Tensor,
    query_results: Optional[Sequence[object]],
) -> Tuple[torch.Tensor, List[str]]:
    coords_pred = _as_coords_tensor(coords_pred)
    if coords_pred.ndim == 3:
        if int(coords_pred.shape[1]) == 0:
            raise ValueError("coords_pred has zero top-k dimension.")
        coords_pred_top1 = coords_pred[:, 0, :]
    elif coords_pred.ndim == 2:
        coords_pred_top1 = coords_pred
    else:
        raise ValueError(f"coords_pred must be [B,K,4] or [B,4], got shape={tuple(coords_pred.shape)}")

    n_query = int(coords_pred_top1.shape[0])
    coords_stage3 = coords_pred_top1.clone()
    coord_sources = ["final_coords_pred"] * n_query
    if query_results is None or len(query_results) != n_query:
        return coords_stage3, coord_sources

    for q_idx, query_result in enumerate(query_results):
        stage_trace = getattr(query_result, "stage_trace", None)
        if stage_trace is None:
            continue
        stage_records = getattr(stage_trace, "stage_records", None)
        if stage_records is None:
            continue
        stage3_records = [
            record
            for record in stage_records
            if str(getattr(record, "stage_name", "")).strip().lower() == "stage3"
            and getattr(record, "coords_topk_raw", None) is not None
            and int(record.coords_topk_raw.shape[0]) > 0
        ]
        if len(stage3_records) == 0:
            continue
        stage3_record = max(stage3_records, key=lambda record: int(getattr(record, "stage_id", 0)))
        coords_stage3[q_idx] = stage3_record.coords_topk_raw[0].detach().to(dtype=torch.float32)
        coord_sources[q_idx] = "stage3_top1"

    return coords_stage3, coord_sources


def export_stage3_retrieval_triplets(
    trainer,
    coords_pred: torch.Tensor,
    coords_gt: torch.Tensor,
    output_root: str,
    use_train_uav: bool = False,
    query_results: Optional[Sequence[object]] = None,
    apply_rotation: bool = True,
    export_batch_size: int = 32,
) -> str:
    """
    Export per-query Stage3 analysis assets:
    - query image
    - predicted satellite crop from Stage3 best coord
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
    coords_stage3_top1, coord_sources = _resolve_stage3_top1_coords(
        coords_pred=coords_pred,
        query_results=query_results,
    )
    coords_stage3_top1 = coords_stage3_top1.reshape(-1, 4)
    if int(coords_gt.shape[0]) != int(coords_stage3_top1.shape[0]):
        raise ValueError(
            f"coords_gt and coords_pred size mismatch: {tuple(coords_gt.shape)} vs {tuple(coords_stage3_top1.shape)}"
        )

    split_name = "train" if use_train_uav else "test"
    query_paths = list(uav_dataset.uavimg_paths_train if use_train_uav else uav_dataset.uavimg_paths_test)
    source_indices = np.asarray(uav_dataset.train_indices if use_train_uav else uav_dataset.test_indices, dtype=np.int64)
    n_query = int(coords_gt.shape[0])
    if len(query_paths) < n_query or int(source_indices.shape[0]) < n_query:
        raise ValueError(
            f"Dataset split length is smaller than exported queries: "
            f"len(paths)={len(query_paths)}, len(indices)={int(source_indices.shape[0])}, n_query={n_query}"
        )

    export_batch_size = max(1, int(export_batch_size))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).expanduser().resolve() / f"stage3_triplets_{split_name}_n{n_query}_{timestamp}"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": timestamp,
        "split": split_name,
        "n_query": n_query,
        "apply_rotation": bool(apply_rotation),
        "n_satmaps": int(sat_dataset.n_satmaps),
        "pred_coord_source": "stage3_top1_from_trace_or_fallback",
        "output_dir": str(run_dir),
        "summary_csv": "stage3_triplets_summary.csv",
        "summary_json": "stage3_triplets_summary.json",
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

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
            pred_crops_batch = sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_stage3_top1[start:end],
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

                pred_img = _denormalize_with_stats(pred_crops_batch[local_idx], pred_mean, pred_std)
                _save_image_uint8(pred_img, sample_dir / "pred_stage3_best.png")

                gt_paths = []
                for satmap_id, gt_crops_batch in enumerate(gt_crops_batches):
                    gt_mean = sat_dataset.satinfo_dict["means_normalized"][satmap_id]
                    gt_std = sat_dataset.satinfo_dict["stds_normalized"][satmap_id]
                    gt_img = _denormalize_with_stats(gt_crops_batch[local_idx], gt_mean, gt_std)
                    gt_filename = f"gt_satmap{satmap_id:02d}.png"
                    _save_image_uint8(gt_img, sample_dir / gt_filename)
                    gt_paths.append(gt_filename)

                pred_coord = coords_stage3_top1[q_idx].detach().cpu()
                gt_coord = coords_gt[q_idx].detach().cpu()
                dist_2d = float(torch.norm(pred_coord[:2] - gt_coord[:2], dim=0).item())
                rot_error_deg = _wrap_angle_error_deg(float(pred_coord[2].item()), float(gt_coord[2].item()))
                pred_scale = max(float(pred_coord[3].item()), 1e-6)
                gt_scale = max(float(gt_coord[3].item()), 1e-6)
                scale_ratio = max(pred_scale / gt_scale, gt_scale / pred_scale)

                sample_meta = {
                    "query_id": int(q_idx),
                    "source_index": int(source_indices[q_idx]),
                    "query_filename": Path(query_path).name,
                    "query_path": str(Path(query_path).resolve()),
                    "sample_dir": str(sample_dir.relative_to(run_dir)),
                    "pred_coord_source": coord_sources[q_idx],
                    "pred_satmap_id": 0,
                    "pred_coord_best": [float(x) for x in pred_coord.tolist()],
                    "gt_coord": [float(x) for x in gt_coord.tolist()],
                    "dist_2d_nrc": dist_2d,
                    "rot_error_deg": rot_error_deg,
                    "scale_ratio": float(scale_ratio),
                    "saved_files": ["query.png", "pred_stage3_best.png", *gt_paths],
                }
                with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(sample_meta, f, ensure_ascii=True, indent=2)

                summary_rows.append(
                    {
                        "query_id": int(q_idx),
                        "source_index": int(source_indices[q_idx]),
                        "query_filename": Path(query_path).name,
                        "query_path": str(Path(query_path).resolve()),
                        "sample_dir": str(sample_dir.relative_to(run_dir)),
                        "pred_coord_source": coord_sources[q_idx],
                        "pred_satmap_id": 0,
                        "pred_nr": float(pred_coord[0].item()),
                        "pred_nc": float(pred_coord[1].item()),
                        "pred_rot_rad": float(pred_coord[2].item()),
                        "pred_rot_deg": float(math.degrees(float(pred_coord[2].item()))),
                        "pred_scale": float(pred_coord[3].item()),
                        "gt_nr": float(gt_coord[0].item()),
                        "gt_nc": float(gt_coord[1].item()),
                        "gt_rot_rad": float(gt_coord[2].item()),
                        "gt_rot_deg": float(math.degrees(float(gt_coord[2].item()))),
                        "gt_scale": float(gt_coord[3].item()),
                        "delta_nr": float((pred_coord[0] - gt_coord[0]).item()),
                        "delta_nc": float((pred_coord[1] - gt_coord[1]).item()),
                        "dist_2d_nrc": dist_2d,
                        "rot_error_deg": rot_error_deg,
                        "scale_ratio": float(scale_ratio),
                    }
                )

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
    if "coords_pred" not in results:
        raise KeyError("results must contain 'coords_pred'")
    if "coords_gt" not in results:
        raise KeyError("results must contain 'coords_gt'")

    return export_stage3_retrieval_triplets(
        trainer=trainer,
        coords_pred=results["coords_pred"],
        coords_gt=results["coords_gt"],
        output_root=output_root,
        use_train_uav=use_train_uav,
        query_results=results.get("query_results", None),
        apply_rotation=bool(apply_rotation),
        export_batch_size=int(export_batch_size),
    )
