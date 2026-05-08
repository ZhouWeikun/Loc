#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GIM_ROOT = REPO_ROOT.parent / "pyproj_gim"
if str(GIM_ROOT) not in sys.path:
    sys.path.insert(0, str(GIM_ROOT))

MATCHING_REFINE_ROOT = (
    REPO_ROOT
    / "gen_fm_exps"
    / "analysis"
    / "stage1_crtl_ckpts2exps"
    / "mathing_refine"
)

from matcher import Gim
from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.config.parser import get_parse
try:
    from trainer_depends.datasets.transform_raster_rcs import raster_rc_to_georc
except Exception:
    raster_rc_to_georc = None
from trainers.util_core_eval import compute_progressive_topk_acc_from_coords
from trainers.util_stage1_retrieval_evaluator import Stage1RetrievalEvaluator


def _json_dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _wrap_angle_rad(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _affine_theta_and_scale(affine_2x3: np.ndarray) -> Tuple[float, float]:
    theta = float(np.arctan2(float(affine_2x3[1, 0]), float(affine_2x3[0, 0])))
    sx = float(np.hypot(float(affine_2x3[0, 0]), float(affine_2x3[1, 0])))
    sy = float(np.hypot(float(affine_2x3[0, 1]), float(affine_2x3[1, 1])))
    return theta, 0.5 * (sx + sy)


def _affine_scale_correction_height(affine_2x3: np.ndarray, query_height: float, patch_size: float) -> float:
    _, affine_scale = _affine_theta_and_scale(affine_2x3)
    nominal_query_to_patch = float(patch_size) / max(float(query_height), 1e-8)
    return float(affine_scale) / max(nominal_query_to_patch, 1e-8)


def _to_float(value, default=None):
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value, default=None):
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(value)
    except Exception:
        return default


def _to_jsonable(obj):
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


def _to_pt_bundleable(obj):
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


def _opts_yaml_from_ckpt(ckpt_path: str) -> Optional[Path]:
    if not ckpt_path:
        return None
    ckpt_abs = Path(ckpt_path)
    if not ckpt_abs.is_absolute():
        ckpt_abs = REPO_ROOT / ckpt_abs
    opts_yaml = ckpt_abs.parent / "opts.yaml"
    return opts_yaml if opts_yaml.is_file() else None


def _make_scene_tag(scene_name: str) -> str:
    return str(scene_name).replace(os.sep, "_").replace("/", "_")


def _make_export_tag(experiment_dir: str, scene_name: str) -> str:
    return f"{experiment_dir}__{_make_scene_tag(scene_name)}"


def _retrieval_result_dir_from_bundle(bundle_path: Path) -> str:
    return str(Path(bundle_path).resolve().parent)


def _save_image_uint8(image: np.ndarray, save_path: Path):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    else:
        pil = Image.fromarray(arr)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(save_path)


@dataclass
class ExperimentSpec:
    bundle_path: Path
    report_path: Path
    experiment_dir: str
    scene_name: str
    dataset_name: str
    aggregator: str
    stage1_ckpt: str
    opts_yaml: Optional[Path]
    query_subset_rule: str = "full"


@dataclass
class MatchResult:
    success: bool
    num_matches: int
    inlier_count: int
    affine_2x3: Optional[List[List[float]]]
    inlier_mask: Optional[List[int]]
    failure_reason: str = ""


@dataclass
class MatchOutput:
    match_result: MatchResult
    raw_match_data: Optional[Dict[str, Any]]


@dataclass
class RefineResult:
    refined_coord_4d: List[float]
    refined_rowcol: List[float]
    refined_georc: Optional[List[float]]
    status: str
    patch_center_xy: Optional[List[float]]
    sat_center_xy: Optional[List[float]]


class RetrievalBundleLoader:
    def __init__(self, summary_csv: Optional[Path] = None):
        self.summary_csv = summary_csv

    def _derive_spec_from_bundle(self, bundle_path: Path) -> ExperimentSpec:
        payload = torch.load(bundle_path, map_location="cpu")
        report = dict(payload["report"])
        config = dict(payload["config"])
        scene_name = str(report["scene_name"])
        experiment_dir = bundle_path.parent.name
        dataset_name = "visloc" if scene_name.startswith("visloc_") else ("wingtra" if "wingtra" in experiment_dir else "neuloc")
        aggregator = experiment_dir.split("_dinov2B2_")[-1].rsplit("_epoch", 1)[0]
        stage1_ckpt = str(config.get("stage1_ckpt", ""))
        return ExperimentSpec(
            bundle_path=bundle_path,
            report_path=bundle_path.with_name("stage1_retrieval_eval_report.json"),
            experiment_dir=experiment_dir,
            scene_name=scene_name,
            dataset_name=dataset_name,
            aggregator=aggregator,
            stage1_ckpt=stage1_ckpt,
            opts_yaml=_opts_yaml_from_ckpt(stage1_ckpt),
            query_subset_rule="interval91subset" if experiment_dir.endswith("__eval_interval91subset") else "full",
        )

    def load_specs(
        self,
        bundle_paths: Optional[Sequence[Path]] = None,
        experiment_dirs: Optional[Sequence[str]] = None,
        aggregators: Optional[Sequence[str]] = None,
        scenes: Optional[Sequence[str]] = None,
    ) -> List[ExperimentSpec]:
        if bundle_paths:
            specs = [self._derive_spec_from_bundle(Path(p)) for p in bundle_paths]
        else:
            if self.summary_csv is None:
                raise ValueError("Either bundle_paths or summary_csv must be provided.")
            rows = []
            with self.summary_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)

            if experiment_dirs:
                wanted = set(experiment_dirs)
                rows = [row for row in rows if row.get("experiment_dir", "") in wanted]
            if aggregators:
                wanted = {str(v).lower() for v in aggregators}
                rows = [row for row in rows if str(row.get("aggregator", "")).lower() in wanted]
            if scenes:
                wanted = {str(v) for v in scenes}
                rows = [row for row in rows if str(row.get("scene", row.get("scene_name", ""))) in wanted]

            specs = []
            for row in rows:
                bundle_path = Path(row["bundle_path"])
                if not bundle_path.is_absolute():
                    bundle_path = REPO_ROOT / bundle_path
                experiment_dir = str(row.get("experiment_dir", bundle_path.parent.name))
                scene_name = str(row.get("scene", row.get("scene_name", "")))
                stage1_ckpt = str(row.get("stage1_ckpt", row.get("best_ckpt_path", "")))
                specs.append(
                    ExperimentSpec(
                        bundle_path=bundle_path,
                        report_path=Path(str(row.get("report_path", bundle_path.with_name("stage1_retrieval_eval_report.json")))),
                        experiment_dir=experiment_dir,
                        scene_name=scene_name,
                        dataset_name=str(row.get("dataset", "")) or ("visloc" if scene_name.startswith("visloc_") else "wingtra"),
                        aggregator=str(row.get("aggregator", "")),
                        stage1_ckpt=stage1_ckpt,
                        opts_yaml=_opts_yaml_from_ckpt(stage1_ckpt),
                        query_subset_rule="interval91subset" if experiment_dir.endswith("__eval_interval91subset") else "full",
                    )
                )

        dedup = []
        seen = set()
        for spec in specs:
            key = (str(spec.bundle_path), spec.scene_name)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(spec)
        return dedup


class QueryDatasetResolver:
    def __init__(self):
        self._dataset_cache: Dict[Tuple[str, str], Tuple[BaseTrainer, object, object]] = {}

    def _load_opt(self, opts_yaml: Path, selected_scene_name: str):
        argv_backup = list(sys.argv)
        try:
            sys.argv = [
                "retrieval_then_gim_refine.py",
                "--p_yaml",
                str(opts_yaml),
                "--selected_scene_name",
                str(selected_scene_name),
            ]
            opt = get_parse(print_summary=False)
        finally:
            sys.argv = argv_backup
        return opt

    def _resolve_dataset_pair(self, spec: ExperimentSpec):
        if spec.opts_yaml is None:
            raise FileNotFoundError(f"Cannot infer opts.yaml from checkpoint: {spec.stage1_ckpt}")
        cache_key = (str(spec.opts_yaml), spec.scene_name)
        if cache_key not in self._dataset_cache:
            opt = self._load_opt(spec.opts_yaml, selected_scene_name=spec.scene_name)
            opt = self._force_single_scene(opt, selected_scene_name=spec.scene_name)
            opt.satmaps_on_cpu = True
            opt.num_worker = 0
            trainer = BaseTrainer(opt)
            trainer._init_datasets(create_train_loader=False)
            sat_dataset = trainer.sat_datasets[spec.scene_name]
            uav_dataset = trainer.uav_datasets_test[spec.scene_name]
            self._dataset_cache[cache_key] = (trainer, sat_dataset, uav_dataset)
        return self._dataset_cache[cache_key]

    @staticmethod
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

    @staticmethod
    def _build_query_indices(
        total_queries: int,
        retrieval_eval_cfg: Dict[str, Any],
        rule: str,
    ) -> List[int]:
        indices = np.arange(int(total_queries), dtype=np.int64)
        query_subset_mode = str(retrieval_eval_cfg.get("query_subset_mode", "") or "").strip().lower()
        query_subset_ratio = _to_float(retrieval_eval_cfg.get("query_subset_train_ratio", None), default=None)

        if query_subset_mode not in {"", "none"} and query_subset_ratio is not None:
            train_indices, test_indices = Stage1RetrievalEvaluator._compute_split_indices(
                n_samples=int(total_queries),
                train_ratio=float(query_subset_ratio),
                split_mode=query_subset_mode,
                random_seed=_to_int(retrieval_eval_cfg.get("query_subset_random_seed", 2026), default=2026),
            )
            query_subset_take = str(retrieval_eval_cfg.get("query_subset_take", "test")).strip().lower()
            if query_subset_take == "train":
                indices = train_indices
            elif query_subset_take == "test":
                indices = test_indices
            else:
                raise ValueError(f"Unsupported query_subset_take: {query_subset_take!r}")
        elif rule == "interval91subset":
            indices = indices[1::2]

        max_queries = _to_int(retrieval_eval_cfg.get("max_queries", None), default=None)
        if max_queries is not None:
            indices = indices[: int(max_queries)]
        return indices.tolist()

    def resolve(self, spec: ExperimentSpec, payload: dict):
        _, sat_dataset, uav_dataset = self._resolve_dataset_pair(spec)
        config = dict(payload.get("config", {}))
        retrieval_eval_cfg = dict(config.get("retrieval_eval_cfg", {}))
        if str(retrieval_eval_cfg.get("use_train_uav", "False")).lower() in {"1", "true", "yes", "y"}:
            raise NotImplementedError("use_train_uav=True is not supported in this script.")

        coords_gt = payload["coords_gt"]
        query_paths = config.get("query_paths", None)
        if isinstance(query_paths, list) and query_paths:
            name_to_indices: Dict[str, List[int]] = {}
            for idx, path in enumerate(uav_dataset.uavimg_paths_test):
                name_to_indices.setdefault(Path(path).name, []).append(idx)
            name_to_all_paths: Dict[str, List[str]] = {}
            for path in getattr(uav_dataset, "uavimg_paths", []):
                name_to_all_paths.setdefault(Path(path).name, []).append(str(path))
            query_indices = []
            missing = []
            ambiguous = []
            for query_path in query_paths:
                filename = Path(str(query_path)).name
                matches = name_to_indices.get(filename, [])
                if matches:
                    if len(matches) > 1:
                        ambiguous.append(filename)
                        continue
                    query_indices.append(int(matches[0]))
                    continue

                path_matches = name_to_all_paths.get(filename, [])
                if not path_matches:
                    fallback_path = self._fallback_query_path(spec.scene_name, filename)
                    if fallback_path is not None:
                        query_indices.append(str(fallback_path))
                        continue
                    missing.append(filename)
                    continue
                if len(path_matches) > 1:
                    ambiguous.append(filename)
                    continue
                query_indices.append(path_matches[0])
            if missing or ambiguous:
                raise ValueError(
                    f"Cannot align query_paths for {spec.experiment_dir}: "
                    f"missing={missing[:10]} ambiguous={ambiguous[:10]}"
                )
        else:
            total_queries = len(uav_dataset.uavimg_paths_test)
            query_indices = self._build_query_indices(
                total_queries=total_queries,
                retrieval_eval_cfg=retrieval_eval_cfg,
                rule=spec.query_subset_rule,
            )
        if len(query_indices) != int(coords_gt.shape[0]):
            raise ValueError(
                f"Query selection mismatch for {spec.experiment_dir}: "
                f"resolved {len(query_indices)} query paths, bundle has {int(coords_gt.shape[0])} queries."
            )
        return sat_dataset, uav_dataset, query_indices

    @staticmethod
    def _fallback_query_path(scene_name: str, filename: str) -> Optional[Path]:
        scene = str(scene_name).strip().lower()
        roots = {
            "zuchwil": [
                Path("/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_h384"),
                Path("/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_org"),
            ],
            "zurich": [
                Path("/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_h384"),
                Path("/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_org"),
            ],
        }.get(scene, [])
        for root in roots:
            path = root / filename
            if path.is_file():
                return path
        return None

    def load_query_rgb(self, uav_dataset, dataset_index: int) -> np.ndarray:
        path = str(dataset_index) if isinstance(dataset_index, str) else uav_dataset.uavimg_paths_test[int(dataset_index)]
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def load_query_path(self, uav_dataset, dataset_index: int) -> str:
        if isinstance(dataset_index, str):
            return dataset_index
        return str(uav_dataset.uavimg_paths_test[int(dataset_index)])


class SatPatchExtractor:
    def __init__(self, sat_dataset):
        self.sat_dataset = sat_dataset

    def extract_patch(self, coord_4d: torch.Tensor) -> np.ndarray:
        coord_4d = coord_4d.detach().cpu().to(torch.float32)
        if hasattr(self.sat_dataset, "crop_satimg_by_4d_coords_fast"):
            sat_patch = self.sat_dataset.crop_satimg_by_4d_coords_fast(coord_4d, apply_rotation=True)
        else:
            sat_patch = self.sat_dataset.crop_satimg_by_4d_coords(coord_4d, apply_rotation=True)
        if sat_patch.ndim == 4:
            sat_patch = sat_patch[0]
        return self.sat_dataset.denormalize_img(sat_patch)

    def coord_to_patch_meta(self, coord_4d: torch.Tensor) -> Dict[str, float]:
        coord_4d = coord_4d.detach().cpu().to(torch.float32)
        nr, nc, rot, scale = [float(v) for v in coord_4d.tolist()]
        row = (nr - float(self.sat_dataset.nr_tiftop)) * float(self.sat_dataset.satmap_hw_max)
        col = (nc - float(self.sat_dataset.nc_tifleft)) * float(self.sat_dataset.satmap_hw_max)
        satimgsize2crop = float(scale) * float(self.sat_dataset.scale_ref_m) / max(float(self.sat_dataset.geo_res_m), 1e-8)
        lower, upper = [float(v) for v in self.sat_dataset.satimgsize2crop_boundary]
        satimgsize2crop = float(np.clip(satimgsize2crop, lower, upper))
        return {
            "nr": nr,
            "nc": nc,
            "rot": rot,
            "scale": scale,
            "row_center": row,
            "col_center": col,
            "satimgsize2crop": satimgsize2crop,
            "patch_size": int(self.sat_dataset.imgsize2net),
        }


class GIMMatcherRunner:
    def __init__(self, match_model: str = "gim_dkm", weights_dir: Optional[str] = None):
        kwargs = {"match_model": match_model}
        if weights_dir:
            kwargs["weights_dir"] = weights_dir
        self.matcher = Gim(**kwargs)

    def _sanitize_match_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        keep_keys = [
            "hw0_i",
            "hw1_i",
            "mkpts0_f",
            "mkpts1_f",
            "mconf",
            "m_bids",
        ]
        sanitized = {}
        for key in keep_keys:
            if key in data and data[key] is not None:
                sanitized[key] = _to_pt_bundleable(data[key])
        return sanitized

    def match(self, query_rgb: np.ndarray, sat_patch_rgb: np.ndarray) -> MatchOutput:
        try:
            data = self.matcher.match(query_rgb, sat_patch_rgb)
        except Exception as exc:
            return MatchOutput(
                match_result=MatchResult(
                    success=False,
                    num_matches=0,
                    inlier_count=0,
                    affine_2x3=None,
                    inlier_mask=None,
                    failure_reason=f"match_failed:{exc}",
                ),
                raw_match_data=None,
            )

        raw_match_data = self._sanitize_match_data(data)
        mkpts0 = data.get("mkpts0_f", None)
        mkpts1 = data.get("mkpts1_f", None)
        if mkpts0 is None or mkpts1 is None:
            return MatchOutput(MatchResult(False, 0, 0, None, None, "missing_keypoints"), raw_match_data)
        mkpts0 = mkpts0.detach().cpu().numpy()
        mkpts1 = mkpts1.detach().cpu().numpy()
        if len(mkpts0) < 4 or len(mkpts1) < 4:
            return MatchOutput(MatchResult(False, int(len(mkpts0)), 0, None, None, "too_few_matches"), raw_match_data)

        affine, inliers = cv2.estimateAffinePartial2D(
            mkpts0,
            mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.999,
            maxIters=10000,
        )
        if affine is None or inliers is None:
            return MatchOutput(MatchResult(False, int(len(mkpts0)), 0, None, None, "affine_failed"), raw_match_data)

        inlier_mask = inliers.reshape(-1).astype(np.uint8)
        inlier_count = int(inlier_mask.sum())
        return MatchOutput(
            match_result=MatchResult(
                success=True,
                num_matches=int(len(mkpts0)),
                inlier_count=inlier_count,
                affine_2x3=affine.astype(np.float32).tolist(),
                inlier_mask=inlier_mask.tolist(),
                failure_reason="",
            ),
            raw_match_data=raw_match_data,
        )


class CandidateReranker:
    def rerank(
        self,
        coords_topk: torch.Tensor,
        match_results: Sequence[MatchResult],
    ) -> Tuple[torch.Tensor, List[int]]:
        topn = len(match_results)
        rerank_keys = [(-int(match_results[idx].inlier_count), idx) for idx in range(topn)]
        rerank_prefix = [idx for _, idx in sorted(rerank_keys)]
        rerank_order = rerank_prefix + list(range(topn, int(coords_topk.shape[0])))
        reranked = coords_topk[torch.tensor(rerank_order, dtype=torch.long)]
        return reranked, rerank_order


class Top1CenterRefiner:
    def __init__(self, sat_dataset, min_inliers: int = 8):
        self.sat_dataset = sat_dataset
        self.min_inliers = int(min_inliers)

    def refine(
        self,
        query_rgb: np.ndarray,
        top1_coord: torch.Tensor,
        match_result: MatchResult,
        patch_meta: Dict[str, float],
    ) -> RefineResult:
        coord_list = [float(v) for v in top1_coord.detach().cpu().tolist()]
        if (not match_result.success) or match_result.affine_2x3 is None:
            return RefineResult(coord_list, [patch_meta["row_center"], patch_meta["col_center"]], None, "fallback_match_failed", None, None)
        if int(match_result.inlier_count) < self.min_inliers:
            return RefineResult(coord_list, [patch_meta["row_center"], patch_meta["col_center"]], None, "fallback_low_inliers", None, None)

        affine = np.asarray(match_result.affine_2x3, dtype=np.float64)
        h_u, w_u = query_rgb.shape[:2]
        query_center = np.array([0.5 * (w_u - 1), 0.5 * (h_u - 1), 1.0], dtype=np.float64)
        patch_xy = affine @ query_center
        x_p, y_p = float(patch_xy[0]), float(patch_xy[1])

        patch_size = float(patch_meta["patch_size"])
        center_patch = 0.5 * (patch_size - 1.0)
        dx = ((x_p / max(patch_size - 1.0, 1e-8)) * 2.0 - 1.0) * (patch_meta["satimgsize2crop"] / 2.0)
        dy = ((y_p / max(patch_size - 1.0, 1e-8)) * 2.0 - 1.0) * (patch_meta["satimgsize2crop"] / 2.0)

        rot = float(patch_meta["rot"])
        cos_v = float(np.cos(rot))
        sin_v = float(np.sin(rot))
        delta_col = cos_v * dx + sin_v * dy
        delta_row = -sin_v * dx + cos_v * dy

        row_refined = float(patch_meta["row_center"] + delta_row)
        col_refined = float(patch_meta["col_center"] + delta_col)
        nr_refined = row_refined / float(self.sat_dataset.satmap_hw_max) + float(self.sat_dataset.nr_tiftop)
        nc_refined = col_refined / float(self.sat_dataset.satmap_hw_max) + float(self.sat_dataset.nc_tifleft)
        theta_affine, _ = _affine_theta_and_scale(affine)
        rot_refined = _wrap_angle_rad(float(coord_list[2]) - theta_affine)
        scale_corr = _affine_scale_correction_height(
            affine_2x3=affine,
            query_height=float(h_u),
            patch_size=patch_size,
        )
        scale_refined = float(coord_list[3]) * float(scale_corr)

        refined_coord = [float(nr_refined), float(nc_refined), float(rot_refined), float(scale_refined)]

        georc = None
        if raster_rc_to_georc is not None:
            geoy, geox = raster_rc_to_georc(
                rows=np.array([row_refined], dtype=np.float64),
                cols=np.array([col_refined], dtype=np.float64),
                source_geotransform=self.sat_dataset.geo_transform,
                source_epsg_code=self.sat_dataset.epsg_code,
                target_epsg_code=self.sat_dataset.epsg_code,
            )
            georc = [float(geoy[0]), float(geox[0])]
        return RefineResult(
            refined_coord_4d=refined_coord,
            refined_rowcol=[row_refined, col_refined],
            refined_georc=georc,
            status="refined",
            patch_center_xy=[x_p, y_p],
            sat_center_xy=[center_patch, center_patch],
        )


class RefineEvaluator:
    @staticmethod
    def evaluate(
        coords_pred: torch.Tensor,
        coords_gt: torch.Tensor,
        thresholds: Dict[str, float],
        k_values: Sequence[int],
    ) -> Dict[str, object]:
        acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
            coords_pred,
            coords_gt,
            dist_th=float(thresholds["norm_dist"]),
            rot_th_deg=_to_float(thresholds.get("rot", None), default=None),
            scale_ratio_th=_to_float(thresholds.get("scale_ratio", None), default=None),
            k_values=tuple(int(v) for v in k_values),
        )
        return {
            "acc_metrics": {
                str(k): float(v)
                for k, v in acc_metrics_raw.items()
                if str(k).startswith("top") and str(k).endswith("_acc")
            },
            "progressive_acc_metrics": _to_jsonable(acc_metrics_raw.get("progressive_acc_metrics", {})),
            "progressive_error_metrics": _to_jsonable(acc_metrics_raw.get("progressive_error_metrics", {})),
            "legacy_acc_metrics_source": str(acc_metrics_raw.get("legacy_acc_metrics_source", "")),
            "progressive_acc_metric_sources": _to_jsonable(acc_metrics_raw.get("progressive_acc_metric_sources", {})),
            "progressive_error_metric_sources": _to_jsonable(acc_metrics_raw.get("progressive_error_metric_sources", {})),
            "err_stats": _to_jsonable(err_stats),
        }


class ResultExporter:
    def __init__(self, output_root: Path, save_intermediates: bool = False):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.save_intermediates = bool(save_intermediates)

    def prepare_experiment_dir(self, spec: ExperimentSpec) -> Dict[str, Any]:
        scene_tag = _make_scene_tag(spec.scene_name)
        export_tag = _make_export_tag(spec.experiment_dir, spec.scene_name)
        exp_dir = self.output_root / export_tag
        exp_dir.mkdir(parents=True, exist_ok=True)
        return {
            "scene_tag": scene_tag,
            "export_tag": export_tag,
            "export_dir": str(exp_dir),
            "exp_dir": exp_dir,
        }

    def save_query_artifacts(
        self,
        exp_dir: Path,
        query_local_idx: int,
        query_path: str,
        query_rgb: np.ndarray,
        candidate_artifacts: Sequence[Dict[str, Any]],
        query_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_dir = exp_dir / "per_query" / f"{int(query_local_idx):06d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        query_img_path = query_dir / "query.png"
        _save_image_uint8(query_rgb, query_img_path)

        candidate_paths = []
        for cand in candidate_artifacts:
            rank_before = int(cand["rank_before"])
            patch_path = query_dir / f"cand_{rank_before:02d}_sat_patch.png"
            _save_image_uint8(cand["patch_rgb"], patch_path)
            match_payload_path = query_dir / f"cand_{rank_before:02d}_match.pt"
            payload = {
                "query_path": str(query_path),
                "rank_before": rank_before,
                "coord_4d": cand["coord_4d"],
                "patch_meta": cand["patch_meta"],
                "match_result": cand["match_result"],
                "raw_match_data": cand["raw_match_data"],
            }
            torch.save(_to_pt_bundleable(payload), match_payload_path)
            candidate_paths.append(
                {
                    "rank_before": rank_before,
                    "patch_image_path": str(patch_path),
                    "match_payload_path": str(match_payload_path),
                }
            )

        query_meta_path = query_dir / "query_meta.json"
        with query_meta_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(query_meta), f, ensure_ascii=False, indent=2)
        return {
            "query_dir": str(query_dir),
            "query_image_path": str(query_img_path),
            "query_meta_path": str(query_meta_path),
            "candidate_paths": candidate_paths,
        }

    def export(
        self,
        exp_info: Dict[str, Any],
        report: Dict[str, object],
        details: Dict[str, object],
    ) -> Dict[str, str]:
        exp_dir = exp_info["exp_dir"]
        report_path = exp_dir / "gim_refine_report.json"
        details_path = exp_dir / "gim_refine_details.pt"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
        torch.save(_to_pt_bundleable(details), details_path)
        return {
            "scene_tag": exp_info["scene_tag"],
            "export_tag": exp_info["export_tag"],
            "export_dir": exp_info["export_dir"],
            "report_path": str(report_path),
            "details_path": str(details_path),
        }

    def append_summary_row(self, summary_csv: Path, row: Dict[str, object]):
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        normalized_row = {k: _to_jsonable(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()}
        if not summary_csv.is_file():
            with summary_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(normalized_row.keys()))
                writer.writeheader()
                writer.writerow(normalized_row)
            return

        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = list(reader.fieldnames or [])

        merged_fields = existing_fields + [k for k in normalized_row.keys() if k not in existing_fields]
        if merged_fields != existing_fields:
            with summary_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({k: old_row.get(k, "") for k in merged_fields})
                writer.writerow({k: normalized_row.get(k, "") for k in merged_fields})
            return

        with summary_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow({k: normalized_row.get(k, "") for k in existing_fields})


class RetrievalThenGIMRefinePipeline:
    def __init__(
        self,
        matcher_model: str,
        topn_match: int,
        min_inliers_for_refine: int,
        output_root: Path,
        summary_csv_out: Path,
        weights_dir: Optional[str] = None,
        limit_queries: Optional[int] = None,
        save_intermediates: bool = False,
    ):
        self.matcher_runner = GIMMatcherRunner(match_model=matcher_model, weights_dir=weights_dir)
        self.query_resolver = QueryDatasetResolver()
        self.reranker = CandidateReranker()
        self.output_root = output_root
        self.summary_csv_out = summary_csv_out
        self.topn_match = int(topn_match)
        self.min_inliers_for_refine = int(min_inliers_for_refine)
        self.exporter = ResultExporter(output_root=output_root, save_intermediates=save_intermediates)
        self.matcher_model = matcher_model
        self.limit_queries = None if limit_queries is None else int(limit_queries)
        self.save_intermediates = bool(save_intermediates)

    def run_experiment(self, spec: ExperimentSpec) -> Dict[str, object]:
        payload = torch.load(spec.bundle_path, map_location="cpu")
        coords_topk = payload["coords_topk"].to(torch.float32)
        coords_gt = payload["coords_gt"].to(torch.float32)
        report = dict(payload["report"])
        thresholds = dict(report["thresholds"])
        k_values = tuple(int(k) for k in report["k_values"])
        refine_target_retrieval_dir = _retrieval_result_dir_from_bundle(spec.bundle_path)

        sat_dataset, uav_dataset, query_indices = self.query_resolver.resolve(spec, payload)
        if self.limit_queries is not None:
            query_indices = query_indices[: self.limit_queries]
            coords_topk = coords_topk[: self.limit_queries]
            coords_gt = coords_gt[: self.limit_queries]

        exp_info = self.exporter.prepare_experiment_dir(spec)
        extractor = SatPatchExtractor(sat_dataset=sat_dataset)
        refiner = Top1CenterRefiner(sat_dataset=sat_dataset, min_inliers=self.min_inliers_for_refine)

        reranked_coords_all = coords_topk.clone()
        refined_coords_all = coords_topk.clone()
        query_details = []

        topn_eff = min(self.topn_match, int(coords_topk.shape[1]))
        pbar = tqdm(range(len(query_indices)), desc=f"{spec.scene_name}/{spec.aggregator}", leave=False)
        for query_local_idx in pbar:
            dataset_index = query_indices[query_local_idx]
            query_rgb = self.query_resolver.load_query_rgb(uav_dataset, dataset_index)
            query_path = self.query_resolver.load_query_path(uav_dataset, dataset_index)
            coords_row = coords_topk[query_local_idx]

            match_results: List[MatchResult] = []
            candidate_records = []
            candidate_artifacts = []
            for cand_idx in range(topn_eff):
                cand_coord = coords_row[cand_idx]
                patch_rgb = extractor.extract_patch(cand_coord)
                patch_meta = extractor.coord_to_patch_meta(cand_coord)
                match_output = self.matcher_runner.match(query_rgb=query_rgb, sat_patch_rgb=patch_rgb)
                match_results.append(match_output.match_result)
                candidate_records.append(
                    {
                        "rank_before": int(cand_idx),
                        "coord_4d": [float(v) for v in cand_coord.tolist()],
                        "patch_meta": patch_meta,
                        "match_result": asdict(match_output.match_result),
                    }
                )
                candidate_artifacts.append(
                    {
                        "rank_before": int(cand_idx),
                        "coord_4d": [float(v) for v in cand_coord.tolist()],
                        "patch_meta": patch_meta,
                        "patch_rgb": patch_rgb,
                        "match_result": asdict(match_output.match_result),
                        "raw_match_data": match_output.raw_match_data,
                    }
                )

            reranked_coords, rerank_order = self.reranker.rerank(coords_row, match_results)
            reranked_coords_all[query_local_idx] = reranked_coords
            refined_coords_all[query_local_idx] = reranked_coords

            if rerank_order[0] < topn_eff:
                top1_match_result = match_results[rerank_order[0]]
                top1_patch_meta = candidate_records[rerank_order[0]]["patch_meta"]
            else:
                top1_match_result = MatchResult(False, 0, 0, None, None, "top1_outside_matched_prefix")
                top1_patch_meta = extractor.coord_to_patch_meta(reranked_coords[0])

            refine_result = refiner.refine(
                query_rgb=query_rgb,
                top1_coord=reranked_coords[0],
                match_result=top1_match_result,
                patch_meta=top1_patch_meta,
            )
            refined_coords_all[query_local_idx, 0] = torch.tensor(refine_result.refined_coord_4d, dtype=torch.float32)

            query_record = {
                "query_local_idx": int(query_local_idx),
                "dataset_index": int(dataset_index) if not isinstance(dataset_index, str) else dataset_index,
                "query_path": query_path,
                "gt_coord_4d": [float(v) for v in coords_gt[query_local_idx].tolist()],
                "rerank_order_prefix": [int(v) for v in rerank_order[:topn_eff]],
                "top1_coord_before": [float(v) for v in coords_row[0].tolist()],
                "top1_coord_after_rerank": [float(v) for v in reranked_coords[0].tolist()],
                "top1_coord_after_refine": list(refine_result.refined_coord_4d),
                "refine_result": asdict(refine_result),
                "candidates": candidate_records,
            }
            if self.save_intermediates:
                artifact_info = self.exporter.save_query_artifacts(
                    exp_dir=exp_info["exp_dir"],
                    query_local_idx=query_local_idx,
                    query_path=query_path,
                    query_rgb=query_rgb,
                    candidate_artifacts=candidate_artifacts,
                    query_meta=query_record,
                )
                query_record["artifact_dir"] = artifact_info["query_dir"]
                query_record["saved_query_image_path"] = artifact_info["query_image_path"]
                query_record["saved_query_meta_path"] = artifact_info["query_meta_path"]
                path_map = {int(v["rank_before"]): v for v in artifact_info["candidate_paths"]}
                for cand in query_record["candidates"]:
                    saved = path_map.get(int(cand["rank_before"]), {})
                    cand.update(saved)

            query_details.append(query_record)

        baseline_eval = RefineEvaluator.evaluate(coords_pred=coords_topk, coords_gt=coords_gt, thresholds=thresholds, k_values=k_values)
        rerank_eval = RefineEvaluator.evaluate(coords_pred=reranked_coords_all, coords_gt=coords_gt, thresholds=thresholds, k_values=k_values)
        refine_eval = RefineEvaluator.evaluate(coords_pred=refined_coords_all, coords_gt=coords_gt, thresholds=thresholds, k_values=k_values)

        report_payload = {
            "scene_name": spec.scene_name,
            "scene_tag": exp_info["scene_tag"],
            "dataset_name": spec.dataset_name,
            "aggregator": spec.aggregator,
            "experiment_dir": spec.experiment_dir,
            "export_tag": exp_info["export_tag"],
            "export_dir": exp_info["export_dir"],
            "refine_target_retrieval_dir": refine_target_retrieval_dir,
            "bundle_path": str(spec.bundle_path),
            "report_path_original": str(spec.report_path),
            "matcher_model": self.matcher_model,
            "topn_match": self.topn_match,
            "min_inliers_for_refine": self.min_inliers_for_refine,
            "query_subset_rule": spec.query_subset_rule,
            "n_queries": int(coords_gt.shape[0]),
            "limit_queries": self.limit_queries,
            "save_intermediates": self.save_intermediates,
            "refine_pose": {
                "rot_from_affine": True,
                "rot_formula": "rot_new=wrap_pi(rot_old-atan2(affine[1,0],affine[0,0]))",
                "scale_from_affine": True,
                "scale_formula": "scale_new=scale_old*(affine_iso_scale/(patch_size/query_height))",
                "scale_norm": "height",
            },
            "thresholds": _to_jsonable(thresholds),
            "k_values": [int(v) for v in k_values],
            "baseline_eval": baseline_eval,
            "rerank_eval": rerank_eval,
            "refine_eval": refine_eval,
        }
        details_payload = {
            "spec": asdict(spec),
            "refine_target_retrieval_dir": refine_target_retrieval_dir,
            "refine_pose": {
                "rot_from_affine": True,
                "rot_formula": "rot_new=wrap_pi(rot_old-atan2(affine[1,0],affine[0,0]))",
                "scale_from_affine": True,
                "scale_formula": "scale_new=scale_old*(affine_iso_scale/(patch_size/query_height))",
                "scale_norm": "height",
            },
            "thresholds": thresholds,
            "k_values": [int(v) for v in k_values],
            "coords_gt": coords_gt,
            "coords_topk_baseline": coords_topk,
            "coords_topk_reranked": reranked_coords_all,
            "coords_topk_refined": refined_coords_all,
            "query_details": query_details,
        }
        export_paths = self.exporter.export(exp_info=exp_info, report=report_payload, details=details_payload)

        def _progressive_top(eval_payload: Dict[str, object], group: str, key: str) -> float:
            progressive = eval_payload.get("progressive_acc_metrics", {})
            if not isinstance(progressive, dict):
                return 0.0
            group_metrics = progressive.get(group, {})
            if not isinstance(group_metrics, dict):
                return 0.0
            return float(group_metrics.get(key, 0.0))

        def _maybe_float(value: object) -> object:
            if value is None:
                return ""
            try:
                return float(value)
            except Exception:
                return ""

        def _err_stat(eval_payload: Dict[str, object], key: str) -> object:
            err_stats = eval_payload.get("err_stats", {})
            if not isinstance(err_stats, dict):
                return ""
            return _maybe_float(err_stats.get(key))

        def _progressive_err(eval_payload: Dict[str, object], group: str, key: str) -> object:
            progressive = eval_payload.get("progressive_error_metrics", {})
            if not isinstance(progressive, dict):
                return ""
            group_metrics = progressive.get(group, {})
            if not isinstance(group_metrics, dict):
                return ""
            return _maybe_float(group_metrics.get(key))

        def _add_error_summary_fields(
            row: Dict[str, object],
            prefix: str,
            eval_payload: Dict[str, object],
            nrc2meter: float,
        ) -> None:
            dist_median = _err_stat(eval_payload, "median_dist_err_top1")
            row[f"{prefix}_dist_median"] = dist_median
            row[f"{prefix}_dist_median_meter"] = (
                "" if dist_median == "" else float(dist_median) * float(nrc2meter)
            )
            row[f"{prefix}_rot_median"] = _err_stat(eval_payload, "median_rot_err_top1")
            row[f"{prefix}_scale_median"] = _err_stat(eval_payload, "median_scale_ratio_top1")

            for group in ("dist_recall", "dist_rot_recall", "dist_rot_scale_recall"):
                base = f"{prefix}_{group}_success"
                cond_dist = _progressive_err(eval_payload, group, "median_dist_err_top1_given_success")
                row[f"{base}_n_top1"] = _progressive_err(eval_payload, group, "n_success_top1")
                row[f"{base}_median_dist"] = cond_dist
                row[f"{base}_median_dist_meter"] = (
                    "" if cond_dist == "" else float(cond_dist) * float(nrc2meter)
                )
                row[f"{base}_median_rot"] = _progressive_err(
                    eval_payload, group, "median_rot_err_top1_given_success"
                )
                row[f"{base}_median_scale"] = _progressive_err(
                    eval_payload, group, "median_scale_ratio_top1_given_success"
                )

        nrc2meter = _to_float(thresholds.get("nrc2meter", None), default=1.0)

        summary_row = {
            "scene": spec.scene_name,
            "scene_tag": export_paths["scene_tag"],
            "dataset": spec.dataset_name,
            "aggregator": spec.aggregator,
            "experiment_dir": spec.experiment_dir,
            "export_tag": export_paths["export_tag"],
            "export_dir": export_paths["export_dir"],
            "refine_target_retrieval_dir": refine_target_retrieval_dir,
            "matcher_model": self.matcher_model,
            "topn_match": self.topn_match,
            "min_inliers_for_refine": self.min_inliers_for_refine,
            "query_subset_rule": spec.query_subset_rule,
            "save_intermediates": self.save_intermediates,
            "n_queries": int(coords_gt.shape[0]),
            "limit_queries": self.limit_queries,
            "baseline_top1_acc": float(baseline_eval["acc_metrics"].get("top1_acc", 0.0)),
            "rerank_top1_acc": float(rerank_eval["acc_metrics"].get("top1_acc", 0.0)),
            "refine_top1_acc": float(refine_eval["acc_metrics"].get("top1_acc", 0.0)),
            "baseline_dist_recall_top1": _progressive_top(baseline_eval, "dist_recall", "top1_acc"),
            "baseline_dist_recall_top5": _progressive_top(baseline_eval, "dist_recall", "top5_acc"),
            "baseline_dist_rot_recall_top1": _progressive_top(baseline_eval, "dist_rot_recall", "top1_acc"),
            "baseline_dist_rot_recall_top5": _progressive_top(baseline_eval, "dist_rot_recall", "top5_acc"),
            "baseline_dist_rot_scale_recall_top1": _progressive_top(
                baseline_eval, "dist_rot_scale_recall", "top1_acc"
            ),
            "baseline_dist_rot_scale_recall_top5": _progressive_top(
                baseline_eval, "dist_rot_scale_recall", "top5_acc"
            ),
            "rerank_dist_recall_top1": _progressive_top(rerank_eval, "dist_recall", "top1_acc"),
            "rerank_dist_recall_top5": _progressive_top(rerank_eval, "dist_recall", "top5_acc"),
            "rerank_dist_rot_recall_top1": _progressive_top(rerank_eval, "dist_rot_recall", "top1_acc"),
            "rerank_dist_rot_recall_top5": _progressive_top(rerank_eval, "dist_rot_recall", "top5_acc"),
            "rerank_dist_rot_scale_recall_top1": _progressive_top(
                rerank_eval, "dist_rot_scale_recall", "top1_acc"
            ),
            "rerank_dist_rot_scale_recall_top5": _progressive_top(
                rerank_eval, "dist_rot_scale_recall", "top5_acc"
            ),
            "refine_dist_recall_top1": _progressive_top(refine_eval, "dist_recall", "top1_acc"),
            "refine_dist_recall_top5": _progressive_top(refine_eval, "dist_recall", "top5_acc"),
            "refine_dist_rot_recall_top1": _progressive_top(refine_eval, "dist_rot_recall", "top1_acc"),
            "refine_dist_rot_recall_top5": _progressive_top(refine_eval, "dist_rot_recall", "top5_acc"),
            "refine_dist_rot_scale_recall_top1": _progressive_top(
                refine_eval, "dist_rot_scale_recall", "top1_acc"
            ),
            "refine_dist_rot_scale_recall_top5": _progressive_top(
                refine_eval, "dist_rot_scale_recall", "top5_acc"
            ),
            "baseline_dist_mean": float(baseline_eval["err_stats"]["mean_dist_err_top1"]),
            "rerank_dist_mean": float(rerank_eval["err_stats"]["mean_dist_err_top1"]),
            "refine_dist_mean": float(refine_eval["err_stats"]["mean_dist_err_top1"]),
            "baseline_progressive_acc_metrics": _json_dump(baseline_eval["progressive_acc_metrics"]),
            "rerank_progressive_acc_metrics": _json_dump(rerank_eval["progressive_acc_metrics"]),
            "refine_progressive_acc_metrics": _json_dump(refine_eval["progressive_acc_metrics"]),
            "baseline_progressive_error_metrics": _json_dump(baseline_eval["progressive_error_metrics"]),
            "rerank_progressive_error_metrics": _json_dump(rerank_eval["progressive_error_metrics"]),
            "refine_progressive_error_metrics": _json_dump(refine_eval["progressive_error_metrics"]),
            "report_path": export_paths["report_path"],
            "details_path": export_paths["details_path"],
            "bundle_path": str(spec.bundle_path),
        }
        _add_error_summary_fields(summary_row, "baseline", baseline_eval, nrc2meter)
        _add_error_summary_fields(summary_row, "rerank", rerank_eval, nrc2meter)
        _add_error_summary_fields(summary_row, "refine", refine_eval, nrc2meter)
        self.exporter.append_summary_row(self.summary_csv_out, summary_row)

        return {
            "report": report_payload,
            "details": details_payload,
            "export_paths": export_paths,
        }


def build_argparser():
    parser = argparse.ArgumentParser(description="Post-hoc retrieval-then-GIM rerank/refine from saved Stage1 bundles.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "gen_fm_exps" / "analysis" / "stage1_interval82_gallery_summary_progressive_multi_cfg.csv",
        help="Summary CSV containing bundle_path rows.",
    )
    parser.add_argument("--bundle-path", type=Path, action="append", default=None, help="Process a specific bundle path. Can be repeated.")
    parser.add_argument("--experiment-dir", type=str, action="append", default=None, help="Filter summary CSV by experiment_dir. Can be repeated.")
    parser.add_argument("--aggregator", type=str, action="append", default=None, help="Filter summary CSV by aggregator. Can be repeated.")
    parser.add_argument("--scene", type=str, action="append", default=None, help="Filter summary CSV by scene name. Can be repeated.")
    parser.add_argument("--matcher-model", type=str, default="gim_dkm")
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--topn-match", type=int, default=10)
    parser.add_argument("--min-inliers-for-refine", type=int, default=8)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--save-intermediates", action="store_true", help="Save query image, cropped reference patches, and per-candidate match payloads.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MATCHING_REFINE_ROOT,
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=MATCHING_REFINE_ROOT / "gim_refine_summary.csv",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    loader = RetrievalBundleLoader(summary_csv=args.summary_csv)
    specs = loader.load_specs(
        bundle_paths=args.bundle_path,
        experiment_dirs=args.experiment_dir,
        aggregators=args.aggregator,
        scenes=args.scene,
    )
    if len(specs) == 0:
        raise ValueError("No experiments selected.")

    pipeline = RetrievalThenGIMRefinePipeline(
        matcher_model=args.matcher_model,
        topn_match=args.topn_match,
        min_inliers_for_refine=args.min_inliers_for_refine,
        output_root=args.output_root,
        summary_csv_out=args.summary_out,
        weights_dir=args.weights_dir,
        limit_queries=args.limit_queries,
        save_intermediates=args.save_intermediates,
    )
    for spec in specs:
        print(f"[GIMRefine] start -> {spec.scene_name} / {spec.aggregator} / {spec.query_subset_rule}")
        result = pipeline.run_experiment(spec)
        print(f"[GIMRefine] done  -> {result['export_paths']['report_path']}")


if __name__ == "__main__":
    main()
