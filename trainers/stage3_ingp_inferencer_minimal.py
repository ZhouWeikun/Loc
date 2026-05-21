#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Stage-3 INGP inference unit.

This file keeps only the Stage-3 inference path:
1. build frozen Stage-1 visual modules and Stage-2 INGP modules
2. load Stage-2 and Stage-1 checkpoints
3. run seed-mode coarse -> mode -> CMA inference
4. print and optionally save full per-stage progressive Recall@K metrics

Excluded on purpose: Stage-3 training, legacy 2D/3D classification wrappers,
PDF visualization, triplet export, and gallery/provenance artifact plumbing.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence

import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainer_depends.config.parser import _expand_selected_scene_config, get_parse, print_config_summary
from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords
from trainers.stage2_INGP_minimal import MinimalStage2INGPTrainer
from trainers.util_stage3_ckpt import (
    _apply_inherit_stage2_yaml as util_apply_inherit_stage2_yaml,
    _get_stage1_checkpoint_path as util_get_stage1_checkpoint_path,
    _get_stage2_checkpoint_path as util_get_stage2_checkpoint_path,
)
from trainers.util_stage3_runtime import (
    _estimate_stage3_grid_shape_from_overlap as util_estimate_stage3_grid_shape_from_overlap,
    _init_stage3_test_runtime as util_init_stage3_test_runtime,
    _resolve_stage3_sampling_config as util_resolve_stage3_sampling_config,
)
from trainers.util_stage3_score import (
    _compute_metric_from_ingp as util_compute_metric_from_ingp,
    _compute_metric_from_query_and_points as util_compute_metric_from_query_and_points,
    _get_feats_fm_INGP as util_get_feats_fm_INGP,
)
from trainers.util_stage3_search import (
    _build_cma_init_seeds as util_build_cma_init_seeds,
    _pack_mode_states_for_eval as util_pack_mode_states_for_eval,
    _pack_stage_records_for_eval as util_pack_stage_records_for_eval,
    _sample_around_candidates as util_sample_around_candidates,
    _sample_indices as util_sample_indices,
    _test_3d_fine_accuracy_hds as util_test_3d_fine_accuracy_hds,
)


def _find_latest_epoch_ckpt(directory):
    if not directory or not os.path.isdir(directory):
        return ""
    candidates = []
    for name in os.listdir(directory):
        if not (name.startswith("epoch") and name.endswith(".pth")):
            continue
        digits = []
        for ch in name[len("epoch"):]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            candidates.append((int("".join(digits)), name))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]))
    return os.path.join(directory, candidates[-1][1])


class MinimalStage3INGPInferencer(MinimalStage2INGPTrainer):
    """INGP-only Stage-3 inferencer with complete progressive metric output."""

    def __init__(self, opt=None):
        should_print_final_config = (opt is None) or not bool(getattr(opt, "_config_summary_printed", False))
        if opt is None:
            opt = get_parse(print_summary=False)
        opt = util_apply_inherit_stage2_yaml(opt)
        opt.scenes_setting = _expand_selected_scene_config(getattr(opt, "scenes_setting", None))
        if should_print_final_config:
            print_config_summary(opt, header="最终生效配置:")

        super().__init__(opt)
        self._setup_inference_modules()
        self.n_coarse = None
        self.n_fine_per_coarse = None
        self.gs_sigma_nrc = None
        self.gs_sigma_radrot = None
        self.gs_sigma_logscale = None
        self.stage3_overlap = None
        self._stage2_checkpoint_loaded = False
        self._stage1_checkpoint_loaded = bool(getattr(self.opt, "load_stage1_ckpt", ""))

    @staticmethod
    def _ensure_param_sequence(value, cast_fn):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(cast_fn(v) for v in value)
        return (cast_fn(value),)

    def _setup_inference_modules(self):
        modules = {
            "vis_encoder": self.vis_encoder,
            "vis_aggregator": self.vis_aggregator,
            "grid": self.grid,
            "grid_mlp": self.grid_mlp,
        }
        if getattr(self, "hash_lod_aggregator", None) is not None:
            modules["hash_lod_aggregator"] = self.hash_lod_aggregator

        for module in modules.values():
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        self.param2optimize = {}
        self.param2freeze = modules
        print("Stage3 minimal frozen modules:", ", ".join(self.param2freeze.keys()))

    @staticmethod
    def _build_stage3_eval_thresh_cfg(
        sat_dataset,
        dist_lambda=1.1 * 0.5,
        rot_th=11.0 * 0.5,
        scale_ratio_th=1.15,
        dist_th_meter=None,
    ):
        cfg = {
            "dist_lambda": float(dist_lambda),
            "rot_th": None if rot_th is None else float(rot_th),
            "scale_ratio_th": None if scale_ratio_th is None else float(scale_ratio_th),
        }
        if dist_th_meter is not None:
            meter2nrc = float(sat_dataset.halfimg_radius_nrc) / max(float(sat_dataset.halfimg_radius_meter), 1e-8)
            cfg["dist_th_meter"] = float(dist_th_meter)
            cfg["dist_th"] = float(dist_th_meter) * meter2nrc
            cfg["dist_lambda"] = cfg["dist_th"] / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
        return cfg

    @staticmethod
    def _resolve_stage3_recall_cfg_path(path):
        path = str(path or "").strip()
        if not path:
            return ""
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    @classmethod
    def _load_stage3_recall_cfg_yaml(cls, path):
        path_abs = cls._resolve_stage3_recall_cfg_path(path)
        if not path_abs:
            return {}
        if not os.path.isfile(path_abs):
            raise FileNotFoundError(f"Stage3 recall cfg yaml not found: {path_abs}")
        with open(path_abs, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise TypeError(f"Stage3 recall cfg yaml must be a dict: {path_abs}")
        return cfg

    def _resolve_stage3_eval_thresh_cfg_from_opt(self):
        cfg_yaml = getattr(self.opt, "stage3_recall_cfg_yaml", "trainer_depends/configs/stage3_recall_thresholds.yaml")
        cfg_selector = str(getattr(self.opt, "stage3_recall_cfg", "per_scene") or "per_scene").strip()
        cfg_root = self._load_stage3_recall_cfg_yaml(cfg_yaml)
        configs = cfg_root.get("configs", {})
        if not isinstance(configs, dict) or not configs:
            raise KeyError(f"Stage3 recall cfg yaml has no non-empty 'configs': {cfg_yaml}")

        scene_name = str(getattr(self.opt, "selected_scene_name", "") or "").strip()
        scenes_setting = getattr(self.opt, "scenes_setting", None)
        if not scene_name and isinstance(scenes_setting, dict):
            scene_name = str(scenes_setting.get("selected_scene_name", "") or "").strip()
            scenes = scenes_setting.get("scenes", [])
            if not scene_name and scenes and isinstance(scenes[0], dict):
                scene_name = str(scenes[0].get("name", "") or "").strip()

        if cfg_selector.lower() in {"per_scene", "scene", "auto"}:
            per_scene = cfg_root.get("per_scene", {})
            cfg_name = str(per_scene.get(scene_name, cfg_root.get("default", "cfg01"))).strip()
        else:
            cfg_name = cfg_selector
        if cfg_name not in configs:
            available = ", ".join(sorted(str(k) for k in configs.keys()))
            raise KeyError(f"Stage3 recall cfg '{cfg_name}' not found. Available: {available}")

        raw_cfg = configs[cfg_name]
        eval_thresh_cfg = self._build_stage3_eval_thresh_cfg(
            sat_dataset=self.sat_dataset,
            dist_lambda=float(raw_cfg.get("dist_lambda", 1.1 * 0.5)),
            rot_th=raw_cfg.get("rot_th", None),
            scale_ratio_th=raw_cfg.get("scale_ratio_th", raw_cfg.get("scale_th", None)),
            dist_th_meter=raw_cfg.get("dist_th_meter", None),
        )
        if raw_cfg.get("dist_th", None) is not None:
            dist_th = float(raw_cfg["dist_th"])
            eval_thresh_cfg["dist_th"] = dist_th
            eval_thresh_cfg["dist_lambda"] = dist_th / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
            nrc2meter = float(self.sat_dataset.halfimg_radius_meter) / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
            eval_thresh_cfg["dist_th_meter"] = dist_th * nrc2meter

        print(
            "[Stage3-RecallCfg] "
            f"yaml={self._resolve_stage3_recall_cfg_path(cfg_yaml)} selector={cfg_selector} "
            f"scene={scene_name} resolved={cfg_name} thresholds={eval_thresh_cfg}"
        )
        return eval_thresh_cfg

    def _resolve_stage2_checkpoint_path(self):
        stage2_ckpt_path = util_get_stage2_checkpoint_path(self, stage3_ckpt_path=None)
        if stage2_ckpt_path:
            return stage2_ckpt_path

        p_yaml = str(getattr(self.opt, "p_yaml", "") or "").strip()
        p_yaml_abs = p_yaml if os.path.isabs(p_yaml) else os.path.join(PROJECT_ROOT, p_yaml)
        if os.path.isfile(p_yaml_abs) and os.path.basename(p_yaml_abs) == "opts.yaml":
            latest_ckpt = _find_latest_epoch_ckpt(os.path.dirname(p_yaml_abs))
            if latest_ckpt:
                print(f"从Stage2 opts.yaml所在目录自动选择checkpoint: {latest_ckpt}")
                self.opt.load_stage2_ckpt = latest_ckpt
                return latest_ckpt
        return None

    def _load_checkpoints_for_inference(self):
        stage2_ckpt_path = self._resolve_stage2_checkpoint_path()
        if not stage2_ckpt_path:
            raise ValueError("未找到Stage 2 checkpoint；请设置 load_stage2_ckpt。")

        modules_stage2 = {"grid": self.grid, "grid_mlp": self.grid_mlp}
        if getattr(self, "hash_lod_aggregator", None) is not None:
            modules_stage2["hash_lod_aggregator"] = self.hash_lod_aggregator
        print(f"Loading Stage-2 checkpoint: {stage2_ckpt_path}")
        self._load_checkpoint(stage2_ckpt_path, modules_stage2, mode="test")
        self._stage2_checkpoint_loaded = True

        stage1_ckpt_path = util_get_stage1_checkpoint_path(self, stage2_ckpt_path)
        if not stage1_ckpt_path:
            raise ValueError("未找到Stage 1 checkpoint；请设置 load_stage1_ckpt 或确保Stage2 opts.yaml记录了该路径。")
        print(f"Loading Stage-1 checkpoint: {stage1_ckpt_path}")
        self._load_checkpoint(
            stage1_ckpt_path,
            {"vis_encoder": self.vis_encoder, "vis_aggregator": self.vis_aggregator},
            mode="test",
        )
        self._stage1_checkpoint_loaded = True

    def _enter_inference_eval_mode(self):
        for model in self.param2freeze.values():
            model.eval()

    def _resolve_hds_inference_config(self, n_bins_max):
        rounds_t = int(getattr(self.opt, "hds_stage2_rounds_t"))
        if rounds_t <= 0:
            raise ValueError("hds_stage2_rounds_t must be positive.")

        eta = float(getattr(self.opt, "hds_stage2_initial_radius_eta"))
        alpha = float(getattr(self.opt, "hds_stage2_radius_decay_alpha"))
        if eta <= 0:
            raise ValueError("hds_stage2_initial_radius_eta must be positive.")
        if not (0 < alpha <= 1):
            raise ValueError("hds_stage2_radius_decay_alpha must be in (0, 1].")

        radius_scale_per_round = tuple(eta * (alpha ** t) for t in range(rounds_t))
        final_radius_scale = radius_scale_per_round[-1]
        cma_sigma0 = (
            float(getattr(self.opt, "hds_stage3_cma_sigma_factor"))
            * final_radius_scale
            * 2.0
            / max(float(n_bins_max), 1.0)
        )

        survival_ratio = getattr(self.opt, "hds_stage2_population_survival_ratio_rho_pop")
        if isinstance(survival_ratio, (int, float)):
            survival_ratio_per_round = tuple(float(survival_ratio) for _ in range(rounds_t))
        else:
            survival_ratio_per_round = tuple(float(v) for v in survival_ratio)
            if len(survival_ratio_per_round) == 0:
                raise ValueError("hds_stage2_population_survival_ratio_rho_pop must not be empty.")
            if len(survival_ratio_per_round) < rounds_t:
                survival_ratio_per_round = survival_ratio_per_round + (
                    survival_ratio_per_round[-1],
                ) * (rounds_t - len(survival_ratio_per_round))
            elif len(survival_ratio_per_round) > rounds_t:
                survival_ratio_per_round = survival_ratio_per_round[:rounds_t]

        particles_m = int(getattr(self.opt, "hds_stage2_particles_m"))
        return {
            "eval_n_samples": getattr(self.opt, "hds_eval_n_samples", None),
            "scale_select_mode": str(getattr(self.opt, "hds_scale_select_mode")),

            "ge_top_ratio_rho0": float(getattr(self.opt, "hds_stage1_top_ratio_rho0")),
            "ge_max_anchors_k0": int(getattr(self.opt, "hds_stage1_max_anchors_k0")),

            "pc_particles_per_round": tuple(particles_m for _ in range(rounds_t)),
            "pc_radius_scale_per_round": radius_scale_per_round,
            "pc_population_survival_ratio_per_round": survival_ratio_per_round,
            "pc_elite_ratio_rho": float(getattr(self.opt, "hds_stage2_elite_ratio_rho")),
            "pc_enable_scale_sampling": bool(getattr(self.opt, "hds_stage2_enable_scale_sampling")),

            "lr_variant": str(getattr(self.opt, "hds_stage3_cma_variant")),
            "lr_max_input_modes_k2": int(getattr(self.opt, "hds_stage3_max_modes_k2")),
            "lr_init_sigma": cma_sigma0,
            "lr_popsize": int(getattr(self.opt, "hds_stage3_cma_popsize")),
            "lr_iters": int(getattr(self.opt, "hds_stage3_cma_iters")),
            "lr_early_stop_patience": int(getattr(self.opt, "hds_stage3_cma_early_stop_patience")),
            "lr_optimize_scale": bool(getattr(self.opt, "hds_stage3_optimize_scale")),

            "score_chunk_size": int(getattr(self.opt, "hds_score_chunk_size")),
        }

    def run_seed_mode_inference(self, use_train_uav=False, eval_thresh_cfg=None):
        eval_thresh_cfg = eval_thresh_cfg or self._resolve_stage3_eval_thresh_cfg_from_opt()
        stage3_test_overlap = float(getattr(self.opt, "hds_stage1_overlap"))
        grid_info = self._estimate_stage3_grid_shape_from_overlap(stage3_test_overlap)
        n_bins_4d = [
            int(grid_info["grid_rows"]),
            int(grid_info["grid_cols"]),
            int(getattr(self.opt, "hds_stage1_n_rot")),
            int(getattr(self.opt, "hds_stage1_n_scale")),
        ]
        n_bins_max = max(n_bins_4d[:2])
        hds_cfg = self._resolve_hds_inference_config(n_bins_max=n_bins_max)

        return self._test_3d_fine_accuracy_hds(
            use_train_uav=use_train_uav,
            n_samples=hds_cfg["eval_n_samples"],
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode="linear",
            scale_select_mode=hds_cfg["scale_select_mode"],

            ge_prob_mode="ingp",
            ge_top_ratio_rho0=hds_cfg["ge_top_ratio_rho0"],
            ge_topk_seed=None,
            ge_max_anchors_k0=hds_cfg["ge_max_anchors_k0"],
            ge_seed_selection_space="3d",

            pc_score_mode="ingp",
            pc_particles_per_round=hds_cfg["pc_particles_per_round"],
            pc_radius_scale_per_round=hds_cfg["pc_radius_scale_per_round"],
            pc_local_sample_method="sobol_deterministic",
            pc_elite_ratio_rho=hds_cfg["pc_elite_ratio_rho"],
            pc_population_survival_ratio_per_round=hds_cfg["pc_population_survival_ratio_per_round"],
            pc_enable_scale_sampling=hds_cfg["pc_enable_scale_sampling"],
            pc_survive_stand="best",
            pc_move_stand="elite_sum",
            chunk_size=hds_cfg["score_chunk_size"],

            lr_score_mode="ingp",
            lr_enable=True,
            lr_variant=hds_cfg["lr_variant"],
            lr_optimize_scale=hds_cfg["lr_optimize_scale"],
            lr_max_input_modes=hds_cfg["lr_max_input_modes_k2"],
            lr_init_sigma=hds_cfg["lr_init_sigma"],
            lr_popsize=hds_cfg["lr_popsize"],
            lr_iters=hds_cfg["lr_iters"],
            lr_enable_early_stop=True,
            lr_early_stop_patience=hds_cfg["lr_early_stop_patience"],
            lr_enable_competition=False,
            lr_competition_interval=2,
            lr_survival_ratio=0.5,
            lr_min_surviving_modes=8,
            lr_elite_ratio=0.2,
            rerank_per_mode_after_lr=True,
            debug_stage_timing=False,
            eval_thresh_cfg=eval_thresh_cfg,
        )

    def compute_seed_mode_recall_from_dict_res(
        self,
        dict_res,
        eval_thresh_cfg=None,
        k_values=None,
        stage_coord_keys=None,
        print_report=True,
    ):
        if "coords_gt" not in dict_res:
            raise KeyError("dict_res missing required key: coords_gt")
        if stage_coord_keys is None:
            stage_coord_keys = {
                "coarse_retrieval": "coords_grid",
                "seed_mode_init": "coords_mode",
                "seed_mode_final": "coords_evo",
            }

        thresholds = self._resolve_recall_thresholds_from_eval_cfg(dict_res, eval_thresh_cfg=eval_thresh_cfg)
        coords_gt = dict_res["coords_gt"].to(torch.float32)
        stage_order = []
        stage_reports = {}

        for stage_name, coords_key in stage_coord_keys.items():
            coords_pred = dict_res.get(coords_key, None)
            if coords_pred is None:
                continue
            coords_pred = coords_pred.to(torch.float32)
            stage_k_values = (
                self._default_recall_k_values(coords_pred.shape[1])
                if k_values is None
                else tuple(int(k) for k in k_values if int(k) <= int(coords_pred.shape[1]))
            ) or (1,)
            metrics, err_stats = compute_progressive_topk_acc_from_coords(
                coords_pred=coords_pred,
                coords_gt=coords_gt,
                dist_th=thresholds["dist_th"],
                rot_th_deg=thresholds["rot_th"],
                scale_ratio_th=thresholds["scale_ratio_th"],
                k_values=stage_k_values,
            )
            stage_order.append(stage_name)
            stage_reports[stage_name] = {
                "coords_key": coords_key,
                "k_values": [int(k) for k in stage_k_values],
                "progressive_acc_metrics": {
                    criterion: {str(k): float(v) for k, v in criterion_metrics.items()}
                    for criterion, criterion_metrics in metrics["progressive_acc_metrics"].items()
                },
                "progressive_error_metrics": metrics.get("progressive_error_metrics", {}),
                "err_stats": {str(key): float(value) for key, value in err_stats.items()},
            }

        report = {
            "thresholds": thresholds,
            "stage_order": stage_order,
            "stages": stage_reports,
        }
        if print_report:
            self._print_seed_mode_recall_summary(report)
        return report

    def _resolve_recall_thresholds_from_eval_cfg(self, dict_res, eval_thresh_cfg=None):
        eval_thresh_cfg = {} if eval_thresh_cfg is None else dict(eval_thresh_cfg)
        seed_mode_eval_config = dict(dict_res.get("seed_mode_eval_config", {}) or {})
        resolved_cfg = dict(seed_mode_eval_config.get("eval_thresh_cfg_resolved", {}) or {})

        nrc2meter = resolved_cfg.get("nrc2meter", None)
        if nrc2meter is None and hasattr(self.sat_dataset, "halfimg_radius_meter"):
            nrc2meter = float(self.sat_dataset.halfimg_radius_meter) / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
        nrc2meter = None if nrc2meter is None else float(nrc2meter)

        dist_th = eval_thresh_cfg.get("dist_th", None)
        if dist_th is None and "dist_th_meter" in eval_thresh_cfg:
            if nrc2meter is None:
                raise ValueError("dist_th_meter requires sat_dataset meter scale or resolved nrc2meter.")
            dist_th = float(eval_thresh_cfg["dist_th_meter"]) / max(nrc2meter, 1e-8)
        if dist_th is None:
            dist_lambda = float(eval_thresh_cfg.get("dist_lambda", resolved_cfg.get("dist_lambda", 1.1)))
            dist_th = float(self.sat_dataset.halfimg_radius_nrc) * dist_lambda

        return {
            "dist_th": float(dist_th),
            "dist_th_meter": None if nrc2meter is None else float(dist_th) * nrc2meter,
            "nrc2meter": nrc2meter,
            "rot_th": None if eval_thresh_cfg.get("rot_th", resolved_cfg.get("rot_th", None)) is None else float(eval_thresh_cfg.get("rot_th", resolved_cfg.get("rot_th"))),
            "scale_ratio_th": None if eval_thresh_cfg.get("scale_ratio_th", resolved_cfg.get("scale_ratio_th", None)) is None else float(eval_thresh_cfg.get("scale_ratio_th", resolved_cfg.get("scale_ratio_th"))),
        }

    @staticmethod
    def _default_recall_k_values(max_k):
        base_k_values = (1, 5, 10, 16, 32, 64, 128, 256, 512, 1024)
        max_k = max(1, int(max_k))
        return tuple(k for k in base_k_values if k <= max_k) or (1,)

    @staticmethod
    def _sort_recall_metric_keys(metric_keys):
        def _metric_key_to_int(metric_key):
            text = str(metric_key)
            if text.startswith("top") and text.endswith("_acc") and text[3:-4].isdigit():
                return int(text[3:-4])
            return 10**9

        return sorted(metric_keys, key=lambda key: (_metric_key_to_int(key), str(key)))

    @classmethod
    def _print_seed_mode_recall_summary(cls, recall_report):
        stages = recall_report.get("stage_order", [])
        stage_reports = recall_report.get("stages", {})
        if not stages or not stage_reports:
            return

        print("\n" + "=" * 96)
        print("INGP Population Contraction Progressive Recall")
        print("=" * 96)
        thresholds = recall_report.get("thresholds", {})
        dist_msg = f"{float(thresholds['dist_th']):.6f} nrc"
        if thresholds.get("dist_th_meter", None) is not None:
            dist_msg += f" / {float(thresholds['dist_th_meter']):.3f} m"
        print(
            f"Thresholds: dist={dist_msg}, "
            f"rot={thresholds.get('rot_th')}, "
            f"scale_ratio={thresholds.get('scale_ratio_th')}"
        )
        print(f"{'Criterion':<16} {'K':>8} " + " ".join(f"{stage:>18}" for stage in stages))
        print("-" * 96)
        for criterion_key, criterion_label in (
            ("dist_recall", "Dist"),
            ("dist_rot_recall", "Dist+Rot"),
            ("dist_rot_scale_recall", "Dist+Rot+Scale"),
        ):
            metric_keys = set()
            for stage in stages:
                metrics = stage_reports.get(stage, {}).get("progressive_acc_metrics", {}).get(criterion_key, {})
                metric_keys.update(metrics.keys())
            for metric_key in cls._sort_recall_metric_keys(metric_keys):
                values = []
                for stage in stages:
                    metrics = stage_reports.get(stage, {}).get("progressive_acc_metrics", {}).get(criterion_key, {})
                    value = metrics.get(metric_key, None)
                    values.append("nan" if value is None else f"{float(value):.2f}")
                print(f"{criterion_label:<16} {str(metric_key):>8} " + " ".join(f"{value:>18}" for value in values))
        print("=" * 96 + "\n")

    @staticmethod
    def _to_jsonable(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): MinimalStage3INGPInferencer._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MinimalStage3INGPInferencer._to_jsonable(v) for v in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _save_progressive_recall_report(self, recall_report):
        output_root = str(getattr(self.opt, "stage3_analysis_export_root", "") or "").strip()
        if not output_root:
            return None
        os.makedirs(output_root, exist_ok=True)
        path = os.path.join(output_root, "stage3_progressive_recall_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(recall_report), f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[Stage3 minimal] saved progressive recall report: {path}")
        return path

    def test(self, use_train_uav=False):
        print("\n" + "=" * 80)
        print("开始 Stage3 minimal INGP-only 推理")
        print("=" * 80 + "\n")

        self._init_stage3_test_runtime(use_train_uav=use_train_uav)
        self._load_checkpoints_for_inference()
        self._enter_inference_eval_mode()

        eval_thresh_cfg = self._resolve_stage3_eval_thresh_cfg_from_opt()
        results = self.run_seed_mode_inference(use_train_uav=use_train_uav, eval_thresh_cfg=eval_thresh_cfg)
        recall_report = self.compute_seed_mode_recall_from_dict_res(
            results,
            eval_thresh_cfg=eval_thresh_cfg,
            print_report=bool(getattr(self.opt, "stage3_print_progressive_recall", False)),
        )
        results["progressive_recall_report"] = recall_report
        self._save_progressive_recall_report(recall_report)
        print("Stage3 minimal INGP-only inference finished.")
        return results

    _init_stage3_test_runtime = util_init_stage3_test_runtime
    _resolve_stage3_sampling_config = util_resolve_stage3_sampling_config
    _estimate_stage3_grid_shape_from_overlap = util_estimate_stage3_grid_shape_from_overlap
    _compute_metric_from_query_and_points = util_compute_metric_from_query_and_points
    _compute_metric_from_ingp = util_compute_metric_from_ingp
    _get_feats_fm_INGP = util_get_feats_fm_INGP
    _sample_around_candidates = util_sample_around_candidates
    _build_cma_init_seeds = util_build_cma_init_seeds
    _pack_mode_states_for_eval = util_pack_mode_states_for_eval
    _pack_stage_records_for_eval = util_pack_stage_records_for_eval
    _sample_indices = util_sample_indices
    _test_3d_fine_accuracy_hds = util_test_3d_fine_accuracy_hds

    def train(self):
        raise RuntimeError("MinimalStage3INGPInferencer is inference-only and does not support train().")


Stage3INGPInferencer = MinimalStage3INGPInferencer


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage3_yaml",
        default=os.path.join(PROJECT_ROOT, "trainer_depends/configs/stage3_ingp_minimal.yaml"),
    )
    parser.add_argument("--stage2_opts_yaml", default=None)
    parser.add_argument("--use_train_uav", action="store_true")
    args, remaining_argv = parser.parse_known_args(argv)

    if "--p_yaml" not in remaining_argv:
        remaining_argv = ["--p_yaml", args.stage3_yaml] + remaining_argv
    else:
        raise ValueError("Use --stage3_yaml for Stage3 minimal config; --p_yaml is kept internal.")

    if args.stage2_opts_yaml:
        if "--inherit_stage2_yaml" not in remaining_argv:
            remaining_argv.extend(["--inherit_stage2_yaml", args.stage2_opts_yaml])
        if "--load_stage2_ckpt" not in remaining_argv:
            maybe_ckpt = _find_latest_epoch_ckpt(os.path.dirname(os.path.abspath(args.stage2_opts_yaml)))
            if maybe_ckpt:
                remaining_argv.extend(["--load_stage2_ckpt", maybe_ckpt])

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining_argv
        inferencer = MinimalStage3INGPInferencer()
        inferencer.test(use_train_uav=args.use_train_uav)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
