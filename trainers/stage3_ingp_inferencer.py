#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3 INGP-only inference.

This module keeps the Stage 3 localization/evaluation workflow but removes the
old MetricNet/Projector training path. It uses:
- Stage 1 visual encoder + aggregator for query features.
- Stage 2 INGP/grid_mlp for map-coordinate features.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence

import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from trainer_depends.config.parser import (  # noqa: E402
    _expand_selected_scene_config,
    get_parse,
    print_config_summary,
)
from trainers.stage2_INGP import GridHashFitTrainer, _find_latest_epoch_ckpt  # noqa: E402
from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords  # noqa: E402
from trainers.util_stage3_ckpt import (  # noqa: E402
    _apply_inherit_stage2_yaml as util_apply_inherit_stage2_yaml,
    _collect_declared_keys_from_cfg as util_collect_declared_keys_from_cfg,
    _collect_stage3_explicit_keys as util_collect_stage3_explicit_keys,
    _get_stage1_checkpoint_path as util_get_stage1_checkpoint_path,
    _get_stage2_checkpoint_path as util_get_stage2_checkpoint_path,
    _load_yaml_dict as util_load_yaml_dict,
    _merge_section_key_sets as util_merge_section_key_sets,
    _normalize_inherit_stage2_scope as util_normalize_inherit_stage2_scope,
)
from trainers.util_stage3_eval import (  # noqa: E402
    _compute_2d_loc_metrics as util_compute_2d_loc_metrics,
    _compute_neighbors_recall as util_compute_neighbors_recall,
    _get_dir2save as util_get_dir2save,
    _save_pred_pdf_3d as util_save_pred_pdf_3d,
    _test_2d_classification_accuracy as util_test_2d_classification_accuracy,
    _test_2d_sequence_localization_accuracy as util_test_2d_sequence_localization_accuracy,
    _test_3d_classification_accuracy as util_test_3d_classification_accuracy,
)
from trainers.util_stage3_runtime import (  # noqa: E402
    _estimate_stage3_grid_shape_from_overlap as util_estimate_stage3_grid_shape_from_overlap,
    _init_stage3_test_runtime as util_init_stage3_test_runtime,
    _resolve_stage3_sampling_config as util_resolve_stage3_sampling_config,
)
from trainers.util_stage3_score import (  # noqa: E402
    _compute_metric_from_ingp as util_compute_metric_from_ingp,
    _compute_metric_from_query_and_points as util_compute_metric_from_query_and_points,
    _get_feats_fm_INGP as util_get_feats_fm_INGP,
)
from trainers.util_stage3_search import (  # noqa: E402
    _build_cma_init_seeds as util_build_cma_init_seeds,
    _opt_coords_topN as util_opt_coords_topN,
    _pack_mode_states_for_eval as util_pack_mode_states_for_eval,
    _pack_stage_records_for_eval as util_pack_stage_records_for_eval,
    _sample_around_candidates as util_sample_around_candidates,
    _sample_indices as util_sample_indices,
    _test_3d_fine_accuracy_CMA_ES as util_test_3d_fine_accuracy_CMA_ES,
    _test_3d_fine_accuracy_coarse2fine as util_test_3d_fine_accuracy_coarse2fine,
    _test_3d_fine_accuracy_seed_mode_CMA_ES as util_test_3d_fine_accuracy_seed_mode_CMA_ES,
)


class Stage3INGPInferencer(GridHashFitTrainer):
    """INGP-only Stage 3 inference/analysis object."""

    def __init__(self, opt=None):
        should_print_final_config = (opt is None) or not bool(getattr(opt, "_config_summary_printed", False))
        if opt is None:
            opt = get_parse(print_summary=False)
        opt = self._apply_inherit_stage2_yaml(opt)
        opt.scenes_setting = _expand_selected_scene_config(getattr(opt, "scenes_setting", None))
        if should_print_final_config:
            print_config_summary(opt, header="最终生效配置:")

        super().__init__(opt)
        self._setup_inference_modules()

        self.energy_temperature = float(getattr(self.opt, "stage3_energy_temperature", 0.05))
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
        """Freeze all Stage 1/Stage 2 modules and expose a single eval-mode module map."""
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

        self.param2optimize = {}
        self.param2freeze = modules
        print("Stage3 INGP inference modules frozen:")
        print(f"  frozen: {', '.join(self.param2freeze.keys())}")

    @staticmethod
    def _build_stage3_eval_thresh_cfg(
        sat_dataset,
        dist_lambda=1.1 * 0.5,
        rot_th=11.0 * 0.5,
        scale_ratio_th=1.15,
        dist_th_meter=None,
    ):
        eval_thresh_cfg = {
            "dist_lambda": float(dist_lambda),
            "rot_th": None if rot_th is None else float(rot_th),
            "scale_ratio_th": None if scale_ratio_th is None else float(scale_ratio_th),
        }
        if dist_th_meter is not None:
            if sat_dataset is None or not hasattr(sat_dataset, "halfimg_radius_meter") or not hasattr(sat_dataset, "halfimg_radius_nrc"):
                raise ValueError("dist_th_meter requires sat_dataset.halfimg_radius_meter and sat_dataset.halfimg_radius_nrc")
            meter2nrc = float(sat_dataset.halfimg_radius_nrc) / max(float(sat_dataset.halfimg_radius_meter), 1e-8)
            dist_th = float(dist_th_meter) * meter2nrc
            eval_thresh_cfg["dist_th_meter"] = float(dist_th_meter)
            eval_thresh_cfg["dist_th"] = float(dist_th)
            eval_thresh_cfg["dist_lambda"] = float(dist_th) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
        return eval_thresh_cfg

    @staticmethod
    def _resolve_stage3_recall_cfg_path(path):
        path = str(path or "").strip()
        if not path:
            return ""
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    @staticmethod
    def _load_stage3_recall_cfg_yaml(path):
        path_abs = Stage3INGPInferencer._resolve_stage3_recall_cfg_path(path)
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
        cfg_yaml = getattr(
            self.opt,
            "stage3_recall_cfg_yaml",
            "trainer_depends/configs/stage3_recall_thresholds.yaml",
        )
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
            if not isinstance(per_scene, dict):
                raise TypeError(f"Stage3 recall cfg 'per_scene' must be a dict: {cfg_yaml}")
            cfg_name = str(per_scene.get(scene_name, cfg_root.get("default", "cfg01"))).strip()
        else:
            cfg_name = cfg_selector

        if cfg_name not in configs:
            available = ", ".join(sorted(str(k) for k in configs.keys()))
            raise KeyError(
                f"Stage3 recall cfg '{cfg_name}' not found for selector='{cfg_selector}', "
                f"scene='{scene_name}'. Available configs: {available}"
            )

        raw_cfg = configs[cfg_name]
        if not isinstance(raw_cfg, dict):
            raise TypeError(f"Stage3 recall cfg '{cfg_name}' must be a dict, got {type(raw_cfg).__name__}")

        eval_thresh_cfg = self._build_stage3_eval_thresh_cfg(
            sat_dataset=getattr(self, "sat_dataset", None),
            dist_lambda=float(raw_cfg.get("dist_lambda", 1.1 * 0.5)),
            rot_th=raw_cfg.get("rot_th", None),
            scale_ratio_th=raw_cfg.get("scale_ratio_th", raw_cfg.get("scale_th", None)),
            dist_th_meter=raw_cfg.get("dist_th_meter", None),
        )
        if raw_cfg.get("dist_th", None) is not None:
            if self.sat_dataset is None or not hasattr(self.sat_dataset, "halfimg_radius_nrc"):
                raise ValueError("dist_th requires sat_dataset.halfimg_radius_nrc")
            dist_th = float(raw_cfg["dist_th"])
            eval_thresh_cfg["dist_th"] = dist_th
            eval_thresh_cfg["dist_lambda"] = dist_th / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
            if hasattr(self.sat_dataset, "halfimg_radius_meter"):
                nrc2meter = float(self.sat_dataset.halfimg_radius_meter) / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
                eval_thresh_cfg["dist_th_meter"] = dist_th * nrc2meter

        print(
            "[Stage3-RecallCfg] "
            f"yaml={self._resolve_stage3_recall_cfg_path(cfg_yaml)} selector={cfg_selector} "
            f"scene={scene_name} resolved={cfg_name} thresholds={eval_thresh_cfg}"
        )
        return eval_thresh_cfg

    def _load_checkpoints_for_inference(self):
        """Load only Stage 2 INGP and Stage 1 visual checkpoints."""
        print("\n" + "=" * 80)
        print("加载 Stage3 INGP-only 推理 checkpoint")
        print("=" * 80)

        stage2_ckpt_path = self._resolve_stage2_checkpoint_path()
        if not stage2_ckpt_path:
            raise ValueError("未找到Stage 2 checkpoint；请设置 load_stage2_ckpt。")

        modules_stage2 = {"grid": self.grid, "grid_mlp": self.grid_mlp}
        if getattr(self, "hash_lod_aggregator", None) is not None:
            modules_stage2["hash_lod_aggregator"] = self.hash_lod_aggregator

        print(f"\n📦 Stage 2 checkpoint: {stage2_ckpt_path}")
        self._load_checkpoint(stage2_ckpt_path, modules_stage2, mode="test")
        self._stage2_checkpoint_loaded = True

        stage1_ckpt_path = self._get_stage1_checkpoint_path(stage2_ckpt_path)
        if not stage1_ckpt_path:
            raise ValueError("未找到Stage 1 checkpoint；请设置 load_stage1_ckpt 或确保Stage2 opts.yaml记录了该路径。")

        print(f"\n📦 Stage 1 checkpoint: {stage1_ckpt_path}")
        self._load_checkpoint(
            stage1_ckpt_path,
            {"vis_encoder": self.vis_encoder, "vis_aggregator": self.vis_aggregator},
            mode="test",
        )
        self._stage1_checkpoint_loaded = True

        print("\n✅ Stage1 + Stage2 checkpoint加载完成")
        print("=" * 80 + "\n")

    def _resolve_stage2_checkpoint_path(self):
        """Resolve Stage 2 checkpoint from explicit option or a Stage 2 opts.yaml path."""
        stage2_ckpt_path = self._get_stage2_checkpoint_path(stage3_ckpt_path=None)
        if stage2_ckpt_path:
            return stage2_ckpt_path

        p_yaml = str(getattr(self.opt, "p_yaml", "") or "").strip()
        if not p_yaml:
            return None
        p_yaml_abs = p_yaml if os.path.isabs(p_yaml) else os.path.join(PROJECT_ROOT, p_yaml)
        if not os.path.isfile(p_yaml_abs):
            return None

        yaml_name = os.path.basename(p_yaml_abs)
        yaml_dir = os.path.dirname(p_yaml_abs)
        if yaml_name == "opts.yaml":
            latest_ckpt = _find_latest_epoch_ckpt(yaml_dir)
            if latest_ckpt:
                print(f"从Stage2 opts.yaml所在目录自动选择checkpoint: {latest_ckpt}")
                self.opt.load_stage2_ckpt = latest_ckpt
                return latest_ckpt

        return None

    def _enter_inference_eval_mode(self):
        for model in self.param2freeze.values():
            model.eval()

    def compute_seed_mode_recall_from_dict_res(
        self,
        dict_res,
        eval_thresh_cfg=None,
        k_values=None,
        stage_coord_keys=None,
        print_report=True,
    ):
        """Recompute threshold-based recall from seed-mode search outputs."""
        if not isinstance(dict_res, dict):
            raise TypeError("dict_res must be a dict returned by seed-mode search.")
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
            if coords_key not in dict_res or dict_res[coords_key] is None:
                continue
            coords_pred = dict_res[coords_key].to(torch.float32)
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
                "err_stats": {str(key): float(value) for key, value in err_stats.items()},
            }

        recall_report = {
            "thresholds": thresholds,
            "stage_order": stage_order,
            "stages": stage_reports,
        }
        if print_report:
            self._print_seed_mode_recall_summary(recall_report)
        return recall_report

    def _resolve_recall_thresholds_from_eval_cfg(self, dict_res, eval_thresh_cfg=None):
        eval_thresh_cfg = {} if eval_thresh_cfg is None else dict(eval_thresh_cfg)
        seed_mode_eval_config = dict(dict_res.get("seed_mode_eval_config", {}) or {})
        resolved_cfg = dict(seed_mode_eval_config.get("eval_thresh_cfg_resolved", {}) or {})

        nrc2meter = resolved_cfg.get("nrc2meter", None)
        if nrc2meter is None and hasattr(self, "sat_dataset"):
            if hasattr(self.sat_dataset, "halfimg_radius_meter") and hasattr(self.sat_dataset, "halfimg_radius_nrc"):
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

    @staticmethod
    def _print_seed_mode_recall_summary(recall_report):
        stages = recall_report.get("stage_order", [])
        stage_reports = recall_report.get("stages", {})
        if not stages or not stage_reports:
            return

        print("\n" + "=" * 96)
        print("INGP Seed-Mode Recall")
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
            for metric_key in Stage3INGPInferencer._sort_recall_metric_keys(metric_keys):
                values = []
                for stage in stages:
                    metrics = stage_reports.get(stage, {}).get("progressive_acc_metrics", {}).get(criterion_key, {})
                    value = metrics.get(metric_key, None)
                    values.append("nan" if value is None else f"{float(value):.2f}")
                print(f"{criterion_label:<16} {str(metric_key):>8} " + " ".join(f"{value:>18}" for value in values))
        print("=" * 96 + "\n")

    def run_seed_mode_inference(self, use_train_uav=False, eval_thresh_cfg=None):
        """Run the current INGP-only seed-mode localization pipeline."""
        eval_thresh_cfg = eval_thresh_cfg or self._resolve_stage3_eval_thresh_cfg_from_opt()
        stage3_test_overlap = float(getattr(self.opt, "stage3_test_overlap", 0.5))
        grid_info = self._estimate_stage3_grid_shape_from_overlap(stage3_test_overlap)
        n_bins_4d = [
            int(grid_info["grid_rows"]),
            int(grid_info["grid_cols"]),
            int(getattr(self.opt, "stage3_test_n_rot", getattr(self.opt, "stage3_n_rot", self.n_coarse[2]))),
            int(getattr(self.opt, "stage3_test_n_scale", 4)),
        ]
        n_bins_max = max(n_bins_4d[:2])

        dict_res = self._test_3d_fine_accuracy_seed_mode_CMA_ES(
            use_train_uav=use_train_uav,
            n_samples=getattr(self.opt, "stage3_test_n_samples", None),
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode="linear",
            l0_seed_selection_space="3d",
            scale_select_mode=getattr(self.opt, "stage3_scale_select_mode", "log_expectation"),
            l0_prob_mode="ingp",
            l0_ratio=float(getattr(self.opt, "stage3_l0_ratio", 0.1)),
            l0_topN=None,

            stage1_5_refiner_mode="multi_start_es",
            stage1_survive_stand="best",
            stage1_move_stand="elite_sum",

            stage1_mode_input_max=int(getattr(self.opt, "stage3_stage1_mode_input_max", 1024)),
            stage1_alpha=1.0,
            stage1_local_sample_method="sobol_deterministic",
            stage1_prob_mode="ingp",

            stage1_samples_per_round=tuple(getattr(self.opt, "stage3_stage1_samples_per_round", (8, 8))),
            stage1_radius_scale_per_round=tuple(getattr(self.opt, "stage3_stage1_radius_scale_per_round", (1.05, 0.5))),
            stage1_survival_ratio_per_round=tuple(getattr(self.opt, "stage3_stage1_survival_ratio_per_round", (0.25, 0.125))),
            stage1_elite_ratio=float(getattr(self.opt, "stage3_stage1_elite_ratio", 0.125)),
            stage1_enable_scale_sampling=False,
            chunk_size=int(getattr(self.opt, "stage3_chunk_size", 8192 * 4)),

            stage3_enable=True,
            cma_variant=getattr(self.opt, "stage3_cma_variant", "Sep-CMA"),
            cma_prob_mode="ingp",
            cma_optimize_scale=False,
            cma_max_input_mode=int(getattr(self.opt, "stage3_cma_max_input_mode", 5)),
            cma_init_sigma_manual=float(getattr(self.opt, "stage3_cma_init_sigma_manual", 0.125 / n_bins_max)),
            cma_popsize=int(getattr(self.opt, "stage3_cma_popsize", 32)),
            cma_iters=int(getattr(self.opt, "stage3_cma_iters", 12)),
            cma_enable_early_stop=True,
            cma_early_stop_patience=int(getattr(self.opt, "stage3_cma_early_stop_patience", 5)),
            rerank_per_mode_after_stage3=True,

            cma_enable_competition=False,
            cma_competition_interval=2,
            cma_survival_ratio=0.5,
            cma_min_surviving_modes=8,
            cma_elite_ratio=0.2,

            stage4_enable=False,
            stage4_input_stage="latest",
            stage4_prob_mode="ingp",
            stage4_opt_space="linear",
            stage4_topk_input=5,
            stage4_n_steps=150,
            stage4_lr_xy=1e-5,
            stage4_lr_rot=1e-5,
            stage4_lr_scale=1e-7,
            stage4_optimize_scale=False,
            debug_stage_timing=False,
            stage4_verbose=True,
            eval_thresh_cfg=eval_thresh_cfg,
        )
        return dict_res

    def export_analysis_triplets(self, results, use_train_uav=False):
        output_root = getattr(self.opt, "stage3_analysis_export_root", "")
        if not output_root:
            return None

        from trainers.leggacy_stage.util_stage3_export_triplets import export_stage3_retrieval_triplets_from_results

        t_export0 = time.perf_counter()
        export_dir = export_stage3_retrieval_triplets_from_results(
            trainer=self,
            results=results,
            output_root=output_root,
            use_train_uav=use_train_uav,
            apply_rotation=bool(getattr(self.opt, "stage3_analysis_apply_rotation", True)),
            export_batch_size=int(getattr(self.opt, "stage3_analysis_export_batch_size", 32)),
        )
        print(f"[Stage3-Analysis] Exported triplets to: {export_dir} | {time.perf_counter() - t_export0:.3f}s")
        return export_dir

    def test(self, use_train_uav=False):
        print("\n" + "=" * 80)
        print("开始 Stage3 INGP-only 推理")
        print("=" * 80 + "\n")

        self._init_stage3_test_runtime(use_train_uav=use_train_uav)
        self._load_checkpoints_for_inference()
        self._enter_inference_eval_mode()

        eval_thresh_cfg = self._resolve_stage3_eval_thresh_cfg_from_opt()

        results = self.run_seed_mode_inference(
            use_train_uav=use_train_uav,
            eval_thresh_cfg=eval_thresh_cfg,
        )
        self.export_analysis_triplets(results, use_train_uav=use_train_uav)
        print("✅ Stage3 INGP-only 推理完成")
        return results

    _normalize_inherit_stage2_scope = staticmethod(util_normalize_inherit_stage2_scope)
    _apply_inherit_stage2_yaml = staticmethod(util_apply_inherit_stage2_yaml)
    _load_yaml_dict = staticmethod(util_load_yaml_dict)
    _merge_section_key_sets = staticmethod(util_merge_section_key_sets)
    _collect_declared_keys_from_cfg = staticmethod(util_collect_declared_keys_from_cfg)
    _collect_stage3_explicit_keys = staticmethod(util_collect_stage3_explicit_keys)
    _get_stage2_checkpoint_path = util_get_stage2_checkpoint_path
    _get_stage1_checkpoint_path = util_get_stage1_checkpoint_path

    _init_stage3_test_runtime = util_init_stage3_test_runtime
    _resolve_stage3_sampling_config = util_resolve_stage3_sampling_config
    _estimate_stage3_grid_shape_from_overlap = util_estimate_stage3_grid_shape_from_overlap

    _compute_metric_from_query_and_points = util_compute_metric_from_query_and_points
    _compute_metric_from_ingp = util_compute_metric_from_ingp
    _get_feats_fm_INGP = util_get_feats_fm_INGP

    _test_2d_classification_accuracy = util_test_2d_classification_accuracy
    _test_2d_sequence_localization_accuracy = util_test_2d_sequence_localization_accuracy
    _compute_2d_loc_metrics = util_compute_2d_loc_metrics
    _compute_neighbors_recall = util_compute_neighbors_recall
    _get_dir2save = util_get_dir2save
    _save_pred_pdf_3d = util_save_pred_pdf_3d
    _test_3d_classification_accuracy = util_test_3d_classification_accuracy

    _opt_coords_topN = util_opt_coords_topN
    _sample_around_candidates = util_sample_around_candidates
    _build_cma_init_seeds = util_build_cma_init_seeds
    _pack_mode_states_for_eval = util_pack_mode_states_for_eval
    _pack_stage_records_for_eval = util_pack_stage_records_for_eval
    _sample_indices = util_sample_indices
    _test_3d_fine_accuracy_seed_mode_CMA_ES = util_test_3d_fine_accuracy_seed_mode_CMA_ES
    _test_3d_fine_accuracy_coarse2fine = util_test_3d_fine_accuracy_coarse2fine
    _test_3d_fine_accuracy_CMA_ES = util_test_3d_fine_accuracy_CMA_ES

    def train(self):
        raise RuntimeError("Stage3INGPInferencer is inference-only and does not support train().")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p_yaml", default=None)
    parser.add_argument("--use_train_uav", action="store_true")
    parser.add_argument("--load_stage2_ckpt", default=None)
    args, remaining_argv = parser.parse_known_args(argv)

    if args.p_yaml is not None and "--p_yaml" not in remaining_argv:
        remaining_argv = ["--p_yaml", args.p_yaml] + remaining_argv
    if "--p_yaml" not in remaining_argv:
        remaining_argv.extend(["--p_yaml", os.path.join(PROJECT_ROOT, "trainer_depends/configs/stage3_visloc.yaml")])

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining_argv
        inferencer = Stage3INGPInferencer()
        inferencer.test(use_train_uav=args.use_train_uav)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
