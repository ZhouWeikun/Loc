#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3: MetricNet Trainer

训练目标：
- metric_net (距离预测网络)

前置条件：
- Stage 1: vis_encoder + vis_aggregator (冻结)
- Stage 2: grid + grid_mlp (可选冻结/微调)

训练策略：
- 使用UDF监督训练MetricNet预测距离
- 使用Eikonal正则化约束距离场平滑性
"""

import torch
import torch.nn.functional as TF
import tqdm
import time
import sys
import os
import numpy as np
import yaml
from collections.abc import Sequence

# 添加项目根目录到路径，确保直接执行脚本时也能导入仓库内模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from losses.CL_losses_w_weight import pairLoss_multiEdge_logSum
from trainers.util_stage3_loc_manager import Stage3FineLocManager
from trainers.util_stage3_multi_stage_refiner import (
    EvoTorchFinalModeOptimizer,
    GradientTopKOptimizer,
    IterativeSeedCloudRefiner,
    LocalSeedCloudBuilder,
    ModeState,
    PassthroughModeDeduper,
    SeedModeSearchConfig,
    SeedModeSearchPipeline,
    SeedRegionConfig,
    Stage3CMAConfig,
    Stage4GradConfig,
    TopNSeedScreening,
)

from trainers.stage2_INGP import GridHashFitTrainer
# from trainer_depends.base.components import NetworkComponents
import torch.nn.functional as F
from trainer_depends.config.parser import (
    _expand_selected_scene_config,
    get_parse,
    print_config_summary,
)
from trainers.util_core_eval import compute_progressive_topk_acc_from_coords

from trainers.util_stage3_ckpt import (
    _apply_inherit_stage2_yaml as util_apply_inherit_stage2_yaml,
    _collect_declared_keys_from_cfg as util_collect_declared_keys_from_cfg,
    _collect_stage3_explicit_keys as util_collect_stage3_explicit_keys,
    _get_stage1_checkpoint_path as util_get_stage1_checkpoint_path,
    _get_stage2_checkpoint_path as util_get_stage2_checkpoint_path,
    _get_stage3_checkpoint_path as util_get_stage3_checkpoint_path,
    _load_checkpoints_for_test as util_load_checkpoints_for_test,
    _load_loss_fn_temperature_from_ckpt as util_load_loss_fn_temperature_from_ckpt,
    _load_stage2_checkpoint as util_load_stage2_checkpoint,
    _load_yaml_dict as util_load_yaml_dict,
    _merge_section_key_sets as util_merge_section_key_sets,
    _normalize_inherit_stage2_scope as util_normalize_inherit_stage2_scope,
)
from trainers.util_stage3_eval import (
    _compute_2d_loc_metrics as util_compute_2d_loc_metrics,
    _compute_neighbors_recall as util_compute_neighbors_recall,
    _get_dir2save as util_get_dir2save,
    _run_epoch_evaluation as util_run_epoch_evaluation,
    _save_pred_pdf_3d as util_save_pred_pdf_3d,
    _test_2d_classification_accuracy as util_test_2d_classification_accuracy,
    _test_2d_sequence_localization_accuracy as util_test_2d_sequence_localization_accuracy,
    _test_3d_classification_accuracy as util_test_3d_classification_accuracy,
)
from trainers.util_stage3_runtime import (
    _estimate_stage3_grid_shape_from_overlap as util_estimate_stage3_grid_shape_from_overlap,
    _init_stage3_test_runtime as util_init_stage3_test_runtime,
    _init_stage3_train_runtime as util_init_stage3_train_runtime,
    _resolve_stage3_sampling_config as util_resolve_stage3_sampling_config,
)
from trainers.util_stage3_score import (
    _compute_metric_from_ingp as util_compute_metric_from_ingp,
    _compute_metric_from_query_and_points as util_compute_metric_from_query_and_points,
    _get_feats_fm_INGP as util_get_feats_fm_INGP,
)
from trainers.util_stage3_search import (
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
from trainers.util_stage3_vis import (
    _compute_energy_field_local as util_compute_energy_field_local,
    _render_energy_field_local as util_render_energy_field_local,
    analyze_energy_field as util_analyze_energy_field,
    analyze_feat_freq_band as util_analyze_feat_freq_band,
    visualize_energy_field_local as util_visualize_energy_field_local,
    visualize_energy_of_coords as util_visualize_energy_of_coords,
)
class MetricNetTrainer(GridHashFitTrainer):
    """
    Stage 3: MetricNet Trainer

    继承自GridHashFitTrainer，在其基础上添加MetricNet
    """

    def __init__(self, opt=None):
        """初始化Stage 3 Trainer"""
        should_print_final_config = (opt is None) or not bool(getattr(opt, '_config_summary_printed', False))
        if opt is None:
            opt = get_parse(print_summary=False)
        opt = self._apply_inherit_stage2_yaml(opt)
        opt.scenes_setting = _expand_selected_scene_config(getattr(opt, 'scenes_setting', None))
        if should_print_final_config:
            print_config_summary(opt, header="最终生效配置:")

        # 调用父类初始化（会初始化vis_encoder, grid等）
        super().__init__(opt)

        # 加载Stage 2的Grid权重（如果指定）
        if self.opt.load_stage2_ckpt:
            self._load_stage2_checkpoint()

        # 初始化MetricNet
        self._init_projector()
        self.energy_temperature = 0.05
        self.loss_fn_state = None
        self.loss_fn_beta = None

        # 重新设置可训练参数
        self._setup_trainable_params_stage3()

        # 这些运行时采样配置依赖 sat_dataset，延后到 _init_stage3_*_runtime 中解析
        self.n_coarse = None
        self.n_fine_per_coarse = None
        self.gs_sigma_nrc = None
        self.gs_sigma_radrot = None
        self.gs_sigma_logscale = None
        self.stage3_overlap = None

    @staticmethod
    def _ensure_param_sequence(value, cast_fn):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(cast_fn(v) for v in value)
        return (cast_fn(value),)

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
        return os.path.join(project_root, path)

    @staticmethod
    def _load_stage3_recall_cfg_yaml(path):
        path_abs = MetricNetTrainer._resolve_stage3_recall_cfg_path(path)
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

        dist_lambda = float(raw_cfg.get("dist_lambda", 1.1 * 0.5))
        rot_th = raw_cfg.get("rot_th", None)
        scale_ratio_th = raw_cfg.get("scale_ratio_th", raw_cfg.get("scale_th", None))
        dist_th_meter = raw_cfg.get("dist_th_meter", None)
        eval_thresh_cfg = self._build_stage3_eval_thresh_cfg(
            sat_dataset=getattr(self, "sat_dataset", None),
            dist_lambda=dist_lambda,
            rot_th=rot_th,
            scale_ratio_th=scale_ratio_th,
            dist_th_meter=dist_th_meter,
        )

        if raw_cfg.get("dist_th", None) is not None:
            if self.sat_dataset is None or not hasattr(self.sat_dataset, "halfimg_radius_nrc"):
                raise ValueError("dist_th requires sat_dataset.halfimg_radius_nrc")
            dist_th = float(raw_cfg["dist_th"])
            eval_thresh_cfg["dist_th"] = dist_th
            eval_thresh_cfg["dist_lambda"] = dist_th / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
            if hasattr(self.sat_dataset, "halfimg_radius_meter"):
                nrc2meter = float(self.sat_dataset.halfimg_radius_meter) / max(
                    float(self.sat_dataset.halfimg_radius_nrc),
                    1e-8,
                )
                eval_thresh_cfg["dist_th_meter"] = dist_th * nrc2meter

        cfg_yaml_abs = self._resolve_stage3_recall_cfg_path(cfg_yaml)
        print(
            "[Stage3-RecallCfg] "
            f"yaml={cfg_yaml_abs} selector={cfg_selector} scene={scene_name} "
            f"resolved={cfg_name} thresholds={eval_thresh_cfg}"
        )
        return eval_thresh_cfg

    def _resolve_recall_thresholds_from_cfg(self, dict_res, eval_thresh_cfg=None):
        eval_thresh_cfg = {} if eval_thresh_cfg is None else dict(eval_thresh_cfg)
        seed_mode_eval_config = dict(dict_res.get("seed_mode_eval_config", {}) or {})
        resolved_cfg = dict(seed_mode_eval_config.get("eval_thresh_cfg_resolved", {}) or {})

        nrc2meter = resolved_cfg.get("nrc2meter", None)
        if nrc2meter is None:
            sat_dataset = getattr(self, "sat_dataset", None)
            if (
                sat_dataset is not None
                and hasattr(sat_dataset, "halfimg_radius_meter")
                and hasattr(sat_dataset, "halfimg_radius_nrc")
            ):
                nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(
                    float(sat_dataset.halfimg_radius_nrc),
                    1e-8,
                )
        nrc2meter = None if nrc2meter is None else float(nrc2meter)

        dist_th = eval_thresh_cfg.get("dist_th", None)
        if dist_th is None and "dist_th_meter" in eval_thresh_cfg:
            if nrc2meter is None:
                raise ValueError("dist_th_meter requires sat_dataset meter scale or resolved nrc2meter in dict_res.")
            dist_th = float(eval_thresh_cfg["dist_th_meter"]) / max(nrc2meter, 1e-8)

        if dist_th is None:
            dist_lambda = float(eval_thresh_cfg.get("dist_lambda", resolved_cfg.get("dist_lambda", 1.1)))
            sat_dataset = getattr(self, "sat_dataset", None)
            if sat_dataset is not None and hasattr(sat_dataset, "halfimg_radius_nrc"):
                halfimg_radius_nrc = float(sat_dataset.halfimg_radius_nrc)
            elif "dist_th" in resolved_cfg and "dist_lambda" in resolved_cfg:
                halfimg_radius_nrc = float(resolved_cfg["dist_th"]) / max(float(resolved_cfg["dist_lambda"]), 1e-8)
            else:
                raise ValueError(
                    "dist_lambda requires sat_dataset.halfimg_radius_nrc or resolved dist_th/dist_lambda in dict_res."
                )
            dist_th = halfimg_radius_nrc * dist_lambda

        rot_th = eval_thresh_cfg.get("rot_th", resolved_cfg.get("rot_th", None))
        scale_ratio_th = eval_thresh_cfg.get("scale_ratio_th", resolved_cfg.get("scale_ratio_th", None))

        dist_th = float(dist_th)
        return {
            "dist_th": dist_th,
            "dist_th_meter": None if nrc2meter is None else dist_th * nrc2meter,
            "nrc2meter": nrc2meter,
            "rot_th": None if rot_th is None else float(rot_th),
            "scale_ratio_th": None if scale_ratio_th is None else float(scale_ratio_th),
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
        print("Seed-Mode Recall From dict_res")
        print("=" * 96)
        thresholds = recall_report.get("thresholds", {})
        dist_th_meter = thresholds.get("dist_th_meter", None)
        dist_msg = f"{float(thresholds['dist_th']):.6f} nrc"
        if dist_th_meter is not None:
            dist_msg += f" / {float(dist_th_meter):.3f} m"
        print(
            f"Thresholds: dist={dist_msg}, "
            f"rot={thresholds.get('rot_th')}, "
            f"scale_ratio={thresholds.get('scale_ratio_th')}"
        )
        print(f"{'Criterion':<16} {'K':>8} " + " ".join(f"{stage:>18}" for stage in stages))
        print("-" * 96)

        criteria = (
            ("dist_recall", "Dist"),
            ("dist_rot_recall", "Dist+Rot"),
            ("dist_rot_scale_recall", "Dist+Rot+Scale"),
        )
        for criterion_key, criterion_label in criteria:
            metric_keys = set()
            for stage in stages:
                metrics = (
                    stage_reports.get(stage, {})
                    .get("progressive_acc_metrics", {})
                    .get(criterion_key, {})
                )
                metric_keys.update(metrics.keys())
            for metric_key in MetricNetTrainer._sort_recall_metric_keys(metric_keys):
                values = []
                for stage in stages:
                    metrics = (
                        stage_reports.get(stage, {})
                        .get("progressive_acc_metrics", {})
                        .get(criterion_key, {})
                    )
                    value = metrics.get(metric_key, None)
                    values.append("nan" if value is None else f"{float(value):.2f}")
                print(f"{criterion_label:<16} {str(metric_key):>8} " + " ".join(f"{value:>18}" for value in values))
        print("=" * 96 + "\n")

    def compute_seed_mode_recall_from_dict_res(
        self,
        dict_res,
        eval_thresh_cfg=None,
        k_values=None,
        stage_coord_keys=None,
        print_report=True,
    ):
        """
        Recompute threshold-based progressive recall from _test_3d_fine_accuracy_seed_mode_CMA_ES output.
        """
        if not isinstance(dict_res, dict):
            raise TypeError("dict_res must be a dict returned by _test_3d_fine_accuracy_seed_mode_CMA_ES.")
        if "coords_gt" not in dict_res:
            raise KeyError("dict_res missing required key: coords_gt")

        if stage_coord_keys is None:
            stage_coord_keys = {
                "coarse_retrieval": "coords_grid",
                "seed_mode_init": "coords_mode",
                "seed_mode_final": "coords_evo",
            }

        coords_gt = dict_res["coords_gt"].to(torch.float32)
        thresholds = self._resolve_recall_thresholds_from_cfg(dict_res, eval_thresh_cfg=eval_thresh_cfg)

        stage_order = []
        stage_reports = {}
        for stage_name, coords_key in stage_coord_keys.items():
            if coords_key not in dict_res or dict_res[coords_key] is None:
                continue
            coords_pred = dict_res[coords_key].to(torch.float32)
            if coords_pred.ndim != 3 or coords_pred.shape[-1] != 4:
                raise ValueError(f"dict_res[{coords_key!r}] must be [B, K, 4], got {tuple(coords_pred.shape)}")

            stage_k_values = (
                self._default_recall_k_values(coords_pred.shape[1])
                if k_values is None
                else tuple(int(k) for k in k_values if int(k) <= int(coords_pred.shape[1]))
            )
            if not stage_k_values:
                stage_k_values = (1,)

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
                "acc_metrics": {
                    str(key): float(value)
                    for key, value in metrics.items()
                    if str(key).startswith("top") and str(key).endswith("_acc")
                },
                "progressive_acc_metrics": {
                    criterion: {str(k): float(v) for k, v in criterion_metrics.items()}
                    for criterion, criterion_metrics in metrics["progressive_acc_metrics"].items()
                },
                "legacy_acc_metrics_source": str(metrics["legacy_acc_metrics_source"]),
                "progressive_acc_metric_sources": dict(metrics["progressive_acc_metric_sources"]),
                "err_stats": {str(key): float(value) for key, value in err_stats.items()},
            }

        baseline_stage = "coarse_retrieval" if "coarse_retrieval" in stage_reports else (stage_order[0] if stage_order else None)
        deltas_vs_baseline = {}
        if baseline_stage is not None:
            criteria = ("dist_recall", "dist_rot_recall", "dist_rot_scale_recall")
            for criterion in criteria:
                baseline_metrics = stage_reports[baseline_stage]["progressive_acc_metrics"].get(criterion, {})
                criterion_delta = {}
                for stage_name in stage_order:
                    stage_metrics = stage_reports[stage_name]["progressive_acc_metrics"].get(criterion, {})
                    common_keys = set(baseline_metrics.keys()) & set(stage_metrics.keys())
                    criterion_delta[stage_name] = {
                        str(metric_key): float(stage_metrics[metric_key]) - float(baseline_metrics[metric_key])
                        for metric_key in self._sort_recall_metric_keys(common_keys)
                    }
                deltas_vs_baseline[criterion] = criterion_delta

        recall_report = {
            "thresholds": thresholds,
            "stage_order": stage_order,
            "stages": stage_reports,
            "baseline_stage": baseline_stage,
            "deltas_vs_baseline": deltas_vs_baseline,
        }

        if print_report:
            self._print_seed_mode_recall_summary(recall_report)
        return recall_report

    def _init_projector(self):
        from models.projector_mlp_sample import Projector
        """初始化Projector"""
        print("\n" + "=" * 80)
        print("初始化 Projector ")
        print("=" * 80)

        # self.projector = Projector(
        #     input_dim = 1024,
        #     hidden_dims = [256,256],
        #     output_dim = 128
        # ).to(self.device)

        from models.projector_mlp_res import AdvancedProjector
        self.projector = AdvancedProjector(
            input_dim=1024,
            hidden_dim=1024,  # 保持宽通道
            expansion_factor=2,  # 内部升维到 2048
            num_res_blocks=3,  # 深度镖人
            output_dim=128,
            use_spectral_norm=True,  # 关键：开启谱归一化matplotlib.use("TkAgg")
        ).to(self.device)

        print("✅ Projector 初始化完成")
        print("=" * 80 + "\n")

    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        for param in self.grid.parameters():
            param.requires_grad = False
        for param in self.grid_mlp.parameters():
            param.requires_grad = False

        self.param2optimize = {
            'projector': self.projector,
        }

        # 始终冻结Stage 1组件和Grid
        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
            'vis_aggregator': self.vis_aggregator,
            'grid': self.grid,
            'grid_mlp': self.grid_mlp,
        }

        # 动态生成参数配置信息
        freeze_grid_status = "freeze_grid=True" if getattr(self.opt, 'freeze_grid', True) else "freeze_grid=False"
        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys())

        print(f"参数配置 ({freeze_grid_status}):")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")

    def _init_loss_modules(self):
        from losses.CL_losses_w_weight import (
            pairLoss_multiEdge_logSum,
        )

        self.active_loss_type = str(
            getattr(self.opt, "loss_type", "pairloss_multiedge_logsum")
        ).lower()
        loss_registry = {
            "pairloss_multiedge_logsum": {
                "factory": lambda: pairLoss_multiEdge_logSum(
                    beta=10.0,
                    margin=0.1,
                    learnable_beta=True,
                ),
                "input_mode": "weights",
                "output_mode": "pair",
                "module_name": "loss_fn",
            }
        }
        if self.active_loss_type not in loss_registry:
            supported = ", ".join(sorted(loss_registry.keys()))
            raise ValueError(
                f"Unsupported loss_type: {self.active_loss_type}. "
                f"Supported: {supported}"
            )

        loss_spec = loss_registry[self.active_loss_type]
        self.active_loss_input_mode = loss_spec["input_mode"]
        self.active_loss_output_mode = loss_spec["output_mode"]
        self.active_loss_module_name = loss_spec["module_name"]
        self.active_loss_module = loss_spec["factory"]().to(self.device)
        self.loss_fn = self.active_loss_module
        self._register_active_loss_module()

    def _register_active_loss_module(self):
        self.param2optimize.pop("loss_fn", None)
        has_trainable_params = any(
            param.requires_grad for param in self.active_loss_module.parameters()
        )
        if has_trainable_params:
            self.param2optimize[self.active_loss_module_name] = self.active_loss_module
        print(
            f"Loss配置: loss_type={self.active_loss_type}, "
            f"input_mode={self.active_loss_input_mode}, "
            f"output_mode={self.active_loss_output_mode}, "
            f"register_trainable={has_trainable_params}"
        )


    _normalize_inherit_stage2_scope = staticmethod(util_normalize_inherit_stage2_scope)
    _apply_inherit_stage2_yaml = staticmethod(util_apply_inherit_stage2_yaml)
    _load_yaml_dict = staticmethod(util_load_yaml_dict)
    _merge_section_key_sets = staticmethod(util_merge_section_key_sets)
    _collect_declared_keys_from_cfg = staticmethod(util_collect_declared_keys_from_cfg)
    _collect_stage3_explicit_keys = staticmethod(util_collect_stage3_explicit_keys)

    _load_checkpoints_for_test = util_load_checkpoints_for_test
    _load_loss_fn_temperature_from_ckpt = util_load_loss_fn_temperature_from_ckpt
    _load_stage2_checkpoint = util_load_stage2_checkpoint
    _get_stage3_checkpoint_path = util_get_stage3_checkpoint_path
    _get_stage2_checkpoint_path = util_get_stage2_checkpoint_path
    _get_stage1_checkpoint_path = util_get_stage1_checkpoint_path

    _init_stage3_test_runtime = util_init_stage3_test_runtime
    _init_stage3_train_runtime = util_init_stage3_train_runtime
    _resolve_stage3_sampling_config = util_resolve_stage3_sampling_config
    _estimate_stage3_grid_shape_from_overlap = util_estimate_stage3_grid_shape_from_overlap

    _compute_metric_from_query_and_points = util_compute_metric_from_query_and_points
    _compute_metric_from_ingp = util_compute_metric_from_ingp
    _get_feats_fm_INGP = util_get_feats_fm_INGP

    _run_epoch_evaluation = util_run_epoch_evaluation
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

    visualize_energy_of_coords = util_visualize_energy_of_coords
    _compute_energy_field_local = util_compute_energy_field_local
    _render_energy_field_local = util_render_energy_field_local
    visualize_energy_field_local = util_visualize_energy_field_local
    analyze_feat_freq_band = util_analyze_feat_freq_band
    analyze_energy_field = util_analyze_energy_field

    def _prepare_energy_field_sat_background(self, query_id=20, use_train_uav=False, local_zoom_wh=None, satimg_id=0):
        """Crop a satellite background patch aligned with analyze_energy_field bounds."""
        sat_dataset = self.sat_dataset

        global_nr_min = float(sat_dataset.nr2sample_min)
        global_nr_max = float(sat_dataset.nr2sample_max)
        global_nc_min = float(sat_dataset.nc2sample_min)
        global_nc_max = float(sat_dataset.nc2sample_max)

        if local_zoom_wh is None:
            start_nr, end_nr = global_nr_min, global_nr_max
            start_nc, end_nc = global_nc_min, global_nc_max
        else:
            coords_attr = "uav_coords_4d_torch_train" if use_train_uav else "uav_coords_4d_torch_test"
            if not hasattr(self.uav_dataset_train if use_train_uav else self.uav_dataset_test, coords_attr):
                dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
                _, coord_q = dataset[int(query_id)]
            else:
                dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
                coord_q = getattr(dataset, coords_attr)[int(query_id)]

            zoom_nr, zoom_nc = local_zoom_wh
            half_nr = (global_nr_max - global_nr_min) * float(zoom_nr) / 2
            half_nc = (global_nc_max - global_nc_min) * float(zoom_nc) / 2
            center_nr = float(coord_q[0].item())
            center_nc = float(coord_q[1].item())

            start_nr = max(global_nr_min, center_nr - half_nr)
            end_nr = min(global_nr_max, center_nr + half_nr)
            start_nc = max(global_nc_min, center_nc - half_nc)
            end_nc = min(global_nc_max, center_nc + half_nc)

        try:
            sat_patch = sat_dataset.crop_rect_satimg(
                nrc_topleft=(start_nr, start_nc),
                nrc_rightbottom=(end_nr, end_nc),
                type="tensor",
                satimg_id=satimg_id,
            )
        except TypeError:
            sat_patch = sat_dataset.crop_rect_satimg(
                nrc_topleft=(start_nr, start_nc),
                nrc_rightbottom=(end_nr, end_nc),
                type="tensor",
            )
        return sat_dataset.denormalize_img(sat_patch)

    @staticmethod
    def _estimate_energy_field_grid_from_satimg(sat_img, long_side=192, min_side=32):
        """Choose an NR/NC grid resolution that follows a satellite patch aspect ratio."""
        sat_shape = sat_img.shape
        if len(sat_shape) == 3 and sat_shape[0] in {1, 3, 4}:
            sat_h, sat_w = int(sat_shape[1]), int(sat_shape[2])
        else:
            sat_h, sat_w = int(sat_shape[0]), int(sat_shape[1])

        long_side = int(long_side)
        min_side = int(min_side)
        if sat_h >= sat_w:
            n_nr = long_side
            n_nc = max(min_side, int(round(long_side * sat_w / max(sat_h, 1))))
        else:
            n_nc = long_side
            n_nr = max(min_side, int(round(long_side * sat_h / max(sat_w, 1))))
        return n_nr, n_nc



    def test(self, use_train_uav=False):
        """
        Stage 3测试函数
        """
        print("\n" + "🧪" * 40)
        print("开始 Stage 3 测试: Projector")
        print("🧪" * 40 + "\n")
        opt = self.opt

        # 1. 初始化数据集与运行时上下文
        self._init_stage3_test_runtime(use_train_uav=use_train_uav)

        # 2. 加载checkpoint（自适应）
        self._load_checkpoints_for_test()

        # 3. 设置为评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 运行3D分类测试 (NR, NC, Rot)
        eval_3d_classify = False
        if eval_3d_classify:
            results_3d = self._test_3d_classification_accuracy(
                n_samples=256,
                use_train_uav=use_train_uav,
                temperature=self.energy_temperature,
                save_pred_pdf=False,
            )

        # 可视化评估
        # self.analyze_feat_freq_band(vis=False)
        eval_vis= False
        if eval_vis:
            zoom=(0.9,0.9)
            query_id=20
            sat_patch_np = self._prepare_energy_field_sat_background(
                query_id=query_id,
                use_train_uav=use_train_uav,
                local_zoom_wh=zoom,
            )
            n_nr, n_nc = self._estimate_energy_field_grid_from_satimg(
                sat_patch_np,
                long_side=192,
                min_side=32,
            )
            self.analyze_energy_field(
                n_nr=n_nr,
                n_nc=n_nc,
                use_train_uav=use_train_uav,
                query_id=query_id,
                plot_mode='ingp',
                local_zoom_wh=zoom,
                vis=True, # 必须开，才会保存 HTML 曲面图。但它不是等比例地图图的设置。
                surface_plot_setting={
                    "colorscale": "Magma",# "Viridis",RdBu_r,
                    "show_axis_info": False,
                    "clean_scene": True,
                    "width": 1000,
                    "height": int(1000 * n_nr / n_nc),
                    "visencoder_show_axis_info": False,
                    "z_aspect": 0.45,
                },
                #在 vis=True 的默认输出里使用。它不是等比例地图图的设置。
                # plot_contour_setting={
                    # "field_name": "energy_ingp",
                    # "save_path": "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/ours_ckpt_best/zurich_infonce_id{query_id}_zoom{zoom[0]:.2f}.html",
                    # "background_img": sat_img,
                    # "background_extent": (nc_min, nc_max, nr_max, nr_min),
                # },
                #开启等比例 raw NR/NC 坐标图
                # 控制“画哪一个 field、保存到哪里、是否叠底图”。
                render_map_contour=True,
                map_contour_setting={
                    "field_name": "energy_ingp",
                    "save_path": f"/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/ours_ckpt_best/zurich_infonce_id{query_id}_zoom{zoom[0]:.2f}.png",
                    "background_img": sat_patch_np,
                },
                #  控制“怎么画”。比如热力图、等高线、颜色、透明度、GT marker、dpi。
                map_plot_setting={
                    "draw_heatmap": True,
                    "draw_contour": False,
                    "cmap": "magma", #viridis，coolwarm，magma
                    "heatmap_alpha": 0.55,
                    "n_fill_levels": 64,
                    "n_line_levels": 32,
                    "contour_alpha": 0.7,
                    "show_gt_marker": True,
                    "dpi": 300,
                    "transparent": False,
                },
            )

        #设置评估相关
        eval_thresh_cfg = self._resolve_stage3_eval_thresh_cfg_from_opt()
        stage3_test_overlap = 0.5
        grid_info = self._estimate_stage3_grid_shape_from_overlap(stage3_test_overlap)
        n_bins_4d = [
            int(grid_info["grid_rows"]),
            int(grid_info["grid_cols"]),
            int(getattr(self.opt, "stage3_test_n_rot", getattr(self.opt, "stage3_n_rot", self.n_coarse[2]))),
            int(getattr(self.opt, "stage3_test_n_scale", 4)),
        ]

        #吸引盆评估
        stage3_basin_enable = getattr(self.opt, "stage3_basin_enable", True)
        if isinstance(stage3_basin_enable, str):
            stage3_basin_enable = stage3_basin_enable.strip().lower() in {"1", "true", "yes", "y"}
        if bool(stage3_basin_enable):
            from trainers.util_stage3_basin_analyzer import Stage3BasinAnalyzer

            def _as_bool(value):
                if isinstance(value, str):
                    return value.strip().lower() not in {"0", "false", "no", "n", "none", "null", ""}
                return bool(value)

            def _parse_optional_int_list(value):
                if value is None or value == "":
                    return None
                if isinstance(value, str):
                    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
                    return [int(p) for p in parts]
                if isinstance(value, Sequence):
                    return [int(p) for p in value]
                return [int(value)]

            stage3_analysis_export_root = getattr(self.opt, "stage3_analysis_export_root", "")
            basin_output_root = getattr(self.opt, "stage3_basin_output_root", "")
            import os
            if not basin_output_root and stage3_analysis_export_root:
                basin_output_root = os.path.join(stage3_analysis_export_root, "stage3_basin")

            stage3_basin_sample_radius_rc = 500/self.sat_dataset.halfimg_radius_meter*self.sat_dataset.halfimg_radius_nrc
            t_basin0 = time.perf_counter()
            basin_analyzer = Stage3BasinAnalyzer(
                trainer=self,
                eval_thresh_cfg=eval_thresh_cfg,
                sample_radius_rc=(stage3_basin_sample_radius_rc,stage3_basin_sample_radius_rc),#getattr(self.opt, "stage3_basin_sample_radius_rc", (stage3_basin_sample_radius_rc,stage3_basin_sample_radius_rc)),
                sample_radius_rot_deg=float(getattr(self.opt, "stage3_basin_sample_radius_rot_deg", 0.0)),
                sample_radius_scale_ratio=float(getattr(self.opt, "stage3_basin_sample_radius_scale_ratio", 0.0)),
                fix_rot=_as_bool(getattr(self.opt, "stage3_basin_fix_rot", True)),
                fix_scale=_as_bool(getattr(self.opt, "stage3_basin_fix_scale", True)),
                seed=getattr(self.opt, "stage3_basin_seed", None),
                chunk_size=int(getattr(self.opt, "stage3_basin_chunk_size", 8192)),
                temperature=self.energy_temperature,
            )
            basin_res = basin_analyzer.run(
                n_samples=getattr(self.opt, "stage3_basin_n_samples", 32),
                use_train_uav=_as_bool(getattr(self.opt, "stage3_basin_use_train_uav", use_train_uav)),
                query_ids=_parse_optional_int_list(getattr(self.opt, "stage3_basin_query_ids", "")),
                query_batch_size=int(getattr(self.opt, "stage3_basin_query_batch_size", 8)),
                shuffle=_as_bool(getattr(self.opt, "stage3_basin_shuffle", False)),
                num_particles=int(getattr(self.opt, "stage3_basin_num_particles", 32)),
                cma_sigma0=stage3_basin_sample_radius_rc,#float(getattr(self.opt, "stage3_basin_cma_sigma0", stage3_basin_sample_radius_rc )),
                cma_iters=int(getattr(self.opt, "stage3_basin_cma_iters", 11)),
                cma_variant=getattr(self.opt, "stage3_basin_cma_variant", "Sep-CMA"),
                cma_prob_mode=getattr(self.opt, "stage3_basin_cma_prob_mode", "product"),
                cma_enable_early_stop=_as_bool(getattr(self.opt, "stage3_basin_cma_enable_early_stop", True)),
                cma_early_stop_patience=int(getattr(self.opt, "stage3_basin_cma_early_stop_patience", 5)),
                optimizer_backend=getattr(self.opt, "stage3_basin_optimizer_backend", "cma_es"),
                stage1_5_iters=getattr(self.opt, "stage3_basin_stage1_5_iters", None),
                stage1_5_elite_ratio=float(getattr(self.opt, "stage3_basin_stage1_5_elite_ratio", 0.125)),
                stage1_5_radius_decay=float(getattr(self.opt, "stage3_basin_stage1_5_radius_decay", 1.0)),
                stage1_5_move_stand=getattr(self.opt, "stage3_basin_stage1_5_move_stand", "elite_sum"),
                save_particles=_as_bool(getattr(self.opt, "stage3_basin_save_particles", False)),
                output_dir=basin_output_root,
                progress=_as_bool(getattr(self.opt, "stage3_basin_progress", True)),
            )
            basin_summary = basin_res["summary"]
            print(
                "[Stage3-Basin] "
                f"backend={basin_res['config'].get('optimizer_backend', 'cma_es')}\n"
                f"queries={basin_summary['n_queries']} particles={basin_summary['total_particles']}\n"
                f"success_particles={basin_summary['total_success']} "
                f"particle_rate={basin_summary['overall_success_rate'] * 100.0:.3f}%\n"
                f"query_success={basin_summary['successful_queries']}/{basin_summary['n_queries']} "
                f"query_rate={basin_summary['query_localization_success_rate'] * 100.0:.3f}%\n"
                f"mean_query_particle_rate={basin_summary['mean_query_success_rate'] * 100.0:.3f}%\n"
                f"json={basin_summary.get('save_path', None)}\n"
                f"txt={basin_summary.get('report_path', None)} | "
                f"{time.perf_counter() - t_basin0:.3f}s"
            )

        #开始优化迭代评估
        use_seed_mode_pipeline = bool(getattr(self.opt, "use_seed_mode_pipeline", True))
        if use_seed_mode_pipeline:
            # n_bins_4d = [
            #     int(self.n_coarse[0])/1.5,
            #     int(self.n_coarse[1])/1.5,
            #     int(getattr(self.opt, "stage3_test_n_rot", getattr(self.opt, "stage3_n_rot", self.n_coarse[2]))),
            #     int(getattr(self.opt, "stage3_test_n_scale", 4)),
            # ]
            n_bins_max = max(n_bins_4d[:2])
            dict_res = self._test_3d_fine_accuracy_seed_mode_CMA_ES(
                use_train_uav=False,
                n_samples=None,
                n_bins_4d=n_bins_4d,
                n_bins_scale_mode="linear",
                scale_select_mode='log_expectation',
                l0_seed_selection_space='3d',
                l0_prob_mode='ingp',

                stage1_5_refiner_mode='multi_start_es',
                stage1_survive_stand='best',
                stage1_move_stand='elite_sum',
                l0_topN=None, #org=1024
                l0_ratio=0.1,
                stage1_mode_input_max=1024,
                stage1_alpha=1.0,
                stage1_local_sample_method='sobol_deterministic',
                stage1_prob_mode='ingp',
                stage1_samples_per_round=(8,8),
                stage1_radius_scale_per_round=(1.05,0.5),
                stage1_survival_ratio_per_round=(0.25,0.125),#序列输入，每轮迭代后剩下的候选mode
                stage1_elite_ratio=0.125, #取ceil(stage1_samples_per_round * stage1_elite_ratio) 个 elite 样本来生成 center_raw
                stage1_enable_scale_sampling=False,
                chunk_size=8192*4,

                stage3_enable=True,
                cma_variant='Sep-CMA',
                cma_prob_mode='ingp',
                cma_max_input_mode=5,
                cma_init_sigma_manual=0.125/n_bins_max, #org=0.75/n_bins_max
                cma_popsize=32, #org=64
                cma_iters=12,
                cma_enable_early_stop=True,
                cma_early_stop_patience=5,
                cma_optimize_scale=False,
                rerank_per_mode_after_stage3=True,

                cma_enable_competition=False,
                cma_competition_interval=2,
                cma_survival_ratio=0.5,
                cma_min_surviving_modes=8,
                cma_elite_ratio=0.2,

                stage4_enable=False,
                stage4_input_stage='latest',
                stage4_prob_mode = "ingp",
                stage4_opt_space='linear',
                stage4_topk_input=5,
                stage4_n_steps=150,
                stage4_lr_xy=1e-5,#1e-5 -> 5e-5,worse ->1./32/40,wrose->1e-6,没变化
                stage4_lr_rot=1e-5,#5e-6->1e-5->1.*0.05/36
                stage4_lr_scale=1e-7,
                stage4_optimize_scale=False,
                debug_stage_timing=False,
                stage4_verbose=True,

                eval_thresh_cfg=eval_thresh_cfg,
            )
            # Example: recompute threshold-based recall from dict_res with custom thresholds.
            # recall_report = self.compute_seed_mode_recall_from_dict_res(
            #     dict_res,
            #     eval_thresh_cfg={"dist_th_meter": 100, "rot_th": 5.5, "scale_ratio_th": 1.15},
            #     k_values=(1, 5, 10, 16, 32, 64, 128),
            #     print_report=True,
            # )

            stage3_analysis_export_root = getattr(self.opt, "stage3_analysis_export_root", "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis")
            if stage3_analysis_export_root:
                from scripts.analysis.util_stage3_export_triplets import (
                    export_stage3_retrieval_triplets_from_results,
                )
                t_export0 = time.perf_counter()
                export_dir = export_stage3_retrieval_triplets_from_results(
                    trainer=self,
                    results=dict_res,
                    output_root=stage3_analysis_export_root,
                    use_train_uav=use_train_uav,
                    apply_rotation=bool(getattr(self.opt, "stage3_analysis_apply_rotation", True)),
                    export_batch_size=int(getattr(self.opt, "stage3_analysis_export_batch_size", 32)),
                )
                print(
                    f"[Stage3-Analysis] Exported triplets to: {export_dir} | "
                    f"{time.perf_counter() - t_export0:.3f}s"
                )
        else:
            dict_res = self._test_3d_fine_accuracy_CMA_ES(
                n_samples=256,
                use_train_uav=use_train_uav,
                temperature=self.energy_temperature,
                save_pred_pdf=False,
                enable_filter=False,
                # n_bins_4d=[int(40 / 1.5), int(30 / 1.5), int(36 / 2), 1],
                # n_bins_4d=[int(40 / 3), int(30 / 3), int(36 / 3), 1],
                # n_bins_4d=[int(40/2), int(30/2), int(36/2), 5],
                n_bins_4d=[int(40), int(30 ), int(36), 4],
                # n_bins_4d=[int(40*1.5 ), int(30*1.5 ), int(36), 4],
                l0_prob_mode='projector',
                l0_topN=128,
                scale_select_mode='log_expectation',
                cma_variant='Sep-CMA',
                cma_prob_mode='ingp',
                # cma_sigma0= 0.75/(40/2),
                cma_sigma0=0.75 / (40),
                # cma_sigma0= 0.75/(40*1.5),
                cma_popsize=64,
                cma_iters=12,
                cma_enable_early_stop=True,
                cma_early_stop_patience=5,
                cma_query_chunk_size=64,
                eval_thresh_cfg={"dist_lambda": 1.1*0.5, "rot_th": 11.0*0.5, "scale_ratio_th": 1.15},
                # eval_thresh_cfg={"dist_lambda": 1.1, "rot_th": 11.0, "scale_ratio_th": 1.25},
            )


        debug_cma_es = True
        if debug_cma_es:
            return 0

        level_resample_cfgs = [
            {"resample_dims": (4, 4, 2, 1), "topN": 256, "space_scale": 1.0},
            {"resample_dims": (5, 5, 2, 1), "topN": 128, "space_scale": 0.75},
            {"resample_dims": (4, 4, 2, 4), "topN": 128, "space_scale": 0.5},
        ]
        dict_res = self._test_3d_fine_accuracy_coarse2fine(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature,
            save_pred_pdf=False,
            enable_filter=False,
            # n_bins_4d=[int(40 / 1.5), int(30 / 1.5), int(36 / 2), 1],
            # n_bins_4d=[int(40 / 3), int(30 / 3), int(36 / 3), 1],
            # n_bins_4d=[int(40/2), int(30/2), int(36/2), 5],
            n_bins_4d=[int(40), int(30), int(36), 4],
            l0_prob_mode='projector',
            l0_topN=512,
            scale_select_mode='log_expectation',
            lk_prob_mode='ingp',
            level_resample_cfgs = level_resample_cfgs,
            # eval_thresh_cfg={"dist_lambda": 1.1, "rot_th": 11.0, "scale_ratio_th": 1.35},
            eval_thresh_cfg={"dist_lambda": 1.1 * 0.5, "rot_th": 11.0 * 0.5, "scale_ratio_th": 1.15},
        )

        # 5. 运行2D分类测试
        results_2d = self._test_2d_classification_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature,
        )

        # 7. 构造可视化保存路径（基于checkpoint信息）
        import os
        import re
        # 获取checkpoint路径
        stage3_ckpt_path = self._get_stage3_checkpoint_path()
        if stage3_ckpt_path:
            # 提取实验目录名（如 stage3_metric_net_31）
            exp_dir = os.path.dirname(stage3_ckpt_path)
            exp_name = os.path.basename(exp_dir)

            # 提取epoch号（如 epoch99.pth -> 99）
            ckpt_filename = os.path.basename(stage3_ckpt_path)
            epoch_match = re.search(r'epoch(\d+)', ckpt_filename)
            epoch_num = epoch_match.group(1) if epoch_match else 'unknown'

            # 构造保存路径
            dataset_type = 'train' if use_train_uav else 'test'
            viz_save_path = os.path.join(exp_dir,
                                         f"energy_field_{exp_name}_epoch{epoch_num}_{dataset_type}.html")
        else:
            # 如果没有找到checkpoint路径，使用默认名称
            dataset_type = 'train' if use_train_uav else 'test'
            viz_dir = self.log_dir2save or os.path.join(
                getattr(self.opt, "dir2save_log", "logs"),
                self.opt.exp_name,
            )
            os.makedirs(viz_dir, exist_ok=True)
            viz_save_path = os.path.join(viz_dir, f"energy_field_{dataset_type}.html")

        print(f"\n📊 生成能量场可视化: {viz_save_path}")
        self.visualize_energy_field_local(
            save_path=viz_save_path,
            use_train_uav=use_train_uav,
            show_grad_field=False,
            delta=0.1,
            argmode='min',
            energy_backend='ingp',
        )

        print("✅ Stage 3 测试完成！")
        # return {'2d': results_2d, '3d': results_3d}

    def train(self):
        """Stage 3训练主循环 (DualStream + Listwise Ranking)"""
        opt = self.opt

        print("\n" + "🚀" * 40)
        print("开始 Stage 3 训练: MetricNet (DualStream + Listwise Ranking)")
        print("🚀" * 40 + "\n")

        # 0. 初始化GradScaler
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

        # 1. 初始化loss模块并注册可学习参数
        self._init_loss_modules()

        # 2. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple

        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam', opt=self.opt)

        # 3. 加载checkpoint
        begin_epoch = self._load_checkpoint(
            opt.load2train,
            self.param2optimize,
            self.optimizer,
            mode='train'
        )

        # 4. 初始化日志
        self._init_logger()

        # 5. 初始化数据集与训练运行时上下文
        self._init_stage3_train_runtime()

        # about mining
        self.use_hard_neg_mining = bool(getattr(opt, 'use_hard_neg_mining', True))
        self.hard_neg_topN = int(getattr(opt, 'hard_neg_topN', 1024))
        self.query_uav_only = bool(getattr(opt, 'query_uav_only', True))
        self.logger.info(
            "Train配置: "
            f"loss_type={self.active_loss_type}, "
            f"query_uav_only={self.query_uav_only}, "
            f"use_hard_neg_mining={self.use_hard_neg_mining}"
        )

        # 6. 训练循环
        loss_fn = self.loss_fn
        num_epochs = opt.num_epochs
        since = time.time()
        cfg_saved = False
        step = 0
        # begin_epoch = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")
        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch_uav in tqdm.tqdm(enumerate(self.uav_dataloader_train)):
                # =================== 1. 数据准备 ===================
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)  # [B, 4]

                if self.query_uav_only:
                    # 仅使用 UAV 作为 query
                    feats_vis = self._get_feats_fm_imgs(uavimgs)  # [B, C]
                    coords_gt = coords_uav  # [B, 4]
                else:
                    batch_sat = next(iter(self.sat_dataloader))
                    satimgs = batch_sat[0].to(self.device)
                    coords_sat = batch_sat[1].to(self.device)  # [B, 4]

                    # 提取视觉特征
                    feats_vis = self._get_feats_fm_imgs(
                        torch.cat([uavimgs, satimgs], dim=0)
                    )  # [2B, C]

                    # Ground truth 坐标
                    coords_gt = torch.cat([coords_uav, coords_sat], dim=0)  # [2B, 4]
                coords_gt_linear = self.coord_normer.raw_to_linear(coords_gt)

                # 邻域正样本采样
                n_neighbors = 32
                coords_neighbor_linear = self.gs_sampler.sample_importance(
                    coords_gt_linear.to(self.device),
                    num_samples=n_neighbors,
                    include_center=True
                )

                # 负样本候选采样
                coords_all_grid, coords_all_grid_labels = self.subspace_sampler.sample_all_subspaces_gpu(
                    n_points_per_subspace=1, use_fine=False, rand_offset=True)
                coords_all_grid_flat = coords_all_grid.view(-1, 4)
                if self.use_hard_neg_mining:
                    coords_all_grid_linear = self.coord_normer.raw_to_linear(coords_all_grid_flat)
                    hard_chunk = 4096
                    dist_all_chunks = []
                    with torch.no_grad():
                        for start_idx in range(0, coords_all_grid_linear.shape[0], hard_chunk):
                            end_idx = min(start_idx + hard_chunk, coords_all_grid_linear.shape[0])
                            coords_chunk = coords_all_grid_linear[start_idx:end_idx]
                            feats_grid_chunk = self._get_feats_fm_INGP(
                                coords_chunk,
                                coord_mode='linear',
                            )
                            dist_chunk = self.projector.compute_energy(
                                feats_vis, feats_grid_chunk, metric='euclidean'
                            )
                            dist_all_chunks.append(dist_chunk)
                        dist_all = torch.cat(dist_all_chunks, dim=1)
                        hard_topN = min(self.hard_neg_topN, dist_all.shape[1])
                        _, topk_idx = torch.topk(dist_all, k=hard_topN, dim=-1, largest=False)
                    coords_rand = coords_all_grid_flat[topk_idx]  # [B, topN, 4]
                else:
                    n_candidates = coords_all_grid_flat.shape[0]
                    n_rand = max(1, int(n_candidates / 4))
                    perm = torch.randperm(n_candidates, device=coords_all_grid_flat.device)
                    perm = perm[:n_rand]
                    coords_rand = coords_all_grid_flat[perm].unsqueeze(0).expand(
                        coords_gt_linear.shape[0], -1, -1
                    )
                coords_rand_linear = self.coord_normer.raw_to_linear(
                    coords_rand.reshape(-1, 4)
                ).reshape(coords_rand.shape[0], coords_rand.shape[1], 4)

                coords_ref_linear = torch.cat([coords_neighbor_linear, coords_rand_linear], dim=1)
                weights_ref = self.coord_normer.compute_weight_matrix_linear(
                    coords_gt_linear.unsqueeze(1),
                    coords_ref_linear,
                    self.normed_sigmas,
                    ignore_dim=[3],
                ).squeeze()

                # compute feat_dist_mat with chunking to avoid large intermediate tensors
                bsz = coords_gt_linear.shape[0]
                feat_dist_chunks = []

                coords_neighbor_linear_flat = coords_neighbor_linear.reshape(-1, 4)
                with torch.no_grad():
                    feats_grid_neighbor = self._get_feats_fm_INGP(
                        coords_neighbor_linear_flat,
                        coord_mode='linear',
                    )
                feats_grid_neighbor = feats_grid_neighbor.view(
                    bsz, coords_neighbor_linear.shape[1], -1
                )
                dist_neighbor = self.projector.compute_energy(
                    feats_vis, feats_grid_neighbor, metric='euclidean'
                )
                feat_dist_chunks.append(dist_neighbor)

                coords_rand_flat = coords_rand_linear.view(-1, 4)
                neg_chunk = 4096
                feats_neg_chunks = []
                for start_idx in range(0, coords_rand_flat.shape[0], neg_chunk):
                    end_idx = min(start_idx + neg_chunk, coords_rand_flat.shape[0])
                    coords_chunk = coords_rand_flat[start_idx:end_idx]
                    with torch.no_grad():
                        feats_grid_chunk = self._get_feats_fm_INGP(
                            coords_chunk,
                            coord_mode='linear',
                        )
                    feats_neg_chunks.append(feats_grid_chunk)
                feats_grid_neg_flat = torch.cat(feats_neg_chunks, dim=0)
                neg_cols = coords_rand_linear.shape[1]
                feats_grid_neg = feats_grid_neg_flat.view(bsz, neg_cols, -1)
                dist_neg = self.projector.compute_energy(
                    feats_vis, feats_grid_neg, metric='euclidean'
                )
                feat_dist_chunks.append(dist_neg)

                feat_dist = torch.cat(feat_dist_chunks, dim=1)
                feat_dist_np = feat_dist.detach().cpu().numpy()
                weights_ref_np = weights_ref.detach().cpu().numpy()

                #compute loss
                loss_pos, loss_neg = loss_fn(feat_dist, weights_ref, 1 - weights_ref)
                loss = loss_pos + loss_neg
                # loss_de = self.wde_loss(feat_dist, weights_ref,1-weights_ref)*0.1
                # loss = loss_de+loss
                # =================== 7. 反向传播 ===================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 日志记录
                if it % 10 == 0:
                    self.logger.info(
                        f'Iter {it}: Loss={loss.item():.6f} | '
                        # f'Pos={loss_pos.item():.6f} | '
                        # f'Neg={loss_neg.item():.6f}'
                    )

                    if self.writer is not None:
                        self.writer.add_scalar('loss/total', loss.item(), step)
                        # self.writer.add_scalar('loss/pos', loss_pos.item(), step)
                        # self.writer.add_scalar('loss/neg', loss_neg.item(), step)

                    # 运行评估
                    self._run_epoch_evaluation(
                        epoch=epoch,
                        run_visualization=False,
                        n_test_samples=256
                    )
                if it % 20 == 0:
                    self.analyze_feat_freq_band()

                step += 1

            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f'epoch {epoch} 完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            self.logger.info('-' * 50)

            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss, epoch)

            # 备份代码
            if not cfg_saved:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(self.exp_dir2save, self.opt)
                cfg_saved = True

            # 每个epoch结束后保存checkpoint
            self._save_checkpoint(
                epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )

            # 每个epoch结束后运行评估
            self._run_epoch_evaluation(
                epoch=epoch,
                run_visualization=False,
                n_test_samples=256
            )

        # 保存最后一个epoch的checkpoint（避免重复保存）
        final_epoch = num_epochs - 1
        if not ((final_epoch % 5 == 0) and (final_epoch > 0)):
            self.logger.info(f"保存最后一个epoch的checkpoint: epoch_{final_epoch}")
            self._save_checkpoint(
                final_epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )

        self.logger.info("✅ Stage 3 训练完成！")


if __name__ == "__main__":
    import argparse
    import sys

    # 添加 --test_only 参数
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid conflict with get_parse
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    args, remaining_argv = parser.parse_known_args()

    # test by manual modification
    args.test_only = True
    # 如果没有显式指定配置文件，回退到默认 stage3 配置。
    # 保持外部传入的 --p_yaml 优先，便于在不同场景间切换测试配置。
    if '--p_yaml' not in remaining_argv:
        remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_wingtra_infonce.yaml'])

    sys.argv[1:] = remaining_argv  # Pass remaining args to get_parse

    trainer = MetricNetTrainer()

    if args.test_only:
        trainer.test(use_train_uav=False)
    else:
        trainer.train()
