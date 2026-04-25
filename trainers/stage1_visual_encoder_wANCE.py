#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1: Visual Encoder Trainer

训练目标：
- vis_aggregator (特征聚合器)

前置条件：
- 预训练的 vis_encoder（冻结）

训练策略：
- 使用Soft Weighted Triplet Loss训练特征聚合
- UAV-Satellite图像对正样本 + 随机负样本
- 支持多场景训练
"""

import os
import re
import sys
import time
import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from functools import partial


def _bootstrap_runtime_env():
    abs_python = os.path.abspath(sys.executable)
    if not abs_python.endswith("/bin/python"):
        return

    env_root = os.path.dirname(os.path.dirname(abs_python))
    env_lib = os.path.join(env_root, "lib")
    if not os.path.isdir(env_lib):
        return

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in old_ld.split(":") if part]
    if ld_parts[:1] == [env_lib]:
        return

    os.environ["LD_LIBRARY_PATH"] = f"{env_lib}:{old_ld}" if old_ld else env_lib
    if os.environ.get("_STAGE1_W_ANCE_REEXECED") == "1":
        return

    os.environ["_STAGE1_W_ANCE_REEXECED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_runtime_env()

import numpy as np
import torch
import torch.nn.functional as TF
import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.base.components import NetworkComponents
from trainers.util_core_eval import compute_topk_acc_from_coords
from trainers.util_stage1_ance import Stage1ANCEHelper
from trainers.util_stage1_gallery_manager import (
    Stage1ReferenceGalleryBank,
    Stage1ReferenceGalleryFeatureConfig,
    Stage1ReferenceGalleryLayoutConfig,
)
from trainers.util_stage1_multi_scene_dataloader import MultiSceneDataLoader
from trainers.util_stage1_retrieval_evaluator import Stage1RetrievalEvalConfig, Stage1RetrievalEvaluator


class VisualEncoderTrainer(BaseTrainer):
    """
    Stage 1: Visual Encoder Trainer

    训练视觉特征聚合器（vis_aggregator）
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, opt=None):
        """初始化Stage 1 Trainer"""
        super().__init__(opt)

        # 初始化网络组件
        self._init_networks()

        # 设置可训练参数
        self._setup_trainable_params()

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"

    def _init_networks(self):
        """初始化所有网络组件"""
        print("\n" + "="*80)
        print("初始化 Stage 1 网络组件")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        # 视觉编码器（冻结）
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel

        # 特征聚合器（可训练）
        self.vis_aggregator = components.create_aggregator(
            self.feat_patch_dim
        )
        self.feat_q_dim = int(getattr(self.vis_aggregator, 'output_dim', self.feat_patch_dim))

        print("="*80 + "\n")

    def _setup_trainable_params(self):
        """设置可训练参数"""
        freeze_backbone = bool(getattr(self.opt, 'freeze_backbone', True))
        adapter_enabled = bool(getattr(self.opt, 'adapter_config', {}).get('enabled', False))
        if freeze_backbone and not adapter_enabled:
            for param in self.vis_encoder.parameters():
                param.requires_grad = False

        self.param2optimize = {'vis_aggregator': self.vis_aggregator}
        self.param2freeze = {}
        n_trainable_backbone_params = sum(
            param.numel() for param in self.vis_encoder.parameters() if param.requires_grad
        )
        n_trainable_adapter_params = sum(
            param.numel()
            for name, param in self.vis_encoder.named_parameters()
            if param.requires_grad and 'adapter' in name
        )
        n_trainable_raw_backbone_params = n_trainable_backbone_params - n_trainable_adapter_params
        if n_trainable_backbone_params > 0:
            self.param2optimize['vis_encoder'] = self.vis_encoder
        else:
            self.param2freeze['vis_encoder'] = self.vis_encoder

        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys()) if self.param2freeze else '(none)'

        print(f"参数配置 (freeze_backbone={freeze_backbone}):")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")
        print(f"  vis_encoder 可训练参数量: {n_trainable_backbone_params}\n")
        if adapter_enabled:
            print(f"  vis_encoder 其中 adapter 可训练参数量: {n_trainable_adapter_params}")
            print(f"  vis_encoder 其中原始 backbone 可训练参数量: {n_trainable_raw_backbone_params}\n")

    def _forward_train_vis_encoder(self, imgs_input):
        has_trainable_params = any(param.requires_grad for param in self.vis_encoder.parameters())
        if not has_trainable_params:
            with torch.no_grad():
                return self.vis_encoder(imgs_input)
        return self.vis_encoder(imgs_input)

    def _log_or_print(self, message):
        if getattr(self, "logger", None) is not None:
            self.logger.info(message)
        else:
            print(message)

    def _autocast_context(self):
        use_amp = bool(getattr(self.opt, "autocast", False)) and getattr(self.device, "type", "") == "cuda"
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------
    # Checkpoint Helpers
    # ------------------------------------------------------------------
    def _resolve_eval_ckpt_config(self, ckpt_config=None):
        ckpt2load = ckpt_config
        if ckpt2load is None:
            ckpt2load = getattr(self.opt, "load2test", "")
            if not ckpt2load:
                ckpt2load = getattr(self.opt, "load2train", "")
        return ckpt2load

    def _pick_ckpt_path_for_log(self, ckpt_config):
        if isinstance(ckpt_config, dict):
            for value in ckpt_config.values():
                if value:
                    return value
            return None
        return ckpt_config or None

    @staticmethod
    def _resolve_gallery_ckpt_tag(ckpt_path):
        if not ckpt_path:
            return None
        exp_name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        ckpt_base = os.path.basename(ckpt_path)
        match = re.search(r"(epoch\d+)", ckpt_base)
        epoch_tag = match.group(1) if match else os.path.splitext(ckpt_base)[0]
        if exp_name:
            return f"{exp_name}_{epoch_tag}"
        return epoch_tag

    def load_eval_checkpoint(self, ckpt_config=None, verbose=True):
        """
        Load the evaluation checkpoint for vis_encoder / vis_aggregator.

        This wraps the stage1 test-time checkpoint logic so demos and evaluators
        do not need to duplicate the same load2test -> load2train fallback.
        """
        ckpt2load = self._resolve_eval_ckpt_config(ckpt_config)
        if not ckpt2load:
            self.last_eval_ckpt_path = None
            if verbose:
                print("[Checkpoint] warning: no load2test/load2train checkpoint configured, using randomly initialized vis_aggregator.")
            return None

        self._load_checkpoint(
            ckpt2load,
            {**self.param2optimize, **self.param2freeze},
            optimizer=None,
            mode="test",
        )
        ckpt_path = self._pick_ckpt_path_for_log(ckpt2load)
        self.last_eval_ckpt_path = ckpt_path
        if verbose:
            print(f"[Checkpoint] loaded stage1 weights from {ckpt_path}")
        return ckpt_path

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def _init_multi_scene_dataloader(self):
        """初始化多场景训练数据加载器"""
        from trainer_depends.datasets.dataset_neuloc_4d_uav_sat_pair import UAVSatPairDataset, collate_uav_sat_pair
        from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor

        opt = self.opt
        scenes = opt.scenes_setting['scenes']
        add_random_satimg_negs = bool(getattr(opt, "add_random_satimg_negs", True))
        reject_sampling = bool(getattr(opt, "reject_sampling", False))
        reject_batch_aware = bool(getattr(opt, "reject_batch_aware", False))
        pair_alignment_mode = str(getattr(opt, "pair_alignment_mode", "full_4d")).strip().lower()

        if getattr(opt, "ance_enabled", False) and add_random_satimg_negs:
            raise ValueError(
                "Configuration conflict: ance_enabled=True cannot be combined with "
                "add_random_satimg_negs=True."
            )

        # 为每个场景创建pair dataloader
        pair_dataloaders = {}
        self.coord_normers = {}
        self.coord_normed_sigmas = {}

        for scene in scenes:
            scene_name = scene['name']

            # 创建该场景的 UAVSatPairDataset
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_train = self.uav_datasets_train[scene_name]

            if getattr(opt, "ance_enabled", False):
                n_neg_per_query = 0
            elif not add_random_satimg_negs:
                n_neg_per_query = 0
            else:
                n_neg_per_query = opt.batchsize_sat // opt.batchsize_uav
            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                device= self.device,
                n_neg_per_query=n_neg_per_query, #控制负样本数=正样本数的倍率
                sat_as_query=opt.sat_as_query,
                nrc_reject_sampling=reject_sampling,
                pair_alignment_mode=pair_alignment_mode,
            )
            pair_dataset.weight = len(uav_dataset_train)

            # 创建DataLoader
            pair_dataloader = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=opt.batchsize_uav,
                num_workers=opt.num_worker,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                collate_fn=partial(
                    collate_uav_sat_pair,
                    sat_dataset=sat_dataset,
                    reject_batch_aware=reject_batch_aware,
                ),
                persistent_workers=(opt.num_worker > 0)
            )

            pair_dataloaders[scene_name] = pair_dataloader
            neg_sampling_mode = "reject" if reject_sampling else "random"
            self.logger.info(
                f"  {scene_name}: {len(pair_dataset)} pairs, {len(pair_dataloader)} batches, "
                f"neg_sampling={neg_sampling_mode}, reject_batch_aware={reject_batch_aware}, "
                f"pair_alignment_mode={pair_alignment_mode}, pos_scale_mean={pair_dataset.pos_scale_mean:.4f}"
            )

            # CoordsNormProcessor for this scene (per pair dataset)
            self.coord_normers[scene_name] = CoordsNormProcessor(sat_dataset)
            gs_sigma_nrc_factor = self.sat_datasets[scene_name].halfimg_radius_nrc*0.5
            gs_sigma_rot_rad_abs = torch.pi / 18 * 0.5
            gs_sigma_scale_log_abs = 0.1
            self.coord_normed_sigmas[scene_name] = torch.tensor(
                [gs_sigma_nrc_factor, gs_sigma_nrc_factor, gs_sigma_rot_rad_abs, gs_sigma_scale_log_abs],
                dtype=torch.float32
            ).to(self.device)
            self.gs_sigma2radius_factor = 2.

        # 使用 MultiSceneDataLoader
        self.dataloader_train = MultiSceneDataLoader(
            pair_dataloaders,
            sampling_strategy=opt.scenes_setting['sampling_strategy']
        )

        self.logger.info(f"\n✅ 多场景训练集: {len(scenes)}个场景, "
                        f"总计{self.dataloader_train.total_batches}个batches, "
                        f"采样策略={opt.scenes_setting['sampling_strategy']}\n")

    # ------------------------------------------------------------------
    # ANCE Integration
    # ------------------------------------------------------------------
    def _init_ance_helper(self):
        self.ance_helper = Stage1ANCEHelper(self)
        self.ance_helper.initialize()
        self.ance_enabled = self.ance_helper.enabled
        self.ance_gallery_info = self.ance_helper.gallery_info

    def _maybe_refresh_ance_gallery(self, epoch):
        if not getattr(self, "ance_enabled", False):
            return
        self.ance_helper.maybe_refresh(epoch)
        self.ance_gallery_info = self.ance_helper.gallery_info

    # ------------------------------------------------------------------
    # Loss & Training Strategy
    # ------------------------------------------------------------------
    def _init_loss_modules(self):
        loss_type = str(getattr(self.opt, "loss_type", "tripleLoss_singleEdge_hardest_fm_weight")).lower()
        if loss_type == "infonce":
            from losses.stage1_infonce_loss import Stage1InfoNCELoss

            self.active_loss_type = loss_type
            self.active_loss_input_mode = "query_positive_infonce"
            self.active_loss_output_mode = "scalar"
            self.loss_w_weight = False
            self.active_loss_module_name = "loss_fn"
            self.active_loss_module = Stage1InfoNCELoss(
                temperature=float(getattr(self.opt, "infonce_temperature", 0.1)),
                negative_mode=str(getattr(self.opt, "infonce_negative_mode", "batch_and_explicit")),
            ).to(self.device)
            self.active_loss_miner = None
            self._register_active_loss_module()
            return

        if loss_type == "msloss_torch":
            from losses.MSLoss_fm_torch import MultiSimilarityLossTorch, MultiSimilarityMinerTorch

            self.active_loss_type = loss_type
            self.active_loss_input_mode = "descriptors_labels"
            self.active_loss_output_mode = "scalar"
            self.loss_w_weight = False
            self.active_loss_module_name = "loss_fn"
            self.active_loss_module = MultiSimilarityLossTorch(
                alpha=1.0,
                beta=50.0,
                base=0.0,
            ).to(self.device)
            self.active_loss_miner = MultiSimilarityMinerTorch(epsilon=0.1)
            self._register_active_loss_module()
            return

        from losses.CL_losses_w_weight import (
            pairLoss_multiEdge_logSum,
            pairLoss_singleEdge_hardest,
            pairLoss_singleEdge_weightedHardest,
        )
        from losses.CL_losses_wo_weight import (
            tripleLoss_singleEdge_hardest_fm_mask,
            tripleLoss_singleEdge_hardest_fm_weight,
        )

        self.active_loss_type = str(getattr(self.opt, "loss_type", "tripleLoss_singleEdge_hardest_fm_weight")).lower()
        loss_registry = {
            "pairloss_singleedge_weightedhardest": {
                "factory": lambda: pairLoss_singleEdge_weightedHardest(
                    beta=5.0,
                    margin=0.0,
                    learnable_beta=True,
                ),
                "input_mode": "weights",
                "output_mode": "pair",
                "module_name": "loss_fm_weight",
            },
            "pairloss_singleedge_hardest": {
                "factory": lambda: pairLoss_singleEdge_hardest(
                    beta=5.0,
                    margin=0.0,
                ),
                "input_mode": "weights",
                "output_mode": "pair",
                "module_name": "loss_fm_weight",
            },
            "pairloss_multiedge_logsum": {
                "factory": lambda: pairLoss_multiEdge_logSum(
                    beta=5.0,
                    margin=0.0,
                    learnable_beta=True,
                ),
                "input_mode": "weights",
                "output_mode": "pair",
                "module_name": "loss_fm_weight",
            },
            "tripleloss_singleedge_hardest_fm_weight": {
                "factory": lambda: tripleLoss_singleEdge_hardest_fm_weight(
                    beta=5.0,
                    margin=0.0,
                    learnable_beta=True,
                ),
                "input_mode": "weights",
                "output_mode": "scalar",
                "module_name": "loss_fm_weight",
            },
            "tripleloss_singleedge_hardest_fm_mask": {
                "factory": lambda: tripleLoss_singleEdge_hardest_fm_mask(),
                "input_mode": "mask",
                "output_mode": "scalar",
                "module_name": "loss_fm_mask",
            },
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
        self.loss_w_weight = self.active_loss_input_mode == "weights"
        self.active_loss_module_name = loss_spec["module_name"]
        self.active_loss_module = loss_spec["factory"]().to(self.device)
        self._register_active_loss_module()

    def _register_active_loss_module(self):
        self.param2optimize.pop("loss_fm_weight", None)
        self.param2optimize.pop("loss_fm_mask", None)
        self.param2optimize.pop("loss_fn", None)
        self.param2optimize[self.active_loss_module_name] = self.active_loss_module
        print(
            f"Loss配置: loss_type={self.active_loss_type}, "
            f"active_loss_module={self.active_loss_module_name}, "
            f"input_mode={self.active_loss_input_mode}, "
            f"output_mode={self.active_loss_output_mode}"
        )

    def _reduce_loss_output(self, loss_output):
        if self.active_loss_output_mode == "pair":
            loss_pos, loss_neg = loss_output
            return loss_pos + loss_neg
        return loss_output

    def _build_train_ckpt_modules(self):
        modules = {
            "vis_aggregator": self.vis_aggregator,
            "vis_encoder": self.vis_encoder,
            self.active_loss_module_name: self.active_loss_module,
            "loss_type": self.active_loss_type,
        }
        if getattr(self.opt, "autocast", False) and hasattr(self, "scaler") and self.scaler is not None:
            modules["amp_scaler"] = self.scaler
        return modules

    def _load_module_state_from_checkpoint(self, checkpoint, module_name, module, strict=True):
        if module_name not in checkpoint:
            return False
        try:
            incompatible = module.load_state_dict(checkpoint[module_name], strict=strict)
            if not strict:
                missing = getattr(incompatible, "missing_keys", [])
                unexpected = getattr(incompatible, "unexpected_keys", [])
                if missing or unexpected:
                    self.logger.info(
                        f"[Checkpoint] relaxed load for {module_name}: "
                        f"missing={missing}, unexpected={unexpected}"
                    ) if self.logger else print(
                        f"[Checkpoint] relaxed load for {module_name}: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        except RuntimeError as exc:
            if strict:
                raise exc
            incompatible = module.load_state_dict(checkpoint[module_name], strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            msg = (
                f"[Checkpoint] non-strict load for {module_name} after mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        return True

    def _load_module_from_file(self, ckpt_path, module_name, module, strict=True):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if isinstance(checkpoint, dict) and module_name in checkpoint:
            state_dict = checkpoint[module_name]
        else:
            state_dict = checkpoint
        module.load_state_dict(state_dict, strict=strict)
        return checkpoint

    def _load_train_checkpoint(self, ckpt_config):
        begin_epoch = 0
        if not ckpt_config:
            return begin_epoch

        if isinstance(ckpt_config, dict):
            module_map = {
                "vis_aggregator": self.vis_aggregator,
                "vis_encoder": self.vis_encoder,
                self.active_loss_module_name: self.active_loss_module,
            }
            for module_name, ckpt_path in ckpt_config.items():
                if not ckpt_path:
                    continue
                if module_name not in module_map:
                    continue
                self._load_module_from_file(ckpt_path, module_name, module_map[module_name], strict=True)
                msg = f"✅ 加载{module_name}模块: {ckpt_path}"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
            return begin_epoch

        checkpoint = torch.load(ckpt_config, map_location=lambda storage, loc: storage)
        self._load_module_state_from_checkpoint(checkpoint, "vis_aggregator", self.vis_aggregator, strict=True)
        self._load_module_state_from_checkpoint(checkpoint, "vis_encoder", self.vis_encoder, strict=True)

        loss_type_in_ckpt = checkpoint.get("loss_type", None)
        if loss_type_in_ckpt is not None and str(loss_type_in_ckpt).lower() != self.active_loss_type:
            msg = (
                f"[Checkpoint] loss_type mismatch: ckpt={loss_type_in_ckpt}, "
                f"current={self.active_loss_type}. Try loading active loss module with fallback."
            )
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        try:
            loaded_loss = self._load_module_state_from_checkpoint(
                checkpoint,
                self.active_loss_module_name,
                self.active_loss_module,
                strict=True,
            )
        except RuntimeError:
            loaded_loss = self._load_module_state_from_checkpoint(
                checkpoint,
                self.active_loss_module_name,
                self.active_loss_module,
                strict=False,
            )

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "amp_scaler" in checkpoint and getattr(self.opt, "autocast", False) and hasattr(self, "scaler"):
            self.scaler.load_state_dict(checkpoint["amp_scaler"])

        if "epoch" in checkpoint:
            begin_epoch = checkpoint["epoch"] + 1

        msg = f"✅ 加载checkpoint: {ckpt_config}, 从epoch {begin_epoch}继续训练"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return begin_epoch

    def _compute_weighted_metric_loss(
            self,
            scene_name,
            feat_dist_mat,
            coords_uav,
            coords_uav_neg_flat,
            has_sat_query=False,
            b_uav=0,
            b_sat=0,
    ):
        # 备用的加权度量学习分支：先用坐标邻近性生成 soft weight，
        # 再把 feature distance 拆成正负项损失，避免这部分细节堆在 train() 主循环里。
        weights_ref = None
        if scene_name in self.coord_normers:
            coord_normer = self.coord_normers[scene_name]
            normed_sigmas = self.coord_normed_sigmas[scene_name]
            if coords_uav_neg_flat is None:
                coords_ref = coords_uav
            else:
                coords_ref = torch.cat([coords_uav, coords_uav_neg_flat], dim=0)
            coords_uav_linear = coord_normer.raw_to_linear(coords_uav)
            coords_ref_linear = coord_normer.raw_to_linear(coords_ref)
            coords_ref_linear = coords_ref_linear.unsqueeze(0).expand(
                coords_uav_linear.shape[0], -1, -1
            )
            weights_ref = coord_normer.compute_weight_matrix_linear(
                coords_uav_linear.unsqueeze(1),
                coords_ref_linear,
                normed_sigmas,
                ignore_dim=None
            ).squeeze(1)

        if weights_ref is None:
            raise ValueError(
                f"Weighted metric loss requires coord_normer for scene '{scene_name}'."
            )

        if has_sat_query and b_sat > 0:
            feat_dist_mat_uav = feat_dist_mat[:b_uav]
            feat_dist_mat_sat = feat_dist_mat[b_uav:]
            weights_ref_uav = weights_ref[:b_uav]
            weights_ref_sat = weights_ref[b_uav:]

            loss_uav = self._reduce_loss_output(
                self.active_loss_module(
                    feat_dist_mat_uav, weights_ref_uav, 1 - weights_ref_uav
                )
            )
            loss_sat = self._reduce_loss_output(
                self.active_loss_module(
                    feat_dist_mat_sat, weights_ref_sat, 1 - weights_ref_sat
                )
            )
            sat_query_loss_weight = float(0.1)
            return loss_uav + sat_query_loss_weight * loss_sat

        return self._reduce_loss_output(
            self.active_loss_module(feat_dist_mat, weights_ref, 1 - weights_ref)
        )

    def _build_pos_mask(self, batch_size, num_refs, device):
        # 当前默认的 hard-positive mask 构造逻辑：
        # query 与同索引正样本配对，其余 reference 视为负样本，便于 SWS/SWT 类损失直接复用。
        if (not hasattr(self, "pos_mask_mat")
                or self.pos_mask_mat.shape[0] != batch_size
                or self.pos_mask_mat.shape[1] != num_refs):
            self.pos_mask_mat = torch.cat([
                torch.eye(batch_size, device=device),
                torch.zeros(
                    batch_size,
                    num_refs - batch_size,
                    device=device,
                )
            ], dim=-1).bool()
        return self.pos_mask_mat

    # ------------------------------------------------------------------
    # Batch Parsing & Forward Helpers
    # ------------------------------------------------------------------
    def _extract_train_batch(self, batch):
        scene_name = batch.get('scene_name', None)
        sat_dataset = self.sat_datasets.get(scene_name, self.sat_dataset)

        uavimgs = batch['uavimgs'].to(self.device)
        satimgs_pos = batch['satimgs_pos'].to(self.device)
        coords_uav = batch['coords_uav'].to(self.device)

        has_sat_query = 'satimgs_query' in batch
        b_uav = uavimgs.shape[0]
        b_sat = 0
        if has_sat_query:
            satimgs_query = batch['satimgs_query'].to(self.device)
            satimgs_pos2satimg_query = batch['satimgs_pos2satimg_query'].to(self.device)
            coords_sat_query = batch['coords_sat_query'].to(self.device)
            b_sat = satimgs_query.shape[0]
            uavimgs = torch.cat([uavimgs, satimgs_query], dim=0)
            satimgs_pos = torch.cat([satimgs_pos, satimgs_pos2satimg_query], dim=0)
            coords_uav = torch.cat([coords_uav, coords_sat_query], dim=0)

        return {
            "scene_name": scene_name,
            "sat_dataset": sat_dataset,
            "uavimgs": uavimgs,
            "satimgs_pos": satimgs_pos,
            "coords_uav": coords_uav,
            "has_sat_query": has_sat_query,
            "b_uav": b_uav,
            "b_sat": b_sat,
        }

    def _load_batch_negatives(self, batch):
        if 'satimgs_neg' not in batch:
            return None, None

        satimgs_neg = batch['satimgs_neg'].to(self.device)
        coords_uav_neg = batch['coords_uav_neg'].to(self.device)
        satimgs_neg_flat = satimgs_neg.reshape(-1, *satimgs_neg.shape[2:])
        coords_uav_neg_flat = coords_uav_neg.reshape(-1, *coords_uav_neg.shape[2:])
        return satimgs_neg_flat, coords_uav_neg_flat

    def _prepare_batch_negatives(self, batch_state, batch):
        satimgs_neg_flat, coords_uav_neg_flat = self._load_batch_negatives(batch)
        if getattr(self, "ance_enabled", False):
            ance_batch = self.ance_helper.prepare_batch(
                scene_name=batch_state["scene_name"],
                sat_dataset=batch_state["sat_dataset"],
                uavimgs=batch_state["uavimgs"],
                coords_uav=batch_state["coords_uav"],
            )
            batch_state["uavimgs"] = ance_batch["uavimgs"]
            batch_state["coords_uav"] = ance_batch["coords_uav"]
            satimgs_neg_flat = ance_batch["satimgs_neg_flat"]
            coords_uav_neg_flat = ance_batch["coords_uav_neg_flat"]

        if satimgs_neg_flat is None and coords_uav_neg_flat is None:
            if getattr(self, "ance_enabled", False):
                raise ValueError("ANCE is enabled but did not return negatives.")
            if bool(getattr(self.opt, "add_random_satimg_negs", True)):
                raise ValueError("Random satimg negatives are enabled but the batch did not provide them.")
            return None, None
        if satimgs_neg_flat is None or coords_uav_neg_flat is None:
            raise ValueError("Training batch does not provide negatives and ANCE is disabled.")
        return satimgs_neg_flat, coords_uav_neg_flat

    def _forward_train_batch(self, uavimgs, satimgs_pos, satimgs_neg_flat):
        if satimgs_neg_flat is None:
            imgs_input = torch.cat([uavimgs, satimgs_pos], dim=0)
        else:
            imgs_input = torch.cat([uavimgs, satimgs_pos, satimgs_neg_flat], dim=0)
        feats_patch = self._forward_train_vis_encoder(imgs_input)

        feats_agg = self.vis_aggregator(feats_patch)
        batch_size = uavimgs.shape[0]
        feats_q = feats_agg[:batch_size]
        feats_ref = feats_agg[batch_size:]
        feat_dist_mat = torch.norm(
            feats_q.unsqueeze(1) - feats_ref.unsqueeze(0),
            p=2,
            dim=-1
        )
        forward_state = {
            "batch_size": batch_size,
            "feats_q": feats_q,
            "feats_ref": feats_ref,
            "feat_dist_mat": feat_dist_mat,
        }
        self._last_forward_state = forward_state
        return forward_state

    def _log_feature_variance_stats(self, feats_q, feats_ref):
        q_var_mean = torch.var(feats_q, dim=0).mean().item()
        ref_var_mean = torch.var(feats_ref, dim=0).mean().item()

        self.logger.info(f"FeatQ方差均值: {q_var_mean:.4e}")
        self.logger.info(f"FeatRef方差均值: {ref_var_mean:.4e}")

    @staticmethod
    def _optimizer_has_any_grad(optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    return True
        return False

    def _after_train_dataloader_init(self):
        """Hook for subclasses that need data-dependent setup before training starts."""
        self._maybe_initialize_netvlad_from_training_data()
        return None

    def _get_netvlad_cluster_init_config(self):
        aggregator_config = dict(getattr(self.opt, "aggregator_config", {}) or {})
        cluster_init_cfg = dict(aggregator_config.get("cluster_init", {}) or {})
        cfg = {
            "enabled": True,
            "sample_uav": True,
            "sample_sat": True,
            "descriptors_per_image": 8,
            "max_batches": 64,
            "max_descriptors": 50000,
            "kmeans_iters": 25,
            "seed": 123,
            "save_artifact": True,
        }
        cfg.update(cluster_init_cfg)
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["sample_uav"] = bool(cfg["sample_uav"])
        cfg["sample_sat"] = bool(cfg["sample_sat"])
        cfg["descriptors_per_image"] = max(1, int(cfg["descriptors_per_image"]))
        cfg["max_batches"] = max(1, int(cfg["max_batches"]))
        cfg["max_descriptors"] = max(1, int(cfg["max_descriptors"]))
        cfg["kmeans_iters"] = max(1, int(cfg["kmeans_iters"]))
        cfg["seed"] = int(cfg["seed"])
        cfg["save_artifact"] = bool(cfg["save_artifact"])
        return cfg

    def _collect_netvlad_patch_descriptors(self, cfg):
        descriptors = []
        total_descriptors = 0
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(cfg["seed"]))
        sample_sources = []
        if cfg["sample_uav"]:
            sample_sources.append("uavimgs")
        if cfg["sample_sat"]:
            sample_sources.append("satimgs_pos")
        if not sample_sources:
            raise ValueError("NetVLAD cluster_init requires at least one of sample_uav/sample_sat to be enabled.")

        self._log_or_print(
            "[NetVLAD Init] collecting patch descriptors: "
            f"sources={sample_sources}, descriptors_per_image={cfg['descriptors_per_image']}, "
            f"max_batches={cfg['max_batches']}, max_descriptors={cfg['max_descriptors']}"
        )

        self.vis_encoder.eval()
        self.vis_aggregator.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(iter(self.dataloader_train)):
                if batch_idx >= cfg["max_batches"] or total_descriptors >= cfg["max_descriptors"]:
                    break

                imgs_to_encode = []
                for source_name in sample_sources:
                    imgs = batch.get(source_name, None)
                    if imgs is not None:
                        imgs_to_encode.append(imgs.to(self.device, non_blocking=True))
                if not imgs_to_encode:
                    continue

                imgs_input = torch.cat(imgs_to_encode, dim=0)
                feats_patch = self._forward_train_vis_encoder(imgs_input)
                if not hasattr(self.vis_aggregator, "backbone") or not hasattr(self.vis_aggregator.backbone, "_tokens_to_feature_map"):
                    raise AttributeError("NetVLAD cluster_init requires aggregator.backbone._tokens_to_feature_map.")
                fmap = self.vis_aggregator.backbone._tokens_to_feature_map(feats_patch)
                if getattr(self.vis_aggregator, "normalize_input", False):
                    fmap = torch.nn.functional.normalize(fmap, p=2, dim=1)

                patch_desc = fmap.flatten(2).transpose(1, 2).detach().cpu().to(dtype=torch.float32)
                n_imgs, n_tokens, dim = patch_desc.shape
                per_image = min(int(cfg["descriptors_per_image"]), n_tokens)
                rand_order = torch.rand((n_imgs, n_tokens), generator=rng)
                token_indices = torch.topk(rand_order, k=per_image, dim=1).indices
                img_indices = torch.arange(n_imgs, dtype=torch.long).unsqueeze(1)
                sampled = patch_desc[img_indices, token_indices].reshape(-1, dim)

                descriptors.append(sampled)
                total_descriptors += int(sampled.shape[0])

        if not descriptors:
            raise RuntimeError("Failed to collect any patch descriptors for NetVLAD cluster initialization.")

        descriptors = torch.cat(descriptors, dim=0)
        if descriptors.shape[0] > cfg["max_descriptors"]:
            keep_idx = torch.randperm(descriptors.shape[0], generator=rng)[: cfg["max_descriptors"]]
            descriptors = descriptors[keep_idx]
        return descriptors.contiguous()

    def _run_faiss_kmeans(self, descriptors, num_clusters, cfg):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("NetVLAD cluster_init requires faiss to be installed.") from exc

        desc_np = np.ascontiguousarray(descriptors.cpu().numpy(), dtype=np.float32)
        if desc_np.shape[0] < num_clusters:
            raise ValueError(
                f"NetVLAD cluster_init needs at least {num_clusters} descriptors, got {desc_np.shape[0]}"
            )

        kmeans = faiss.Kmeans(
            desc_np.shape[1],
            num_clusters,
            niter=int(cfg["kmeans_iters"]),
            verbose=False,
            gpu=False,
            seed=int(cfg["seed"]),
        )
        kmeans.train(desc_np)
        centroids = torch.from_numpy(np.ascontiguousarray(kmeans.centroids)).to(dtype=torch.float32)
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
        return centroids

    def _maybe_save_netvlad_init_artifact(self, cfg, descriptors, centroids):
        if not cfg["save_artifact"] or not getattr(self, "log_dir2save", None):
            return
        save_path = os.path.join(self.log_dir2save, "netvlad_cluster_init.pt")
        payload = {
            "centroids": centroids.cpu(),
            "num_descriptors": int(descriptors.shape[0]),
            "descriptor_dim": int(descriptors.shape[1]),
            "config": cfg,
            "alpha": float(getattr(self.vis_aggregator, "alpha", 0.0)),
        }
        try:
            torch.save(payload, save_path)
            self._log_or_print(f"[NetVLAD Init] saved centroids to {save_path}")
        except Exception as exc:
            self._log_or_print(f"[NetVLAD Init] warning: failed to save init artifact to {save_path}: {exc}")

    def _maybe_initialize_netvlad_from_training_data(self):
        agg_type = str(getattr(self.opt, "aggregator_type", "")).lower()
        if agg_type != "netvlad" or not hasattr(self.vis_aggregator, "initialize_centroids"):
            return
        if getattr(self.opt, "load2train", ""):
            self._log_or_print("[NetVLAD Init] skip cluster initialization because load2train is configured.")
            return

        cfg = self._get_netvlad_cluster_init_config()
        if not cfg["enabled"]:
            self._log_or_print("[NetVLAD Init] cluster initialization disabled by config.")
            return

        prev_encoder_mode = self.vis_encoder.training
        prev_agg_mode = self.vis_aggregator.training
        try:
            descriptors = self._collect_netvlad_patch_descriptors(cfg)
            num_clusters = int(getattr(self.vis_aggregator, "num_clusters"))
            centroids = self._run_faiss_kmeans(descriptors, num_clusters=num_clusters, cfg=cfg)
            self.vis_aggregator.initialize_centroids(centroids.to(self.device))
            self._log_or_print(
                "[NetVLAD Init] initialized centroids from training descriptors: "
                f"num_descriptors={descriptors.shape[0]}, descriptor_dim={descriptors.shape[1]}, "
                f"num_clusters={num_clusters}, alpha={float(getattr(self.vis_aggregator, 'alpha', 0.0)):.2f}"
            )
            self._maybe_save_netvlad_init_artifact(cfg, descriptors, centroids)
        finally:
            self.vis_encoder.train(prev_encoder_mode)
            self.vis_aggregator.train(prev_agg_mode)

    def _compute_train_loss(self, batch_state, feat_dist_mat, coords_uav_neg_flat):
        if self.active_loss_input_mode == "query_positive_infonce":
            forward_state = getattr(self, "_last_forward_state", None)
            if forward_state is None:
                raise RuntimeError(f"Missing forward state for {self.active_loss_type}.")

            batch_size = int(forward_state["batch_size"])
            feats_q = forward_state["feats_q"]
            feats_ref = forward_state["feats_ref"]
            feats_pos = feats_ref[:batch_size]
            feats_neg = feats_ref[batch_size:]

            if feats_pos.shape[0] != batch_size:
                raise ValueError(
                    f"{self.active_loss_type} expects {batch_size} positive references, got {feats_pos.shape[0]}."
                )
            return self.active_loss_module(
                feats_q,
                feats_pos,
                explicit_negative_keys=feats_neg,
            )

        if self.active_loss_input_mode == "descriptors_labels":
            forward_state = getattr(self, "_last_forward_state", None)
            if forward_state is None:
                raise RuntimeError(f"Missing forward state for {self.active_loss_type}.")

            batch_size = int(forward_state["batch_size"])
            feats_q = forward_state["feats_q"]
            feats_ref = forward_state["feats_ref"]
            feats_pos = feats_ref[:batch_size]

            if feats_pos.shape[0] != batch_size:
                raise ValueError(
                    f"{self.active_loss_type} expects {batch_size} positive references, got {feats_pos.shape[0]}."
                )

            query_labels = torch.arange(batch_size, device=feats_q.device, dtype=torch.long)
            ref_labels = query_labels.clone()
            num_explicit_neg_refs = max(0, int(feats_ref.shape[0] - batch_size))
            if num_explicit_neg_refs > 0:
                neg_labels = torch.arange(
                    batch_size,
                    batch_size + num_explicit_neg_refs,
                    device=feats_ref.device,
                    dtype=torch.long,
                )
                ref_labels = torch.cat([ref_labels, neg_labels], dim=0)
                if not getattr(self, "_logged_msloss_explicit_negs", False):
                    msg = (
                        f"{self.active_loss_type} uses {num_explicit_neg_refs} explicit satimgs_neg references "
                        "as labeled negatives in the ref set."
                    )
                    if getattr(self, "logger", None) is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                    self._logged_msloss_explicit_negs = True

            miner_outputs = self.active_loss_miner(
                feats_q,
                query_labels,
                ref_emb=feats_ref,
                ref_labels=ref_labels,
            )
            return self.active_loss_module(
                feats_q,
                query_labels,
                miner_outputs,
                ref_emb=feats_ref,
                ref_labels=ref_labels,
            )

        if self.active_loss_input_mode == "mask":
            pos_mask_mat = self._build_pos_mask(
                batch_size=feat_dist_mat.shape[0],
                num_refs=feat_dist_mat.shape[1],
                device=feat_dist_mat.device,
            )
            return self._reduce_loss_output(self.active_loss_module(feat_dist_mat, pos_mask_mat))

        return self._compute_weighted_metric_loss(
            scene_name=batch_state["scene_name"],
            feat_dist_mat=feat_dist_mat,
            coords_uav=batch_state["coords_uav"],
            coords_uav_neg_flat=coords_uav_neg_flat,
            has_sat_query=batch_state["has_sat_query"],
            b_uav=batch_state["b_uav"],
            b_sat=batch_state["b_sat"],
        )

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def train(self):
        """Stage 1训练主循环"""
        opt = self.opt
        use_amp = bool(opt.autocast) and getattr(self.device, "type", "") == "cuda"

        print("\n" + "🚀"*40)
        print("开始 Stage 1 训练: Visual Encoder (vis_aggregator)")
        print("🚀"*40 + "\n")

        # 0. 初始化GradScaler（如果使用autocast）
        if use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=True)
            print("✅ 启用混合精度训练 (AMP)")
        elif opt.autocast:
            self.scaler = None
            print("⚠️ autocast=True 但当前设备不是 CUDA，AMP 已禁用。")
        else:
            self.scaler = None

        self._init_loss_modules()

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam', opt=self.opt)

        # 2. 加载checkpoint（如果继续训练）
        begin_epoch = self._load_train_checkpoint(opt.load2train)

        # 3. 初始化日志
        self._init_logger()

        # 4. 初始化数据集
        self._init_datasets(create_train_loader=False)

        # 5. 创建多场景DataLoader
        self._init_multi_scene_dataloader()
        self._after_train_dataloader_init()

        # 6. 初始化 ANCE helper（可选）
        self._init_ance_helper()

        # 7. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")
        if getattr(opt, "val", False):
            val_freq = max(1, int(getattr(opt, "val_freq", 1) or 1))
            self.logger.info(f"Stage 1 Recall评估已启用: val_freq={val_freq}")
        else:
            self.logger.info("Stage 1 Recall评估已禁用: val=False")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            self._maybe_refresh_ance_gallery(epoch)

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                batch_state = self._extract_train_batch(batch)
                satimgs_neg_flat, coords_uav_neg_flat = self._prepare_batch_negatives(batch_state, batch)
                with self._autocast_context():
                    forward_state = self._forward_train_batch(
                        uavimgs=batch_state["uavimgs"],
                        satimgs_pos=batch_state["satimgs_pos"],
                        satimgs_neg_flat=satimgs_neg_flat,
                    )
                    feat_dist_mat = forward_state["feat_dist_mat"]
                    loss = self._compute_train_loss(
                        batch_state=batch_state,
                        feat_dist_mat=feat_dist_mat,
                        coords_uav_neg_flat=coords_uav_neg_flat,
                    )

                # 反向传播
                self.optimizer.zero_grad()
                if use_amp:
                    self.scaler.scale(loss).backward()
                    if not self._optimizer_has_any_grad(self.optimizer):
                        self.logger.warning(
                            "Skip AMP optimizer step at epoch=%d iter=%d because no gradients were produced. loss=%.6f",
                            epoch,
                            it,
                            float(loss.detach().item()),
                        )
                        continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 记录loss和recall
                if it % 10 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss.item(), step)

                    # 计算Recall@1
                    batch_size = forward_state["batch_size"]
                    recall1 = (torch.argmin(feat_dist_mat, dim=-1) == torch.arange(
                        0, batch_size, device=feat_dist_mat.device
                    )).sum() / batch_size

                    # 显示场景信息
                    scene_info = f" [{batch.get('scene_name', 'unknown')}]" if len(opt.scenes_setting['scenes']) > 1 else ""
                    self.logger.info(f'training set{scene_info} recall1={recall1.item():.4f}')
                    self._log_feature_variance_stats(
                        forward_state["feats_q"],
                        forward_state["feats_ref"],
                    )

                step += 1

            # 每个epoch结束后（固定间隔 + 最后一个epoch）
            if ((epoch % opt.save_freq == 0) and (epoch > 0)) or (epoch == num_epochs - 1):
                self._save_checkpoint(
                    epoch,
                    self._build_train_ckpt_modules(),
                    self.optimizer
                )

            if self._should_run_epoch_eval(epoch):
                self.eval_recall(
                    use_train_uav=False,
                    init_datasets=False,
                    load_ckpt=False,
                    restore_train=True,
                    **self._build_eval_configs(),
                )

            # 日志
            self.logger.info(f'loss={loss.item():.6f}')
            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f'epoch {epoch} 完成，耗时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
            self.logger.info('-' * 50)

            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss.item(), epoch)

            # 备份代码
            if epoch == 0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(self.exp_dir2save, self.opt)

        self.logger.info("✅ Stage 1 训练完成！")


    # ------------------------------------------------------------------
    # Evaluation APIs
    # 这组是“把外部参数整理成评测可执行对象”
    # ------------------------------------------------------------------
    def _build_eval_configs(self):
        return {
            "chunk_size_vis": getattr(self.opt, "val_chunk_size", 1024),
            "n_bins_4d": getattr(self.opt, "val_n_bins_4d", None),
            "overlap": getattr(self.opt, "val_overlap", 0.5),
            "scale_mode": getattr(self.opt, "val_scale_mode", "linear"),
            "ref_wo_rot_var": getattr(self.opt, "val_ref_wo_rot_var", True),
            "rot_list": getattr(self.opt, "val_rot_list", None),
            "rot_rad_resolution": getattr(self.opt, "val_rot_rad_resolution", None),
            "ref_wo_scale_var": getattr(self.opt, "val_ref_wo_scale_var", True),
            "query_rot2uniform": getattr(self.opt, "val_query_rot2uniform", True),
            "query_scale2uniform": getattr(self.opt, "val_query_scale2uniform", False),
        }

    def _should_run_epoch_eval(self, epoch):
        if not getattr(self.opt, "val", False):
            return False
        val_freq = max(1, int(getattr(self.opt, "val_freq", 1) or 1))
        return ((epoch + 1) % val_freq == 0) or (epoch == getattr(self.opt, "num_epochs", 0) - 1)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _get_eval_models(self):
        return list(self.param2optimize.values()) + list(self.param2freeze.values())

    def _eval_log(self, msg, eval_log_lines=None):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        if eval_log_lines is not None:
            eval_log_lines.append(msg)

    @staticmethod
    def _resolve_eval_log_path(ckpt_path):
        if not ckpt_path:
            return None
        ckpt_dir = os.path.dirname(ckpt_path)
        base = os.path.basename(ckpt_path)
        match = re.search(r"epoch(\d+)", base)
        ep_tag = match.group(1) if match else "latest"
        return os.path.join(ckpt_dir, f"eval_res_{ep_tag}.txt")

    def _save_eval_log(self, eval_log_lines, ckpt_path):
        if not ckpt_path or not eval_log_lines:
            return
        try:
            out_path = self._resolve_eval_log_path(ckpt_path)
            if out_path is None:
                return
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(line) for line in eval_log_lines))
        except Exception as exc:
            self._eval_log(f"[eval_recall] failed to save log: {exc}")

    @staticmethod
    def _to_jsonable(value):
        if hasattr(value, "__dataclass_fields__"):
            return VisualEncoderTrainer._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(k): VisualEncoderTrainer._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [VisualEncoderTrainer._to_jsonable(v) for v in value]
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _to_pt_bundleable(value):
        if hasattr(value, "__dataclass_fields__"):
            return VisualEncoderTrainer._to_pt_bundleable(asdict(value))
        if isinstance(value, dict):
            return {str(k): VisualEncoderTrainer._to_pt_bundleable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [VisualEncoderTrainer._to_pt_bundleable(v) for v in value]
        if isinstance(value, tuple):
            return tuple(VisualEncoderTrainer._to_pt_bundleable(v) for v in value)
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _normalize_stage1_layout_cfg_for_report(layout_cfg):
        if layout_cfg is None:
            return None
        cfg = layout_cfg if isinstance(layout_cfg, Stage1ReferenceGalleryLayoutConfig) else (
            Stage1ReferenceGalleryLayoutConfig(**layout_cfg)
        )
        return VisualEncoderTrainer._to_jsonable(asdict(cfg))

    @staticmethod
    def _normalize_stage1_feature_cfg_for_report(feature_cfg):
        if feature_cfg is None:
            return None
        cfg = feature_cfg if isinstance(feature_cfg, Stage1ReferenceGalleryFeatureConfig) else (
            Stage1ReferenceGalleryFeatureConfig(**feature_cfg)
        )
        return VisualEncoderTrainer._to_jsonable(asdict(cfg))

    @staticmethod
    def _normalize_stage1_eval_cfg_for_report(eval_cfg):
        if eval_cfg is None:
            return None
        cfg = eval_cfg if isinstance(eval_cfg, Stage1RetrievalEvalConfig) else (
            Stage1RetrievalEvalConfig(**eval_cfg)
        )
        return VisualEncoderTrainer._to_jsonable(asdict(cfg))

    @staticmethod
    def _resolve_stage1_dist_th_meter(sat_dataset, dist_th_nrc):
        if sat_dataset is None or dist_th_nrc is None:
            return None
        if hasattr(sat_dataset, "halfimg_radius_meter") and hasattr(sat_dataset, "halfimg_radius_nrc"):
            nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
            return float(dist_th_nrc) * nrc2meter
        return None

    @staticmethod
    def _resolve_stage1_gallery_eval_paths(save_dir):
        return {
            "manifest": os.path.join(save_dir, "stage1_retrieval_eval_manifest.json"),
            "config_json": os.path.join(save_dir, "stage1_retrieval_eval_config.json"),
            "report_json": os.path.join(save_dir, "stage1_retrieval_eval_report.json"),
            "bundle_pt": os.path.join(save_dir, "stage1_retrieval_eval_bundle.pt"),
            "feats_query_pt": os.path.join(save_dir, "feats_query.pt"),
        }

    @classmethod
    def _build_stage1_gallery_eval_config_payload(
            cls,
            scene_name,
            gallery_bank,
            gallery_save_dir,
            retrieval_eval_cfg,
            ckpt_path,
    ):
        retrieval_eval_cfg_payload = cls._normalize_stage1_eval_cfg_for_report(retrieval_eval_cfg)
        sat_dataset = getattr(gallery_bank, "sat_dataset", None)
        if retrieval_eval_cfg_payload is not None:
            dist_th_nrc = retrieval_eval_cfg_payload.get("dist_th", None)
            if dist_th_nrc is None and sat_dataset is not None and hasattr(sat_dataset, "halfimg_radius_nrc"):
                dist_th_nrc = float(sat_dataset.halfimg_radius_nrc) * 1.1
            retrieval_eval_cfg_payload["dist_th_m"] = cls._resolve_stage1_dist_th_meter(sat_dataset, dist_th_nrc)

        return {
            "schema_version": 1,
            "scene_name": str(scene_name),
            "stage1_ckpt": ckpt_path,
            "load2test": getattr(getattr(gallery_bank, "trainer", None), "opt", None).load2test
            if getattr(getattr(gallery_bank, "trainer", None), "opt", None) is not None and hasattr(gallery_bank.trainer.opt, "load2test")
            else "",
            "gallery_save_dir": gallery_save_dir,
            "layout_cfg": cls._normalize_stage1_layout_cfg_for_report(getattr(gallery_bank, "layout_cfg", None)),
            "feature_cfg": cls._normalize_stage1_feature_cfg_for_report(gallery_bank.meta.get("feature_cfg", None)),
            "retrieval_eval_cfg": retrieval_eval_cfg_payload,
            "gallery_summary": cls._to_jsonable(gallery_bank.summary()),
            "gallery_meta": cls._to_jsonable(gallery_bank.meta),
        }

    @classmethod
    def _build_stage1_gallery_eval_report_payload(cls, eval_res):
        return {
            "schema_version": 1,
            "scene_name": str(eval_res.get("scene_name", "")),
            "report_title": str(eval_res.get("report_title", "")),
            "n_queries": int(eval_res.get("n_queries", 0)),
            "n_eval": int(eval_res.get("n_eval", eval_res.get("n_queries", 0))),
            "k_values": cls._to_jsonable(eval_res.get("k_values", [])),
            "thresholds": cls._to_jsonable(eval_res.get("thresholds", {})),
            "report_meta": cls._to_jsonable(eval_res.get("report_meta", {})),
            "acc_metrics": cls._to_jsonable(eval_res.get("acc_metrics", eval_res.get("metrics", {}))),
            "progressive_acc_metrics": cls._to_jsonable(eval_res.get("progressive_acc_metrics", {})),
            "err_stats": cls._to_jsonable(eval_res.get("err_stats", eval_res.get("errors", {}))),
            "runtime_gallery_summary": cls._to_jsonable(eval_res.get("runtime_gallery_summary", {})),
        }

    @classmethod
    def _save_stage1_gallery_eval_artifacts(
            cls,
            save_dir,
            scene_name,
            gallery_bank,
            eval_res,
            retrieval_eval_cfg,
            ckpt_path,
    ):
        os.makedirs(save_dir, exist_ok=True)
        paths = cls._resolve_stage1_gallery_eval_paths(save_dir)
        config_payload = cls._build_stage1_gallery_eval_config_payload(
            scene_name=scene_name,
            gallery_bank=gallery_bank,
            gallery_save_dir=save_dir,
            retrieval_eval_cfg=retrieval_eval_cfg,
            ckpt_path=ckpt_path,
        )
        report_payload = cls._build_stage1_gallery_eval_report_payload(eval_res)
        manifest_payload = {
            "schema_version": 1,
            "saved_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "scene_name": str(scene_name),
            "gallery_save_dir": save_dir,
            "files": {
                "config_json": os.path.basename(paths["config_json"]),
                "report_json": os.path.basename(paths["report_json"]),
                "bundle_pt": os.path.basename(paths["bundle_pt"]),
                "feats_query_pt": os.path.basename(paths["feats_query_pt"]),
            },
        }
        bundle_payload = {
            "schema_version": 1,
            "config": config_payload,
            "report": report_payload,
            "coords_topk": eval_res.get("coords_topk", None),
            "coords_gt": eval_res.get("coords_gt", None),
        }

        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(cls._to_jsonable(manifest_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(paths["config_json"], "w", encoding="utf-8") as f:
            json.dump(cls._to_jsonable(config_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(paths["report_json"], "w", encoding="utf-8") as f:
            json.dump(cls._to_jsonable(report_payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        torch.save(cls._to_pt_bundleable(bundle_payload), paths["bundle_pt"])
        feats_query = eval_res.get("feats_query", None)
        if feats_query is not None:
            torch.save(feats_query.cpu(), paths["feats_query_pt"])
        return paths

    def _build_eval_layout_cfg(
            self,
            overlap=0.5,
            rot_rad_resolution=None,
            rot_list=None,
            n_bins_4d=None,
            scale_mode="linear",
            ref_wo_rot_var=True,
            ref_wo_scale_var=True,
    ):
        if n_bins_4d is not None:
            return Stage1ReferenceGalleryLayoutConfig(
                mode="n_bins_4d",
                n_bins_4d=n_bins_4d,
                scale_mode=scale_mode,
            )

        if not ref_wo_scale_var:
            raise NotImplementedError("ref_wo_scale_var=False is not supported in overlap-mode evaluation.")

        rot_values = None
        n_rot = 1
        if not ref_wo_rot_var:
            if rot_list is not None:
                rot_values = np.asarray(rot_list, dtype=np.float32).reshape(-1)
                if rot_values.size == 0:
                    raise ValueError("rot_list must be non-empty.")
                rot_values = rot_values.tolist()
            else:
                if rot_rad_resolution is None:
                    raise ValueError("ref_wo_rot_var=False requires rot_list or rot_rad_resolution.")
                rot_rad_resolution = float(rot_rad_resolution)
                if rot_rad_resolution <= 0 or rot_rad_resolution > 2 * np.pi:
                    raise ValueError("rot_rad_resolution must be in (0, 2*pi].")
                rot_values = torch.arange(-np.pi, np.pi, rot_rad_resolution, dtype=torch.float32).tolist()
            n_rot = len(rot_values)

        return Stage1ReferenceGalleryLayoutConfig(
            mode="overlap",
            overlap=float(overlap),
            n_rot=int(n_rot),
            n_scale=1,
            scale_mode=scale_mode,
            rot_values=rot_values,
        )

    @staticmethod
    def _build_eval_feature_cfg(chunk_size_vis):
        return Stage1ReferenceGalleryFeatureConfig(
            chunk_size_vis=int(chunk_size_vis),
            normalize_feats=True,
            build_faiss=True,
            show_progress=False,
        )

    def _build_retrieval_eval_cfg(
            self,
            use_train_uav,
            query_rot2uniform,
            query_scale2uniform,
            rot_th_deg=None,
            scale_ratio_th=None,
    ):
        return Stage1RetrievalEvalConfig(
            use_train_uav=bool(use_train_uav),
            batch_size=int(self.opt.batchsize_uav),
            num_workers=int(getattr(self.opt, "num_worker_eval", 0)),
            query_rot2uniform=bool(query_rot2uniform),
            query_scale2uniform=bool(query_scale2uniform),
            k_values=(1, 5, 10, 20, 50, 256, 512, 1024),
            dist_th=None,
            rot_th_deg=None if rot_th_deg is None else float(rot_th_deg),
            scale_ratio_th=None if scale_ratio_th is None else float(scale_ratio_th),
            max_queries=None,
            gallery_downsample_cfg=None,
            print_results=False,
            report_title="Stage1 Retrieval Eval",
        )


    # ------------------------------------------------------------------
    # Gallery Bank Orchestration
    # 这组是“库的路径、构建、载入、基于库做单场景 retrieval eval”
    # ------------------------------------------------------------------
    def resolve_gallery_bank_save_dir(self, scene_name, layout_cfg, root_dir=None, name_prefix=None, ckpt_path=None):
        if not hasattr(self, "sat_datasets"):
            self._init_datasets(create_train_loader=False)
        if scene_name not in self.sat_datasets:
            raise KeyError(f"Unknown scene_name: {scene_name}")

        sat_dataset = self.sat_datasets[scene_name]
        layout_name_info = Stage1ReferenceGalleryBank.resolve_layout_name_info(
            sat_dataset=sat_dataset,
            layout_cfg=layout_cfg,
        )
        overlap_percent = int(round(float(layout_name_info["overlap"]) * 100.0))
        if overlap_percent == 0:
            overlap_tag = "overlap0"
        else:
            overlap_tag = f"overlap{overlap_percent:03d}"
        n_bins_4d_tag = "x".join(str(int(v)) for v in layout_name_info["n_bins_4d"])
        root_dir = root_dir or os.path.join(project_root, "gen_fm_exps", "gallery_bank_stage1")
        name_prefix = name_prefix or scene_name
        if layout_name_info["mode"] == "overlap":
            layout_tag = overlap_tag
        else:
            layout_tag = f"{layout_name_info['mode']}_{overlap_tag}"
        base_dir = os.path.join(
            root_dir,
            f"{name_prefix}_"
            f"{layout_tag}_"
            f"bins{n_bins_4d_tag}_"
            f"{layout_name_info['scale_mode']}",
        )
        ckpt_tag = self._resolve_gallery_ckpt_tag(
            ckpt_path or getattr(self, "last_eval_ckpt_path", None)
        )
        if ckpt_tag:
            return os.path.join(base_dir, ckpt_tag)
        return base_dir

    def build_or_load_gallery_bank(
            self,
            scene_name,
            layout_cfg,
            feature_cfg=None,
            gallery_save_dir=None,
            load_if_exists=True,
            save_gallery=True,
            init_datasets=True,
            load_ckpt=False,
            gallery_root_dir=None,
            gallery_name_prefix=None,
    ):
        if init_datasets or (not hasattr(self, "sat_datasets")):
            self._init_datasets(create_train_loader=False)
        ckpt_path = self.load_eval_checkpoint(verbose=False) if load_ckpt else getattr(self, "last_eval_ckpt_path", None)

        if scene_name not in self.sat_datasets:
            raise KeyError(f"Unknown scene_name: {scene_name}")
        sat_dataset = self.sat_datasets[scene_name]

        if feature_cfg is None:
            feature_cfg = Stage1ReferenceGalleryFeatureConfig()
        elif not isinstance(feature_cfg, Stage1ReferenceGalleryFeatureConfig):
            feature_cfg = Stage1ReferenceGalleryFeatureConfig(**feature_cfg)

        if gallery_save_dir is None and save_gallery:
            gallery_save_dir = self.resolve_gallery_bank_save_dir(
                scene_name=scene_name,
                layout_cfg=layout_cfg,
                root_dir=gallery_root_dir,
                name_prefix=gallery_name_prefix,
                ckpt_path=ckpt_path,
            )

        coords_path = None if gallery_save_dir is None else os.path.join(gallery_save_dir, "coords_gallery.pt")
        can_load = bool(load_if_exists and coords_path and os.path.exists(coords_path))

        if can_load:
            gallery_bank = Stage1ReferenceGalleryBank.load(
                gallery_save_dir,
                sat_dataset=sat_dataset,
                trainer=self,
                build_faiss=bool(feature_cfg.build_faiss),
            )
            self._eval_log(f"[Gallery Bank] loaded from {gallery_save_dir}")
            if gallery_bank.feats_gallery is None:
                self._eval_log("[Gallery Bank] cached gallery has no features, rebuilding them.")
                gallery_bank.build_features(feature_cfg)
                if gallery_save_dir is not None and save_gallery:
                    gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
        else:
            gallery_bank = Stage1ReferenceGalleryBank(sat_dataset=sat_dataset, trainer=self)
            gallery_bank.build_coords(layout_cfg)
            self._eval_log(
                f"[Gallery Bank] scene={scene_name}, n_points={gallery_bank.coords_gallery.shape[0]}"
            )
            gallery_bank.build_features(feature_cfg)
            if gallery_save_dir is not None and save_gallery:
                gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
                self._eval_log(f"[Gallery Bank] saved to {gallery_save_dir}")

        ckpt_tag = self._resolve_gallery_ckpt_tag(ckpt_path)
        if ckpt_path is not None:
            gallery_bank.meta["ckpt_path"] = ckpt_path
        if ckpt_tag is not None:
            gallery_bank.meta["ckpt_tag"] = ckpt_tag

        return {
            "scene_name": scene_name,
            "gallery_bank": gallery_bank,
            "gallery_save_dir": gallery_save_dir,
            "ckpt_path": ckpt_path,
        }

    def eval_gallery_bank(
            self,
            scene_name,
            layout_cfg,
            feature_cfg=None,
            retrieval_eval_cfg=None,
            gallery_save_dir=None,
            load_if_exists=True,
            save_gallery=True,
            init_datasets=True,
            load_ckpt=False,
            gallery_root_dir=None,
            gallery_name_prefix=None,
    ):
        if retrieval_eval_cfg is not None and not isinstance(retrieval_eval_cfg, Stage1RetrievalEvalConfig):
            retrieval_eval_cfg = Stage1RetrievalEvalConfig(**retrieval_eval_cfg)

        if feature_cfg is None:
            feature_cfg = Stage1ReferenceGalleryFeatureConfig()
        elif not isinstance(feature_cfg, Stage1ReferenceGalleryFeatureConfig):
            feature_cfg = Stage1ReferenceGalleryFeatureConfig(**feature_cfg)

        if retrieval_eval_cfg is not None:
            feature_cfg.build_faiss = True

        gallery_state = self.build_or_load_gallery_bank(
            scene_name=scene_name,
            layout_cfg=layout_cfg,
            feature_cfg=feature_cfg,
            gallery_save_dir=gallery_save_dir,
            load_if_exists=load_if_exists,
            save_gallery=save_gallery,
            init_datasets=init_datasets,
            load_ckpt=load_ckpt,
            gallery_root_dir=gallery_root_dir,
            gallery_name_prefix=gallery_name_prefix,
        )
        gallery_bank = gallery_state["gallery_bank"]

        eval_res = None
        eval_artifact_paths = None
        if retrieval_eval_cfg is not None:
            eval_log_lines = []
            retrieval_evaluator = Stage1RetrievalEvaluator(trainer=self, gallery_bank=gallery_bank, logger=self.logger)
            eval_res = retrieval_evaluator.evaluate_scene(
                scene_name=scene_name,
                eval_cfg=retrieval_eval_cfg,
                eval_log_lines=eval_log_lines,
            )
            self._eval_log(f"[Gallery Eval] n_queries={eval_res['n_queries']}")
            ckpt_path = gallery_state.get("ckpt_path") or getattr(self, "last_eval_ckpt_path", None)
            self._save_eval_log(eval_log_lines, ckpt_path)
            if gallery_state.get("gallery_save_dir", None):
                try:
                    eval_artifact_paths = self._save_stage1_gallery_eval_artifacts(
                        save_dir=gallery_state["gallery_save_dir"],
                        scene_name=scene_name,
                        gallery_bank=gallery_bank,
                        eval_res=eval_res,
                        retrieval_eval_cfg=retrieval_eval_cfg,
                        ckpt_path=ckpt_path,
                    )
                    self._eval_log(
                        f"[Gallery Eval] saved structured report to {gallery_state['gallery_save_dir']}"
                    )
                except Exception as exc:
                    self._eval_log(f"[Gallery Eval] failed to save structured report: {exc}")

        gallery_state["eval_res"] = eval_res
        gallery_state["eval_artifact_paths"] = eval_artifact_paths
        return gallery_state


    # ------------------------------------------------------------------
    # Recall Result Postprocess
    # 这组职责很单一，就是“把 retrieval evaluator 的结果转成旧 recall 风格指标并打印
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_scene_recall_metrics(eval_res, k_values, dist_th, rot_th_deg, gallery_has_rot, sat_dataset=None):
        thresholds = dict(eval_res.get("thresholds", {}))
        report_meta = dict(eval_res.get("report_meta", {}))
        progressive_acc_metrics = eval_res.get("progressive_acc_metrics", None)
        errors = eval_res.get("err_stats", eval_res.get("errors", {}))
        recall_dict = {
            "thresholds": thresholds,
            "report_meta": report_meta,
            "acc_metrics": dict(eval_res.get("acc_metrics", eval_res.get("metrics", {}))),
            "progressive_acc_metrics": (
                dict(progressive_acc_metrics) if isinstance(progressive_acc_metrics, dict) else {}
            ),
        }

        def _append_group(group_metrics, prefix):
            for k in k_values:
                top_key = f"top{int(k)}_acc"
                if top_key in group_metrics:
                    recall_dict[f"{prefix}@{int(k)}"] = float(group_metrics[top_key]) / 100.0

        if isinstance(progressive_acc_metrics, dict) and len(progressive_acc_metrics) > 0:
            _append_group(progressive_acc_metrics.get("dist_recall", {}), "recall")
            if gallery_has_rot or thresholds.get("rot") is not None:
                _append_group(progressive_acc_metrics.get("dist_rot_recall", {}), "recall_rot")
            _append_group(progressive_acc_metrics.get("dist_rot_scale_recall", {}), "recall_rot_scale")
            if "legacy_acc_metrics_source" in report_meta:
                recall_dict["legacy_acc_metrics_source"] = str(report_meta["legacy_acc_metrics_source"])
            if "progressive_acc_metric_sources" in report_meta:
                recall_dict["progressive_acc_metric_sources"] = dict(report_meta["progressive_acc_metric_sources"])
            if "progressive_recall_policy" in report_meta:
                recall_dict["progressive_recall_policy"] = dict(report_meta["progressive_recall_policy"])
        else:
            coords_topk = eval_res["coords_topk"]
            coords_gt = eval_res["coords_gt"]
            metrics_nrc, _ = compute_topk_acc_from_coords(
                coords_topk,
                coords_gt,
                dist_th=dist_th,
                rot_th_deg=None,
                scale_ratio_th=None,
                k_values=k_values,
            )
            recall_dict.update({
                f"recall@{int(k)}": float(metrics_nrc[f"top{int(k)}_acc"]) / 100.0
                for k in k_values
            })
            if gallery_has_rot:
                metrics_rot, _ = compute_topk_acc_from_coords(
                    coords_topk,
                    coords_gt,
                    dist_th=dist_th,
                    rot_th_deg=rot_th_deg,
                    scale_ratio_th=None,
                    k_values=k_values,
                )
                recall_dict.update({
                    f"recall_rot@{int(k)}": float(metrics_rot[f"top{int(k)}_acc"]) / 100.0
                    for k in k_values
                })

        if "mean_dist_err_top1" in errors:
            recall_dict["top1_dist_nrc_mean"] = float(errors["mean_dist_err_top1"])
        if "median_dist_err_top1" in errors:
            recall_dict["top1_dist_nrc_median"] = float(errors["median_dist_err_top1"])
        if "mean_rot_err_top1" in errors:
            recall_dict["top1_rot_deg_mean"] = float(errors["mean_rot_err_top1"])
        if "median_rot_err_top1" in errors:
            recall_dict["top1_rot_deg_median"] = float(errors["median_rot_err_top1"])
        if "mean_scale_ratio_top1" in errors:
            recall_dict["top1_scale_ratio_mean"] = float(errors["mean_scale_ratio_top1"])
        if "median_scale_ratio_top1" in errors:
            recall_dict["top1_scale_ratio_median"] = float(errors["median_scale_ratio_top1"])
        if sat_dataset is not None and hasattr(sat_dataset, "halfimg_radius_meter") and hasattr(sat_dataset, "halfimg_radius_nrc"):
            nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
            if "top1_dist_nrc_mean" in recall_dict:
                recall_dict["top1_dist_meter_mean"] = float(recall_dict["top1_dist_nrc_mean"] * nrc2meter)
            if "top1_dist_nrc_median" in recall_dict:
                recall_dict["top1_dist_meter_median"] = float(recall_dict["top1_dist_nrc_median"] * nrc2meter)
        return recall_dict

    def _log_scene_recall_metrics(self, scene_name, recall_dict, n_queries, thresholds, eval_log_lines):
        dist_th = float(thresholds["norm_dist"])
        rot_th_deg = thresholds.get("rot", None)
        scale_ratio_th = thresholds.get("scale_ratio", None)

        info2log_nrc = " | ".join(
            [f"nrc:R@{int(k)}={recall_dict[f'recall@{int(k)}'] * 100:.3f}%" for k in sorted(
                int(key.split("@")[1]) for key in recall_dict if key.startswith("recall@")
            )]
        )
        self._eval_log(
            f"[Scene: {scene_name}] {info2log_nrc} "
            f"(N={n_queries}, nrc_thr={float(dist_th):.3f})",
            eval_log_lines,
        )
        if any(key.startswith("recall_rot@") for key in recall_dict):
            info2log_rot = " | ".join(
                [f"nrc+rot:R@{int(k)}={recall_dict[f'recall_rot@{int(k)}'] * 100:.3f}%" for k in sorted(
                    int(key.split('@')[1]) for key in recall_dict if key.startswith("recall_rot@")
                )]
            )
            self._eval_log(
                f"[Scene: {scene_name}] {info2log_rot} "
                f"(N={n_queries}, nrc_thr={float(dist_th):.3f}, "
                f"rot_thr={'None' if rot_th_deg is None else f'{float(rot_th_deg):.1f}deg'})",
                eval_log_lines,
            )
        if any(key.startswith("recall_rot_scale@") for key in recall_dict):
            info2log_scale = " | ".join(
                [f"nrc+rot+scale:R@{int(k)}={recall_dict[f'recall_rot_scale@{int(k)}'] * 100:.3f}%" for k in sorted(
                    int(key.split('@')[1]) for key in recall_dict if key.startswith("recall_rot_scale@")
                )]
            )
            scale_msg = "None(alias_of_nrc+rot)" if scale_ratio_th is None else f"{float(scale_ratio_th):.3f}x"
            self._eval_log(
                f"[Scene: {scene_name}] {info2log_scale} "
                f"(N={n_queries}, nrc_thr={float(dist_th):.3f}, "
                f"rot_thr={'None' if rot_th_deg is None else f'{float(rot_th_deg):.1f}deg'}, "
                f"scale_thr={scale_msg})",
                eval_log_lines,
            )
        if "top1_dist_nrc_mean" in recall_dict:
            meter_msg = ""
            if "top1_dist_meter_mean" in recall_dict and "top1_dist_meter_median" in recall_dict:
                meter_msg = (
                    f" | meter_mean={recall_dict['top1_dist_meter_mean']:.3f}m"
                    f" | meter_median={recall_dict['top1_dist_meter_median']:.3f}m"
                )
            self._eval_log(
                f"[Scene: {scene_name}] top1_dist_mean={recall_dict['top1_dist_nrc_mean']:.6f} nrc"
                f" | top1_dist_median={recall_dict['top1_dist_nrc_median']:.6f} nrc"
                f"{meter_msg}",
                eval_log_lines,
            )
        if "top1_rot_deg_mean" in recall_dict:
            self._eval_log(
                f"[Scene: {scene_name}] top1_rot_mean={recall_dict['top1_rot_deg_mean']:.6f} deg"
                f" | top1_rot_median={recall_dict['top1_rot_deg_median']:.6f} deg",
                eval_log_lines,
            )
        if "top1_scale_ratio_mean" in recall_dict:
            self._eval_log(
                f"[Scene: {scene_name}] top1_scale_ratio_mean={recall_dict['top1_scale_ratio_mean']:.6f}x"
                f" | top1_scale_ratio_median={recall_dict['top1_scale_ratio_median']:.6f}x",
                eval_log_lines,
            )


    # ------------------------------------------------------------------
    # High-level Recall APIs
    # 这组是给 trainer 外部调用的高层入口
    # ------------------------------------------------------------------
    def eval_recall(self, use_train_uav=False, gallery_cfg=None, query_cfg=None,
                    overlap=0.5, chunk_size_vis=1024,
                    ref_wo_rot_var=True, ref_wo_scale_var=True,
                    rot_rad_resolution=None, rot_list=None, n_bins_4d=None, scale_mode="linear",
                    query_rot2uniform=False, query_scale2uniform=False,
                    init_datasets=True, load_ckpt=True, restore_train=True):
        """Stage 1 Recall 评估入口，内部采用 GalleryBank + RetrievalEvaluator。"""
        chunk_size_vis = self._cfg_get(gallery_cfg, "chunk_size_vis", chunk_size_vis)
        n_bins_4d = self._cfg_get(gallery_cfg, "n_bins_4d", n_bins_4d)
        scale_mode = self._cfg_get(gallery_cfg, "scale_mode", scale_mode)
        sampling_cfg = self._cfg_get(gallery_cfg, "sampling_cfg", None)
        overlap = self._cfg_get(sampling_cfg, "overlap", overlap)
        ref_wo_rot_var = self._cfg_get(sampling_cfg, "ref_wo_rot_var", ref_wo_rot_var)
        rot_list = self._cfg_get(sampling_cfg, "rot_list", rot_list)
        rot_rad_resolution = self._cfg_get(sampling_cfg, "rot_rad_resolution", rot_rad_resolution)
        ref_wo_scale_var = self._cfg_get(sampling_cfg, "ref_wo_scale_var", ref_wo_scale_var)
        query_rot2uniform = self._cfg_get(query_cfg, "query_rot2uniform", query_rot2uniform)
        query_scale2uniform = self._cfg_get(query_cfg, "query_scale2uniform", query_scale2uniform)

        if init_datasets or (not hasattr(self, "sat_datasets")):
            self._init_datasets(create_train_loader=False)

        ckpt_path = self.load_eval_checkpoint(verbose=False) if load_ckpt else None
        eval_log_lines = []
        layout_cfg = self._build_eval_layout_cfg(
            overlap=overlap,
            rot_rad_resolution=rot_rad_resolution,
            rot_list=rot_list,
            n_bins_4d=n_bins_4d,
            scale_mode=scale_mode,
            ref_wo_rot_var=ref_wo_rot_var,
            ref_wo_scale_var=ref_wo_scale_var,
        )
        feature_cfg = self._build_eval_feature_cfg(chunk_size_vis=chunk_size_vis)
        rot_th_deg = float(torch.rad2deg(torch.tensor(torch.pi / 18.0 * 1.1)).item())
        retrieval_eval_cfg = self._build_retrieval_eval_cfg(
            use_train_uav=use_train_uav,
            query_rot2uniform=query_rot2uniform,
            query_scale2uniform=query_scale2uniform,
            rot_th_deg=rot_th_deg,
        )

        if not restore_train:
            for model in self._get_eval_models():
                model.eval()

        self._eval_log("\n" + "=" * 80, eval_log_lines)
        self._eval_log("开始 Stage 1 Recall 评估", eval_log_lines)
        self._eval_log(
            f"use_train_uav={bool(use_train_uav)}, chunk_size_vis={int(chunk_size_vis)}, "
            f"layout_mode={layout_cfg.mode}, n_bins_4d={layout_cfg.n_bins_4d}",
            eval_log_lines,
        )
        self._eval_log(
            f"gallery_sampling: overlap={float(overlap):.4f}, ref_wo_rot_var={bool(ref_wo_rot_var)}, "
            f"ref_wo_scale_var={bool(ref_wo_scale_var)}, rot_rad_resolution={rot_rad_resolution}, "
            f"rot_list={rot_list}",
            eval_log_lines,
        )
        self._eval_log(
            f"query_transform: query_rot2uniform={bool(query_rot2uniform)}, "
            f"query_scale2uniform={bool(query_scale2uniform)}",
            eval_log_lines,
        )
        self._eval_log("=" * 80, eval_log_lines)

        results_all = {}
        for scene in self.opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.sat_datasets[scene_name]
            layout_summary = Stage1ReferenceGalleryBank.estimate_layout_summary(
                sat_dataset=sat_dataset,
                layout_cfg=layout_cfg,
            )
            self._eval_log(
                f"[Scene: {scene_name}] gallery grid={layout_summary['n_bins_4d']} "
                f"({layout_summary['total_points_4d']} pts, n_rot={layout_summary['n_rot']}, "
                f"n_scale={layout_summary['n_scale']}), crop={layout_summary['crop_size_px']}px, "
                f"overlap={layout_summary['overlap']:.4f}, scale={layout_summary['gallery_scale']:.3f}, "
                f"scale_mode={layout_summary['scale_mode']}",
                eval_log_lines,
            )

            gallery_bank = Stage1ReferenceGalleryBank(sat_dataset=sat_dataset, trainer=self)
            gallery_bank.build_coords(layout_cfg)
            gallery_bank.build_features(feature_cfg)
            retrieval_evaluator = Stage1RetrievalEvaluator(
                trainer=self,
                gallery_bank=gallery_bank,
                logger=self.logger,
            )
            eval_res = retrieval_evaluator.evaluate_scene(
                scene_name=scene_name,
                eval_cfg=retrieval_eval_cfg,
            )

            dist_th = float(eval_res["thresholds"]["norm_dist"])
            gallery_has_rot = bool(gallery_bank.meta.get("gallery_has_rot", False))
            recall_dict = self._compute_scene_recall_metrics(
                eval_res=eval_res,
                k_values=retrieval_eval_cfg.k_values,
                dist_th=dist_th,
                rot_th_deg=rot_th_deg,
                gallery_has_rot=gallery_has_rot,
                sat_dataset=sat_dataset,
            )
            results_all[scene_name] = recall_dict
            self._log_scene_recall_metrics(
                scene_name=scene_name,
                recall_dict=recall_dict,
                n_queries=eval_res["n_queries"],
                thresholds=eval_res["thresholds"],
                eval_log_lines=eval_log_lines,
            )

        self._eval_log("=" * 80, eval_log_lines)
        self._eval_log("Recall 评估完成", eval_log_lines)
        self._eval_log("=" * 80, eval_log_lines)
        self._save_eval_log(eval_log_lines, ckpt_path)
        return results_all

    def get_eval_gallery_sampling_point_config(
            self,
            scene_name=None,
            overlap=0.5,
            ref_wo_rot_var=True,
            rot_list=None,
            rot_rad_resolution=None,
            ref_wo_scale_var=True,
            init_datasets=True,
            as_dict=False,
    ):
        """估算当前评测 layout 的参考库密度，不执行完整检索评估。"""
        if init_datasets or (not hasattr(self, "sat_datasets")):
            self._init_datasets(create_train_loader=False)

        layout_cfg = self._build_eval_layout_cfg(
            overlap=overlap,
            rot_rad_resolution=rot_rad_resolution,
            rot_list=rot_list,
            n_bins_4d=None,
            scale_mode="linear",
            ref_wo_rot_var=ref_wo_rot_var,
            ref_wo_scale_var=ref_wo_scale_var,
        )

        def _estimate_for_scene(name):
            sat_dataset = self.sat_datasets[name]
            return Stage1ReferenceGalleryBank.estimate_layout_summary(
                sat_dataset=sat_dataset,
                layout_cfg=layout_cfg,
            )

        if scene_name is None:
            results = {
                scene["name"]: _estimate_for_scene(scene["name"])
                for scene in self.opt.scenes_setting["scenes"]
            }
            return results if as_dict else results

        result = _estimate_for_scene(scene_name)
        return result if as_dict else result


if __name__ == "__main__":
    def _parse_bool_arg(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "n"}

    # 如果没有指定配置文件，使用 stage1 的默认配置
    if '--p_yaml' not in ' '.join(sys.argv):
        sys.argv.extend(['--p_yaml', 'trainer_depends/configs/stage1_visual_encoder_visloc4exp.yaml'])
        # sys.argv.extend(['--p_yaml', 'trainer_depends/configs/stage1_visual_encoder_wingtra.yaml'])
        # sys.argv.extend(['--p_yaml', 'trainer_depends/configs/stage1_visual_encoder.yaml'])

    default_test_only = False
    default_test_mode = "gallery_bank"
    default_scene_name = ""
    default_exp_name_override = ""

    # 调试入口参数：可直接修改默认值，便于在 PyCharm Debug 中切换 train / test 行为。
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid conflict with get_parse
    parser.add_argument(
        '--test_only',
        nargs='?',
        const=True,
        default=default_test_only,
        type=_parse_bool_arg,
        help='是否只运行测试模式；可写 --test_only 或 --test_only true/false',
    )
    parser.add_argument(
        '--test_mode',
        type=str,
        default=default_test_mode,
        choices=('gallery_bank', 'eval_recall'),
        help='测试模式：gallery_bank 为显式建库评测，eval_recall 为复用 trainer.eval_recall 的高层入口',
    )
    parser.add_argument(
        '--scene_name',
        type=str,
        default=default_scene_name,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；为空则默认取第一个场景。',
    )
    parser.add_argument(
        '--exp_name_override',
        type=str,
        default=default_exp_name_override,
        help='可选：覆盖最终 opt.exp_name，便于本地调试时快速区分实验目录。',
    )
    parser.add_argument(
        '--gallery_root_dir',
        type=str,
        default="",
        help='仅在 --test_only --test_mode=gallery_bank 时使用；若指定则覆盖 gallery 输出根目录。',
    )
    parser.add_argument(
        '--gallery_overlap',
        type=float,
        default=0.5,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；gallery overlap 配置。',
    )
    parser.add_argument(
        '--gallery_n_rot',
        type=int,
        default=36,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；gallery rotation bin 数。',
    )
    parser.add_argument(
        '--gallery_n_scale',
        type=int,
        default=4,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；gallery scale bin 数。',
    )
    parser.add_argument(
        '--gallery_scale_mode',
        type=str,
        default="linear",
        choices=("linear", "log"),
        help='仅在 --test_only --test_mode=gallery_bank 时使用；gallery scale 采样模式。',
    )
    parser.add_argument(
        '--dist_th_meter',
        type=float,
        default=None,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；若指定则按米制阈值换算为归一化 dist_th。',
    )
    parser.add_argument(
        '--eval_query_subset_mode',
        type=str,
        default="",
        choices=("", "segment", "interval", "random"),
        help='仅在 --test_only --test_mode=gallery_bank 时使用；对当前 query split 再做一次子切分。',
    )
    parser.add_argument(
        '--eval_query_subset_ratio',
        type=float,
        default=None,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；对子切分的 train_ratio，例如 0.5。',
    )
    parser.add_argument(
        '--eval_query_subset_take',
        type=str,
        default="test",
        choices=("train", "test"),
        help='仅在 --test_only --test_mode=gallery_bank 时使用；对子切分后保留哪一半。',
    )
    parser.add_argument(
        '--eval_query_subset_seed',
        type=int,
        default=2026,
        help='仅在 --test_only --test_mode=gallery_bank 时使用；random 子切分时的随机种子。',
    )
    args, remaining_argv = parser.parse_known_args()

    # 是否直接指定实验名称
    sys.argv = [sys.argv[0]] + remaining_argv
    from trainer_depends.config.parser import get_parse
    opt = get_parse()
    if args.exp_name_override:
        opt.exp_name = args.exp_name_override

    trainer = VisualEncoderTrainer(opt=opt)
    if not args.test_only:
        trainer.train()
    else:
        if args.test_mode == 'eval_recall':
            trainer.eval_recall(
                use_train_uav=True,
                init_datasets=True,
                load_ckpt=True,
                restore_train=True,
                **trainer._build_eval_configs(),
            )
        elif args.test_mode == 'gallery_bank':
            if not hasattr(trainer, "sat_datasets"):
                trainer._init_datasets(create_train_loader=False)

            scene_name = str(args.scene_name).strip()
            if not scene_name:
                scene_name = next(iter(trainer.sat_datasets.keys()))
            if scene_name not in trainer.sat_datasets:
                available = ", ".join(sorted(trainer.sat_datasets.keys()))
                raise KeyError(f"Unknown scene_name: {scene_name}. Available scenes: {available}")
            sat_dataset = trainer.sat_datasets[scene_name]

            trainer.load_eval_checkpoint()
            gallery_layout_cfg = Stage1ReferenceGalleryLayoutConfig(
                mode="overlap",
                overlap=float(args.gallery_overlap),
                n_rot=int(args.gallery_n_rot),
                n_scale=int(args.gallery_n_scale),
                scale_mode=str(args.gallery_scale_mode),
            )

            gallery_feature_cfg = trainer._build_eval_feature_cfg(chunk_size_vis=1024 + 256)
            gallery_feature_cfg.build_faiss = True
            gallery_feature_cfg.show_progress = True

            retrieval_eval_cfg = trainer._build_retrieval_eval_cfg(
                use_train_uav=False,
                query_rot2uniform=False,
                query_scale2uniform=False,
            )
            retrieval_eval_cfg.k_values = (1, 5, 10, 20, 50, 128, 256, 512, 1024)
            dist_th_meter = args.dist_th_meter
            if dist_th_meter is not None:
                dist_th = (
                    float(dist_th_meter)
                    / max(float(sat_dataset.halfimg_radius_meter), 1e-8)
                    * float(sat_dataset.halfimg_radius_nrc)
                )
            else:
                dist_th = float(sat_dataset.halfimg_radius_nrc) * 1.1 * 0.5
            retrieval_eval_cfg.dist_th = dist_th
            retrieval_eval_cfg.rot_th_deg = 11 * 0.5
            retrieval_eval_cfg.scale_ratio_th = 1.15
            if args.eval_query_subset_mode:
                if args.eval_query_subset_ratio is None:
                    raise ValueError("--eval_query_subset_mode requires --eval_query_subset_ratio")
                retrieval_eval_cfg.query_subset_mode = str(args.eval_query_subset_mode)
                retrieval_eval_cfg.query_subset_train_ratio = float(args.eval_query_subset_ratio)
                retrieval_eval_cfg.query_subset_take = str(args.eval_query_subset_take)
                retrieval_eval_cfg.query_subset_random_seed = int(args.eval_query_subset_seed)
            retrieval_eval_cfg.print_results = True
            retrieval_eval_cfg.report_title = "Stage1 Retrieval Eval"

            gallery_state = trainer.eval_gallery_bank(
                scene_name=scene_name,
                layout_cfg=gallery_layout_cfg,
                feature_cfg=gallery_feature_cfg,
                retrieval_eval_cfg=retrieval_eval_cfg,
                gallery_save_dir=None,
                load_if_exists=True,
                save_gallery=True,
                init_datasets=False,
                load_ckpt=False,
                gallery_root_dir=(str(args.gallery_root_dir).strip() or None),
                gallery_name_prefix=f"{scene_name}",
            )
            print(f"[Gallery Demo] save_dir={gallery_state['gallery_save_dir']}")
            if gallery_state.get("eval_artifact_paths", None):
                print(f"[Gallery Demo] eval_report_json={gallery_state['eval_artifact_paths']['report_json']}")
                print(f"[Gallery Demo] eval_bundle_pt={gallery_state['eval_artifact_paths']['bundle_pt']}")
        else:
            raise ValueError(f"Unknown test_mode: {args.test_mode}")
