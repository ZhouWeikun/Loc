#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2 APR MLP trainer.

This is a PoseNet-style baseline for the existing Stage-2 data path:
image feature -> pure MLP -> normalized absolute pose.

Default target parameterization is euc5d:
    [nr_norm, nc_norm, cos(theta), sin(theta), log_scale_norm]
which avoids the rotation discontinuity of directly regressing theta.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as TF
import tqdm


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.multi_mlp import create_mlp, init_weights
from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.base.components import NetworkComponents
from trainer_depends.config.parser import get_parse, print_config_summary
from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
from trainers.stage2_INGP import GridHashFitTrainer, _find_latest_epoch_ckpt


class APRMLP(nn.Module):
    """Plain MLP regressor for image features -> normalized pose."""

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=512,
            num_layers=3,
            norm_type="layer",
            dropout_p=0.1,
            activation="relu",
            output_tanh=False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_fn = activation_map.get(str(activation).lower())
        if activation_fn is None:
            raise ValueError(f"Unsupported APR MLP activation: {activation}")

        hidden_dims = [int(hidden_dim)] * max(0, int(num_layers) - 1)
        dims = [int(input_dim)] + hidden_dims + [int(output_dim)]
        norm_type = None if str(norm_type).lower() in {"", "none", "null"} else norm_type
        dropout_p = None if dropout_p is None else float(dropout_p)

        self.net = create_mlp(
            dims,
            activation_fn=activation_fn,
            norm_type=norm_type,
            dropout_p=dropout_p,
        )
        self.net.apply(lambda m: init_weights(m, method="kaiming", nonlinearity="relu"))
        self.output_tanh = bool(output_tanh)

        print("APR MLP:")
        print(f"   dims: {dims}")
        print(f"   activation: {activation_fn.__name__}")
        print(f"   norm: {norm_type}")
        print(f"   dropout: {dropout_p}")
        print(f"   output_tanh: {self.output_tanh}")

    def forward(self, x):
        out = self.net(x)
        if self.output_tanh:
            out = torch.tanh(out)
        return out


class APRMLPTrainer(BaseTrainer):
    """PoseNet-style APR baseline using frozen Stage-1 image features."""

    _apply_inherit_stage1_yaml = staticmethod(GridHashFitTrainer._apply_inherit_stage1_yaml)
    _collect_stage2_explicit_keys = staticmethod(GridHashFitTrainer._collect_stage2_explicit_keys)
    _load_yaml_dict = staticmethod(GridHashFitTrainer._load_yaml_dict)
    _collect_declared_keys_from_cfg = staticmethod(GridHashFitTrainer._collect_declared_keys_from_cfg)
    _merge_section_key_sets = staticmethod(GridHashFitTrainer._merge_section_key_sets)

    def __init__(self, opt=None):
        should_print_final_config = (opt is None) or not bool(getattr(opt, "_config_summary_printed", False))
        if opt is None:
            opt = get_parse(print_summary=False)
        opt = self._apply_inherit_stage1_yaml(opt)
        if should_print_final_config:
            print_config_summary(opt, header="最终生效配置:")

        super().__init__(opt)
        self._init_networks()

        if self.opt.load_stage1_ckpt:
            self._load_stage1_checkpoint()

        self._setup_trainable_params()

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"

    def _init_networks(self):
        print("\n" + "=" * 80)
        print("Initializing Stage 2 APR MLP networks")
        print("=" * 80)

        components = NetworkComponents(self.opt, self.device)
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel
        self.vis_aggregator = components.create_aggregator(self.feat_patch_dim)
        self.feat_q_dim = int(getattr(self.vis_aggregator, "output_dim", self.feat_patch_dim))

        self.apr_target_mode = str(getattr(self.opt, "apr_target_mode", "euc5d")).lower()
        if self.apr_target_mode not in {"euc5d", "raw4d_norm"}:
            raise ValueError("apr_target_mode must be 'euc5d' or 'raw4d_norm'")
        output_dim = 5 if self.apr_target_mode == "euc5d" else 4

        self.apr_mlp = APRMLP(
            input_dim=self.feat_q_dim,
            output_dim=output_dim,
            hidden_dim=int(getattr(self.opt, "apr_mlp_hidden_dim", 512)),
            num_layers=int(getattr(self.opt, "apr_mlp_num_layers", 3)),
            norm_type=getattr(self.opt, "apr_mlp_norm", "layer"),
            dropout_p=float(getattr(self.opt, "apr_mlp_dropout", 0.1)),
            activation=getattr(self.opt, "apr_mlp_activation", "relu"),
            output_tanh=bool(getattr(self.opt, "apr_output_tanh", False)),
        ).to(self.device)

        print(f"   image feature dim: {self.feat_q_dim}")
        print(f"   target mode: {self.apr_target_mode}")
        print(f"   output dim: {output_dim}")
        print("=" * 80 + "\n")

    def _load_stage1_checkpoint(self):
        print(f"\nLoading Stage 1 checkpoint: {self.opt.load_stage1_ckpt}")
        self._load_checkpoint(
            self.opt.load_stage1_ckpt,
            {
                "vis_encoder": self.vis_encoder,
                "vis_aggregator": self.vis_aggregator,
            },
        )
        print("Stage 1 model loaded.\n")

    @staticmethod
    def _count_module_params(module):
        total_params = sum(param.numel() for param in module.parameters())
        trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
        return total_params, trainable_params

    def _setup_trainable_params(self):
        for module in [self.vis_encoder, self.vis_aggregator]:
            for param in module.parameters():
                param.requires_grad = False

        self.param2freeze = {
            "vis_encoder": self.vis_encoder,
            "vis_aggregator": self.vis_aggregator,
        }
        self.param2optimize = {
            "apr_mlp": self.apr_mlp,
        }

        print("Parameter configuration:")
        for module_name, module in {**self.param2freeze, **self.param2optimize}.items():
            total_params, trainable_params = self._count_module_params(module)
            status = "trainable" if trainable_params > 0 else "frozen"
            print(
                f"  {module_name}: {status}, "
                f"trainable_params={trainable_params:,}, total_params={total_params:,}"
            )
        print("")

    def _make_train_checkpoint_modules(self):
        modules = {"apr_mlp": self.apr_mlp}
        if getattr(self.opt, "autocast", False) and hasattr(self, "scaler") and self.scaler is not None:
            modules["amp_scaler"] = self.scaler
        return modules

    def _make_optimizer(self):
        optimizer_name = str(getattr(self.opt, "apr_optimizer", "adamw")).lower()
        lr = float(getattr(self.opt, "apr_lr", 1e-3))
        weight_decay = float(getattr(self.opt, "apr_weight_decay", 1e-4))
        betas = tuple(getattr(self.opt, "apr_betas", (0.9, 0.999)))
        params = [p for p in self.apr_mlp.parameters() if p.requires_grad]

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            raise ValueError(f"Unsupported apr_optimizer: {optimizer_name}")

        print(f"APR optimizer: {optimizer_name}, lr={lr:.2e}, weight_decay={weight_decay:.2e}")
        return optimizer

    def _ensure_coord_normer(self):
        if not hasattr(self, "coord_normer") or self.coord_normer is None:
            self.coord_normer = CoordsNormProcessor(self.sat_dataset)

    def _raw_to_norm4d(self, coords_raw_4d):
        self._ensure_coord_normer()
        if coords_raw_4d.device != self.coord_normer.device:
            coords_raw_4d = coords_raw_4d.to(self.coord_normer.device)

        nrc_raw = coords_raw_4d[..., 0:2]
        nrc_norm = 2.0 * (nrc_raw - self.coord_normer.nrc_min) / self.coord_normer.nrc_diff - 1.0
        theta_norm = coords_raw_4d[..., 2:3] / torch.pi
        scale_raw = coords_raw_4d[..., 3:4]
        scale_raw = torch.clamp(
            scale_raw,
            min=torch.exp(self.coord_normer.scale_log_min),
            max=torch.exp(self.coord_normer.scale_log_max),
        )
        scale_log = torch.log(scale_raw)
        scale_norm = 2.0 * (scale_log - self.coord_normer.scale_log_min) / self.coord_normer.scale_log_diff - 1.0
        return torch.cat([nrc_norm, theta_norm, scale_norm], dim=-1)

    def _norm4d_to_raw(self, coords_norm_4d):
        self._ensure_coord_normer()
        if coords_norm_4d.device != self.coord_normer.device:
            coords_norm_4d = coords_norm_4d.to(self.coord_normer.device)

        coords_norm_4d = coords_norm_4d.clone()
        coords_norm_4d = torch.clamp(coords_norm_4d, -1.0, 1.0)
        nrc_raw = (coords_norm_4d[..., 0:2] + 1.0) / 2.0 * self.coord_normer.nrc_diff + self.coord_normer.nrc_min
        theta_raw = coords_norm_4d[..., 2:3] * torch.pi
        scale_log = (coords_norm_4d[..., 3:4] + 1.0) / 2.0 * self.coord_normer.scale_log_diff + self.coord_normer.scale_log_min
        scale_raw = torch.exp(scale_log)
        return torch.cat([nrc_raw, theta_raw, scale_raw], dim=-1)

    def _make_target(self, coords_raw_4d):
        self._ensure_coord_normer()
        coords_raw_4d = coords_raw_4d.to(self.device, dtype=torch.float32)
        if self.apr_target_mode == "euc5d":
            target = self.coord_normer.raw_to_norm(coords_raw_4d, append_linear_rot=False)
        else:
            target = self._raw_to_norm4d(coords_raw_4d)
        return target.to(self.device)

    def _postprocess_prediction(self, pred):
        pred = pred.clone()
        if self.apr_target_mode == "euc5d":
            pred[..., 2:4] = TF.normalize(pred[..., 2:4], dim=-1, eps=1e-6)
            if bool(getattr(self.opt, "apr_eval_clamp", True)):
                pred[..., 0:2] = torch.clamp(pred[..., 0:2], -1.0, 1.0)
                pred[..., 4:5] = torch.clamp(pred[..., 4:5], -1.0, 1.0)
        else:
            if bool(getattr(self.opt, "apr_eval_clamp", True)):
                pred = torch.clamp(pred, -1.0, 1.0)
        return pred

    def _prediction_to_raw(self, pred):
        pred = self._postprocess_prediction(pred)
        if self.apr_target_mode == "euc5d":
            return self.coord_normer.norm_to_raw(pred)
        return self._norm4d_to_raw(pred)

    def _compute_loss(self, pred, target):
        if self.apr_target_mode == "euc5d":
            pred_rot = TF.normalize(pred[:, 2:4], dim=-1, eps=1e-6)
            loss_nrc = TF.mse_loss(pred[:, 0:2], target[:, 0:2])
            loss_rot = TF.mse_loss(pred_rot, target[:, 2:4])
            loss_scale = TF.mse_loss(pred[:, 4:5], target[:, 4:5])
        else:
            loss_nrc = TF.mse_loss(pred[:, 0:2], target[:, 0:2])
            loss_rot = TF.mse_loss(pred[:, 2:3], target[:, 2:3])
            loss_scale = TF.mse_loss(pred[:, 3:4], target[:, 3:4])

        w_nrc = float(getattr(self.opt, "apr_loss_weight_nrc", 1.0))
        w_rot = float(getattr(self.opt, "apr_loss_weight_rot", 1.0))
        w_scale = float(getattr(self.opt, "apr_loss_weight_scale", 1.0))
        loss = w_nrc * loss_nrc + w_rot * loss_rot + w_scale * loss_scale
        return loss, {
            "loss_nrc": loss_nrc.detach(),
            "loss_rot": loss_rot.detach(),
            "loss_scale": loss_scale.detach(),
        }

    def _extract_image_features(self, imgs):
        feats = self._get_feats_fm_imgs(imgs)
        if bool(getattr(self.opt, "apr_normalize_input_feat", True)):
            feats = TF.normalize(feats, dim=-1)
        return feats

    def _make_uav_loader(self, dataset, shuffle, drop_last, batch_size=None):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or self.opt.batchsize_uav,
            num_workers=self.opt.num_worker,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=(self.opt.num_worker > 0),
        )

    def _set_train_modes(self):
        self.apr_mlp.train()
        self.vis_encoder.eval()
        self.vis_aggregator.eval()

    def train(self):
        opt = self.opt
        print("\n" + "=" * 80)
        print("Starting Stage 2 APR MLP training")
        print("=" * 80 + "\n")

        if not getattr(opt, "load_stage1_ckpt", ""):
            msg = "APR MLP training needs a trained Stage-1 image encoder checkpoint."
            if bool(getattr(opt, "apr_require_stage1_ckpt", True)):
                raise ValueError(f"{msg} Set exp_setting.load_stage1_ckpt or apr_require_stage1_ckpt=False.")
            print(f"Warning: {msg} Continuing with current encoder weights.")

        amp_enabled = bool(getattr(opt, "autocast", False) and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if amp_enabled:
            print("AMP enabled.")

        self.optimizer = self._make_optimizer()
        train_ckpt_modules = self._make_train_checkpoint_modules()
        begin_epoch = self._load_checkpoint(
            opt.load2train,
            train_ckpt_modules,
            self.optimizer,
            mode="train",
        )

        self._init_logger()
        self._init_datasets(create_train_loader=False)
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        self.uav_dataloader_train = self._make_uav_loader(
            self.uav_dataset_train,
            shuffle=True,
            drop_last=True,
        )
        self.uav_dataloader_test = self._make_uav_loader(
            self.uav_dataset_test,
            shuffle=False,
            drop_last=False,
            batch_size=int(getattr(opt, "batchsize_uav_test", opt.batchsize_uav)),
        )

        num_epochs = int(opt.num_epochs)
        save_freq = max(1, int(getattr(opt, "save_freq", 10)))
        val_freq = max(1, int(getattr(opt, "val_freq", 1)))
        max_eval_batches = getattr(opt, "apr_max_eval_batches", None)
        if max_eval_batches is not None:
            max_eval_batches = int(max_eval_batches)

        step = 0
        since = time.time()
        self.logger.info(f"Start APR MLP training for {num_epochs} epochs.")

        for epoch in range(begin_epoch, num_epochs):
            self._set_train_modes()
            epoch_loss = 0.0
            epoch_count = 0

            for it, batch in tqdm.tqdm(enumerate(self.uav_dataloader_train), total=len(self.uav_dataloader_train)):
                imgs = batch[0].to(self.device, non_blocking=True)
                coords_raw = batch[1].to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    feats = self._extract_image_features(imgs)
                    pred = self.apr_mlp(feats)
                    target = self._make_target(coords_raw)
                    loss, loss_parts = self._compute_loss(pred, target)

                if amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_item = float(loss.detach().item())
                epoch_loss += loss_item
                epoch_count += 1

                if self.writer is not None and it % 10 == 0:
                    self.writer.add_scalar("loss_it", loss_item, step)
                    self.writer.add_scalar("loss_it_nrc", float(loss_parts["loss_nrc"].item()), step)
                    self.writer.add_scalar("loss_it_rot", float(loss_parts["loss_rot"].item()), step)
                    self.writer.add_scalar("loss_it_scale", float(loss_parts["loss_scale"].item()), step)
                step += 1

            mean_loss = epoch_loss / max(epoch_count, 1)
            self.logger.info(f"epoch={epoch} loss={mean_loss:.6f}")
            if self.writer is not None:
                self.writer.add_scalar("loss_epoch", mean_loss, epoch)

            if bool(getattr(opt, "val", True)) and (epoch % val_freq == 0 or epoch == num_epochs - 1):
                eval_res = self.evaluate(self.uav_dataloader_test, max_batches=max_eval_batches)
                self._log_eval_result(eval_res, prefix=f"[Val][epoch={epoch}]")
                if self.writer is not None:
                    self.writer.add_scalar("val/error_rc_meter_mean", eval_res["error_rc_meter_mean"], epoch)
                    self.writer.add_scalar("val/error_rot_deg_mean", eval_res["error_rot_deg_mean"], epoch)
                    self.writer.add_scalar("val/error_scale_rel_mean", eval_res["error_scale_rel_mean"], epoch)

            is_last_epoch = epoch == num_epochs - 1
            should_save_ckpt = ((epoch % save_freq == 0) and (epoch > 0)) or is_last_epoch
            if should_save_ckpt:
                self._save_checkpoint(epoch, train_ckpt_modules, self.optimizer)

            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f"epoch {epoch} done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            self.logger.info("-" * 50)

        self.logger.info("Stage 2 APR MLP training completed.")

    @staticmethod
    def _angle_error_deg(pred_rad, gt_rad):
        diff = torch.remainder(pred_rad - gt_rad + torch.pi, 2.0 * torch.pi) - torch.pi
        return torch.abs(diff) * 180.0 / torch.pi

    def collect_predictions(self, dataloader=None, max_batches=None):
        if dataloader is None:
            dataloader = self.uav_dataloader_test

        self.apr_mlp.eval()
        self.vis_encoder.eval()
        self.vis_aggregator.eval()

        coords_pred_all = []
        coords_gt_all = []
        loss_vals = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                imgs = batch[0].to(self.device, non_blocking=True)
                coords_gt = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)

                feats = self._extract_image_features(imgs)
                pred = self.apr_mlp(feats)
                target = self._make_target(coords_gt)
                loss, _ = self._compute_loss(pred, target)
                coords_pred = self._prediction_to_raw(pred).to(self.device)

                coords_pred_all.append(coords_pred.detach().cpu())
                coords_gt_all.append(coords_gt.detach().cpu())
                loss_vals.append(torch.full((coords_gt.shape[0],), float(loss.item()), dtype=torch.float32))

        def _cat(values, shape):
            return torch.cat(values, dim=0) if values else torch.empty(shape, dtype=torch.float32)

        return {
            "coords_pred": _cat(coords_pred_all, (0, 4)),
            "coords_gt": _cat(coords_gt_all, (0, 4)),
            "loss": _cat(loss_vals, (0,)),
        }

    def evaluate(self, dataloader=None, max_batches=None):
        if dataloader is None:
            dataloader = self.uav_dataloader_test

        self.apr_mlp.eval()
        self.vis_encoder.eval()
        self.vis_aggregator.eval()

        rc_errors_norm = []
        rc_errors_meter = []
        rot_errors_deg = []
        scale_abs_errors = []
        scale_rel_errors = []
        losses = []
        n_samples = 0
        meter_per_norm = float(self.sat_dataset.satmap_hw_max) * float(self.sat_dataset.geo_res_m)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                imgs = batch[0].to(self.device, non_blocking=True)
                coords_gt = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)

                feats = self._extract_image_features(imgs)
                pred = self.apr_mlp(feats)
                target = self._make_target(coords_gt)
                loss, _ = self._compute_loss(pred, target)
                coords_pred = self._prediction_to_raw(pred).to(self.device)

                rc_error_norm = torch.linalg.norm(coords_pred[:, 0:2] - coords_gt[:, 0:2], dim=-1)
                rc_error_meter = rc_error_norm * meter_per_norm
                rot_error_deg = self._angle_error_deg(coords_pred[:, 2], coords_gt[:, 2])
                scale_abs_error = torch.abs(coords_pred[:, 3] - coords_gt[:, 3])
                scale_rel_error = scale_abs_error / torch.clamp(torch.abs(coords_gt[:, 3]), min=1e-6)

                rc_errors_norm.append(rc_error_norm.detach().cpu())
                rc_errors_meter.append(rc_error_meter.detach().cpu())
                rot_errors_deg.append(rot_error_deg.detach().cpu())
                scale_abs_errors.append(scale_abs_error.detach().cpu())
                scale_rel_errors.append(scale_rel_error.detach().cpu())
                losses.append(torch.full_like(rc_error_norm.detach().cpu(), float(loss.item())))
                n_samples += int(coords_gt.shape[0])

        def _cat(values):
            return torch.cat(values, dim=0) if values else torch.empty(0)

        rc_norm = _cat(rc_errors_norm)
        rc_meter = _cat(rc_errors_meter)
        rot_deg = _cat(rot_errors_deg)
        scale_abs = _cat(scale_abs_errors)
        scale_rel = _cat(scale_rel_errors)
        loss_vals = _cat(losses)

        def _mean(x):
            return float(x.mean().item()) if x.numel() else 0.0

        def _median(x):
            return float(x.median().item()) if x.numel() else 0.0

        return {
            "n_queries": n_samples,
            "loss_mean": _mean(loss_vals),
            "error_rc_norm_mean": _mean(rc_norm),
            "error_rc_norm_median": _median(rc_norm),
            "error_rc_meter_mean": _mean(rc_meter),
            "error_rc_meter_median": _median(rc_meter),
            "error_rot_deg_mean": _mean(rot_deg),
            "error_rot_deg_median": _median(rot_deg),
            "error_scale_abs_mean": _mean(scale_abs),
            "error_scale_abs_median": _median(scale_abs),
            "error_scale_rel_mean": _mean(scale_rel),
            "error_scale_rel_median": _median(scale_rel),
        }

    def _log_eval_result(self, eval_res, prefix="[Eval]"):
        msg = (
            f"{prefix} N={eval_res['n_queries']} loss={eval_res['loss_mean']:.6f} "
            f"rc={eval_res['error_rc_meter_mean']:.3f}m/"
            f"{eval_res['error_rc_meter_median']:.3f}m(med) "
            f"rot={eval_res['error_rot_deg_mean']:.3f}deg/"
            f"{eval_res['error_rot_deg_median']:.3f}deg(med) "
            f"scale_rel={eval_res['error_scale_rel_mean']:.4f}/"
            f"{eval_res['error_scale_rel_median']:.4f}(med)"
        )
        if self.logger:
            self.logger.info(msg)
        print(msg)

    def _get_apr_checkpoint_path(self):
        if getattr(self.opt, "load2test", ""):
            return self.opt.load2test
        if self.exp_dir2save and os.path.exists(self.exp_dir2save):
            ckpt_path = _find_latest_epoch_ckpt(self.exp_dir2save)
            if ckpt_path:
                return ckpt_path
        return None

    def _get_stage1_checkpoint_path(self, apr_ckpt_path=None):
        if getattr(self.opt, "load_stage1_ckpt", ""):
            return self.opt.load_stage1_ckpt
        if apr_ckpt_path:
            opts_path = os.path.join(os.path.dirname(apr_ckpt_path), "opts.yaml")
            if os.path.exists(opts_path):
                import yaml
                with open(opts_path, "r", encoding="utf-8") as f:
                    opts = yaml.safe_load(f) or {}
                stage1_path = (opts.get("exp_setting") or {}).get("load_stage1_ckpt", "")
                if stage1_path:
                    return stage1_path
        return None

    def _load_checkpoints_for_test(self):
        apr_ckpt_path = self._get_apr_checkpoint_path()
        if not apr_ckpt_path:
            raise ValueError("No APR checkpoint found. Set load2test.")

        print(f"Loading APR checkpoint: {apr_ckpt_path}")
        self._load_checkpoint(
            {"apr_mlp": apr_ckpt_path},
            {"apr_mlp": self.apr_mlp},
            mode="test",
        )

        stage1_ckpt_path = self._get_stage1_checkpoint_path(apr_ckpt_path)
        if not stage1_ckpt_path:
            raise ValueError("No Stage 1 checkpoint found. Set load_stage1_ckpt or keep opts.yaml with it.")

        print(f"Loading Stage 1 checkpoint: {stage1_ckpt_path}")
        self._load_checkpoint(
            {
                "vis_encoder": stage1_ckpt_path,
                "vis_aggregator": stage1_ckpt_path,
            },
            self.param2freeze,
            mode="test",
        )

    def test(self):
        print("\n" + "=" * 80)
        print("Starting Stage 2 APR MLP test")
        print("=" * 80 + "\n")

        self._init_datasets(create_train_loader=False)
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        self.uav_dataloader_test = self._make_uav_loader(
            self.uav_dataset_test,
            shuffle=False,
            drop_last=False,
            batch_size=int(getattr(self.opt, "batchsize_uav_test", self.opt.batchsize_uav)),
        )
        self._load_checkpoints_for_test()

        max_eval_batches = getattr(self.opt, "apr_max_eval_batches", None)
        if max_eval_batches is not None:
            max_eval_batches = int(max_eval_batches)
        eval_res = self.evaluate(self.uav_dataloader_test, max_batches=max_eval_batches)
        self._log_eval_result(eval_res, prefix="[Test]")

        report_path = getattr(self.opt, "apr_eval_report_path", "")
        if report_path:
            report_path = report_path if os.path.isabs(report_path) else os.path.join(project_root, report_path)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(eval_res, f, indent=2)
            print(f"Saved APR eval report: {report_path}")
        return eval_res


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", action="store_true", help="Run test only.")
    args, remaining_argv = parser.parse_known_args()

    if "--p_yaml" not in remaining_argv:
        remaining_argv.extend(["--p_yaml", os.path.join(project_root, "trainer_depends/configs/stage2_arp_mlp.yaml")])
    sys.argv = [sys.argv[0]] + remaining_argv

    opt = get_parse(print_summary=False)
    trainer = APRMLPTrainer(opt=opt)
    if args.test_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
