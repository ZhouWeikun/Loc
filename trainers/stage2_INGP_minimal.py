#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Stage-2 INGP train/test unit.

This file keeps only the core Stage-2 path:
1. load frozen Stage-1 visual encoder + aggregator
2. train grid/grid_mlp to regress visual descriptors from 4D coordinates
3. test by building a temporary coordinate gallery and reporting Recall@K

Excluded on purpose: gallery-bank cache/artifacts, provenance export,
visualization tools, and legacy multi-mode eval wrappers.
"""

import argparse
import copy
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import tqdm
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from trainer_depends.base.components import NetworkComponents
from trainer_depends.base.trainer_base import BaseTrainer


class MinimalStage2INGPTrainer(BaseTrainer):
    """Readable Stage-2 trainer with only the essential train/test logic."""

    def __init__(self, opt=None):
        opt = self._prepare_stage2_options(opt)
        super().__init__(opt)
        self._init_networks()
        if getattr(self.opt, "load_stage1_ckpt", ""):
            self._load_stage1_checkpoint(self.opt.load_stage1_ckpt)
        self._setup_trainable_params()

    @staticmethod
    def _resolve_yaml_path(path):
        path = str(path or "").strip()
        if not path:
            return ""
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    @staticmethod
    def _load_yaml_dict(path):
        path = MinimalStage2INGPTrainer._resolve_yaml_path(path)
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}

    @classmethod
    def _collect_explicit_keys_from_yaml(cls, path, _visited=None):
        path_abs = cls._resolve_yaml_path(path)
        if not path_abs:
            return {}
        if _visited is None:
            _visited = set()
        if path_abs in _visited:
            return {}
        _visited.add(path_abs)

        cfg = cls._load_yaml_dict(path_abs)
        if not cfg:
            return {}

        explicit = {}
        base_yaml = cfg.get("p_yaml") or cfg.get("exp_setting", {}).get("p_yaml")
        if base_yaml:
            explicit = cls._collect_explicit_keys_from_yaml(base_yaml, _visited=_visited)

        for section_name in ("exp_setting", "data_setting", "network_setting", "hardware_setting"):
            section_cfg = cfg.get(section_name, None)
            if isinstance(section_cfg, dict):
                explicit.setdefault(section_name, set()).update(section_cfg.keys())
        if isinstance(cfg.get("scenes_setting", None), dict):
            explicit.setdefault("scenes_setting", set()).add("__section__")
        return explicit

    @classmethod
    def _apply_stage1_yaml_inheritance(cls, opt):
        """Small Stage1 YAML inheritance needed by Stage2 minimal configs."""
        inherit_yaml = getattr(opt, "inherit_stage1_yaml", "")
        if not inherit_yaml:
            return opt

        stage1_cfg = cls._load_yaml_dict(inherit_yaml)
        if not stage1_cfg:
            raise FileNotFoundError(f"inherit_stage1_yaml not found or empty: {inherit_yaml}")
        explicit_keys = cls._collect_explicit_keys_from_yaml(getattr(opt, "p_yaml", ""))

        scopes_text = str(getattr(opt, "inherit_stage1_scope", "network,data") or "network,data")
        scopes = [item.strip() for item in scopes_text.split(",") if item.strip()]
        section_by_scope = {
            "network": "network_setting",
            "data": "data_setting",
            "hardware": "hardware_setting",
        }
        data_key_whitelist = {
            "pad_mode",
            "imgsize2net",
            "satimgsize2crop",
            "n_rand2sample_per_pos",
            "split_train_ratio",
            "split_mode",
        }

        inherited = []
        for scope in scopes:
            if scope == "scenes":
                if "scenes_setting" not in explicit_keys and stage1_cfg.get("scenes_setting"):
                    opt.scenes_setting = copy.deepcopy(stage1_cfg["scenes_setting"])
                    inherited.append("scenes_setting")
                continue

            section_name = section_by_scope.get(scope)
            if section_name is None:
                continue
            section_cfg = stage1_cfg.get(section_name, {})
            if not isinstance(section_cfg, dict):
                continue

            section_explicit_keys = explicit_keys.get(section_name, set())
            for key, value in section_cfg.items():
                if section_name == "data_setting" and key not in data_key_whitelist:
                    continue
                if key not in section_explicit_keys:
                    setattr(opt, key, value)
                    inherited.append(key)

        if inherited:
            print(f"Stage2 minimal inherited from Stage1 YAML: {', '.join(inherited)}")
        return opt

    @staticmethod
    def _fill_required_stage2_defaults(opt):
        """Defaults only for fields that BaseTrainer / datasets require."""
        defaults = {
            "imgsize2net": 224,
            "split_train_ratio": 0.9,
            "split_mode": "segment",
            "batchsize_sat": 512,
            "batchsize_uav": 64,
            "num_worker": 0,
            "satmaps_on_cpu": True,
            "autocast": False,
            "freeze_backbone": True,
            "aggregator_type": "salad",
            "num_epochs": 1000,
            "save_freq": 10,
        }
        for key, value in defaults.items():
            if not hasattr(opt, key) or getattr(opt, key) is None:
                setattr(opt, key, value)
        return opt

    @classmethod
    def _prepare_stage2_options(cls, opt):
        if opt is None:
            from trainer_depends.config.parser import get_parse

            opt = get_parse()
        opt = cls._apply_stage1_yaml_inheritance(opt)
        opt = cls._fill_required_stage2_defaults(opt)
        return opt

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"

    def _init_networks(self):
        components = NetworkComponents(self.opt, self.device)

        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel
        self.vis_aggregator = components.create_aggregator(self.feat_patch_dim)
        self.feat_q_dim = int(getattr(self.vis_aggregator, "output_dim", self.feat_patch_dim))

        self.pos_encoder_grid = components.create_coords_5d_encoder(
            multires_rc=int(getattr(self.opt, "posenc_multires_rc", 8)),
            multires_rot=int(getattr(self.opt, "posenc_multires_rot", 6)),
            multires_scale=int(getattr(self.opt, "posenc_multires_scale", 4)),
        )
        self.grid_mlp_use_coord_condition = bool(getattr(self.opt, "grid_mlp_use_coord_condition", True))

        self.grid = components.create_grid()
        self.feat_grid_dim = int(getattr(self.grid, "output_dim", self.feat_q_dim))

        self.hash_lod_aggregator = components.create_hash_lod_aggregator()
        if self.hash_lod_aggregator is not None:
            if self.feat_grid_dim != self.hash_lod_aggregator.input_dim:
                raise ValueError(
                    f"HashGrid output_dim={self.feat_grid_dim} does not match "
                    f"hash_lod_aggregator input_dim={self.hash_lod_aggregator.input_dim}"
                )
            self.feat_grid_dim = self.hash_lod_aggregator.output_dim

        condition_dim = self.pos_encoder_grid.out_dim if self.grid_mlp_use_coord_condition else 0
        self.grid_mlp = components.create_grid_mlp(
            self.feat_grid_dim,
            condition_dim,
            hidden_dim=int(getattr(self.opt, "grid_mlp_hidden_dim", 512)),
            num_blocks=int(getattr(self.opt, "grid_mlp_num_blocks", 1)),
            output_dim=self.feat_q_dim,
        )

    def _load_stage1_checkpoint(self, ckpt_path):
        print(f"Loading Stage-1 checkpoint: {ckpt_path}")
        self._load_checkpoint(
            ckpt_path,
            {
                "vis_encoder": self.vis_encoder,
                "vis_aggregator": self.vis_aggregator,
            },
            optimizer=None,
            mode="test",
        )

    def _setup_trainable_params(self):
        for module in (self.vis_encoder, self.vis_aggregator):
            for param in module.parameters():
                param.requires_grad = False

        self.param2optimize = {
            "grid": self.grid,
            "grid_mlp": self.grid_mlp,
        }
        if self.hash_lod_aggregator is not None:
            self.param2optimize["hash_lod_aggregator"] = self.hash_lod_aggregator

        self.param2freeze = {
            "vis_encoder": self.vis_encoder,
            "vis_aggregator": self.vis_aggregator,
        }

        print("Stage2 minimal trainable modules:", ", ".join(self.param2optimize.keys()))
        print("Stage2 minimal frozen modules:", ", ".join(self.param2freeze.keys()))

    def _autocast_context(self):
        use_amp = bool(getattr(self.opt, "autocast", False)) and self.device.type == "cuda"
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _checkpoint_modules(self):
        return dict(self.param2optimize)

    def _init_runtime_data(self):
        self._init_datasets(create_train_loader=False)
        from trainer_depends.utils.util_core_coords_translater import CoordsNormProcessor

        self.coord_normer = CoordsNormProcessor(self.sat_dataset)

    def _init_train_dataloaders(self):
        opt = self.opt
        self.sat_dataloader = torch.utils.data.DataLoader(
            self.sat_dataset,
            batch_size=int(opt.batchsize_sat),
            num_workers=int(opt.num_worker),
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            persistent_workers=(int(opt.num_worker) > 0),
        )
        self.uav_dataloader_train = torch.utils.data.DataLoader(
            self.uav_dataset_train,
            batch_size=int(opt.batchsize_uav),
            num_workers=int(opt.num_worker),
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            persistent_workers=(int(opt.num_worker) > 0),
        )

    def _get_feats_fm_grid(self, grid_coords_normed, z_padding=0.025):
        input_shape = grid_coords_normed.shape
        coords_flat = grid_coords_normed.flatten(0, 1) if len(input_shape) == 3 else grid_coords_normed

        grid_input = coords_flat.clone()
        if z_padding > 0.0:
            grid_input[:, 2] = grid_input[:, 2] * (1.0 - 2.0 * float(z_padding))
        grid_input = torch.clamp(grid_input, -1.0, 1.0)

        feats_grid = self.grid.interpolate(grid_input, len(self.grid.active_lods) - 1)
        if len(input_shape) == 3:
            feats_grid = feats_grid.view(input_shape[0], input_shape[1], -1)
        return feats_grid

    def _postprocess_grid_feats(self, feats_grid, coords_6d):
        if self.hash_lod_aggregator is None:
            return feats_grid
        return self.hash_lod_aggregator(feats_grid, coords_6d)

    def _encode_grid_mlp_condition(self, coords_6d):
        if not self.grid_mlp_use_coord_condition:
            return None
        return self.pos_encoder_grid(coords_6d[..., :5])

    def _encode_coords_with_grid(self, coords_4d, normalize=True):
        coords_4d = coords_4d.to(self.device, dtype=torch.float32)
        coords_6d = self.coord_normer.raw_to_norm(coords_4d, append_linear_rot=True)
        grid_coords_3d = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)

        feats_grid = self._get_feats_fm_grid(grid_coords_3d)
        feats_grid = self._postprocess_grid_feats(feats_grid, coords_6d)
        coords_encoded = self._encode_grid_mlp_condition(coords_6d)
        feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
        if normalize:
            feats_grid = F.normalize(feats_grid, dim=-1)
        return feats_grid

    def _encode_imgs_with_stage1(self, imgs, normalize=False):
        with torch.no_grad():
            feats = self._get_feats_fm_imgs(imgs)
            if normalize:
                feats = F.normalize(feats, dim=-1)
        return feats

    def train(self):
        opt = self.opt
        use_amp = bool(getattr(opt, "autocast", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        from tool.util_mk_optimizer import create_optimizer_w_temple

        self.optimizer = create_optimizer_w_temple(self.param2optimize, "adam", opt=opt)
        begin_epoch = self._load_checkpoint(
            getattr(opt, "load2train", ""),
            self._checkpoint_modules(),
            self.optimizer,
            mode="train",
        )

        self._init_logger()
        self._init_runtime_data()
        self._init_train_dataloaders()

        loss_mse = torch.nn.MSELoss(reduction="mean")
        save_freq = max(1, int(getattr(opt, "save_freq", 10)))
        step = 0
        since = time.time()

        for epoch in range(begin_epoch, int(opt.num_epochs)):
            self.logger.info("Epoch %d/%d", epoch, int(opt.num_epochs) - 1)
            uav_iter = iter(self.uav_dataloader_train)
            last_loss = None

            for it, sat_batch in tqdm.tqdm(enumerate(self.sat_dataloader), total=len(self.sat_dataloader)):
                satimgs = sat_batch[0].to(self.device)
                coords_sat = sat_batch[1].to(self.device)

                try:
                    uav_batch = next(uav_iter)
                except StopIteration:
                    uav_iter = iter(self.uav_dataloader_train)
                    uav_batch = next(uav_iter)
                uavimgs = uav_batch[0].to(self.device)
                coords_uav = uav_batch[1].to(self.device)

                coords_all = torch.cat([coords_sat, coords_uav], dim=0)
                imgs_all = torch.cat([satimgs, uavimgs], dim=0)

                with self._autocast_context():
                    feats_grid = self._encode_coords_with_grid(coords_all, normalize=True)
                    feats_vis = self._encode_imgs_with_stage1(imgs_all, normalize=False)
                    loss = loss_mse(feats_grid.squeeze(), feats_vis.squeeze()) * 1000.0

                self.optimizer.zero_grad()
                if use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if it % 10 == 0:
                    self.logger.info("iter=%d loss=%.6f", it, float(loss.detach().item()))
                    if self.writer is not None:
                        self.writer.add_scalar("loss_it", loss.item(), step)
                last_loss = loss
                step += 1

            if ((epoch + 1) % save_freq == 0) or (epoch == int(opt.num_epochs) - 1):
                self._save_checkpoint(epoch, self._checkpoint_modules(), self.optimizer)

            if bool(getattr(opt, "val", False)):
                val_freq = max(1, int(getattr(opt, "val_freq", 1)))
                if ((epoch + 1) % val_freq == 0) or (epoch == int(opt.num_epochs) - 1):
                    self.test(init_datasets=False, load_ckpt=False, restore_train=True)

            elapsed = time.time() - since
            since = time.time()
            if last_loss is not None:
                self.logger.info("epoch=%d loss=%.6f elapsed=%.0fm %.0fs", epoch, last_loss.item(), elapsed // 60, elapsed % 60)
                if self.writer is not None:
                    self.writer.add_scalar("loss_epoch", last_loss.item(), epoch)

        self.logger.info("Stage2 minimal training finished.")

    def _load_stage2_eval_ckpt_if_needed(self):
        ckpt_path = getattr(self.opt, "load2test", "") or getattr(self.opt, "load2train", "")
        if not ckpt_path:
            return
        self._load_checkpoint(ckpt_path, self._checkpoint_modules(), optimizer=None, mode="test")
        print(f"Loaded Stage-2 eval checkpoint: {ckpt_path}")

    @staticmethod
    def _warp_uav_imgs(imgs, rot_rad=None, scale_f=None):
        if rot_rad is None and scale_f is None:
            return imgs
        bsz = imgs.shape[0]
        device = imgs.device
        dtype = imgs.dtype
        if rot_rad is None:
            rot_rad = torch.zeros(bsz, device=device, dtype=dtype)
        else:
            rot_rad = rot_rad.to(device=device, dtype=dtype)
        if scale_f is None:
            scale_f = torch.ones(bsz, device=device, dtype=dtype)
        else:
            scale_f = scale_f.to(device=device, dtype=dtype)

        theta = torch.zeros(bsz, 2, 3, device=device, dtype=dtype)
        cos_v = torch.cos(rot_rad)
        sin_v = torch.sin(rot_rad)
        theta[:, 0, 0] = cos_v * scale_f
        theta[:, 0, 1] = sin_v * scale_f
        theta[:, 1, 0] = -sin_v * scale_f
        theta[:, 1, 1] = cos_v * scale_f
        grid = F.affine_grid(theta, imgs.size(), align_corners=False)
        return F.grid_sample(imgs, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _build_rc_gallery_coords(self, overlap):
        gallery_scale = float(self.sat_dataset.satimgsize_scale_to_ref_m_mean)
        crop_size = float(self.sat_dataset.satimgsize2crop_mean)
        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=crop_size,
            overlap=float(overlap),
            only_nrcs=True,
        )
        nrcs_flat = torch.as_tensor(nrcs_gallery, dtype=torch.float32).flatten(0, 1)
        return torch.cat(
            [
                nrcs_flat,
                torch.zeros(nrcs_flat.shape[0], 1),
                torch.full((nrcs_flat.shape[0], 1), gallery_scale),
            ],
            dim=-1,
        )

    def _build_grid_gallery_features(self, coords_gallery, chunk_size):
        feats = []
        with torch.no_grad():
            for start in range(0, coords_gallery.shape[0], int(chunk_size)):
                end = min(start + int(chunk_size), coords_gallery.shape[0])
                feats.append(self._encode_coords_with_grid(coords_gallery[start:end], normalize=True).cpu())
        return torch.cat(feats, dim=0)

    def test(self, init_datasets=True, load_ckpt=True, restore_train=True):
        """Simple RC Recall@K test: UAV visual query vs. Stage-2 coordinate gallery."""
        if init_datasets or not hasattr(self, "sat_dataset"):
            self._init_runtime_data()
        if load_ckpt:
            self._load_stage2_eval_ckpt_if_needed()

        models = list(self.param2optimize.values()) + list(self.param2freeze.values())
        old_modes = [model.training for model in models]
        for model in models:
            model.eval()

        overlap = float(getattr(self.opt, "val_overlap", 0.5))
        chunk_size = int(getattr(self.opt, "val_chunk_size", 4096))
        k_values = [1, 5, 10, 20, 50]
        query_rot2uniform = bool(getattr(self.opt, "val_query_rot2uniform", True))

        coords_gallery = self._build_rc_gallery_coords(overlap=overlap)
        feats_gallery = self._build_grid_gallery_features(coords_gallery, chunk_size=chunk_size)

        uav_loader = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=int(getattr(self.opt, "batchsize_uav_test", self.opt.batchsize_uav)),
            num_workers=int(getattr(self.opt, "num_worker_eval", getattr(self.opt, "num_worker", 0))),
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        success_counts = {k: 0 for k in k_values}
        top1_dist_nrc = []
        total = 0
        top_k = min(max(k_values), feats_gallery.shape[0])
        gallery_scale = float(self.sat_dataset.satimgsize_scale_to_ref_m_mean)

        for batch in tqdm.tqdm(uav_loader, desc="stage2 test"):
            uavimgs, coords_uav = batch[0], batch[1]
            uavimgs = uavimgs.to(self.device)
            coords_uav = coords_uav.to(self.device)

            if query_rot2uniform:
                uavimgs = self._warp_uav_imgs(uavimgs, rot_rad=-coords_uav[:, 2])
                coords_uav = coords_uav.clone()
                coords_uav[:, 2] = 0.0
                coords_uav[:, 3] = gallery_scale

            feats_q = self._encode_imgs_with_stage1(uavimgs, normalize=True).cpu()
            dist_feat = torch.cdist(feats_q, feats_gallery)
            indices = torch.topk(dist_feat, k=top_k, dim=1, largest=False).indices

            coords_topk = coords_gallery[indices]
            dist_nrc = torch.norm(coords_uav[:, None, :2].cpu() - coords_topk[:, :, :2], p=2, dim=-1)
            hits = dist_nrc < float(self.sat_dataset.halfimg_radius_nrc)

            for k in k_values:
                if k <= hits.shape[1]:
                    success_counts[k] += hits[:, :k].any(dim=1).sum().item()
            top1_dist_nrc.append(dist_nrc[:, 0])
            total += coords_uav.shape[0]

        result = {f"recall@{k}": success_counts[k] / max(1, total) for k in k_values}
        if top1_dist_nrc:
            top1_dist_nrc = torch.cat(top1_dist_nrc)
            nrc2meter = float(self.sat_dataset.halfimg_radius_meter) / max(float(self.sat_dataset.halfimg_radius_nrc), 1e-8)
            result["top1_dist_meter_mean"] = float(top1_dist_nrc.mean().item() * nrc2meter)
            result["top1_dist_meter_median"] = float(torch.median(top1_dist_nrc).item() * nrc2meter)

        msg = " | ".join(f"R@{k}={result[f'recall@{k}'] * 100:.3f}%" for k in k_values)
        print(f"[Stage2 minimal] {msg} | N={total}")
        if self.logger is not None:
            self.logger.info("[Stage2 minimal] %s | N=%d", msg, total)

        if restore_train:
            for model, was_train in zip(models, old_modes):
                model.train(was_train)
        return result


GridHashFitTrainer = MinimalStage2INGPTrainer


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n"}


if __name__ == "__main__":
    if "--p_yaml" not in " ".join(sys.argv):
        sys.argv.extend(["--p_yaml", "trainer_depends/configs/stage2_INGP_wingtra.yaml"])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", nargs="?", const=True, default=False, type=_parse_bool_arg)
    args, remaining_argv = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    from trainer_depends.config.parser import get_parse

    opt = get_parse()
    trainer = MinimalStage2INGPTrainer(opt=opt)
    if args.test_only:
        trainer.test(init_datasets=True, load_ckpt=True, restore_train=False)
    else:
        trainer.train()
