#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Stage-1 train/test unit.

This file is intentionally small.  It keeps only the core path:
1. build frozen/trainable visual encoder + aggregator
2. train with UAV query / satellite positive / satellite negatives
3. test by building a simple satellite gallery and reporting Recall@K

Excluded on purpose: ANCE, gallery-bank cache/artifacts, advanced eval reports,
NetVLAD data-dependent init, code backup, and control-experiment branches.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from trainer_depends.base.components import NetworkComponents
from trainer_depends.base.trainer_base import BaseTrainer
from trainers.util_stage1_multi_scene_dataloader import MultiSceneDataLoader


class MinimalStage1VisualEncoderTrainer(BaseTrainer):
    """A readable Stage-1 trainer with only the essential train/test logic."""

    def __init__(self, opt=None):
        super().__init__(opt)
        self._init_networks()
        self._setup_trainable_params()

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"

    def _init_networks(self):
        components = NetworkComponents(self.opt, self.device)
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel
        self.vis_aggregator = components.create_aggregator(self.feat_patch_dim)
        self.feat_q_dim = int(getattr(self.vis_aggregator, "output_dim", self.feat_patch_dim))

    def _setup_trainable_params(self):
        freeze_backbone = bool(getattr(self.opt, "freeze_backbone", True))
        adapter_enabled = bool(getattr(self.opt, "adapter_config", {}).get("enabled", False))
        if freeze_backbone and not adapter_enabled:
            for param in self.vis_encoder.parameters():
                param.requires_grad = False

        self.param2optimize = {"vis_aggregator": self.vis_aggregator}
        self.param2freeze = {}
        if any(param.requires_grad for param in self.vis_encoder.parameters()):
            self.param2optimize["vis_encoder"] = self.vis_encoder
        else:
            self.param2freeze["vis_encoder"] = self.vis_encoder

        print("Stage1 minimal trainable modules:", ", ".join(self.param2optimize.keys()))
        print("Stage1 minimal frozen modules:", ", ".join(self.param2freeze.keys()) or "(none)")

    def _autocast_context(self):
        use_amp = bool(getattr(self.opt, "autocast", False)) and self.device.type == "cuda"
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _encode_for_train(self, imgs):
        if any(param.requires_grad for param in self.vis_encoder.parameters()):
            return self.vis_encoder(imgs)
        with torch.no_grad():
            return self.vis_encoder(imgs)

    def _init_train_dataloader(self):
        from trainer_depends.datasets.dataset_neuloc_4d_uav_sat_pair import (
            UAVSatPairDataset,
            collate_uav_sat_pair,
        )

        opt = self.opt
        pair_dataloaders = {}
        for scene in opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_train = self.uav_datasets_train[scene_name]

            n_neg_per_query = 0
            if bool(getattr(opt, "add_random_satimg_negs", True)):
                n_neg_per_query = max(1, int(opt.batchsize_sat) // max(1, int(opt.batchsize_uav)))

            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                device=self.device,
                n_neg_per_query=n_neg_per_query,
                sat_as_query=bool(getattr(opt, "sat_as_query", False)),
                nrc_reject_sampling=bool(getattr(opt, "reject_sampling", False)),
                pair_alignment_mode=str(getattr(opt, "pair_alignment_mode", "full_4d")),
            )
            pair_dataset.weight = len(uav_dataset_train)

            pair_dataloaders[scene_name] = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=int(opt.batchsize_uav),
                num_workers=int(opt.num_worker),
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                collate_fn=partial(
                    collate_uav_sat_pair,
                    sat_dataset=sat_dataset,
                    reject_batch_aware=bool(getattr(opt, "reject_batch_aware", False)),
                ),
                persistent_workers=(int(opt.num_worker) > 0),
            )
            self.logger.info(
                "%s: %d pairs, %d batches, n_neg_per_query=%d",
                scene_name,
                len(pair_dataset),
                len(pair_dataloaders[scene_name]),
                n_neg_per_query,
            )

        self.dataloader_train = MultiSceneDataLoader(
            pair_dataloaders,
            sampling_strategy=self.opt.scenes_setting.get("sampling_strategy", "round_robin"),
        )

    def _init_loss(self):
        loss_type = str(getattr(self.opt, "loss_type", "tripleLoss_singleEdge_hardest_fm_mask")).lower()
        if loss_type == "infonce":
            from losses.stage1_infonce_loss import Stage1InfoNCELoss

            self.loss_type = "infonce"
            self.loss_fn = Stage1InfoNCELoss(
                temperature=float(getattr(self.opt, "infonce_temperature", 0.1)),
                negative_mode=str(getattr(self.opt, "infonce_negative_mode", "batch_and_explicit")),
            ).to(self.device)
            return

        if loss_type == "tripleloss_singleedge_hardest_fm_mask":
            from losses.CL_losses_wo_weight import tripleLoss_singleEdge_hardest_fm_mask

            self.loss_type = "tripleloss_singleedge_hardest_fm_mask"
            self.loss_fn = tripleLoss_singleEdge_hardest_fm_mask().to(self.device)
            return

        raise ValueError(
            f"Unsupported stage1 minimal loss_type={getattr(self.opt, 'loss_type', None)!r}. "
            "Supported values: infonce, tripleLoss_singleEdge_hardest_fm_mask."
        )

    def _extract_batch(self, batch):
        uavimgs = batch["uavimgs"].to(self.device)
        satimgs_pos = batch["satimgs_pos"].to(self.device)

        if "satimgs_query" in batch:
            uavimgs = torch.cat([uavimgs, batch["satimgs_query"].to(self.device)], dim=0)
            satimgs_pos = torch.cat([satimgs_pos, batch["satimgs_pos2satimg_query"].to(self.device)], dim=0)

        satimgs_neg = batch.get("satimgs_neg", None)
        if satimgs_neg is not None:
            satimgs_neg = satimgs_neg.to(self.device)
            satimgs_neg = satimgs_neg.reshape(-1, *satimgs_neg.shape[2:])

        return uavimgs, satimgs_pos, satimgs_neg

    def _forward_train_batch(self, uavimgs, satimgs_pos, satimgs_neg):
        refs = [satimgs_pos]
        if satimgs_neg is not None:
            refs.append(satimgs_neg)
        imgs_input = torch.cat([uavimgs] + refs, dim=0)

        feats_patch = self._encode_for_train(imgs_input)
        feats = self.vis_aggregator(feats_patch)

        batch_size = uavimgs.shape[0]
        feats_q = feats[:batch_size]
        feats_ref = feats[batch_size:]
        feat_dist_mat = torch.norm(feats_q[:, None, :] - feats_ref[None, :, :], p=2, dim=-1)
        return feats_q, feats_ref, feat_dist_mat

    @staticmethod
    def _build_pos_mask(batch_size, num_refs, device):
        pos_mask = torch.zeros(batch_size, num_refs, dtype=torch.bool, device=device)
        pos_mask[:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=device)
        return pos_mask

    def _compute_loss(self, feats_q, feats_ref, feat_dist_mat):
        batch_size = feats_q.shape[0]
        feats_pos = feats_ref[:batch_size]
        feats_neg = feats_ref[batch_size:]

        if self.loss_type == "infonce":
            return self.loss_fn(feats_q, feats_pos, explicit_negative_keys=feats_neg)

        pos_mask = self._build_pos_mask(batch_size, feats_ref.shape[0], feat_dist_mat.device)
        return self.loss_fn(feat_dist_mat, pos_mask)

    def _checkpoint_modules(self):
        return {**self.param2optimize, **self.param2freeze}

    def train(self):
        opt = self.opt
        use_amp = bool(getattr(opt, "autocast", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        self._init_loss()
        from tool.util_mk_optimizer import create_optimizer_w_temple

        self.optimizer = create_optimizer_w_temple(self.param2optimize, "adam", opt=opt)
        begin_epoch = self._load_checkpoint(
            getattr(opt, "load2train", ""),
            self._checkpoint_modules(),
            self.optimizer,
            mode="train",
        )

        self._init_logger()
        self._init_datasets(create_train_loader=False)
        self._init_train_dataloader()

        step = 0
        since = time.time()
        for epoch in range(begin_epoch, int(opt.num_epochs)):
            self.logger.info("Epoch %d/%d", epoch, int(opt.num_epochs) - 1)
            last_loss = None

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train)):
                uavimgs, satimgs_pos, satimgs_neg = self._extract_batch(batch)
                with self._autocast_context():
                    feats_q, feats_ref, feat_dist_mat = self._forward_train_batch(
                        uavimgs,
                        satimgs_pos,
                        satimgs_neg,
                    )
                    loss = self._compute_loss(feats_q, feats_ref, feat_dist_mat)

                self.optimizer.zero_grad()
                if use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if it % 10 == 0:
                    batch_size = feats_q.shape[0]
                    recall1 = (
                        torch.argmin(feat_dist_mat, dim=-1)
                        == torch.arange(batch_size, device=feat_dist_mat.device)
                    ).float().mean()
                    self.logger.info(
                        "iter=%d scene=%s loss=%.6f train_recall@1=%.4f",
                        it,
                        batch.get("scene_name", "unknown"),
                        float(loss.detach().item()),
                        float(recall1.detach().item()),
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("loss_it", loss.item(), step)
                        self.writer.add_scalar("train_recall1_it", recall1.item(), step)

                last_loss = loss
                step += 1

            save_freq = max(1, int(getattr(opt, "save_freq", 1)))
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

        self.logger.info("Stage1 minimal training finished.")

    def _load_eval_ckpt_if_needed(self):
        ckpt_path = getattr(self.opt, "load2test", "") or getattr(self.opt, "load2train", "")
        if ckpt_path:
            self._load_checkpoint(ckpt_path, self._checkpoint_modules(), optimizer=None, mode="test")
            print(f"Loaded eval checkpoint: {ckpt_path}")

    def _encode_imgs_eval(self, imgs):
        feats = self._get_feats_fm_imgs(imgs)
        return F.normalize(feats, dim=-1)

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

    def _build_scene_gallery(self, sat_dataset, overlap, chunk_size):
        gallery_scale = float(sat_dataset.satimgsize_scale_to_ref_m_mean)
        crop_size = float(sat_dataset.satimgsize2crop_mean)
        nrcs_gallery = sat_dataset.crop_sat_unifrom(
            size2clip=crop_size,
            overlap=float(overlap),
            only_nrcs=True,
        )
        nrcs_flat = torch.as_tensor(nrcs_gallery, dtype=torch.float32).flatten(0, 1)
        coords_gallery = torch.cat(
            [
                nrcs_flat,
                torch.zeros(nrcs_flat.shape[0], 1),
                torch.full((nrcs_flat.shape[0], 1), gallery_scale),
            ],
            dim=-1,
        )

        feats = []
        with torch.no_grad():
            for start in range(0, coords_gallery.shape[0], int(chunk_size)):
                end = min(start + int(chunk_size), coords_gallery.shape[0])
                coords_chunk = coords_gallery[start:end]
                if hasattr(sat_dataset, "crop_satimg_by_4d_coords_fast"):
                    satimgs = sat_dataset.crop_satimg_by_4d_coords_fast(
                        coords_chunk,
                        apply_rotation=False,
                        chunk_size=int(chunk_size),
                    )
                else:
                    satimgs = sat_dataset.crop_satimg_by_4d_coords(coords_chunk, apply_rotation=False)
                feats.append(self._encode_imgs_eval(satimgs.to(self.device)).cpu())
        return coords_gallery.cpu(), torch.cat(feats, dim=0), gallery_scale

    def test(self, init_datasets=True, load_ckpt=True, restore_train=True):
        """Simple Recall@K test against a freshly built satellite grid gallery."""
        if init_datasets or not hasattr(self, "sat_datasets"):
            self._init_datasets(create_train_loader=False)
        if load_ckpt:
            self._load_eval_ckpt_if_needed()

        models = list(self._checkpoint_modules().values())
        old_modes = [model.training for model in models]
        for model in models:
            model.eval()

        overlap = float(getattr(self.opt, "val_overlap", 0.5))
        chunk_size = int(getattr(self.opt, "val_chunk_size", 1024))
        k_values = [1, 5, 10, 20, 50]
        query_rot2uniform = bool(getattr(self.opt, "val_query_rot2uniform", True))
        query_scale2uniform = bool(getattr(self.opt, "val_query_scale2uniform", False))
        results = {}

        for scene in self.opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset = self.uav_datasets_test[scene_name]
            coords_gallery, feats_gallery, gallery_scale = self._build_scene_gallery(
                sat_dataset,
                overlap=overlap,
                chunk_size=chunk_size,
            )

            uav_loader = torch.utils.data.DataLoader(
                uav_dataset,
                batch_size=int(self.opt.batchsize_uav),
                num_workers=int(getattr(self.opt, "num_worker_eval", 0)),
                shuffle=False,
                drop_last=False,
                pin_memory=False,
            )

            success_counts = {k: 0 for k in k_values}
            top1_dist_nrc = []
            total = 0
            top_k = min(max(k_values), feats_gallery.shape[0])
            for batch in tqdm.tqdm(uav_loader, desc=f"test {scene_name}"):
                uavimgs, coords_uav = batch[0], batch[1]
                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)

                rot_align = -coords_uav[:, 2] if query_rot2uniform else None
                scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6) if query_scale2uniform else None
                if query_rot2uniform or query_scale2uniform:
                    uavimgs = self._warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)

                with torch.no_grad():
                    feats_q = self._encode_imgs_eval(uavimgs).cpu()

                dist_feat = torch.cdist(feats_q, feats_gallery)
                indices = torch.topk(dist_feat, k=top_k, dim=1, largest=False).indices
                coords_topk = coords_gallery[indices]
                dist_nrc = torch.norm(coords_uav[:, None, :2].cpu() - coords_topk[:, :, :2], p=2, dim=-1)
                hits = dist_nrc < float(sat_dataset.halfimg_radius_nrc)

                for k in k_values:
                    if k <= hits.shape[1]:
                        success_counts[k] += (hits[:, :k].any(dim=1)).sum().item()
                top1_dist_nrc.append(dist_nrc[:, 0])
                total += coords_uav.shape[0]

            scene_result = {f"recall@{k}": success_counts[k] / max(1, total) for k in k_values}
            if top1_dist_nrc:
                top1_dist_nrc = torch.cat(top1_dist_nrc)
                nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
                scene_result["top1_dist_meter_mean"] = float(top1_dist_nrc.mean().item() * nrc2meter)
                scene_result["top1_dist_meter_median"] = float(torch.median(top1_dist_nrc).item() * nrc2meter)
            results[scene_name] = scene_result

            msg = " | ".join(f"R@{k}={scene_result[f'recall@{k}'] * 100:.3f}%" for k in k_values)
            print(f"[{scene_name}] {msg} | N={total}")
            if self.logger is not None:
                self.logger.info("[%s] %s | N=%d", scene_name, msg, total)

        if restore_train:
            for model, was_train in zip(models, old_modes):
                model.train(was_train)
        return results


VisualEncoderTrainer = MinimalStage1VisualEncoderTrainer


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n"}


if __name__ == "__main__":
    if "--p_yaml" not in " ".join(sys.argv):
        sys.argv.extend(["--p_yaml", "trainer_depends/configs/stage1_visual_encoder_wingtra.yaml"])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", nargs="?", const=True, default=False, type=_parse_bool_arg)
    parser.add_argument("--exp_name_override", default="", type=str,help=" 如果命令行传入了exp_name_override 参数，则用该值替换 opt.exp_name；在不修改 yaml 配置文件的情况下，通过命令行为同一套配置启动不同命名的实验，方便对比实验的日志/checkpoint 目录区分")
    args, remaining_argv = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    from trainer_depends.config.parser import get_parse

    opt = get_parse()
    if args.exp_name_override:
        opt.exp_name = args.exp_name_override

    trainer = MinimalStage1VisualEncoderTrainer(opt=opt)
    if args.test_only:
        trainer.test(init_datasets=True, load_ckpt=True, restore_train=False)
    else:
        trainer.train()
