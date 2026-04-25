#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 control-experiment trainer entrypoint.

This file intentionally keeps the original Stage-1 trainer flow untouched.
For now it mirrors `trainers/stage1_visual_encoder_w_ANCE.py` by subclassing it,
so future control-experiment changes can be isolated here without modifying the
default training path.
"""

import argparse
import os
import sys


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
    if os.environ.get("_STAGE1_CONTROL_REEXECED") == "1":
        return

    os.environ["_STAGE1_CONTROL_REEXECED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_runtime_env()

import numpy as np
import torch
import torch.nn.functional as F


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainer_depends.base.components import NetworkComponents
from trainer_depends.config.parser import get_parse
from trainers.stage1_visual_encoder_wANCE import (
    Stage1ReferenceGalleryLayoutConfig,
    VisualEncoderTrainer as _BaseVisualEncoderTrainer,
)


class VisualEncoderTrainer(_BaseVisualEncoderTrainer):
    """Isolated trainer shell for stage-1 control experiments."""

    @staticmethod
    def _warp_uav_imgs(imgs, rot_rad=None, scale_f=None):
        if rot_rad is None and scale_f is None:
            return imgs
        b = imgs.shape[0]
        device = imgs.device
        dtype = imgs.dtype
        if rot_rad is None:
            rot_rad = torch.zeros(b, device=device, dtype=dtype)
        else:
            rot_rad = rot_rad.to(device=device, dtype=dtype)
        if scale_f is None:
            scale_f = torch.ones(b, device=device, dtype=dtype)
        else:
            scale_f = scale_f.to(device=device, dtype=dtype)

        cos_v = torch.cos(rot_rad)
        sin_v = torch.sin(rot_rad)
        theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos_v * scale_f
        theta[:, 0, 1] = sin_v * scale_f
        theta[:, 1, 0] = -sin_v * scale_f
        theta[:, 1, 1] = cos_v * scale_f
        grid = F.affine_grid(theta, imgs.size(), align_corners=False)
        return F.grid_sample(imgs, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def _should_canonicalize_train_uav_query(self):
        pair_alignment_mode = str(getattr(self.opt, "pair_alignment_mode", "full_4d")).strip().lower()
        active_loss_type = str(getattr(self, "active_loss_type", "")).strip().lower()
        return pair_alignment_mode == "xy_only" and active_loss_type == "msloss_torch"

    def _get_scene_train_scale_mean(self, scene_name, coords_uav=None):
        if scene_name in getattr(self, "uav_datasets_train", {}):
            dataset = self.uav_datasets_train[scene_name]
            coords_train = getattr(dataset, "uav_coords_4d_torch_train", None)
            if torch.is_tensor(coords_train) and coords_train.numel() > 0:
                return float(coords_train[:, 3].detach().cpu().to(dtype=torch.float32).mean().item())

        if scene_name in getattr(self, "sat_datasets", {}):
            sat_dataset = self.sat_datasets[scene_name]
            if hasattr(sat_dataset, "satimgsize_scale_to_ref_m_mean"):
                return float(sat_dataset.satimgsize_scale_to_ref_m_mean)

        if torch.is_tensor(coords_uav) and coords_uav.numel() > 0:
            return float(coords_uav[:, 3].detach().cpu().to(dtype=torch.float32).mean().item())

        return 1.0

    def _maybe_canonicalize_train_uav_query(self, batch_state):
        if not self._should_canonicalize_train_uav_query():
            return batch_state

        b_uav = int(batch_state.get("b_uav", 0))
        if b_uav <= 0:
            return batch_state

        scene_name = batch_state.get("scene_name", None)
        uavimgs = batch_state["uavimgs"]
        coords_uav = batch_state["coords_uav"]

        target_scale = self._get_scene_train_scale_mean(scene_name, coords_uav=coords_uav[:b_uav])
        rot_align = -coords_uav[:b_uav, 2]
        scale_f = float(target_scale) / coords_uav[:b_uav, 3].clamp(min=1e-6)

        uavimgs_aligned = uavimgs.clone()
        coords_aligned = coords_uav.clone()
        uavimgs_aligned[:b_uav] = self._warp_uav_imgs(
            uavimgs[:b_uav],
            rot_rad=rot_align,
            scale_f=scale_f,
        )
        coords_aligned[:b_uav, 2] = 0.0
        coords_aligned[:b_uav, 3] = float(target_scale)

        if not getattr(self, "_logged_xy_only_query_canonicalization", False):
            self._log_or_print(
                "[xy_only] canonicalize train UAV queries for MSLoss_torch: "
                f"rot->0, scale->{float(target_scale):.4f} (train-mean)."
            )
            self._logged_xy_only_query_canonicalization = True

        updated = dict(batch_state)
        updated["uavimgs"] = uavimgs_aligned
        updated["coords_uav"] = coords_aligned
        return updated

    def _after_train_dataloader_init(self):
        super()._after_train_dataloader_init()

    def _extract_train_batch(self, batch):
        batch_state = super()._extract_train_batch(batch)
        return self._maybe_canonicalize_train_uav_query(batch_state)

    def _init_networks(self):
        print("\n" + "=" * 80)
        print("初始化 Stage 1 Control-Exps 网络组件")
        print("=" * 80)

        components = NetworkComponents(self.opt, self.device)
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel

        agg_type = str(getattr(self.opt, "aggregator_type", "salad")).lower()
        if agg_type in {"gem", "netvlad", "g2m", "g2m_scalar_p", "g2m_channelwise_p", "fsra", "lpn"}:
            self.vis_aggregator = self._create_control_aggregator(
                agg_type=agg_type,
                feat_dim=self.feat_patch_dim,
            )
        else:
            self.vis_aggregator = components.create_aggregator(self.feat_patch_dim)

        self.feat_q_dim = int(getattr(self.vis_aggregator, "output_dim", self.feat_patch_dim))
        print("=" * 80 + "\n")

    def _create_control_aggregator(self, agg_type: str, feat_dim: int):
        from models.Head.token_vpr_aggregators import (
            TokenFSRA,
            TokenLPN,
            TokenG2MChannelwiseP,
            TokenG2MScalarP,
            TokenGeM,
            TokenNetVLAD,
        )

        aggregator_config = dict(getattr(self.opt, "aggregator_config", {}) or {})
        imgsize2net = int(getattr(self.opt, "imgsize2net", 224))
        backbone_name = str(getattr(self.opt, "backbone", "")).lower()
        patchsize = 14 if "dinov2" in backbone_name else 16

        def _resolve_aggregator_output_dim(default=None):
            value = aggregator_config.get("output_dim", aggregator_config.get("out_channels", default))
            return int(value) if value is not None else None

        output_dim = _resolve_aggregator_output_dim()
        rank = int(aggregator_config.get("rank", 1024))
        p = float(aggregator_config.get("p", 3.0))
        eps = float(aggregator_config.get("eps", 1e-6))
        block = int(aggregator_config.get("block", 3))
        num_bottleneck = int(aggregator_config.get("num_bottleneck", 256))
        droprate = float(aggregator_config.get("droprate", 0.0))
        fuse_mode = str(aggregator_config.get("fuse_mode", "concat"))
        use_cls_token = bool(aggregator_config.get("use_cls_token", True))

        if agg_type == "gem":
            aggregator = TokenGeM(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                p=p,
                eps=eps,
                output_dim=_resolve_aggregator_output_dim(feat_dim),
            ).to(self.device)
            print("✅ 创建 GeM 聚合器")
            print(
                f"   GeM配置: p={p}, "
                f"eps={eps}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        if agg_type in {"g2m", "g2m_scalar_p"}:
            aggregator = TokenG2MScalarP(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                output_dim=_resolve_aggregator_output_dim(feat_dim),
                rank=rank,
                p=p,
                eps=eps,
            ).to(self.device)
            print("✅ 创建 G2M 聚合器 (scalar p)")
            print(
                f"   G2M配置: rank={rank}, p={p}, eps={eps}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        if agg_type == "g2m_channelwise_p":
            aggregator = TokenG2MChannelwiseP(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                output_dim=_resolve_aggregator_output_dim(feat_dim),
                rank=rank,
                p=p,
                eps=eps,
            ).to(self.device)
            print("✅ 创建 G2M 聚合器 (channelwise p)")
            print(
                f"   G2M配置: rank={rank}, p={p}, eps={eps}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        if agg_type == "fsra":
            aggregator = TokenFSRA(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                block=block,
                num_bottleneck=num_bottleneck,
                droprate=droprate,
                fuse_mode=fuse_mode,
                use_cls_token=use_cls_token,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建 FSRA 聚合器")
            print(
                f"   FSRA配置: block={block}, num_bottleneck={num_bottleneck}, "
                f"droprate={droprate}, fuse_mode={fuse_mode}, use_cls_token={use_cls_token}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        if agg_type == "lpn":
            aggregator = TokenLPN(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                block=block,
                num_bottleneck=num_bottleneck,
                droprate=droprate,
                fuse_mode=fuse_mode,
                use_cls_token=use_cls_token,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建 LPN 聚合器")
            print(
                f"   LPN配置: block={block}, num_bottleneck={num_bottleneck}, "
                f"droprate={droprate}, fuse_mode={fuse_mode}, use_cls_token={use_cls_token}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        if agg_type == "netvlad":
            aggregator = TokenNetVLAD(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                num_clusters=int(aggregator_config.get("num_clusters", 16)),
                alpha=float(aggregator_config.get("alpha", 100.0)),
                normalize_input=bool(aggregator_config.get("normalize_input", True)),
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建 NetVLAD 聚合器")
            print(
                f"   NetVLAD配置: num_clusters={int(aggregator_config.get('num_clusters', 16))}, "
                f"alpha={float(aggregator_config.get('alpha', 100.0))}, "
                f"normalize_input={bool(aggregator_config.get('normalize_input', True))}, "
                f"patchsize={patchsize}, output_dim={aggregator.output_dim}"
            )
            return aggregator

        raise ValueError(f"Unsupported control-experiment aggregator_type: {agg_type}")

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

    def _log_or_print(self, message):
        if getattr(self, "logger", None) is not None:
            self.logger.info(message)
        else:
            print(message)

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

    def _init_loss_modules(self):
        loss_type = str(getattr(self.opt, "loss_type", "tripleLoss_singleEdge_hardest_fm_weight")).lower()
        if loss_type != "msloss_torch":
            self.active_loss_miner = None
            return super()._init_loss_modules()

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

    def _forward_train_batch(self, uavimgs, satimgs_pos, satimgs_neg_flat):
        forward_state = super()._forward_train_batch(uavimgs, satimgs_pos, satimgs_neg_flat)
        self._last_forward_state = forward_state
        return forward_state

    def _compute_train_loss(self, batch_state, feat_dist_mat, coords_uav_neg_flat):
        if str(getattr(self, "active_loss_type", "")).lower() != "msloss_torch":
            return super()._compute_train_loss(batch_state, feat_dist_mat, coords_uav_neg_flat)

        forward_state = getattr(self, "_last_forward_state", None)
        if forward_state is None:
            raise RuntimeError("Missing forward state for MSLoss_torch.")

        batch_size = int(forward_state["batch_size"])
        feats_q = forward_state["feats_q"]
        feats_ref = forward_state["feats_ref"]
        feats_pos = feats_ref[:batch_size]

        if feats_pos.shape[0] != batch_size:
            raise ValueError(
                f"MSLoss_torch expects {batch_size} positive references, got {feats_pos.shape[0]}."
            )

        if feats_ref.shape[0] > batch_size and not getattr(self, "_warned_msloss_ignore_negs", False):
            msg = (
                "MSLoss_torch ignores explicit satimgs_neg samples and uses other places in the batch "
                "as negatives. Set add_random_satimg_negs=False for a cleaner SALAD-style control experiment."
            )
            if getattr(self, "logger", None) is not None:
                self.logger.warning(msg)
            else:
                print(f"[Warning] {msg}")
            self._warned_msloss_ignore_negs = True

        descriptors = torch.cat([feats_q, feats_pos], dim=0)
        labels = torch.arange(batch_size, device=descriptors.device, dtype=torch.long)
        labels = torch.cat([labels, labels], dim=0)

        miner_outputs = self.active_loss_miner(descriptors, labels)
        return self.active_loss_module(descriptors, labels, miner_outputs)


def main():
    if "--p_yaml" not in " ".join(sys.argv):
        sys.argv.extend(["--p_yaml", "/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_visloc_control_exps.yaml"])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", action="store_true", help="是否只运行测试模式")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="gallery_bank",
        choices=("gallery_bank", "eval_recall"),
        help="测试模式：gallery_bank 为显式建库评测，eval_recall 为复用 trainer.eval_recall 的高层入口",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="",
        help="仅在 --test_only --test_mode=gallery_bank 时使用；为空则默认取第一个场景。",
    )
    args, _ = parser.parse_known_args()

    opt = get_parse()
    trainer = VisualEncoderTrainer(opt=opt)

    if not args.test_only:
        trainer.train()
        return

    if args.test_mode == "eval_recall":
        trainer.eval_recall(
            use_train_uav=True,
            init_datasets=True,
            load_ckpt=True,
            restore_train=True,
            **trainer._build_eval_configs(),
        )
        return

    if args.test_mode != "gallery_bank":
        raise ValueError(f"Unknown test_mode: {args.test_mode}")

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
        overlap=0.25,
        n_rot=36,
        n_scale=4,
        scale_mode="linear",
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
    retrieval_eval_cfg.dist_th = float(sat_dataset.halfimg_radius_nrc) * 1.1 * 0.5
    retrieval_eval_cfg.rot_th_deg = 11 * 0.5
    retrieval_eval_cfg.scale_ratio_th = 1.15
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
        gallery_name_prefix=f"{scene_name}",
    )
    print(f"[Gallery Demo] save_dir={gallery_state['gallery_save_dir']}")


if __name__ == "__main__":
    main()
