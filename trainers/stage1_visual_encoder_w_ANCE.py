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

import torch
import torch.nn.functional as TF
import tqdm
import time
import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.base.components import NetworkComponents
from trainer_depends.miners import MultiSceneANCEMiner, SatGalleryProvider, SceneNegMasker

class MultiSceneDataLoader:
    """
    多场景数据加载器

    支持多种采样策略：
    - round_robin: 轮流采样各场景
    - random: 随机采样场景
    - weighted: 按权重采样场景
    """

    def __init__(self, dataloaders, sampling_strategy='round_robin'):
        """
        Args:
            dataloaders: dict, {scene_name: DataLoader}
            sampling_strategy: str, 采样策略 ('round_robin', 'random', 'weighted')
        """
        self.dataloaders = dataloaders
        self.scene_names = list(dataloaders.keys())
        self.num_scenes = len(self.scene_names)
        self.sampling_strategy = sampling_strategy

        # 计算总迭代次数（所有场景的总batch数）
        self.total_batches = sum(len(dl) for dl in dataloaders.values())

        # 初始化迭代器
        self.scene_iters = {}
        self.current_scene_idx = 0
        self.current_iter = 0

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        """每个epoch开始时重置所有迭代器"""
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        self.current_scene_idx = 0
        self.current_iter = 0
        return self

    def __next__(self):
        """获取下一个batch"""
        if self.current_iter >= self.total_batches:
            raise StopIteration

        # 选择场景
        if self.sampling_strategy == 'round_robin':
            scene_name = self.scene_names[self.current_scene_idx]
            self.current_scene_idx = (self.current_scene_idx + 1) % self.num_scenes
        elif self.sampling_strategy == 'random':
            scene_name = np.random.choice(self.scene_names)
        elif self.sampling_strategy == 'weighted':
            weights = [self.dataloaders[name].dataset.weight for name in self.scene_names]
            scene_name = np.random.choice(self.scene_names, p=np.array(weights)/sum(weights))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # 获取该场景的 batch
        try:
            batch = next(self.scene_iters[scene_name])
        except StopIteration:
            # 该场景遍历完，重新开始（在epoch内循环）
            self.scene_iters[scene_name] = iter(self.dataloaders[scene_name])
            batch = next(self.scene_iters[scene_name])

        # 在 batch 中添加场景名称标记
        batch['scene_name'] = scene_name

        # 更新迭代计数器
        self.current_iter += 1

        return batch


class VisualEncoderTrainer(BaseTrainer):
    """
    Stage 1: Visual Encoder Trainer

    训练视觉特征聚合器（vis_aggregator）
    """

    def __init__(self, opt=None):
        """初始化Stage 1 Trainer"""
        super().__init__(opt)

        # 初始化网络组件
        self._init_networks()

        # 设置可训练参数
        self._setup_trainable_params()

    def _init_networks(self):
        """初始化所有网络组件"""
        print("\n" + "="*80)
        print("初始化 Stage 1 网络组件")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        # 视觉编码器（冻结）
        self.vis_encoder = components.create_visual_encoder()
        self.feat_q_dim = self.vis_encoder.output_channel

        # 特征聚合器（可训练）
        self.vis_aggregator = components.create_aggregator(
            self.feat_q_dim
        )

        print("="*80 + "\n")


    def _setup_trainable_params(self):
        """设置可训练参数"""
        # 冻结视觉编码器
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        # 训练聚合器
        self.param2optimize = {
            'vis_aggregator': self.vis_aggregator
        }

        self.param2freeze = {
            'vis_encoder': self.vis_encoder
        }

        # 动态生成参数配置信息
        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys())

        print("参数配置:")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")


    def _init_multi_scene_dataloader(self):
        """初始化多场景训练数据加载器"""
        from trainer_depends.datasets.dataset_neuloc_4d_uav_sat_pair import UAVSatPairDataset, collate_uav_sat_pair
        from trainer_depends.utils.util_sample_neg_nrcs import BoundedNegativeCoordinateSampler
        from trainer_depends.datasets.util_coords_translater import CoordsNormProcessor

        opt = self.opt
        scenes = opt.scenes_setting['scenes']

        # 为每个场景创建pair dataloader
        pair_dataloaders = {}
        self.coord_normers = {}
        self.coord_normed_sigmas = {}

        for scene in scenes:
            scene_name = scene['name']

            # 创建该场景的 UAVSatPairDataset
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_train = self.uav_datasets_train[scene_name]

            satmap_sampler = BoundedNegativeCoordinateSampler(self.device)
            if getattr(opt, "ance_enabled", False):
                n_neg_per_query = 0
            else:
                n_neg_per_query = opt.batchsize_sat // opt.batchsize_uav
            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                satmap_sampler=satmap_sampler,
                device= self.device,
                n_neg_per_query=n_neg_per_query, #控制负样本数=正样本数的倍率
                sat_as_query=opt.sat_as_query,
            )
            pair_dataset.weight = len(uav_dataset_train)

            # 创建DataLoader
            pair_dataloader = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=opt.batchsize_uav,
                num_workers=opt.num_worker,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_uav_sat_pair,
                persistent_workers=True
            )

            pair_dataloaders[scene_name] = pair_dataloader
            self.logger.info(f"  {scene_name}: {len(pair_dataset)} pairs, {len(pair_dataloader)} batches")

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

    def _warp_uav_imgs(self, imgs, rot_rad=None, scale_f=None):
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
        grid = TF.affine_grid(theta, imgs.size(), align_corners=False)
        return TF.grid_sample(imgs, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def _jitter_ance_neg_coords(self, coords, sat_dataset):
        if coords is None:
            return coords
        radii = getattr(sat_dataset, "ance_filter_radius", None)
        if not radii:
            radii = getattr(sat_dataset, "ance_neg_radii", None)
        if not radii:
            return coords

        coords_t = coords if torch.is_tensor(coords) else torch.as_tensor(coords, dtype=torch.float32)
        coords_t = coords_t.clone()

        radius_nrc = float(radii.get("radius_nrc", 0.0) or 0.0)
        radius_rot = float(radii.get("radius_rot_rad", 0.0) or 0.0)
        radius_scale_log = radii.get("radius_scale_log", None)

        if radius_nrc > 0 and coords_t.shape[-1] >= 2:
            noise = (torch.rand_like(coords_t[..., :2]) * 2.0 - 1.0) * radius_nrc
            coords_t[..., :2] = coords_t[..., :2] + noise
            nr_min, nr_max = sat_dataset.nr2sample_range
            nc_min, nc_max = sat_dataset.nc2sample_range
            coords_t[..., 0] = coords_t[..., 0].clamp(min=float(nr_min), max=float(nr_max))
            coords_t[..., 1] = coords_t[..., 1].clamp(min=float(nc_min), max=float(nc_max))

        if radius_rot > 0 and coords_t.shape[-1] >= 3:
            noise = (torch.rand_like(coords_t[..., 2]) * 2.0 - 1.0) * radius_rot
            coords_t[..., 2] = coords_t[..., 2] + noise
            coords_t[..., 2] = (coords_t[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi

        if radius_scale_log is not None and coords_t.shape[-1] >= 4:
            radius_scale_log = float(radius_scale_log)
            noise = (torch.rand_like(coords_t[..., 3]) * 2.0 - 1.0) * radius_scale_log
            coords_t[..., 3] = coords_t[..., 3] * (1.0 + noise)
            s_min, s_max = sat_dataset.satimgsize_scale_to_ref_m_boundary
            coords_t[..., 3] = coords_t[..., 3].clamp(min=float(s_min), max=float(s_max))

        return coords_t

    def _init_ance_miners(self):
        opt = self.opt
        self.ance_enabled = getattr(opt, "ance_enabled", False)
        if not self.ance_enabled:
            return
        self.ance_backend = getattr(opt, "ance_backend", "faiss")
        self.ance_metric = getattr(opt, "ance_metric", "l2")
        self.ance_use_gpu_index = getattr(opt, "ance_use_gpu_index", False)
        self.ance_top_k = getattr(opt, "ance_top_k", 1024)  #top_k 决定 从 gallery 里先取多少个最相似候选
        self.ance_n_neg = getattr(opt, "ance_n_neg", opt.batchsize_sat // opt.batchsize_uav) #再用 neg_mask 过滤掉“正样本半径内”的候选，最后从剩下的里取 n_neg
        self.ance_refresh_epoch = getattr(opt, "ance_refresh_epoch", 2)
        self.ance_gallery_chunk_size = getattr(opt, "ance_gallery_chunk_size", int(1024))
        #about how to construct Gallery, controlling the sampling ratio
        self.ance_overlap = getattr(opt, "ance_overlap", 0.5)
        self.ance_rot_rad_resolution = getattr(opt, "ance_rot_rad_resolution", np.float32(np.pi / 18.))
        self.ance_rot_list = getattr(opt, "ance_rot_list", None)
        self.ance_consider_scale = getattr(opt, "ance_consider_scale", False) #负样本挖掘时是否考虑scale轴的影响
        #about SatGalleryProvider
        self.ance_ref_wo_rot_var = getattr(opt, "ance_ref_wo_rot_var", False)
        self.ance_ref_wo_scale_var = getattr(opt, "ance_ref_wo_scale_var", True)
        #about how to handling query when mining
        self.ance_query_rot2uniform = getattr(opt, "ance_query_rot2uniform", False)
        self.ance_query_scale2uniform = getattr(opt, "ance_query_scale2uniform", False)
        self.ance_recompute_pos = getattr(opt, "ance_recompute_pos", True)

        #about how to define the negatives
        maskers_by_scene = {}
        scenes = self.opt.scenes_setting['scenes']
        for scene in scenes:
            scene_name = scene['name']
            sat_dataset = self.sat_datasets[scene_name]
            sigma = self.coord_normed_sigmas[scene_name]
            sigma_nrc = float(sigma[0].item())
            sigma_rot = float(sigma[2].item())
            sigma_scale_log = float(sigma[3].item())
            gs_factor = float(getattr(self, "gs_sigma2radius_factor", 2.0))
            radius_nrc = sigma_nrc * sat_dataset.halfimg_radius_nrc * gs_factor
            radius_rot = sigma_rot * gs_factor
            radius_scale_log = sigma_scale_log * gs_factor
            sat_dataset.ance_filter_radius = {
                "radius_nrc": radius_nrc,
                "radius_rot_rad": radius_rot,
                "radius_scale_log": radius_scale_log if self.ance_consider_scale else None,
            }
            sat_dataset.ance_neg_radii = sat_dataset.ance_filter_radius
            maskers_by_scene[scene_name] = SceneNegMasker(
                radius_nrc=radius_nrc,
                radius_rot_rad=radius_rot,
                radius_scale_log=radius_scale_log if self.ance_consider_scale else None,
            )

        self.ance_miners = MultiSceneANCEMiner(
            backend=self.ance_backend,
            use_gpu=self.ance_use_gpu_index,
            metric=self.ance_metric,
            maskers_by_scene=maskers_by_scene,
        )
        self.ance_gallery_info = {}
        self.ance_last_refresh_epoch = None

    def _refresh_ance_gallery(self, scene_name=None):
        if not self.ance_enabled:
            return
        if not self.ance_ref_wo_scale_var:
            raise NotImplementedError("ANCE gallery with ref_wo_scale_var=False is not implemented yet.")
        scenes = self.opt.scenes_setting['scenes']
        if scene_name is None:
            scene_names = [s['name'] for s in scenes]
        else:
            scene_names = [scene_name]

        for name in scene_names:
            sat_dataset = self.sat_datasets[name]
            provider = SatGalleryProvider(
                sat_dataset,
                overlap=self.ance_overlap,
                ref_wo_rot_var=self.ance_ref_wo_rot_var,
                ref_wo_scale_var=self.ance_ref_wo_scale_var,
                rot_rad_resolution=self.ance_rot_rad_resolution,
                rot_list=self.ance_rot_list
            )
            coords_gallery = provider.build_coords()
            feats_gallery_chunks = []
            apply_rot_gallery = not self.ance_ref_wo_rot_var
            chunk_iter = tqdm.tqdm(
                range(0, coords_gallery.shape[0], self.ance_gallery_chunk_size),
                desc=f"[ANCE] gallery {name}",
                leave=False
            )
            with torch.inference_mode():
                for start in chunk_iter:
                    end = min(start + self.ance_gallery_chunk_size, coords_gallery.shape[0])
                    coords_chunk = coords_gallery[start:end]
                    satimgs = sat_dataset.crop_satimg_by_4d_coords_fast(
                        coords_chunk, apply_rotation=apply_rot_gallery, chunk_size=self.ance_gallery_chunk_size, random_satmap=True,
                    )
                    satimgs = satimgs.to(self.device)
                    if self.opt.autocast:
                        with torch.cuda.amp.autocast():
                            feats = self._get_feats_fm_imgs(satimgs)
                    else:
                        feats = self._get_feats_fm_imgs(satimgs)
                    feats_gallery_chunks.append(feats.detach().cpu())
                    del satimgs, feats
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            feats_gallery = torch.cat(feats_gallery_chunks, dim=0).numpy()
            self.ance_miners.update_scene(name, feats_gallery, coords_gallery)
            n_rot = int(provider.rot_list.numel()) if getattr(provider, "rot_list", None) is not None else 1
            self.ance_gallery_info[name] = {
                "scale": float(provider.ref_scale),
                "rot": float(provider.rot_list[0]) if n_rot == 1 else None,
                "n_rot": n_rot,
                "ref_wo_rot_var": bool(self.ance_ref_wo_rot_var),
                "ref_wo_scale_var": bool(self.ance_ref_wo_scale_var),
                "rot_rad_resolution": self.ance_rot_rad_resolution,
                "rot_list": self.ance_rot_list,
                "overlap": float(self.ance_overlap),
                "size": int(coords_gallery.shape[0])
            }
            if self.logger is not None:
                self.logger.info(
                    f"[ANCE] refresh gallery {name}: {coords_gallery.shape[0]} pts"
                )


    def train(self):
        """Stage 1训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 1 训练: Visual Encoder (vis_aggregator)")
        print("🚀"*40 + "\n")

        # 0. 初始化GradScaler（如果使用autocast）
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

        # 0.5 初始化可学习的权重损失（用于学习beta）
        from losses.CL_loss_fm_weight import SoftMultiSimLoss_WeightedMax
        self.sms_loss = SoftMultiSimLoss_WeightedMax(
            init_beta=10.0,
            margin=0.0,
            learnable_beta=True
        ).to(self.device)
        self.param2optimize['loss_fn'] = self.sms_loss
        # 其他候选loss：
        # from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        # loss_swt = SWTLoss_fm_mat(decoupling=False)

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam')

        # 2. 加载checkpoint（如果继续训练）
        begin_epoch = self._load_checkpoint(
            opt.load2train,
            self.param2optimize,
            self.optimizer,
            mode='train'
        )

        # 3. 初始化日志
        self._init_logger()

        # 4. 初始化数据集
        self._init_datasets(create_train_loader=False)

        # 5. 创建多场景DataLoader
        self._init_multi_scene_dataloader()

        # 6. 初始化 ANCE miners（可选）
        self._init_ance_miners()

        # 7. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            #刷新 ANCE gallery
            if self.ance_enabled and (
                    self.ance_last_refresh_epoch is None
                    or (epoch - self.ance_last_refresh_epoch) >= self.ance_refresh_epoch
            ):
                self._refresh_ance_gallery()
                self.ance_last_refresh_epoch = epoch

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                scene_name = batch.get('scene_name', None)
                sat_dataset = self.sat_datasets.get(scene_name, self.sat_dataset)

                # basic
                uavimgs = batch['uavimgs'].to(self.device)  # [B, C, H, W]
                satimgs_pos = batch['satimgs_pos'].to(self.device)  # [B, C, H, W]
                coords_uav = batch['coords_uav'].to(self.device)

                # 处理satimgs as query的情况（区分 UAV/SAT query）
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

                # sample neg for per query (may be None when ANCE is enabled)
                if 'satimgs_neg' in batch:
                    satimgs_neg = batch.get('satimgs_neg')
                    coords_uav_neg = batch.get('coords_uav_neg')
                    satimgs_neg = satimgs_neg.to(self.device)
                    coords_uav_neg = coords_uav_neg.to(self.device)
                    satimgs_neg_flat = satimgs_neg.reshape(-1, *satimgs_neg.shape[2:])
                    coords_uav_neg_flat = coords_uav_neg.reshape(-1, *coords_uav_neg.shape[2:])

                # ANCE hard negative mining for gening satimgs_neg
                if self.ance_enabled and scene_name in self.ance_miners.miners:
                    gallery_info = self.ance_gallery_info.get(scene_name, None)
                    gallery_scale = gallery_info["scale"] if gallery_info is not None else float(sat_dataset.satimgsize_scale_to_ref_m_mean)

                    rot_align = -coords_uav[:, 2] if self.ance_query_rot2uniform else None
                    scale_f = None
                    if self.ance_query_scale2uniform:
                        scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)

                    if self.ance_query_rot2uniform or self.ance_query_scale2uniform:
                        uavimgs = self._warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)
                        coords_uav = coords_uav.clone()
                        if self.ance_query_rot2uniform:
                            coords_uav[:, 2] = 0
                        if self.ance_query_scale2uniform:
                            coords_uav[:, 3] = gallery_scale

                        if self.ance_recompute_pos:
                            #query 是对齐后的图像，而sat正样本还是原始姿态的 sat 裁剪，两者不一致，所以对sat正样本也进行重剪裁
                            coords_uav_cpu = coords_uav.detach().cpu()
                            apply_rot = not self.ance_ref_wo_rot_var
                            satimgs_pos = sat_dataset.crop_satimg_by_4d_coords_fast(
                                coords_uav_cpu, apply_rotation=apply_rot, chunk_size=self.ance_gallery_chunk_size, random_satmap=True,
                            )
                            satimgs_pos = satimgs_pos.to(self.device)

                    # mine negatives
                    with torch.no_grad():
                        feats_q_mining = self._get_feats_fm_imgs(uavimgs).detach().cpu().numpy()
                    coords_uav_neg = self.ance_miners.mine(
                        scene_name,
                        feats_q_mining,
                        coords_uav.detach().cpu(),
                        top_k=self.ance_top_k,
                        n_neg=self.ance_n_neg,
                    )
                    coords_uav_neg_jittered = self._jitter_ance_neg_coords(coords_uav_neg, sat_dataset)

                    coords_uav_neg_flat = coords_uav_neg_jittered.reshape(-1, coords_uav_neg.shape[-1])
                    if gallery_info is not None:
                        apply_rot_neg = gallery_info.get("n_rot", 1) > 1
                    else:
                        apply_rot_neg = not self.ance_ref_wo_rot_var
                    satimgs_neg_flat = sat_dataset.crop_satimg_by_4d_coords_fast(
                        coords_uav_neg_flat, apply_rotation=apply_rot_neg, chunk_size=self.ance_gallery_chunk_size,random_satmap=True,
                    )
                    satimgs_neg_flat = satimgs_neg_flat.to(self.device)
                    coords_uav_neg_flat = coords_uav_neg_flat.to(self.device)

                # debug2vis
                # id2vis = 13
                # img2vis_sat1 =  self.sat_datasets[scene_name].denormalize_img(satimgs_pos[id2vis])
                # img2vis_sat2 =  self.sat_datasets[scene_name].denormalize_img(satimgs_neg[id2vis*self.ance_n_neg])
                # img2vis_uav =  self.uav_datasets_train[scene_name].denormalize_img(uavimgs[id2vis])
                # from matplotlib import pyplot as plt
                # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                # ax1.imshow(img2vis_sat1)
                # ax2.imshow(img2vis_sat2)
                # ax3.imshow(img2vis_uav)
                # plt.show()

                #=====================处理图像特征=====================
                imgs_input = torch.cat([uavimgs, satimgs_pos, satimgs_neg_flat], dim=0)
                with torch.no_grad():
                    feats_patch = self.vis_encoder(imgs_input)

                # 聚合器处理（可训练）
                feats_agg = self.vis_aggregator(feats_patch)  # [3B, feat_dim]

                # 分离query和reference特征
                B = uavimgs.shape[0]
                feats_q = feats_agg[:B]  # UAV特征
                feats_ref = feats_agg[B:]  # SAT特征（正样本+负样本）

                # 计算特征距离矩阵
                feat_dist_mat = torch.norm(
                    feats_q.unsqueeze(1) - feats_ref.unsqueeze(0),
                    p=2,
                    dim=-1
                )  # [B, 2B]

                # =====================处理坐标&权重&loss=====================
                # 由坐标计算坐标权重矩阵（与 feats_ref 顺序对齐：pos 在前，neg 在后）
                weights_ref = None
                scene_name = batch.get('scene_name', None)
                if scene_name in self.coord_normers:
                    coord_normer = self.coord_normers[scene_name]
                    normed_sigmas = self.coord_normed_sigmas[scene_name]
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

                if has_sat_query and b_sat > 0:
                    feat_dist_mat_uav = feat_dist_mat[:b_uav]
                    feat_dist_mat_sat = feat_dist_mat[b_uav:]
                    weights_ref_uav = weights_ref[:b_uav] if weights_ref is not None else None
                    weights_ref_sat = weights_ref[b_uav:] if weights_ref is not None else None

                    loss_pos_uav, loss_neg_uav = self.sms_loss(
                        feat_dist_mat_uav, weights_ref_uav, 1 - weights_ref_uav
                    )
                    loss_pos_sat, loss_neg_sat = self.sms_loss(
                        feat_dist_mat_sat, weights_ref_sat, 1 - weights_ref_sat
                    )

                    loss_uav = loss_pos_uav + loss_neg_uav
                    loss_sat = loss_pos_sat + loss_neg_sat

                    sat_query_loss_weight = float(0.1)
                    loss = loss_uav + sat_query_loss_weight * loss_sat
                else:
                    loss_pos, loss_neg = self.sms_loss(feat_dist_mat, weights_ref, 1 - weights_ref)
                    loss = loss_pos + loss_neg
                # version0,sml_loss,不使用距离权重，硬HardMining:
                # if not hasattr(self, 'pos_mask_mat'):
                #     self.pos_mask_mat = torch.cat([
                #         torch.eye(B),
                #         torch.zeros(feat_dist_mat.shape[0],feat_dist_mat.shape[1]-B)
                #     ], dim=-1).bool()  # [B, 2B]
                # loss = loss_swt(feat_dist_mat, self.pos_mask_mat)

                # 反向传播
                self.optimizer.zero_grad()
                if opt.autocast:
                    self.scaler.scale(loss).backward()
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
                    recall1 = (torch.argmin(feat_dist_mat, dim=-1) == torch.arange(
                        0, B, device=feat_dist_mat.device
                    )).sum() / B

                    # 显示场景信息
                    scene_info = f" [{batch.get('scene_name', 'unknown')}]" if len(opt.scenes_setting['scenes']) > 1 else ""
                    self.logger.info(f'training set{scene_info} recall1={recall1.item():.4f}')

                step += 1

            # 每个epoch结束后（固定间隔 + 最后一个epoch）
            # if ((epoch % 5 == 0) and (epoch > 0)) or (epoch == num_epochs - 1):
            self._save_checkpoint(
                epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )

            # epoch 结束后刷新 ANCE gallery（按频次）
            if self.ance_enabled and (
                self.ance_last_refresh_epoch is None
                or (epoch - self.ance_last_refresh_epoch) >= self.ance_refresh_epoch
            ):
                self._refresh_ance_gallery()
                self.ance_last_refresh_epoch = epoch

            # 评估 Recall（可选）
            # if getattr(opt, "val", False) and ((epoch + 1) % getattr(opt, "val_freq", 1) == 0):
            if self.ance_enabled:
                eval_overlap = self.ance_overlap
                eval_chunk = self.ance_gallery_chunk_size
                eval_rot_res = self.ance_rot_rad_resolution
                eval_rot_list = self.ance_rot_list
                eval_ref_wo_rot = self.ance_ref_wo_rot_var
                eval_ref_wo_scale = self.ance_ref_wo_scale_var
                eval_q_rot = self.ance_query_rot2uniform
                eval_q_scale = self.ance_query_scale2uniform
            else:
                eval_overlap = getattr(opt, "val_overlap", 0.5)
                eval_chunk = getattr(opt, "val_chunk_size", 1024)
                eval_rot_res = getattr(opt, "val_rot_rad_resolution", None)
                eval_rot_list = getattr(opt, "val_rot_list", None)
                eval_ref_wo_rot = getattr(opt, "val_ref_wo_rot_var", True)
                eval_ref_wo_scale = getattr(opt, "val_ref_wo_scale_var", True)
                eval_q_rot = getattr(opt, "val_query_rot2uniform", True)
                eval_q_scale = getattr(opt, "val_query_scale2uniform", False)
            self.eval_recall(
                use_train_uav=False,
                init_datasets=False,
                load_ckpt=False,
                restore_train=True,
                overlap=eval_overlap,
                chunk_size_vis=eval_chunk,
                rot_rad_resolution=eval_rot_res,
                rot_list=eval_rot_list,
                ref_wo_rot_var=eval_ref_wo_rot,
                ref_wo_scale_var=eval_ref_wo_scale,
                query_rot2uniform=eval_q_rot,
                query_scale2uniform=eval_q_scale
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


    def eval_recall(self, use_train_uav=False, overlap=0.5, chunk_size_vis=1024,
                    rot_rad_resolution=None, rot_list=None,
                    init_datasets=True, load_ckpt=True, restore_train=True,
                    ref_wo_rot_var=True, ref_wo_scale_var=True,
                    query_rot2uniform=False, query_scale2uniform=False):
        """
        Recall评估：
        - 参考库只构建一次（不旋转参考库图像）
        - 默认使用 sat_dataset.satimgsize2crop_mean 作为裁剪大小
        - ref_wo_rot_var=True: 参考库仅2D网格采样，不考虑旋转
        - ref_wo_rot_var=False: 参考库按 rot_list 或 rot_rad_resolution 采样多个旋转角度
        - ref_wo_scale_var=True: 参考库固定为 mean scale
        - ref_wo_scale_var=False: 暂未实现
        - query_rot2uniform=True: UAV图像反向旋转到正北；rot置0
        - query_scale2uniform=True: UAV尺度统一到 satimgsize_scale_to_ref_m_mean
        """
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1).")
        if not ref_wo_rot_var and rot_rad_resolution is None and rot_list is None:
            raise ValueError("ref_wo_rot_var=False requires rot_list or rot_rad_resolution.")
        if not ref_wo_scale_var:
            raise NotImplementedError("ref_wo_scale_var=False is not implemented yet.")
        if not ref_wo_rot_var:
            if rot_list is not None:
                rot_arr = np.asarray(rot_list, dtype=np.float32).reshape(-1)
                if rot_arr.size == 0:
                    raise ValueError("rot_list must be non-empty.")
                rot_list = rot_arr.tolist()
            else:
                rot_rad_resolution = float(rot_rad_resolution)
                if rot_rad_resolution <= 0 or rot_rad_resolution > 2 * np.pi:
                    raise ValueError("rot_rad_resolution must be in (0, 2*pi].")

        # 初始化数据集（评估时也需要）
        if init_datasets:
            self._init_datasets(create_train_loader=False)

        # 加载测试用 checkpoint（优先 load2test，其次 load2train）
        ckpt2load_path = None
        if load_ckpt:
            ckpt2load = getattr(self.opt, "load2test", "")
            if not ckpt2load:
                ckpt2load = getattr(self.opt, "load2train", "")
            if ckpt2load:
                if isinstance(ckpt2load, dict):
                    for _v in ckpt2load.values():
                        if _v:
                            ckpt2load_path = _v
                            break
                else:
                    ckpt2load_path = ckpt2load
                self._load_checkpoint(
                    ckpt2load,
                    {**self.param2optimize, **self.param2freeze},
                    mode='test'
                )

        # 切换到 eval 模式（记录原状态以便恢复）
        models_all = list(self.param2optimize.values()) + list(self.param2freeze.values())
        orig_modes = [m.training for m in models_all]
        for model in models_all:
            model.eval()

        eval_log_lines = []

        def _log(msg):
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
            eval_log_lines.append(msg)

        _log("\n" + "=" * 80)
        _log("开始 Stage 1 Recall 评估")
        _log(
            f"use_train_uav={use_train_uav}, overlap={overlap}, chunk_size_vis={chunk_size_vis}, "
            f"ref_wo_rot_var={ref_wo_rot_var}, ref_wo_scale_var={ref_wo_scale_var}, "
            f"query_rot2uniform={query_rot2uniform}, query_scale2uniform={query_scale2uniform}, "
            f"rot_rad_resolution={rot_rad_resolution}, rot_list={rot_list}"
        )
        _log("=" * 80)

        # 评估参数
        k_values = [1, 5, 10, 20, 50, 256, 512, 1024]
        results_all = {}

        # 多场景逐个评估
        scenes = self.opt.scenes_setting['scenes']
        for scene in scenes:
            scene_name = scene['name']
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset = self.uav_datasets_train[scene_name] if use_train_uav else self.uav_datasets_test[scene_name]

            eval_num_workers = getattr(self.opt, "num_worker_eval", 0)
            eval_persistent_workers = getattr(self.opt, "persistent_workers_eval", False)
            uav_dataloader = torch.utils.data.DataLoader(
                uav_dataset,
                batch_size=self.opt.batchsize_uav,
                num_workers=eval_num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                persistent_workers=(eval_persistent_workers and eval_num_workers > 0)
            )

            # ---------- 构建/复用参考库（每个scene一次） ----------
            gallery_scale = float(sat_dataset.satimgsize_scale_to_ref_m_mean)
            use_ance_gallery = False
            miner = None
            if (ref_wo_scale_var
                and getattr(self, "ance_enabled", False)
                and hasattr(self, "ance_miners")
                and self.ance_miners is not None):
                miner = self.ance_miners.miners.get(scene_name, None)
                if miner is not None and miner.coords_gallery is not None and miner.index is not None:
                    gallery_info = self.ance_gallery_info.get(scene_name, None)
                    overlap_ok = True
                    if gallery_info is not None:
                        overlap_ok = abs(float(overlap) - float(gallery_info.get("overlap", overlap))) < 1e-6
                    rot_flag_ok = True
                    if gallery_info is not None:
                        rot_flag_ok = bool(ref_wo_rot_var) == bool(gallery_info.get("ref_wo_rot_var", True))
                    rot_res_ok = True
                    if not ref_wo_rot_var:
                        if rot_list is not None:
                            if gallery_info is None or gallery_info.get("rot_list", None) is None:
                                rot_res_ok = False
                            else:
                                g_list = np.asarray(gallery_info.get("rot_list"), dtype=np.float32).reshape(-1)
                                q_list = np.asarray(rot_list, dtype=np.float32).reshape(-1)
                                rot_res_ok = (
                                    g_list.shape == q_list.shape
                                    and np.allclose(g_list, q_list, rtol=0, atol=1e-6)
                                )
                        else:
                            rot_res_ok = (
                                gallery_info is not None
                                and gallery_info.get("rot_rad_resolution", None) is not None
                                and rot_rad_resolution is not None
                                and abs(float(rot_rad_resolution) - float(gallery_info.get("rot_rad_resolution"))) < 1e-6
                            )
                    if overlap_ok and rot_flag_ok and rot_res_ok:
                        use_ance_gallery = True
                    else:
                        _log(
                            f"[Scene: {scene_name}] gallery mismatch (overlap/rot settings), "
                            f"fallback to rebuild gallery."
                        )

            if use_ance_gallery:
                coords_gallery_cpu = miner.coords_gallery.cpu()
                gallery_scale = float(
                    self.ance_gallery_info.get(scene_name, {}).get("scale", gallery_scale)
                )
                satimgsize2crop = float(sat_dataset.satimgsize2crop_mean)
                _log(
                    f"[Scene: {scene_name}] reuse ANCE gallery "
                    f"({coords_gallery_cpu.shape[0]} pts), crop={satimgsize2crop:.1f}px, scale={gallery_scale:.3f}"
                )
                feat_gallery_index = miner.index
            else:
                satimgsize2crop = float(sat_dataset.satimgsize2crop_mean)
                nrcs_gallery = sat_dataset.crop_sat_unifrom(
                    size2clip=satimgsize2crop,
                    overlap=overlap,
                    only_nrcs=True
                )
                nrcs_flat = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
                if ref_wo_rot_var:
                    rot_gallery = torch.zeros((nrcs_flat.shape[0], 1), dtype=torch.float32)
                    scale_gallery = torch.full((nrcs_flat.shape[0], 1), gallery_scale, dtype=torch.float32)
                    coords_gallery = torch.cat([nrcs_flat, rot_gallery, scale_gallery], dim=-1)
                    n_rot = 1
                else:
                    if rot_list is not None:
                        rot_vals = torch.tensor(rot_list, dtype=torch.float32)
                    else:
                        rot_vals = torch.arange(
                            -np.pi, np.pi, float(rot_rad_resolution), dtype=torch.float32
                        )
                    n_rot = rot_vals.numel()
                    nrcs_rep = nrcs_flat[:, None, :].repeat(1, n_rot, 1).reshape(-1, 2)
                    rot_rep = rot_vals[None, :, None].repeat(nrcs_flat.shape[0], 1, 1).reshape(-1, 1)
                    scale_rep = torch.full((nrcs_rep.shape[0], 1), gallery_scale, dtype=torch.float32)
                    coords_gallery = torch.cat([nrcs_rep, rot_rep, scale_rep], dim=-1)

                _log(
                    f"[Scene: {scene_name}] gallery grid={nrcs_gallery.shape[0]}x{nrcs_gallery.shape[1]} "
                    f"({coords_gallery.shape[0]} pts, n_rot={n_rot}), "
                    f"crop={satimgsize2crop:.1f}px, scale={gallery_scale:.3f}"
                )

                feats_gallery_list = []
                with torch.no_grad():
                    for start in range(0, coords_gallery.shape[0], chunk_size_vis):
                        end = min(start + chunk_size_vis, coords_gallery.shape[0])
                        coords_chunk = coords_gallery[start:end]
                        apply_rot_gallery = not ref_wo_rot_var
                        satimgs_refs = sat_dataset.crop_satimg_by_4d_coords_fast(
                            coords_chunk, apply_rotation=apply_rot_gallery, chunk_size=chunk_size_vis
                        )
                        satimgs_refs = satimgs_refs.to(self.device)
                        feats_ref = TF.normalize(self._get_feats_fm_imgs(satimgs_refs), dim=-1)
                        feats_gallery_list.append(feats_ref.detach().cpu())

                feats_gallery = torch.cat(feats_gallery_list, dim=0)
                coords_gallery_cpu = coords_gallery.cpu()

                # 构建 Faiss 索引
                import faiss
                feat_gallery_index = faiss.IndexFlatL2(self.feat_q_dim)
                feat_gallery_index.add(feats_gallery.numpy())

            # 统计 Recall
            success_counts_nrc = {k: 0 for k in k_values}
            success_counts_rot = {k: 0 for k in k_values}
            total_queries = 0

            _log(f"\n[Scene: {scene_name}] 开始评估，共 {len(uav_dataset)} 个 queries")

            for batch in uav_dataloader:
                if isinstance(batch, (list, tuple)):
                    uavimgs, coords_uav = batch[0], batch[1]
                else:
                    uavimgs, coords_uav = batch

                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)

                # UAV query 对齐
                rot_align = -coords_uav[:, 2] if query_rot2uniform else None
                scale_f = None
                if query_scale2uniform:
                    scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)

                if query_rot2uniform or query_scale2uniform:
                    uavimgs = self._warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)
                    coords_uav = coords_uav.clone()
                    if query_rot2uniform:
                        coords_uav[:, 2] = 0
                    if query_scale2uniform:
                        coords_uav[:, 3] = gallery_scale

                with torch.no_grad():
                    if use_ance_gallery:
                        feats_q = self._get_feats_fm_imgs(uavimgs)
                    else:
                        feats_q = TF.normalize(self._get_feats_fm_imgs(uavimgs), dim=-1)

                top_k = min(max(k_values), coords_gallery_cpu.shape[0])
                if top_k <= 0:
                    continue
                _, indices = feat_gallery_index.search(
                    feats_q.detach().cpu().numpy(),
                    k=top_k
                )

                coords_topk = coords_gallery_cpu[torch.from_numpy(indices)]  # [B, K, 4]
                dist_nrc = torch.norm(
                    coords_uav[:, None, :2].cpu() - coords_topk[:, :, :2],
                    p=2, dim=-1
                )
                hits_nrc = dist_nrc < sat_dataset.halfimg_radius_nrc*1.1
                hits_rot = None
                if not ref_wo_rot_var:
                    rot_thr = float(torch.pi / 18.0)
                    rot_diff = coords_topk[:, :, 2] - coords_uav[:, None, 2].cpu()
                    rot_diff = (rot_diff + torch.pi) % (2 * torch.pi) - torch.pi
                    hits_rot = hits_nrc & (rot_diff.abs() < rot_thr*1.1)

                for k in k_values:
                    if k <= hits_nrc.shape[1]:
                        success_counts_nrc[k] += (hits_nrc[:, :k].sum(dim=-1) > 0).sum().item()
                        if hits_rot is not None:
                            success_counts_rot[k] += (hits_rot[:, :k].sum(dim=-1) > 0).sum().item()
                total_queries += coords_uav.shape[0]

            # 汇总结果
            if total_queries == 0:
                _log(f"[Scene: {scene_name}] 无有效 query，跳过。")
                continue

            recall_dict = {f"recall@{k}": success_counts_nrc[k] / total_queries for k in k_values}
            if not ref_wo_rot_var:
                recall_rot_dict = {f"recall_rot@{k}": success_counts_rot[k] / total_queries for k in k_values}
                recall_dict.update(recall_rot_dict)
            results_all[scene_name] = recall_dict

            nrc_thr = float(sat_dataset.halfimg_radius_nrc)
            info2log_nrc = " | ".join([f"nrc:R@{k}={recall_dict[f'recall@{k}']*100:.3f}%" for k in k_values])
            if not ref_wo_rot_var:
                rot_thr_deg = float(rot_thr * 180.0 / np.pi)
                info2log_rot = "| ".join([f"nrc+rot:R@{k}={recall_dict[f'recall_rot@{k}']*100:.3f}%" for k in k_values])
                _log(
                    f"[Scene: {scene_name}] {info2log_nrc} "
                    f"(N={total_queries}, nrc_thr={nrc_thr:.3f})"
                )
                _log(
                    f"[Scene: {scene_name}] {info2log_rot} "
                    f"(N={total_queries}, nrc_thr={nrc_thr:.3f}, rot_thr={rot_thr_deg:.1f}deg)"
                )
            else:
                _log(
                    f"[Scene: {scene_name}] {info2log_nrc} "
                    f"(N={total_queries}, nrc_thr={nrc_thr:.3f})"
                )

        _log("=" * 80)
        _log("Recall 评估完成")
        _log("=" * 80)

        if restore_train:
            for model, was_train in zip(models_all, orig_modes):
                model.train(was_train)

        # 保存评估结果到 ckpt 同目录
        if ckpt2load_path:
            try:
                import re
                ckpt_dir = os.path.dirname(ckpt2load_path)
                eval_dir = os.path.join(ckpt_dir, "eval_recall")
                os.makedirs(eval_dir, exist_ok=True)
                base = os.path.basename(ckpt2load_path)
                m = re.search(r"epoch(\d+)", base)
                ep_tag = m.group(1) if m else "latest"
                out_path = os.path.join(eval_dir, f"ep{ep_tag}.log")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(eval_log_lines))
            except Exception as e:
                _log(f"[eval_recall] failed to save log: {e}")

        return results_all


if __name__ == "__main__":
    # 如果没有指定配置文件，使用 stage1 的默认配置
    import sys
    if '--p_yaml' not in ' '.join(sys.argv):
        sys.argv.extend(['--p_yaml', 'trainer_depends/configs/stage1_visual_encoder_wingtra.yaml'])

    trainer = VisualEncoderTrainer()
    # trainer.eval_recall(query_rot2uniform=True,ref_wo_rot_var=True, overlap=0.5, rot_rad_resolution=np.pi / 18.*1.)
    trainer.train()
