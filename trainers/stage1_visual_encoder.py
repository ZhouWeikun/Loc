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
        if freeze_backbone:
            for param in self.vis_encoder.parameters():
                param.requires_grad = False

        self.param2optimize = {'vis_aggregator': self.vis_aggregator}
        self.param2freeze = {}
        if freeze_backbone:
            self.param2freeze['vis_encoder'] = self.vis_encoder
        else:
            self.param2optimize['vis_encoder'] = self.vis_encoder

        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys()) if self.param2freeze else '(none)'
        n_trainable_backbone_params = sum(
            param.numel() for param in self.vis_encoder.parameters() if param.requires_grad
        )

        print(f"参数配置 (freeze_backbone={freeze_backbone}):")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")
        if not freeze_backbone:
            print(f"  vis_encoder 可训练参数量: {n_trainable_backbone_params}\n")

    def _forward_train_vis_encoder(self, imgs_input):
        if getattr(self.opt, 'freeze_backbone', True):
            with torch.no_grad():
                return self.vis_encoder(imgs_input)
        return self.vis_encoder(imgs_input)


    def _init_multi_scene_dataloader(self):
        """初始化多场景训练数据加载器"""
        from trainer_depends.datasets.dataset_neuloc_4d_uav_sat_pair import UAVSatPairDataset, collate_uav_sat_pair
        from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor

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

            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                device=self.device,
                n_neg_per_query=opt.batchsize_sat // opt.batchsize_uav, #控制负样本数=正样本数的倍率
                nrc_reject_sampling=False,
            )

            # 创建DataLoader
            pair_dataloader = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=opt.batchsize_uav,
                num_workers=opt.num_worker,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_uav_sat_pair,
                persistent_workers=(opt.num_worker > 0)
            )

            pair_dataloaders[scene_name] = pair_dataloader
            self.logger.info(f"  {scene_name}: {len(pair_dataset)} pairs, {len(pair_dataloader)} batches")

            # CoordsNormProcessor for this scene (per pair dataset)
            if not hasattr(sat_dataset, 'satimgsize_scale_to_refm_boundary') and hasattr(
                sat_dataset, 'satimgsize_scale_to_ref_m_boundary'
            ):
                sat_dataset.satimgsize_scale_to_refm_boundary = sat_dataset.satimgsize_scale_to_ref_m_boundary
            self.coord_normers[scene_name] = CoordsNormProcessor(sat_dataset)
            gs_sigma_nrc = self.sat_datasets[scene_name].halfimg_radius_nrc*0.5
            gs_sigma_rot_rad = 0.5 * torch.pi / 18
            gs_sigma_scale_log = 0.1
            self.coord_normed_sigmas[scene_name] = torch.tensor(
                [gs_sigma_nrc, gs_sigma_nrc, gs_sigma_rot_rad, gs_sigma_scale_log],
                dtype=torch.float32
            ).to(self.device)

        # 使用 MultiSceneDataLoader
        self.dataloader_train = MultiSceneDataLoader(
            pair_dataloaders,
            sampling_strategy=opt.scenes_setting['sampling_strategy']
        )

        self.logger.info(f"\n✅ 多场景训练集: {len(scenes)}个场景, "
                        f"总计{self.dataloader_train.total_batches}个batches, "
                        f"采样策略={opt.scenes_setting['sampling_strategy']}\n")


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
        from losses.CL_losses_w_weight import pairLoss_singleEdge_weightedHardest
        self.sms_loss = pairLoss_singleEdge_weightedHardest(
            beta=10.0,
            margin=0.0,
            learnable_beta=True
        ).to(self.device)
        self.param2optimize['loss_fn'] = self.sms_loss
        # 其他候选loss：
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False)

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam')

        # 2. 加载checkpoint（如果继续训练）
        begin_epoch = self._load_checkpoint(
            opt.load2train,
            {**self.param2optimize, **self.param2freeze},
            self.optimizer,
            mode='train'
        )

        # 3. 初始化日志
        self._init_logger()

        # 4. 初始化数据集
        self._init_datasets(create_train_loader=False)

        # 5. 创建多场景DataLoader
        self._init_multi_scene_dataloader()

        # 7. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                # 获取图像数据
                coords_uav = batch['coords_uav'].to(self.device)
                uavimgs = batch['uavimgs'].to(self.device)  # [B, C, H, W]
                satimgs_pos = batch['satimgs_pos'].to(self.device)  # [B, C, H, W]
                # 获取坐标数据
                satimgs_neg = batch['satimgs_neg'].to(self.device)  # [B, C, H, W]
                coords_neg = batch['coords_neg'].to(self.device)
                if len(satimgs_neg.shape)>4:
                    satimgs_neg = satimgs_neg.reshape(-1,*satimgs_neg.shape[2:])
                    coords_neg = coords_neg.reshape(-1,*coords_neg.shape[2:])

                #=====================处理图像特征=====================
                imgs_input = torch.cat([uavimgs, satimgs_pos, satimgs_neg], dim=0)
                feats_patch = self._forward_train_vis_encoder(imgs_input)

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
                compute_weights=False
                weights_ref = None
                scene_name = batch.get('scene_name', None)
                if scene_name in self.coord_normers and compute_weights:
                    coord_normer = self.coord_normers[scene_name]
                    normed_sigmas = self.coord_normed_sigmas[scene_name]
                    coords_ref = torch.cat([coords_uav, coords_neg], dim=0)
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

                #version1:
                # loss_pos, loss_neg = self.sms_loss(feat_dist_mat, weights_ref, 1 - weights_ref)
                # loss = loss_pos + loss_neg
                # version0,sml_loss,不使用距离权重，硬HardMining:
                if not hasattr(self, 'pos_mask_mat'):
                    self.pos_mask_mat = torch.cat([
                        torch.eye(B),
                        torch.zeros(feat_dist_mat.shape[0],feat_dist_mat.shape[1]-B)
                    ], dim=-1).bool()  # [B, 2B]
                loss = loss_swt(feat_dist_mat, self.pos_mask_mat)

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

            # 每个epoch结束后
            if (epoch % 5 == 0) and (epoch > 0):
                self._save_checkpoint(
                    epoch,
                    {**self.param2optimize, **self.param2freeze},
                    self.optimizer
                )
            # 评估 Recall（可选）
            # if getattr(opt, "val", False) and ((epoch + 1) % getattr(opt, "val_freq", 1) == 0):
                self.eval_recall(
                    use_train_uav=False,
                    overlap=getattr(opt, "val_overlap", 0.5),
                    chunk_size_vis=getattr(opt, "val_chunk_size", 1024),
                    init_datasets=False,
                    load_ckpt=False,
                    restore_train=True,
                    fixed_rot=getattr(opt, "val_fixed_rot", True),
                    fixed_scale=getattr(opt, "val_fixed_scale", False)
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
                    init_datasets=True, load_ckpt=True, restore_train=True,
                    fixed_rot=True, fixed_scale=False):
        """
        Recall评估：
        - 参考库只构建一次（不旋转参考库图像）
        - 默认使用 sat_dataset.satimgsize2crop_mean 作为裁剪大小
        - fixed_rot=True: UAV图像反向旋转到正北；rot置0
        - fixed_scale=True: UAV尺度统一到 satimgsize_scale_to_ref_m_mean
        """
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1).")

        # 初始化数据集（评估时也需要）
        if init_datasets:
            self._init_datasets(create_train_loader=False)

        # 加载测试用 checkpoint（优先 load2test，其次 load2train）
        if load_ckpt:
            ckpt2load = getattr(self.opt, "load2test", "")
            if not ckpt2load:
                ckpt2load = getattr(self.opt, "load2train", "")
            if ckpt2load:
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

        def _log(msg):
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)

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
            grid = TF.affine_grid(theta, imgs.size(), align_corners=False)
            return TF.grid_sample(imgs, grid, mode='bilinear', padding_mode='border', align_corners=False)

        _log("\n" + "=" * 80)
        _log("开始 Stage 1 Recall 评估")
        _log(
            f"use_train_uav={use_train_uav}, overlap={overlap}, chunk_size_vis={chunk_size_vis}, "
            f"fixed_rot={fixed_rot}, fixed_scale={fixed_scale}"
        )
        _log("=" * 80)

        # 评估参数
        k_values = [1, 5, 10, 20, 50]
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

            # ---------- 构建参考库（每个scene一次） ----------
            gallery_scale = float(sat_dataset.satimgsize_scale_to_ref_m_mean)
            satimgsize2crop = float(sat_dataset.satimgsize2crop_mean)

            nrcs_gallery = sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap,
                only_nrcs=True
            )
            nrcs_flat = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
            rot_gallery = torch.zeros((nrcs_flat.shape[0], 1), dtype=torch.float32)
            scale_gallery = torch.full((nrcs_flat.shape[0], 1), gallery_scale, dtype=torch.float32)
            coords_gallery = torch.cat([nrcs_flat, rot_gallery, scale_gallery], dim=-1)

            _log(
                f"[Scene: {scene_name}] gallery grid={nrcs_gallery.shape[0]}x{nrcs_gallery.shape[1]} "
                f"({coords_gallery.shape[0]} pts), crop={satimgsize2crop:.1f}px, scale={gallery_scale:.3f}"
            )

            feats_gallery_list = []
            with torch.no_grad():
                for start in range(0, coords_gallery.shape[0], chunk_size_vis):
                    end = min(start + chunk_size_vis, coords_gallery.shape[0])
                    coords_chunk = coords_gallery[start:end]
                    if hasattr(sat_dataset, "crop_satimg_by_4d_coords_fast"):
                        satimgs_refs = sat_dataset.crop_satimg_by_4d_coords_fast(
                            coords_chunk, apply_rotation=False, chunk_size=chunk_size_vis
                        )
                    else:
                        satimgs_refs = sat_dataset.crop_satimg_by_4d_coords(coords_chunk, apply_rotation=False)
                    satimgs_refs = satimgs_refs.to(self.device)
                    feats_ref = TF.normalize(self._get_feats_fm_imgs(satimgs_refs), dim=-1)
                    feats_gallery_list.append(feats_ref.detach().cpu())

            feats_gallery = torch.cat(feats_gallery_list, dim=0)
            coords_gallery_cpu = coords_gallery.cpu()

            # 构建 Faiss 索引
            import faiss
            feat_gallery_index = faiss.IndexFlatL2(int(feats_gallery.shape[1]))
            feat_gallery_index.add(feats_gallery.numpy())

            # 统计 Recall
            success_counts = {k: 0 for k in k_values}
            total_queries = 0

            _log(f"\n[Scene: {scene_name}] 开始评估，共 {len(uav_dataset)} 个 queries")

            for batch in uav_dataloader:
                if isinstance(batch, (list, tuple)):
                    uavimgs, coords_uav = batch[0], batch[1]
                else:
                    uavimgs, coords_uav = batch

                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)

                # 固定旋转：将 UAV 旋转到正北方向
                rot_align = -coords_uav[:, 2] if fixed_rot else None
                scale_f = None
                if fixed_scale:
                    scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)

                if fixed_rot or fixed_scale:
                    uavimgs = _warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)
                    coords_uav = coords_uav.clone()
                    if fixed_rot:
                        coords_uav[:, 2] = 0
                    if fixed_scale:
                        coords_uav[:, 3] = gallery_scale

                with torch.no_grad():
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
                hits = dist_nrc < sat_dataset.halfimg_radius_nrc

                for k in k_values:
                    if k <= hits.shape[1]:
                        success_counts[k] += (hits[:, :k].sum(dim=-1) > 0).sum().item()
                total_queries += coords_uav.shape[0]

            # 汇总结果
            if total_queries == 0:
                _log(f"[Scene: {scene_name}] 无有效 query，跳过。")
                continue

            recall_dict = {f"recall@{k}": success_counts[k] / total_queries for k in k_values}
            results_all[scene_name] = recall_dict

            info2log = " | ".join([f"R@{k}={recall_dict[f'recall@{k}']*100:.3f}%" for k in k_values])
            _log(f"[Scene: {scene_name}] {info2log} (N={total_queries})")

        _log("=" * 80)
        _log("Recall 评估完成")
        _log("=" * 80)

        if restore_train:
            for model, was_train in zip(models_all, orig_modes):
                model.train(was_train)

        return results_all


if __name__ == "__main__":
    # 如果没有指定配置文件，使用 stage1 的默认配置
    import sys
    if '--p_yaml' not in ' '.join(sys.argv):
        sys.argv.extend(['--p_yaml', 'trainer_depends/configs/stage1_visual_encoder.yaml'])

    trainer = VisualEncoderTrainer()
    # trainer.eval_recall(fixed_scale=False)
    trainer.train()
