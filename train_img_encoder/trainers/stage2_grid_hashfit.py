#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2: Grid HashFit Trainer

训练目标：
- grid (INGP)
- grid_mlp (条件调制器)

前置条件：
- 需要Stage 1训练好的 vis_encoder + vis_aggregator（冻结使用）

训练策略：
- 使用MSE Loss将Grid特征拟合到视觉特征
- 使用4D坐标的位置编码作为条件
"""

import torch
import torch.nn.functional as TF
import tqdm
import time
import sys
import os

# 添加路径：train_img_encoder目录 和 项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
train_img_encoder_dir = os.path.dirname(current_dir)  # train_img_encoder/
project_root = os.path.dirname(train_img_encoder_dir)  # pyproj_neuloc_v0/

sys.path.insert(0, train_img_encoder_dir)  # 用于导入core, models等
sys.path.insert(0, project_root)  # 用于导入tool模块

from core.base.trainer_base import BaseTrainer
from core.base.components import NetworkComponents
from core.config.parser import get_parse
from util_udf_computer import UDFComputer
from models.pos_encoder import encode_4d_coords


class GridHashFitTrainer(BaseTrainer):
    """
    Stage 2: Grid HashFit Trainer

    训练Grid拟合视觉特征场
    """

    def __init__(self, opt=None):
        """初始化Stage 2 Trainer"""
        super().__init__(opt)

        # 初始化网络组件
        self._init_networks()

        # 加载Stage 1的预训练权重（如果指定）
        if self.opt.load_stage1_ckpt:
            self._load_stage1_checkpoint()

        # 设置可训练参数
        self._setup_trainable_params()


    def _init_networks(self):
        """初始化所有网络组件"""
        print("\n" + "="*80)
        print("初始化 Stage 2 网络组件")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        # Stage 1组件（将被冻结）
        self.vis_encoder = components.create_visual_encoder()
        self.vis_aggregator = components.create_aggregator(
            self.vis_encoder.output_channel
        )
        self.feat_q_dim = self.vis_encoder.output_channel

        # 位置编码器
        pos_encoders = components.create_positional_encoders(
            multires_rc=8,
            multires_rot=6,
            multires_scale=4
        )
        self.rc_pos_encoder = pos_encoders['rc']
        self.rot_pos_encoder = pos_encoders['rot']
        self.scale_pos_encoder = pos_encoders['scale']
        self.coord_encoded_dim = (
            self.rc_pos_encoder.out_dim +
            self.rot_pos_encoder.out_dim +
            self.scale_pos_encoder.out_dim
        )

        # Stage 2组件（将被训练）
        self.grid = components.create_grid()
        self.grid_mlp = components.create_grid_mlp(
            self.feat_q_dim,
            self.coord_encoded_dim
        )

        # 保存grid_args供后续使用
        self.grid_args = components.grid_args

        print("="*80 + "\n")


    def _load_stage1_checkpoint(self):
        """加载Stage 1训练好的模型"""
        print(f"\n加载Stage 1 checkpoint: {self.opt.load_stage1_ckpt}")

        self._load_checkpoint(
            self.opt.load_stage1_ckpt,
            {
                'vis_encoder': self.vis_encoder,
                'vis_aggregator': self.vis_aggregator
            }
        )

        print("✅ Stage 1模型加载完成\n")


    def _setup_trainable_params(self):
        """设置可训练参数"""
        # 冻结Stage 1组件
        for module in [self.vis_encoder, self.vis_aggregator]:
            for param in module.parameters():
                param.requires_grad = False

        # 训练Stage 2组件
        self.param2optimize = {
            'grid': self.grid,
            'grid_mlp': self.grid_mlp
        }

        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
            'vis_aggregator': self.vis_aggregator
        }

        print("参数配置:")
        print("  可训练: grid, grid_mlp")
        print("  冻结:   vis_encoder, vis_aggregator\n")


    def _get_feats_fm_grid(self, coords_4d):
        """
        从Grid中查询特征

        Args:
            coords_4d: [N, 4] 4D坐标 [nr, nc, rot_rad, scale]

        Returns:
            feats: [N, feat_dim] Grid特征
        """
        # 将4D坐标转换为3D坐标（Grid只支持3D）
        # 使用 [nr, nc, rot] 作为3D坐标
        nrcs = coords_4d[:, :2]  # [N, 2]
        rots = coords_4d[:, 2:3]  # [N, 1]

        # 旋转归一化到 [-1, 1]
        grid_rot_coords = rots / torch.pi  # [N, 1]

        # 组合为3D坐标
        gird_3d_coords = torch.cat([nrcs, grid_rot_coords], dim=-1)  # [N, 3]

        # 缩放到Grid的范围
        gird_3d_coords *= (180 / self.grid_args.grid.max_grid_res)

        # 从Grid中插值
        n_gird_lod = len(self.grid.active_lods)
        feats_grid = self.grid.interpolate(gird_3d_coords.to(self.device), n_gird_lod - 1)

        return feats_grid


    def train(self):
        """Stage 2训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 2 训练: Grid HashFit")
        print("🚀"*40 + "\n")

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

        # 创建DataLoader
        self.sat_dataloader = torch.utils.data.DataLoader(
            self.sat_dataset,
            batch_size=opt.batchsize_sat,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        self.uav_dataloader_train = torch.utils.data.DataLoader(
            self.uav_dataset_train,
            batch_size=opt.batchsize_uav,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=opt.batchsize_uav,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        # 5. 配置Loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        loss_mse = torch.nn.MSELoss(reduction='mean')

        # 6. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch in tqdm.tqdm(enumerate(self.sat_dataloader)):
                # 获取sat数据
                satimgs = batch[0].to(self.device)
                coords_sat = batch[1].to(self.device)  # [B, 4]

                # 获取uav数据
                batch_uav = next(iter(self.uav_dataloader_train))
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)  # [B, 4]

                # 合并坐标
                coords_all = torch.cat([coords_sat, coords_uav], dim=0)  # [2B, 4]

                # 提取视觉特征（冻结）
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([satimgs, uavimgs], dim=0)
                )  # [2B, feat_dim]

                # 从Grid提取特征
                feats_grid = self._get_feats_fm_grid(coords_all)  # [2B, feat_dim]

                # 位置编码
                coords_encoded = encode_4d_coords(
                    coords_all,
                    rc_encoder=self.rc_pos_encoder,
                    rot_endcoder=self.rot_pos_encoder,
                    scale_encoder=self.scale_pos_encoder
                )  # [2B, coord_encoded_dim]

                # Grid MLP调制
                feats_grid = self.grid_mlp(
                    inputs=feats_grid,
                    condition_features=coords_encoded
                )  # [2B, feat_dim]

                # L2归一化
                feats_grid = TF.normalize(feats_grid, dim=-1)

                # 计算loss（Grid特征拟合视觉特征）
                loss = loss_mse(feats_grid.squeeze(), feats_vis.squeeze()) * 1000

                # 反向传播
                self.optimizer.zero_grad()
                if opt.autocast:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 记录loss
                if it % 10 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss.item(), step)
                step += 1

            # 每个epoch结束后
            # 保存checkpoint
            if (epoch % 5 == 0) and (epoch > 0):
                # 统计信息
                grid_feats_grad = self.grid.codebook.feats.grad
                self.logger.info(f"Grid feats grad L2 norm: {torch.linalg.norm(grid_feats_grad).item():.6f}")
                self.logger.info(f"Grid feats value.max(): {feats_grid.max().item():.4f}")

                mean_dino = torch.mean(feats_vis, dim=0)
                mean_grid = torch.mean(feats_grid, dim=0)
                l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                cosine_sim = TF.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                self.logger.info(f"mean值- L2距离: {l2_distance:.4f}")
                self.logger.info(f"mean值- 余弦相似度: {cosine_sim:.4f}")

                var_per_dim_grid = torch.var(feats_grid, dim=0)
                var_per_dim_dino = torch.var(feats_vis, dim=0)
                self.logger.info(f"Grid方差均值: {var_per_dim_grid.mean().item():.4e}")
                self.logger.info(f"Vis方差均值: {var_per_dim_dino.mean().item():.4e}")

                # 保存
                self._save_checkpoint(
                    epoch,
                    {**self.param2optimize, **self.param2freeze},
                    self.optimizer
                )

            # 日志
            self.logger.info(f'loss={loss.item():.6f}')
            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f'epoch {epoch} 完成，耗时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
            self.logger.info('-' * 50)

            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss.item(), epoch)

            # 备份代码（第一个epoch后）
            if epoch == 0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(self.exp_dir2save, self.opt)

        self.logger.info("✅ Stage 2 训练完成！")

    # ==================== 测试辅助函数 ====================
    def _test_ready(self):
        """准备测试环境"""
        opt = self.opt

        # 使用统一的checkpoint加载方法
        self._load_checkpoint(opt.load2test, mode='test')

        # 设置为评估模式
        for k, v in self.param2optimize.items():
            v.eval()
        for k, v in self.param2freeze.items():
            v.eval()

        # config the datalaoder
        # 初始化多场景数据集（包括训练dataloader）
        self._init_datasets(create_train_loader=False)
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset,
                                                          batch_size=1,
                                                          num_workers=self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=False,
                                                          persistent_workers=True)
        self.uav_dataloader_test = torch.utils.data.DataLoader(
                    self.uav_datset_test,
                    batch_size=128,
                    num_workers=opt.num_worker,
                    shuffle=True, drop_last=False, pin_memory=True,
                    persistent_workers=True)

    def _get_feat_gallery_fm_grid(self, overlap=0.5, delta_rot_rangle=10, scale=None, include_rotation=True):
        """
        从grid生成特征库

        Args:
            overlap: 裁剪重叠度
            delta_rot_rangle: 旋转角度间隔（仅当include_rotation=True时使用）
            scale: 尺度值，None则使用默认值
            include_rotation: 是否包含旋转维度。False时所有rot=0

        Returns:
            dict: 包含特征库和坐标信息的字典
        """
        import numpy as np

        with torch.no_grad():
            # construct coords_gallery
            scale = self.sat_dataset.satimgsize_scale_to_refm_mean if scale is None else scale
            satimgsize2crop = scale*self.sat_dataset.scale_ref_m/self.sat_dataset.geo_res_m
            nrcs_gallery = self.sat_dataset.crop_sat_unifrom(size2clip=satimgsize2crop, overlap=overlap, only_nrcs=True)
            nrcs_gallery_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)

            if include_rotation:
                # 包含多个旋转角度
                rots_angle = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
                rots_rad = torch.tensor(np.deg2rad(np.stack(rots_angle)), dtype=torch.float32)
                coords_gallery = torch.concatenate([nrcs_gallery_flatten.unsqueeze(1).expand(-1, rots_rad.shape[0], -1),
                                                    rots_rad[None, :, None].expand(nrcs_gallery_flatten.shape[0], -1, 1),
                                                    torch.ones([nrcs_gallery_flatten.shape[0], rots_rad.shape[0], 1],
                                                               dtype=torch.float32) * scale
                                                    ], dim=-1)
                coords_gallery_flatten = coords_gallery.flatten(start_dim=0, end_dim=1)
                gallery_shape = torch.Size([nrcs_gallery.shape[0], nrcs_gallery.shape[1], rots_rad.shape[0]])
            else:
                # 不包含旋转，所有rot=0
                rots_rad = torch.tensor([0.0])  # 只有一个角度：0
                coords_gallery_flatten = torch.cat([
                    nrcs_gallery_flatten,  # [N, 2]
                    torch.zeros(nrcs_gallery_flatten.shape[0], 1),  # rot=0 [N, 1]
                    torch.ones(nrcs_gallery_flatten.shape[0], 1) * scale  # [N, 1]
                ], dim=-1)  # [N, 4]
                gallery_shape = torch.Size([nrcs_gallery.shape[0], nrcs_gallery.shape[1]])

            coords_gallery_encoded_flatten = encode_4d_coords(coords_gallery_flatten,
                                                              rc_encoder=self.rc_pos_encoder,
                                                              rot_endcoder=self.rot_pos_encoder,
                                                              scale_encoder=self.scale_pos_encoder)

            # construct feat_gallery form grid
            feat_gallery_flatten = []
            chunk_size = 512  # 定义块大小
            coords_chunks = torch.split(coords_gallery_flatten, chunk_size)
            encoded_coords_chunks = torch.split(coords_gallery_encoded_flatten, chunk_size)
            for coords_4d, encoded_coords_4d in zip(coords_chunks, encoded_coords_chunks):
                feat = self._get_feats_fm_grid(coords_4d)
                feat = self.grid_mlp(inputs=feat, condition_features=encoded_coords_4d.to(feat.device))
                feat_gallery_flatten.append(feat.detach().cpu())
            feat_gallery_flatten = torch.concatenate(feat_gallery_flatten, dim=0)
            feat_gallery_flatten = TF.normalize(feat_gallery_flatten, dim=-1, p=2)

            if include_rotation:
                feat_gallery = feat_gallery_flatten.reshape(*nrcs_gallery.shape[:2], rots_rad.shape[0], -1)
                print(f"Gallery shape (with rot): {feat_gallery.shape}")
            else:
                feat_gallery = feat_gallery_flatten.reshape(*nrcs_gallery.shape[:2], -1)
                print(f"Gallery shape (wo rot): {feat_gallery.shape}")
            print(f"Total candidates: {feat_gallery_flatten.shape[0]}")

            dict2ret = {
                'gallery_shape': gallery_shape,
                'feat_gallery_flatten': feat_gallery_flatten,
                'nrc_gallery': nrcs_gallery,
                "rots_rad": rots_rad,
                "scale": scale,
                "coords_gallery_flatten": coords_gallery_flatten,
                'coords_gallery_encoded_flatten':coords_gallery_encoded_flatten
            }
            return dict2ret

    def test_xy_scale_fm_INGP_wUAV(self, overlap=0.5, scale=None, use_multiscale=False, n_scales=3, vis=False):
        """
        简化版本：对UAV图像逆向旋转，特征库不包含旋转（rot=0）

        Args:
            overlap: 裁剪重叠度
            scale: 尺度值，None则使用默认值（仅当use_multiscale=False时使用）
            use_multiscale: 是否使用多尺度
            n_scales: 尺度数量（仅当use_multiscale=True时使用）
            vis: 是否可视化
        """
        self._test_ready()

        # ==================== 生成特征库（rot=0） ====================
        if use_multiscale:
            # 多尺度模式
            scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
            print(f"\n尺度列表:")
            for i, (scale_val, imgsize) in enumerate(zip(scale_list, satimgsize_list)):
                print(f"  Level {i}: scale={scale_val:.3f}, imgsize={imgsize:.1f}px")

            gallery_features = []
            gallery_coords = []
            for scale_idx, scale_val in enumerate(scale_list):
                print(f"\n{'=' * 60}")
                print(f"处理尺度 {scale_idx + 1}/{n_scales}: scale={scale_val:.3f}")
                print(f"{'=' * 60}")

                feat_gallery_dict = self._get_feat_gallery_fm_grid(
                    overlap=overlap,
                    scale=scale_val,
                    include_rotation=False
                )
                gallery_features.append(feat_gallery_dict['feat_gallery_flatten'])
                gallery_coords.append(feat_gallery_dict['coords_gallery_flatten'])

            feat_gallery_flatten = torch.cat(gallery_features, dim=0)
            coords_gallery_flatten = torch.cat(gallery_coords, dim=0)
            print(f"\n多尺度特征库总数: {feat_gallery_flatten.shape[0]}")
        else:
            # 单尺度模式
            feat_gallery_dict = self._get_feat_gallery_fm_grid(
                overlap=overlap,
                scale=scale,
                include_rotation=False
            )
            feat_gallery_flatten = feat_gallery_dict['feat_gallery_flatten']
            coords_gallery_flatten = feat_gallery_dict['coords_gallery_flatten']

        # ==================== 采样UAV query并逆向旋转 ====================
        for it, data in tqdm.tqdm(enumerate(self.uav_dataloader_test)):
            uavimgs, coords_uav = data[0].to(self.device), data[1].to(self.device)
            break

        # 逆向旋转UAV图像（使其与rot=0的卫星图对齐）
        rot_to_align_deg = torch.rad2deg(-coords_uav[:, 2]).cpu().numpy()  # 逆向旋转角度
        from util_batch_rotation import batch_rotate_images_per_sample
        uavimgs_wo_rot = batch_rotate_images_per_sample(
            uavimgs,  # [B, C, H, W]
            rot_to_align_deg  # [B]
        )

        # 调整坐标：rot=0
        coords_uav_wo_rot = coords_uav.clone()
        coords_uav_wo_rot[:, 2] = 0  # rot = 0

        # 提取query特征
        with torch.no_grad():
            feats_q = self._get_feats_fm_imgs(uavimgs_wo_rot, feat_fm_agg=True)
            feats_q = TF.normalize(feats_q, dim=-1, p=2)

        # ==================== 检索和评估 ====================
        import faiss
        topN = 50
        feat_gallery_index_l2 = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index_l2.add(feat_gallery_flatten.detach().cpu().numpy())
        feat_dist_l2, indices_l2 = feat_gallery_index_l2.search(feats_q.detach().cpu().numpy(), k=topN)

        coords_gallery_topN = coords_gallery_flatten[torch.from_numpy(indices_l2[:, :topN])]
        dist_nrc_topN = torch.norm(
            coords_uav_wo_rot[:, None, :2].to(coords_gallery_topN.device) - coords_gallery_topN[:, :, :2], p=2, dim=-1
        )
        nrc_loc_success = dist_nrc_topN < self.sat_dataset.halfimg_radius_nrc

        k_values = [1, 5, 10, 20, 50]
        recall_res = []
        for k in k_values:
            recall = (nrc_loc_success[:, :k].sum(dim=-1) > 0).float().mean()
            recall_res.append(recall)

        mode_str = f"multiscale({n_scales})" if use_multiscale else "single scale"
        info2log = f"Recall@K ({mode_str}, wo rot): " + " | ".join([f"R@{k}={r.item() * 100:.2f}%" for k, r in zip(k_values, recall_res)])
        print(info2log)
        if hasattr(self, 'logger'):
            self.logger.info(info2log)

        if vis:
            res_maps = torch.norm(feats_q.unsqueeze(1)-feat_gallery_flatten.unsqueeze(0).to(feats_q.device),dim=-1,p=2)
            res_map = res_maps[0].reshape(*feat_gallery_dict['gallery_shape'])
            from util_vis_retrieval_in_2d import visualize_response_map_3d
            visualize_response_map_3d(res_map)
            visualize_response_map_3d(torch.exp(-res_map))


if __name__ == "__main__":
    trainer = GridHashFitTrainer()
    trainer.train()
