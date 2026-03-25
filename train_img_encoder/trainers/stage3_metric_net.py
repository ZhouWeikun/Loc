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

# 添加路径：train_img_encoder目录 和 项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
train_img_encoder_dir = os.path.dirname(current_dir)  # train_img_encoder/
project_root = os.path.dirname(train_img_encoder_dir)  # pyproj_neuloc_v0/

sys.path.insert(0, train_img_encoder_dir)  # 用于导入core, models等
sys.path.insert(0, project_root)  # 用于导入tool模块

from trainers.stage2_grid_hashfit_v0 import GridHashFitTrainer
from core.base.components import NetworkComponents
from util_udf_computer import UDFComputer
from models.pos_encoder import encode_4d_coords


class MetricNetTrainer(GridHashFitTrainer):
    """
    Stage 3: MetricNet Trainer

    继承自GridHashFitTrainer，在其基础上添加MetricNet
    """

    def __init__(self, opt=None):
        """初始化Stage 3 Trainer"""
        # 调用父类初始化（会初始化vis_encoder, grid等）
        super().__init__(opt)

        # 加载Stage 2的Grid权重（如果指定）
        if self.opt.load_stage2_ckpt:
            self._load_stage2_checkpoint()

        # 初始化MetricNet
        self._init_metric_net()

        # 重新设置可训练参数
        self._setup_trainable_params_stage3()


    def _load_stage2_checkpoint(self):
        """加载Stage 2训练好的Grid"""
        print(f"\n加载Stage 2 checkpoint: {self.opt.load_stage2_ckpt}")

        self._load_checkpoint(
            self.opt.load_stage2_ckpt,
            {
                'grid': self.grid,
                'grid_mlp': self.grid_mlp
            }
        )

        print("✅ Stage 2模型加载完成\n")


    def _init_metric_net(self):
        """初始化MetricNet"""
        print("\n" + "="*80)
        print("初始化 MetricNet")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        self.metric_net = components.create_metric_net(
            query_dim=self.feat_q_dim,
            ref_dim=self.feat_q_dim,
            coord_dim=self.coord_encoded_dim,
            hidden_dim=256,
            num_layers=3
        )

        print("="*80 + "\n")


    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        # 选项1：仅训练metric_net（Grid冻结）
        if getattr(self.opt, 'freeze_grid', True):
            for param in self.grid.parameters():
                param.requires_grad = False
            for param in self.grid_mlp.parameters():
                param.requires_grad = False

            self.param2optimize = {
                'metric_net': self.metric_net
            }

            print("参数配置 (freeze_grid=True):")
            print("  可训练: metric_net")
            print("  冻结:   vis_encoder, vis_aggregator, grid, grid_mlp\n")

        # 选项2：同时微调grid_mlp
        else:
            for param in self.grid.parameters():
                param.requires_grad = False

            self.param2optimize = {
                'grid_mlp': self.grid_mlp,
                'metric_net': self.metric_net
            }

            print("参数配置 (freeze_grid=False):")
            print("  可训练: grid_mlp, metric_net")
            print("  冻结:   vis_encoder, vis_aggregator, grid\n")

        # 始终冻结Stage 1组件和Grid
        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
            'vis_aggregator': self.vis_aggregator,
            'grid': self.grid
        }


    def train(self):
        """Stage 3训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 3 训练: MetricNet")
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

        # 5. 配置Loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False)

        # 6. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch_uav in tqdm.tqdm(enumerate(self.uav_dataloader_train)):
                # 获取UAV数据
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)  # [B, 4]

                # 获取SAT数据
                batch_sat = next(iter(self.sat_dataloader))
                satimgs = batch_sat[0].to(self.device)
                coords_sat = batch_sat[1].to(self.device)  # [B, 4]

                # 提取视觉特征（冻结）
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([uavimgs, satimgs], dim=0)
                )  # [2B, feat_dim]

                # Ground truth坐标
                coords_gt = torch.cat([coords_uav, coords_sat], dim=0)  # [2B, 4]

                # 随机采样负样本坐标
                coords_rand = self.sat_dataset.mk_rand_coords_4d(
                    n_rand=1024,
                    return_tensor=True
                ).to(self.device)  # [1024, 4]

                # 所有坐标
                coords_all = torch.cat([coords_gt, coords_rand], dim=0)  # [2B+1024, 4]

                # 位置编码
                coords_all_encoded = encode_4d_coords(
                    coords_all,
                    rc_encoder=self.rc_pos_encoder,
                    rot_endcoder=self.rot_pos_encoder,
                    scale_encoder=self.scale_pos_encoder
                )  # [2B+1024, coord_encoded_dim]

                # 从Grid提取特征
                feats_all_grid = self._get_feats_fm_grid(coords_all)  # [2B+1024, feat_dim]
                feats_all_grid = self.grid_mlp(
                    feats_all_grid,
                    coords_all_encoded
                )
                feats_all_grid = TF.normalize(feats_all_grid, dim=-1)

                # === MetricNet距离预测 ===
                B_vis = feats_vis.shape[0]  # 2B
                N_grid = feats_all_grid.shape[0]  # 2B+1024

                # 扩展为MetricNet输入格式 [B_vis, N_grid, C]
                feats_vis_expanded = feats_vis.unsqueeze(1).expand(B_vis, N_grid, -1)
                feats_grid_expanded = feats_all_grid.unsqueeze(0).expand(B_vis, N_grid, -1)
                coords_all_encoded_expanded = coords_all_encoded.unsqueeze(0).expand(B_vis, N_grid, -1)

                # MetricNet前向
                metric_dist_mat = self.metric_net(
                    feats_vis_expanded,
                    feats_grid_expanded,
                    coords_all_encoded_expanded
                )  # [B_vis, N_grid]

                metric_dist_mat_act = TF.softplus(metric_dist_mat)

                # === 计算UDF Loss ===
                udf_dist_mat = self.udf_compter.compute_udf(coords_gt, coords_all)
                positive_mat = udf_dist_mat < self.sat_dataset.halfimg_radius_nrc

                loss = loss_swt(metric_dist_mat_act, positive_mat, udf_dist_mat, w_weight=True)

                # === Eikonal正则化 ===
                eikonal_points = self.sat_dataset.mk_rand_coords_4d(
                    n_rand=1024,
                    return_tensor=True
                ).to(self.device)
                eikonal_points.requires_grad = True

                # 固定query特征
                query_feat_eikonal = feats_vis[0:1].detach()

                # 冻结网络获取grid特征
                with torch.no_grad():
                    feats_grid_eikonal_raw = self._get_feats_fm_grid(eikonal_points)
                    coords_eikonal_encoded_frozen = encode_4d_coords(
                        eikonal_points,
                        rc_encoder=self.rc_pos_encoder,
                        rot_endcoder=self.rot_pos_encoder,
                        scale_encoder=self.scale_pos_encoder
                    )
                    feats_grid_eikonal = self.grid_mlp(
                        feats_grid_eikonal_raw,
                        coords_eikonal_encoded_frozen
                    )
                    feats_grid_eikonal = TF.normalize(feats_grid_eikonal, dim=-1)
                    feats_grid_exp = feats_grid_eikonal.unsqueeze(0)

                # 非冻结的坐标编码
                coords_eikonal_encoded = encode_4d_coords(
                    eikonal_points,
                    rc_encoder=self.rc_pos_encoder,
                    rot_endcoder=self.rot_pos_encoder,
                    scale_encoder=self.scale_pos_encoder
                )
                coords_enc_exp = coords_eikonal_encoded.unsqueeze(0)

                # 扩展query
                query_feat_exp = query_feat_eikonal.unsqueeze(1).expand(
                    1, eikonal_points.shape[0], -1
                )

                # MetricNet前向
                dist_eikonal = self.metric_net(
                    query_feat_exp,
                    feats_grid_exp,
                    coords_enc_exp
                )
                dist_eikonal_act = TF.softplus(dist_eikonal)

                # 计算梯度
                grad_outputs = torch.ones_like(dist_eikonal_act)
                grad_coords = torch.autograd.grad(
                    outputs=dist_eikonal_act,
                    inputs=eikonal_points,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]

                # Eikonal Loss
                grad_norm = grad_coords.norm(dim=-1)
                loss_eikonal = ((grad_norm - 1.0) ** 2).mean()

                # 总Loss
                lambda_eikonal = 0.01
                loss = loss + lambda_eikonal * loss_eikonal

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
            if (epoch % 99 == 0) and (epoch > 0):
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

            # 备份代码
            if epoch == 0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(self.exp_dir2save, self.opt)

        self.logger.info("✅ Stage 3 训练完成！")

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

    def _opt_coords_topN(self, coords_topN, feat_q, n_step=500):
        """
        使用metric_net作为损失函数进行坐标优化

        Args:
            coords_topN: [B, N, 4] - topN候选坐标
            feat_q: [B, C] - query特征
            n_step: 优化步数

        Returns:
            coords_sorted: [B, N, 4] - 优化并重排序后的坐标
        """
        feat_q = feat_q.to(self.device)
        coords_opted_topN = []
        loss_topN = []

        for id in range(coords_topN.shape[1]):
            coords2opt = torch.tensor(coords_topN[:, id, :], dtype=torch.float32,
                                     device=feat_q.device, requires_grad=True)
            optimizer = torch.optim.Adam([coords2opt], lr=1e-4)

            for i in range(n_step):
                optimizer.zero_grad()

                # 获取grid特征
                feat_ref = self._get_feats_fm_grid(coords2opt)
                encoded_coords_opted = encode_4d_coords(coords2opt,
                                                        rc_encoder=self.rc_pos_encoder,
                                                        rot_endcoder=self.rot_pos_encoder,
                                                        scale_encoder=self.scale_pos_encoder)
                feat_ref = self.grid_mlp(inputs=feat_ref, condition_features=encoded_coords_opted)
                feat_ref = TF.normalize(feat_ref, dim=-1, p=2)

                # 使用metric_net计算距离作为损失
                metric_dist = self.metric_net(
                    feat_q,  # [B, C] -> 会自动broadcast
                    feat_ref,  # [B, C]
                    encoded_coords_opted  # [B, D]
                )  # 输出 [B]
                metric_dist = torch.nn.functional.softplus(metric_dist)
                loss = metric_dist.mean()

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Candidate {id}, Step {i}, Metric Loss: {loss.item():.4f}")

            # 保存结果
            with torch.no_grad():
                feat_ref_final = self._get_feats_fm_grid(coords2opt)
                encoded_coords_final = encode_4d_coords(coords2opt,
                                                        rc_encoder=self.rc_pos_encoder,
                                                        rot_endcoder=self.rot_pos_encoder,
                                                        scale_encoder=self.scale_pos_encoder)
                feat_ref_final = self.grid_mlp(inputs=feat_ref_final, condition_features=encoded_coords_final)
                feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)

                # 使用metric_net计算最终距离
                final_metric_dist = self.metric_net(
                    feat_q,
                    feat_ref_final,
                    encoded_coords_final
                )
                final_metric_dist = torch.nn.functional.softplus(final_metric_dist)
                loss_topN.append(final_metric_dist)

            coords_opted_topN.append(coords2opt.detach())

        coords_opted_topN = torch.stack(coords_opted_topN, dim=1)
        loss_topN = torch.stack(loss_topN, dim=1)

        # 候选重排序
        sorted_indices = loss_topN.argsort(dim=1)  # [B, N] 从小到大排序
        # 重新排列坐标和loss
        coords_sorted = coords_opted_topN[
            torch.arange(coords_opted_topN.shape[0]).unsqueeze(1),  # [B, 1]
            sorted_indices  # [B, N]
        ]  # [B, N, 4] - 按metric loss从小到大排序
        return coords_sorted

    def _print_error_comparison(self, coords_topN, coords_topN_opted, coords_gt):
        """
        打印优化前后的误差对比

        Args:
            coords_topN: [B, N, 4] - 优化前的topN候选坐标
            coords_topN_opted: [B, N, 4] - 优化后的topN候选坐标
            coords_gt: [B, 4] - ground truth坐标
        """
        import math

        coords_best = coords_topN_opted[:, 0, :]

        # ==================== Recall@1 ====================
        dist_nrc_top1 = torch.norm(coords_topN[:, 0, :2].cpu() - coords_gt[:, :2].cpu(), dim=-1, p=2)
        nrc_loc_success_top1 = dist_nrc_top1 < self.sat_dataset.halfimg_radius_nrc
        dist_nrc_topN_opted = torch.norm(coords_best[:, :2].cpu() - coords_gt[:, :2].cpu(), dim=-1, p=2)
        nrc_loc_success_opt1_opted = dist_nrc_topN_opted < self.sat_dataset.halfimg_radius_nrc
        print(f'nrc_recall@1: {nrc_loc_success_top1.sum() / coords_gt.shape[0]:.5f}; '
              f'nrc_recall@1_opted: {nrc_loc_success_opt1_opted.sum() / coords_gt.shape[0]:.5f}')

        # ==================== 位置误差 ====================
        err_nrc_top1 = torch.norm(coords_topN[:, 0, :2].cpu() - coords_gt[:, :2].cpu(), dim=-1)
        err_met_top1_mean = self.sat_dataset.halfimg_radius_meter * err_nrc_top1.mean() / self.sat_dataset.halfimg_radius_nrc
        err_nrc_top1_opted = torch.norm(coords_best[:, :2].cpu() - coords_gt[:, :2].cpu(), dim=-1)
        err_met_top1_opted_mean = self.sat_dataset.halfimg_radius_meter * err_nrc_top1_opted.mean() / self.sat_dataset.halfimg_radius_nrc
        print(f'err_nrc_top1: {err_nrc_top1.mean().item():.5f}; '
              f'err_nrc_top1_opted: {err_nrc_top1_opted.mean().item():.5f}')
        print(f'err_meter_top1: {err_met_top1_mean.item():.2f}m; '
              f'err_meter_top1_opted: {err_met_top1_opted_mean.item():.2f}m')

        # ==================== 旋转误差 ====================
        rot_diff_top1 = coords_topN[:, 0, 2].cpu() - coords_gt[:, 2].cpu()
        rot_err_top1 = torch.abs(torch.atan2(torch.sin(rot_diff_top1), torch.cos(rot_diff_top1)))
        rot_err_top1_mean = torch.rad2deg(rot_err_top1).mean()
        rot_diff_top1_opted = coords_best[:, 2].cpu() - coords_gt[:, 2].cpu()
        rot_err_top1_opted = torch.abs(torch.atan2(torch.sin(rot_diff_top1_opted), torch.cos(rot_diff_top1_opted)))
        rot_err_top1_opted_mean = torch.rad2deg(rot_err_top1_opted).mean()
        print(f'err_rot_top1: {rot_err_top1_mean.item():.2f}°; '
              f'err_rot_top1_opted: {rot_err_top1_opted_mean.item():.2f}°')

        # ==================== 尺度误差 ====================
        norm_factor_scale = math.log(
            self.sat_dataset.satimgsize_scale_to_refm_boundary[1] /
            self.sat_dataset.satimgsize_scale_to_refm_boundary[0])
        scale_err_top1 = torch.abs(torch.log(coords_topN[:, 0, 3].cpu() / coords_gt[:, 3].cpu()))
        scale_err_top1 = scale_err_top1 / norm_factor_scale
        scale_err_top1_opted = torch.abs(torch.log(coords_topN_opted[:, 0, 3].cpu() / coords_gt[:, 3].cpu()))
        scale_err_top1_opted = scale_err_top1_opted / norm_factor_scale
        print(f'err_normed_scale_top1: {scale_err_top1.mean().item():.5f}; '
              f'err_normed_scale_top1_opted: {scale_err_top1_opted.mean().item():.5f}')

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

    def test_xy_scale_fm_metric_net_wUAV(self, overlap=0.5, scale=None, use_multiscale=False, n_scales=3, opt=False, vis=False):
        """
        使用metric_net替代L2距离进行检索测试（简化版：对UAV图像逆向旋转，特征库不包含旋转）

        与test_xy_scale_fm_INGP_wUAV的主要区别是使用可学习的metric_net计算距离

        Args:
            overlap: 裁剪重叠度
            scale: 尺度值，None则使用默认值（仅当use_multiscale=False时使用）
            use_multiscale: 是否使用多尺度
            n_scales: 尺度数量（仅当use_multiscale=True时使用）
            opt: 是否进行迭代优化
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
            gallery_coords_encoded = []

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
                gallery_coords_encoded.append(feat_gallery_dict['coords_gallery_encoded_flatten'])

            feat_gallery_flatten = torch.cat(gallery_features, dim=0)
            coords_gallery_flatten = torch.cat(gallery_coords, dim=0)
            coords_gallery_encoded_flatten = torch.cat(gallery_coords_encoded, dim=0)
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
            coords_gallery_encoded_flatten = feat_gallery_dict['coords_gallery_encoded_flatten']

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

        # 计算ground truth特征（用于对比）
        feats_pos = self._get_feats_fm_grid(coords_uav_wo_rot)
        coords_gt_encoded = encode_4d_coords(coords_uav_wo_rot, rc_encoder=self.rc_pos_encoder,
                                            rot_endcoder=self.rot_pos_encoder,
                                            scale_encoder=self.scale_pos_encoder)
        feats_pos = self.grid_mlp(inputs=feats_pos, condition_features=coords_gt_encoded)
        feats_pos = TF.normalize(feats_pos, dim=1, p=2)

        # ==================== 使用metric_net计算距离矩阵 ====================
        mode_str = f"multiscale({n_scales})" if use_multiscale else "single scale"
        print("\n" + "="*60)
        print(f"使用Metric Net计算距离 ({mode_str})...")
        print("="*60)

        B_q = feats_q.shape[0]  # query数量
        N_gallery = feat_gallery_flatten.shape[0]  # gallery数量
        print(f"Query数量: {B_q}, Gallery数量: {N_gallery}")

        # 分批计算以避免OOM
        batch_size_gallery = 2048  # 每次处理2048个gallery candidates
        metric_dist_mat = []

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, N_gallery, batch_size_gallery), desc="Computing metric distances"):
                end_idx = min(i + batch_size_gallery, N_gallery)
                batch_size = end_idx - i

                # 获取当前批次的gallery特征和坐标
                feat_gallery_batch = feat_gallery_flatten[i:end_idx].to(self.device)  # [batch_size, C]
                coords_gallery_encoded_batch = coords_gallery_encoded_flatten[i:end_idx].to(self.device)  # [batch_size, D]

                # 扩展为 [B_q, batch_size, C] 格式
                feats_q_expanded = feats_q.unsqueeze(1).expand(B_q, batch_size, -1)  # [B_q, batch_size, C]
                feat_gallery_expanded = feat_gallery_batch.unsqueeze(0).expand(B_q, batch_size, -1)  # [B_q, batch_size, C]
                coords_gallery_encoded_expanded = coords_gallery_encoded_batch.unsqueeze(0).expand(B_q, batch_size, -1)  # [B_q, batch_size, D]

                # 使用metric_net计算距离
                metric_dist_batch = self.metric_net(
                    feats_q_expanded,  # [B_q, batch_size, C]
                    feat_gallery_expanded,  # [B_q, batch_size, C]
                    coords_gallery_encoded_expanded  # [B_q, batch_size, D]
                )  # 输出 [B_q, batch_size]
                metric_dist_batch = torch.nn.functional.softplus(metric_dist_batch)

                metric_dist_mat.append(metric_dist_batch.cpu())  # 移到CPU节省GPU内存

        # 拼接所有批次
        metric_dist_mat = torch.cat(metric_dist_mat, dim=1)  # [B_q, N_gallery]
        print(f"Metric distance matrix shape: {metric_dist_mat.shape}")

        # 获取topN候选
        n_top2eval = 100
        metric_dist_topN, indices_topN = torch.topk(metric_dist_mat, k=n_top2eval, dim=-1, largest=False)  # 距离越小越好

        print(f'metric_dist(f_pred_best,f_q).mean:{metric_dist_topN[:, 0].mean():.3f}')

        # 评估recall
        coords_gallery_topN = coords_gallery_flatten[indices_topN.cpu()]  # [B_q, topN, 4]
        dist_nrc_topN = torch.norm(coords_uav_wo_rot[:, None, :2].cpu() - coords_gallery_topN[:, :, :2], p=2, dim=-1)
        nrc_loc_success = dist_nrc_topN < self.sat_dataset.halfimg_radius_nrc

        k_values = [1, 5, 10, 20, 50, 100]
        recall_res = []
        for k in k_values:
            recall = (nrc_loc_success[:, :k].sum(dim=-1) > 0).sum() / nrc_loc_success.shape[0]
            recall_res.append(recall)

        info2log = f"Recall@K (Metric Net, {mode_str}, wo rot): " + " | ".join([f"R@{k}={r.item() * 100:.2f}%" for k, r in zip(k_values, recall_res)])
        print(info2log)
        if hasattr(self, 'logger'):
            self.logger.info(info2log)

        # vis the res_map
        if vis:
            res_map = metric_dist_mat[0].reshape(*feat_gallery_dict['gallery_shape'])
            from util_vis_retrieval_in_2d  import  visualize_response_map_3d
            visualize_response_map_3d(torch.exp(-res_map))

        if not opt:
            return 0
        # ==================== 迭代优化（使用metric_net作为损失） ====================
        print("\n" + "="*60)
        print("开始Metric Net引导的迭代优化...")
        print("="*60)

        n2opt = 10
        coords_topN = coords_gallery_topN[:, :n2opt].to(self.device)
        coords_topN_opted = self._opt_coords_topN(coords_topN, feats_q)

        dist_nrc_topN = torch.norm(coords_uav_wo_rot[:, None, :2].cpu() - coords_topN_opted[:, :, :2].cpu(), p=2, dim=-1)
        nrc_loc_success_opted = dist_nrc_topN < self.sat_dataset.halfimg_radius_nrc
        k_values = [1, 5, 10]
        recall_res = []
        for k in k_values:
            recall = (nrc_loc_success_opted[:, :k].sum(dim=-1) > 0).sum() / nrc_loc_success_opted.shape[0]
            recall_res.append(recall)

        info2log_opted = f"opted_Recall@K (Metric Net, {mode_str}, wo rot): " + " | ".join([f"R@{k}={r.item() * 100:.2f}%" for k, r in zip(k_values, recall_res)])
        print(info2log_opted)
        if hasattr(self, 'logger'):
            self.logger.info(info2log_opted)

        # 打印详细误差指标
        self._print_error_comparison(coords_topN, coords_topN_opted, coords_uav_wo_rot)


if __name__ == "__main__":
    trainer = MetricNetTrainer()
    trainer.train()
