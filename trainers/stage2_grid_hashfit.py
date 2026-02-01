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

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.base.components import NetworkComponents
from trainer_depends.utils.util_udf_computer import UDFComputer
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
        # version1:
        self.pos_encoder_864 = components.create_coords_5d_encoder(
            multires_rc=8,
            multires_rot=6,
            multires_scale=4
        )
        self.pos_encoder_grid = self.pos_encoder_864

        # Stage 2组件（将被训练）
        self.grid = components.create_grid()
        # version1：
        self.grid_mlp = components.create_grid_mlp(
            self.feat_q_dim,
            self.pos_encoder_grid.out_dim,
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

        # 动态生成参数配置信息
        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys())

        print("参数配置:")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")


    def _get_feats_fm_grid(self, grid_coords_normed,z_padding=0.025):
        """
        纯粹的 Grid 查表函数

        Args:
            grid_coords_normed: [N, 3] or [B, N, 3]
                                对应 (nr, nc, z_axis)，z_axis=log_scale or nrot
                                范围必须是 [-1, 1] (来自 CoordsNormProcessor)
            z_padding: float, Z轴边界留空比例 (0.0 ~ 0.5).
                       例如 0.05 表示将 Z轴映射到 [0.05, 0.95]。
                       这对 Z=Rotation 方案至关重要，防止 0/1 边界断裂问题。
        Returns:
            feats_grid: [N, feat_dim] or [B, N, feat_dim]
        """
        # 1. 维度展平 (Handle Batch)
        input_shape = grid_coords_normed.shape
        if len(input_shape) == 3:
            coords_flat = grid_coords_normed.flatten(0, 1)  # [B*N, 3]
        else:
            coords_flat = grid_coords_normed

        # 2. 范围映射: [-1, 1] -> [0, 1]
        # (Instant-NGP) 的 HashGrid 要求输入严格在 [0, 1] 之间
        # 这一步放在这里做最安全，保证进入 TCNN 前一刻数据是合法的
        grid_input_01 = (coords_flat + 1.0) * 0.5

        # 3. [新增] Z轴 Padding (解决旋转断裂 & 边界数值稳定问题)
        # 仅对最后一维 (Z轴) 进行压缩映射: [0, 1] -> [p, 1-p]
        if z_padding > 0.0:
            # 缩放系数: 原长 1.0 -> 新长 (1.0 - 2*p)
            scale_factor = 1.0 - 2.0 * z_padding
            # 原地修改最后一列 (Z轴)
            # 公式: z_new = z_old * scale + padding
            # 例 (p=0.05): 0.0 -> 0.05, 1.0 -> 0.95, 0.5 -> 0.5 (中心不变)
            grid_input_01[:, 2] = grid_input_01[:, 2] * scale_factor + z_padding

            # 4. 安全钳位 (Double Safety)
            # 虽然有了 Padding 应该不会出界，但为了防止 float32 精度误差导致 1.0000001
            # 显式 clamp 是最稳妥的，TCNN 对越界非常敏感
            grid_input_01 = torch.clamp(grid_input_01, 0.0, 1.0)

        # 5. 查表 (自动处理多分辨率插值)
        feats_grid = self.grid.interpolate(grid_input_01, len(self.grid.active_lods) - 1) #拼接得到的多尺度特征

        # 6. 恢复维度
        if len(input_shape) == 3:
            feats_grid = feats_grid.view(input_shape[0], input_shape[1], -1)

        return feats_grid




    def train(self):
        """Stage 2训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 2 训练: Grid HashFit")
        print("🚀"*40 + "\n")

        # 0. 初始化GradScaler（如果使用autocast）
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

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

        # 4.5 初始化4d坐标归一化器
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)
        # 5. 配置Loss
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

                # 合并坐标并转到欧式空间
                coords_all = torch.cat([coords_sat, coords_uav], dim=0)  # [2B, 4]
                # coords_all_5d = self.coord_normer.raw_to_norm(coords_all)
                coords_all_6d = self.coord_normer.raw_to_norm(coords_all,append_linear_rot=True)
                # coords_all_rot_rad = coords_all[:,2:3]

                # 从Grid提取特征
                feats_grid = self._get_feats_fm_grid(torch.concatenate([coords_all_6d[:,:2],coords_all_6d[:,-1:]],dim=-1))  # [2B, feat_dim]
                # 位置编码
                # version1:
                coords_all_encoded = self.pos_encoder_grid(
                    coords_all_6d[:,:5],
                )  # [2B, coord_encoded_dim]
                # Grid MLP调制
                feats_grid = self.grid_mlp(
                    inputs=feats_grid,
                    condition_features=coords_all_encoded
                )  # [2B, feat_dim]

                # 提取视觉特征（冻结）
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([satimgs, uavimgs], dim=0)
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
                if (epoch % 10 == 0):
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



    def test(self):
        """
        测试并可视化 INGP 的拟合能力
        """
        import yaml
        print("\n" + "🧪"*40)
        print("开始 Stage 2 测试: INGP 拟合能力可视化")
        print("🧪"*40 + "\n")

        # 1. 初始化数据集和坐标归一化器 (test模式下也需要)
        self._init_datasets(create_train_loader=False)
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=self.opt.batchsize_uav,
            num_workers=self.opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        # 2. 加载checkpoint
        self._load_checkpoints_for_test()

        # 3. 开始测试可视化
        # self.visualize_similarity_in_3d(delta=0.1)

        # 4. 开始测试检索定位
        # 4.1 测试RC定位（消除旋转差异）
        # print("\n" + "🔹"*40)
        # print("测试1: RC定位（消除旋转差异）")
        # print("🔹"*40)
        # self.test_rc_localization(overlap=0.5, fixed_scale=None)

        # 4.2 测试RC + 旋转定位
        # print("\n" + "🔹"*40)
        # print("测试2: RC + 旋转定位")
        # print("🔹"*40)
        # self.test_rc_rot_localization(overlap=0.5, delta_rot_deg=10, fixed_scale=None)

        # 4.3 测试RC + 尺度定位
        # print("\n" + "🔹"*40)
        # print("测试3: RC + 尺度定位")
        # print("🔹"*40)
        # self.test_rc_scale_localization(overlap=0.5, n_scales=3)

        # 4.4 测试RC + 旋转 + 尺度定位
        print("\n" + "🔹"*40)
        print("测试4: RC + 旋转 + 尺度定位")
        print("🔹"*40)
        self.test_rc_rot_scale_localization(overlap=0.5, delta_rot_deg=10, n_scales=3)




    def _load_checkpoints_for_test(self):
        """
        测试时加载checkpoint的统一方法

        加载逻辑：
        1. Stage 2自身的checkpoint (grid, grid_mlp)
        2. Stage 1的预训练模型 (vis_encoder, vis_aggregator)
        """
        import yaml

        print("\n" + "="*80)
        print("加载测试用的checkpoint")
        print("="*80)

        # --- 1. 加载Stage 2的checkpoint (当前stage) ---
        stage2_ckpt_path = self._get_stage2_checkpoint_path()

        if stage2_ckpt_path:
            print(f"\n📦 Stage 2 checkpoint: {stage2_ckpt_path}")
            self._load_checkpoint(
                {'grid': stage2_ckpt_path, 'grid_mlp': stage2_ckpt_path},
                self.param2optimize,
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 2的checkpoint，无法进行测试。")

        # --- 2. 加载Stage 1的checkpoint (依赖的预训练模型) ---
        stage1_ckpt_path = self._get_stage1_checkpoint_path(stage2_ckpt_path)

        if stage1_ckpt_path:
            print(f"\n📦 Stage 1 checkpoint: {stage1_ckpt_path}")
            self._load_checkpoint(
                {'vis_encoder': stage1_ckpt_path, 'vis_aggregator': stage1_ckpt_path},
                self.param2freeze,
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 1的checkpoint，无法进行测试。")

        print("\n" + "="*80)
        print("✅ 所有checkpoint加载完成")
        print("="*80 + "\n")


    def _get_stage2_checkpoint_path(self):
        """获取Stage 2的checkpoint路径"""
        # 优先使用命令行参数指定的路径
        if hasattr(self.opt, 'load2test') and self.opt.load2test:
            print(f"从opt.load2test读取: {self.opt.load2test}")
            return self.opt.load2test

        # 否则从实验目录中找最新的checkpoint
        if self.exp_dir2save and os.path.exists(self.exp_dir2save):
            ckpts = [f for f in os.listdir(self.exp_dir2save) if f.startswith('epoch')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace('epoch','').split('.')[0]))
                ckpt_path = os.path.join(self.exp_dir2save, ckpts[-1])
                print(f"从实验目录读取: {ckpt_path}")
                return ckpt_path

        print(f"⚠️  未找到Stage 2 checkpoint:")
        print(f"   opt.load2test = {getattr(self.opt, 'load2test', 'NOT SET')}")
        print(f"   exp_dir2save = {self.exp_dir2save}")
        return None


    def _get_stage1_checkpoint_path(self, stage2_ckpt_path):
        """
        获取Stage 1的checkpoint路径

        优先级：
        1. 命令行参数 (opt.load_stage1_ckpt)
        2. Stage 2实验目录中的opts.yaml
        """
        import yaml

        # 优先使用命令行参数
        if self.opt.load_stage1_ckpt:
            return self.opt.load_stage1_ckpt

        # 从Stage 2的opts.yaml中读取
        if stage2_ckpt_path:
            stage2_exp_dir = os.path.dirname(stage2_ckpt_path)
            stage2_opts_path = os.path.join(stage2_exp_dir, 'opts.yaml')

            if os.path.exists(stage2_opts_path):
                try:
                    with open(stage2_opts_path, 'r') as f:
                        stage2_opts = yaml.safe_load(f)

                    if 'exp_setting' in stage2_opts:
                        stage1_path = stage2_opts['exp_setting'].get('load_stage1_ckpt')
                        if stage1_path:
                            print(f"从opts.yaml读取Stage 1路径: {stage2_opts_path}")
                            return stage1_path
                except Exception as e:
                    print(f"⚠️  读取opts.yaml失败: {e}")

        return None

    def visualize_similarity_in_3d(self, metric='euclidean',delta=0.1):
        """
        在3D空间(r, c, rot)中可视化相似度分布

        Args:
            metric: 可视化的度量指标
                - 'cosine': 余弦相似度 (越大越好，范围[-1, 1])
                - 'euclidean': 欧式距离 (越小越好，范围[0, +∞))
        """
        # 0. 确保所有模型处于评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        print(f"\n{'='*80}")
        print(f"可视化度量指标: {metric.upper()}")
        print(f"{'='*80}\n")

        # 1. 从测试集中抽取一个UAV样本
        uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        uav_img = uav_img[0].to(self.device).unsqueeze(0)
        uav_coords_4d = uav_coords_4d[0].to(self.device)

        # 2. 提取"真值"视觉特征
        feats_vis_gt = self._get_feats_fm_imgs(uav_img)
        feats_vis_gt = TF.normalize(feats_vis_gt, dim=-1)

        # 3. 创建评估网格
        n_pts_grid = 32
        nr_center, nc_center, rot_center, scale_val = uav_coords_4d

        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_pts_grid, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_pts_grid, device=self.device)
        rot_range = torch.linspace(rot_center - torch.pi/2, rot_center + torch.pi/2, n_pts_grid, device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        grid_coords_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_val)
        ], dim=-1)

        # 4. 遍历网格，计算INGP特征和度量值
        with torch.no_grad():
            # 转换到6D空间（包含线性旋转）
            coords_all_6d = self.coord_normer.raw_to_norm(grid_coords_4d, append_linear_rot=True)

            # Grid输入：前2维(nr, nc) + 最后1维(log_scale)
            grid_coords_3d = torch.cat([coords_all_6d[:, 0:2], coords_all_6d[:, -1:]], dim=-1)

            # 从Grid提取特征
            feats_grid = self._get_feats_fm_grid(grid_coords_3d)

            # 位置编码：使用前5维（nr, nc, cos, sin, log_scale）
            coords_all_encoded = self.pos_encoder_grid(coords_all_6d[:, :5])

            # Grid MLP调制
            feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_all_encoded)
            feats_grid = TF.normalize(feats_grid, dim=-1)

            # 计算两种度量
            similarities = TF.cosine_similarity(feats_grid, feats_vis_gt.expand_as(feats_grid), dim=-1)
            dists = torch.norm(feats_grid - feats_vis_gt.expand_as(feats_grid), p=2, dim=-1)

        # 5. 根据度量类型选择可视化的值
        if metric == 'cosine':
            metric_values = similarities
            best_idx = torch.argmax(metric_values)
            best_value = metric_values[best_idx]
            metric_name = 'Cosine Similarity'
            colorscale = 'Viridis'  # 黄色表示高相似度
        elif metric == 'euclidean':
            metric_values = dists
            best_idx = torch.argmin(metric_values)
            best_value = metric_values[best_idx]
            metric_name = 'Euclidean Distance'
            colorscale = 'Viridis_r'  # 反转颜色，使蓝色表示小距离
        else:
            raise ValueError(f"不支持的度量类型: {metric}，请使用 'cosine' 或 'euclidean'")

        coord_pred_best = grid_coords_4d[best_idx]

        # 8. 可视化
        try:
            import plotly.graph_objects as go

            colors = metric_values.cpu().numpy()

            # Trace 1: 度量值点云 (背景)
            fig = go.Figure(data=go.Scatter3d(
                x=grid_coords_4d[:, 0].cpu().numpy(),
                y=grid_coords_4d[:, 1].cpu().numpy(),
                z=grid_coords_4d[:, 2].cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale=colorscale,
                    colorbar=dict(title=metric_name),
                    opacity=0.6
                ),
                name=f'{metric_name} Cloud'
            ))

            # Trace 2: GT 真值 (红色菱形)
            fig.add_trace(go.Scatter3d(
                x=[nr_center.cpu().item()],
                y=[nc_center.cpu().item()],
                z=[rot_center.cpu().item()],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name='Ground Truth',
                text=['GT'],
                textposition="top left"
            ))

            # Trace 3: 预测最佳值 (青色叉号)
            fig.add_trace(go.Scatter3d(
                x=[coord_pred_best[0].cpu().item()],
                y=[coord_pred_best[1].cpu().item()],
                z=[coord_pred_best[2].cpu().item()],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='cyan',
                    symbol='cross',
                    line=dict(width=2, color='blue')
                ),
                name=f'Best {metric_name}',
                text=[f'{best_value:.3f}'],
                textposition="top right"
            ))

            # 更新布局
            fig.update_layout(
                title=f'{metric_name} Distribution (Best: {best_value:.4f})',
                scene=dict(
                    xaxis_title='Normalized Row',
                    yaxis_title='Normalized Col',
                    zaxis_title='Rotation (rad)'
                )
            )

            fig.show()

        except ImportError:
            print("请安装 plotly 以进行3D可视化: pip install plotly")

        # 6. 打印分析结果
        print(f"\n=== Peak Analysis ({metric_name}) ===")
        print(f"GT Center : r={nr_center:.4f}, c={nc_center:.4f}, rot={rot_center:.4f}")
        print(f"Predicted Best : r={coord_pred_best[0]:.4f}, c={coord_pred_best[1]:.4f}, rot={coord_pred_best[2]:.4f}")
        print(f"Best {metric_name}: {best_value:.4f}")

        # 额外打印两种度量的对比
        max_sim_val = similarities[best_idx]
        min_dist_val = dists[best_idx]
        print(f"\n对应位置的两种度量:")
        print(f"  Cosine Similarity: {max_sim_val:.4f}")
        print(f"  Euclidean Distance: {min_dist_val:.4f}")


    def test_rc_localization(self, overlap=0.5, fixed_scale=None):
        """
        测试RC定位精度（通过逆向旋转UAV图像消除旋转差异）

        策略：
        - 特征库：固定rot=0，固定scale
        - UAV图像：逆向旋转到rot=0

        Args:
            overlap: 特征库采样的重叠度（0-1之间）
            fixed_scale: 固定的尺度值，None则使用数据集默认值
        """
        print("\n" + "="*80)
        print("测试 RC 定位精度 (通过逆向旋转消除rot差异)")
        print("="*80)
        print(f"策略: 特征库 rot=0, UAV图像逆向旋转到 rot=0")
        print(f"重叠度: {overlap}, 固定scale: {fixed_scale}")

        # 确保模型处于评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 1. 构建特征库 (rot=0)
        print("\n" + "-"*80)
        print("步骤1: 构建特征库 (rot=0, fixed_scale)")
        print("-"*80)

        feat_gallery_dict = self._build_feature_gallery_rc(
            overlap=overlap,
            fixed_rot=0.0,  # 固定rot=0
            fixed_scale=fixed_scale
        )

        feat_gallery = feat_gallery_dict['features']  # [N, feat_dim]
        coords_gallery = feat_gallery_dict['coords']  # [N, 4]
        grid_shape = feat_gallery_dict['grid_shape']  # (H, W)

        print(f"✅ 特征库大小: {feat_gallery.shape}")
        print(f"   网格形状: {grid_shape}")

        # 2. 采样UAV query并逆向旋转
        print("\n" + "-"*80)
        print("步骤2: 采样UAV测试集并逆向旋转")
        print("-"*80)

        for it, data in enumerate(self.uav_dataloader_test):
            uavimgs, coords_uav = data[0].to(self.device), data[1].to(self.device)
            break #只使用数据的前一小部分进行测试

        # 逆向旋转UAV图像（使其与rot=0的卫星图对齐）
        rot_to_align_deg = torch.rad2deg(-coords_uav[:, 2]).cpu().numpy()  # 逆向旋转角度

        from trainer_depends.utils.util_batch_rotation import batch_rotate_images_per_sample
        uavimgs_aligned = batch_rotate_images_per_sample(
            uavimgs,  # [B, C, H, W]
            rot_to_align_deg  # [B]
        )

        # 调整坐标：rot=0
        coords_uav_aligned = coords_uav.clone()
        coords_uav_aligned[:, 2] = 0  # rot = 0

        print(f"✅ UAV图像逆向旋转完成")

        # 3. 提取query特征
        with torch.no_grad():
            feats_query = self._get_feats_fm_imgs(uavimgs_aligned)

        print(f"✅ Query特征: {feats_query.shape}")

        # 4. Faiss检索
        print("\n" + "-"*80)
        print("步骤3: Faiss最近邻检索")
        print("-"*80)

        import faiss
        feat_gallery_index = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index.add(feat_gallery.detach().cpu().numpy())

        topK = 50
        feat_dist, indices = feat_gallery_index.search(feats_query.detach().cpu().numpy(), k=topK)

        print(f"✅ 检索完成，Top-{topK}结果")

        # 5. 计算Recall@K
        print("\n" + "-"*80)
        print("步骤4: 计算定位精度")
        print("-"*80)

        coords_gallery_topK = coords_gallery[torch.from_numpy(indices[:, :topK])]  # [B, K, 4]

        # 计算rc距离（使用对齐后的UAV坐标）
        dist_rc_topK = torch.norm(
            coords_uav_aligned[:, None, :2].cpu() - coords_gallery_topK[:, :, :2],
            p=2, dim=-1
        )  # [B, K]

        # 定位成功的阈值
        success_threshold = self.sat_dataset.halfimg_radius_nrc
        rc_loc_success = dist_rc_topK < success_threshold

        # 计算Recall@K
        k_values = [1, 5, 10, 20, 50]
        recalls = [(rc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log = f"Recall@K: " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)])
        print(f"\n📊 {info2log}")

        # 6. 计算定位误差
        print(f"\n📊 定位误差 (Top-1):")

        # RC误差（归一化坐标）
        err_rc_top1 = dist_rc_topK[:, 0]
        print(f"   RC误差 (归一化): {err_rc_top1.mean().item():.5f} ± {err_rc_top1.std().item():.5f}")

        # RC误差（米）
        err_meter_top1 = self.sat_dataset.halfimg_radius_meter * err_rc_top1 / self.sat_dataset.halfimg_radius_nrc
        print(f"   RC误差 (米):     {err_meter_top1.mean().item():6.2f}m ± {err_meter_top1.std().item():.2f}m")

        print("\n" + "="*80)
        print("✅ RC定位测试完成")
        print("="*80)

        return {
            'recall@k': {k: r for k, r in zip(k_values, recalls)},
            'error_rc_norm': err_rc_top1.mean().item(),
            'error_rc_meter': err_meter_top1.mean().item(),
        }


    def test_rc_rot_localization(self, overlap=0.5, delta_rot_deg=10, fixed_scale=None):
        """
        测试RC和旋转定位精度（支持旋转维度）

        与test_rc_localization的区别：
        - 特征库包含多个旋转角度
        - UAV图像不需要逆向旋转，直接用原始图像
        - 同时评估RC和旋转的定位精度

        Args:
            overlap: 特征库采样的重叠度（0-1之间）
            delta_rot_deg: 旋转角度间隔（度）
            fixed_scale: 固定的尺度值，None则使用数据集默认值
        """
        print("\n" + "="*80)
        print("测试 RC + 旋转 定位精度")
        print("="*80)
        print(f"策略: 特征库包含多个旋转角度，直接使用原始UAV图像")
        print(f"重叠度: {overlap}, 旋转间隔: {delta_rot_deg}°, 固定scale: {fixed_scale}")

        # 确保模型处于评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 1. 构建包含旋转的特征库
        print("\n" + "-"*80)
        print("步骤1: 构建特征库 (包含多个旋转角度)")
        print("-"*80)

        feat_gallery_dict = self._build_feature_gallery_rc_rot(
            overlap=overlap,
            delta_rot_deg=delta_rot_deg,
            fixed_scale=fixed_scale
        )

        feat_gallery = feat_gallery_dict['features']  # [N, feat_dim]
        coords_gallery = feat_gallery_dict['coords']  # [N, 4]
        gallery_shape = feat_gallery_dict['gallery_shape']  # (H, W, n_rots)

        print(f"✅ 特征库大小: {feat_gallery.shape}")
        print(f"   网格形状: {gallery_shape}")

        # 2. 采样UAV query（不需要旋转）
        print("\n" + "-"*80)
        print("步骤2: 采样UAV测试集")
        print("-"*80)

        for it, data in enumerate(self.uav_dataloader_test):
            uavimgs, coords_uav = data[0].to(self.device), data[1].to(self.device)
            break  # 快速验证，只用第一个batch

        print(f"✅ UAV图像: {uavimgs.shape}")

        # 3. 提取query特征
        with torch.no_grad():
            feats_query = self._get_feats_fm_imgs(uavimgs)

        print(f"✅ Query特征: {feats_query.shape}")

        # 4. Faiss检索
        print("\n" + "-"*80)
        print("步骤3: Faiss最近邻检索")
        print("-"*80)

        import faiss
        feat_gallery_index = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index.add(feat_gallery.detach().cpu().numpy())

        topK = 50
        feat_dist, indices = feat_gallery_index.search(feats_query.detach().cpu().numpy(), k=topK)

        print(f"✅ 检索完成，Top-{topK}结果")

        # 5. 计算Recall@K（RC定位）
        print("\n" + "-"*80)
        print("步骤4: 计算RC定位精度")
        print("-"*80)

        coords_gallery_topK = coords_gallery[torch.from_numpy(indices[:, :topK])]  # [B, K, 4]

        # 计算rc距离
        dist_rc_topK = torch.norm(
            coords_uav[:, None, :2].cpu() - coords_gallery_topK[:, :, :2],
            p=2, dim=-1
        )  # [B, K]

        # RC定位成功的阈值
        success_threshold_rc = self.sat_dataset.halfimg_radius_nrc
        rc_loc_success = dist_rc_topK < success_threshold_rc

        # 计算Recall@K
        k_values = [1, 5, 10, 20, 50]
        recalls = [(rc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log = f"Recall@K (RC): " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)])
        print(f"\n📊 {info2log}")

        # 6. 计算旋转误差
        print("\n" + "-"*80)
        print("步骤5: 计算旋转定位精度")
        print("-"*80)

        # 旋转误差（弧度差）
        rot_diff_topK = torch.abs(
            coords_uav[:, None, 2].cpu() - coords_gallery_topK[:, :, 2]
        )  # [B, K]

        # 处理周期性：旋转误差应该在[0, π]之间
        rot_diff_topK = torch.minimum(rot_diff_topK, 2*torch.pi - rot_diff_topK)

        # Top-1旋转误差
        rot_err_top1_rad = rot_diff_topK[:, 0]
        rot_err_top1_deg = torch.rad2deg(rot_err_top1_rad)

        print(f"📊 旋转误差 (Top-1):")
        print(f"   平均: {rot_err_top1_deg.mean().item():6.2f}° ± {rot_err_top1_deg.std().item():.2f}°")
        print(f"   中位数: {rot_err_top1_deg.median().item():6.2f}°")

        # 7. 计算RC定位误差
        print(f"\n📊 RC定位误差 (Top-1):")

        # RC误差（归一化坐标）
        err_rc_top1 = dist_rc_topK[:, 0]
        print(f"   RC误差 (归一化): {err_rc_top1.mean().item():.5f} ± {err_rc_top1.std().item():.5f}")
        print(f"   中位数: {err_rc_top1.median().item():6.2f}")

        # RC误差（米）
        err_meter_top1 = self.sat_dataset.halfimg_radius_meter * err_rc_top1 / self.sat_dataset.halfimg_radius_nrc
        print(f"   RC误差 (米):     {err_meter_top1.mean().item():6.2f}m ± {err_meter_top1.std().item():.2f}m")
        print(f"   中位数: {err_meter_top1.median().item():6.2f}m")

        print("\n" + "="*80)
        print("✅ RC + 旋转 定位测试完成")
        print("="*80)

        return {
            'recall@k': {k: r for k, r in zip(k_values, recalls)},
            'error_rc_norm': err_rc_top1.mean().item(),
            'error_rc_meter': err_meter_top1.mean().item(),
            'error_rot_deg': rot_err_top1_deg.mean().item(),
        }


    def test_rc_scale_localization(self, overlap=0.5, n_scales=3):
        """
        测试RC和尺度定位精度

        Args:
            overlap: 特征库采样的重叠度
            n_scales: 尺度数量
        """
        print("\n" + "="*80)
        print("测试 RC + 尺度 定位精度")
        print("="*80)
        print(f"策略: 特征库包含多个尺度，UAV图像逆向旋转到 rot=0")
        print(f"重叠度: {overlap}, 尺度数: {n_scales}")

        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        print("\n" + "-"*80)
        print("步骤1: 构建特征库 (包含多个尺度)")
        print("-"*80)
        
        feat_gallery_dict = self._build_feature_gallery_rc_scale(
            overlap=overlap,
            n_scales=n_scales
        )
        feat_gallery = feat_gallery_dict['features']
        coords_gallery = feat_gallery_dict['coords']
        
        print(f"✅ 特征库大小: {feat_gallery.shape}")

        print("\n" + "-"*80)
        print("步骤2: 采样UAV测试集并逆向旋转")
        print("-"*80)

        for it, data in enumerate(self.uav_dataloader_test):
            uavimgs, coords_uav = data[0].to(self.device), data[1].to(self.device)
            break

        rot_to_align_deg = torch.rad2deg(-coords_uav[:, 2]).cpu().numpy()
        from trainer_depends.utils.util_batch_rotation import batch_rotate_images_per_sample
        uavimgs_aligned = batch_rotate_images_per_sample(uavimgs, rot_to_align_deg)
        coords_uav_aligned = coords_uav.clone()
        coords_uav_aligned[:, 2] = 0

        with torch.no_grad():
            feats_query = self._get_feats_fm_imgs(uavimgs_aligned)

        print(f"✅ Query特征: {feats_query.shape}")

        print("\n" + "-"*80)
        print("步骤3: Faiss最近邻检索")
        print("-"*80)

        import faiss
        feat_gallery_index = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index.add(feat_gallery.detach().cpu().numpy())

        topK = 50
        feat_dist, indices = feat_gallery_index.search(feats_query.detach().cpu().numpy(), k=topK)

        print(f"✅ 检索完成，Top-{topK}结果")

        print("\n" + "-"*80)
        print("步骤4: 计算RC定位精度")
        print("-"*80)

        coords_gallery_topK = coords_gallery[torch.from_numpy(indices[:, :topK])]

        dist_rc_topK = torch.norm(
            coords_uav_aligned[:, None, :2].cpu() - coords_gallery_topK[:, :, :2],
            p=2, dim=-1
        )
        success_threshold_rc = self.sat_dataset.halfimg_radius_nrc
        rc_loc_success = dist_rc_topK < success_threshold_rc

        k_values = [1, 5, 10, 20, 50]
        recalls = [(rc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log = f"Recall@K (RC): " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)])
        print(f"\n📊 {info2log}")

        # 计算RC定位误差
        print(f"\n📊 RC定位误差 (Top-1):")
        err_rc_top1 = dist_rc_topK[:, 0]
        print(f"   RC误差 (归一化): {err_rc_top1.mean().item():.5f} ± {err_rc_top1.std().item():.5f}")
        print(f"   中位数: {err_rc_top1.median().item():6.2f}")
        
        print("\n" + "-"*80)
        print("步骤5: 计算尺度误差")
        print("-"*80)

        scale_err_top1 = torch.abs(torch.log(coords_gallery_topK[:, 0, 3] / coords_uav_aligned[:, 3].cpu()))
        norm_factor_scale = torch.log(torch.tensor(self.sat_dataset.satimgsize_scale_to_refm_boundary[1] / self.sat_dataset.satimgsize_scale_to_refm_boundary[0]))
        scale_err_top1_normed = scale_err_top1 / norm_factor_scale
        
        print(f"📊 尺度误差 (Top-1):")
        print(f"   归一化误差: {scale_err_top1_normed.mean().item():.5f} ± {scale_err_top1_normed.std().item():.5f}")

        print("\n" + "="*80)
        print("✅ RC + 尺度 定位测试完成")
        print("="*80)

        return {
            'recall@k': {k: r for k, r in zip(k_values, recalls)},
            'error_scale_normed': scale_err_top1_normed.mean().item(),
        }


    def test_rc_rot_scale_localization(self, overlap=0.5, delta_rot_deg=10, n_scales=3):
        """
        测试RC、旋转和尺度定位精度

        Args:
            overlap: 特征库采样的重叠度
            delta_rot_deg: 旋转角度间隔（度）
            n_scales: 尺度数量
        """
        print("\n" + "="*80)
        print("测试 RC + 旋转 + 尺度 定位精度")
        print("="*80)
        print(f"策略: 特征库包含多尺度和多旋转角度，直接使用原始UAV图像")
        print(f"重叠度: {overlap}, 旋转间隔: {delta_rot_deg}°, 尺度数: {n_scales}")

        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        print("\n" + "-"*80)
        print("步骤1: 构建特征库 (包含RC、旋转、尺度)")
        print("-"*80)

        feat_gallery_dict = self._build_feature_gallery_rc_rot_scale(
            overlap=overlap,
            delta_rot_deg=delta_rot_deg,
            n_scales=n_scales
        )

        feat_gallery = feat_gallery_dict['features']
        coords_gallery = feat_gallery_dict['coords']

        print(f"✅ 特征库大小: {feat_gallery.shape}")

        print("\n" + "-"*80)
        print("步骤2: 采样UAV测试集 (原始图像)")
        print("-"*80)

        for it, data in enumerate(self.uav_dataloader_test):
            uavimgs, coords_uav = data[0].to(self.device), data[1].to(self.device)
            break

        print(f"✅ UAV图像: {uavimgs.shape}")

        print("\n" + "-"*80)
        print("步骤3: 提取query特征")
        print("-"*80)

        with torch.no_grad():
            feats_query = self._get_feats_fm_imgs(uavimgs)

        print(f"✅ Query特征: {feats_query.shape}")

        print("\n" + "-"*80)
        print("步骤4: Faiss最近邻检索")
        print("-"*80)

        import faiss
        feat_gallery_index = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index.add(feat_gallery.detach().cpu().numpy())

        topK = 50
        feat_dist, indices = feat_gallery_index.search(feats_query.detach().cpu().numpy(), k=topK)

        print(f"✅ 检索完成，Top-{topK}结果")

        coords_gallery_topK = coords_gallery[torch.from_numpy(indices[:, :topK])]

        print("\n" + "-"*80)
        print("步骤5: 计算RC定位精度")
        print("-"*80)

        dist_rc_topK = torch.norm(
            coords_uav[:, None, :2].cpu() - coords_gallery_topK[:, :, :2],
            p=2, dim=-1
        )
        success_threshold_rc = self.sat_dataset.halfimg_radius_nrc
        rc_loc_success = dist_rc_topK < success_threshold_rc

        k_values = [1, 5, 10, 20, 50]
        recalls_rc = [(rc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log_rc = f"Recall@K (RC): " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls_rc)])
        print(f"\n📊 {info2log_rc}")

        err_rc_top1 = dist_rc_topK[:, 0]
        print(f"\n📊 RC定位误差 (Top-1):")
        print(f"   RC误差 (归一化): {err_rc_top1.mean().item():.5f} ± {err_rc_top1.std().item():.5f}")
        print(f"   中位数: {err_rc_top1.median().item():6.2f}")

        print("\n" + "-"*80)
        print("步骤6: 计算旋转定位误差")
        print("-"*80)

        rot_diff_topK = torch.abs(
            coords_uav[:, None, 2].cpu() - coords_gallery_topK[:, :, 2]
        )
        rot_diff_topK = torch.minimum(rot_diff_topK, 2*torch.pi - rot_diff_topK)
        rot_err_top1_rad = rot_diff_topK[:, 0]
        rot_err_top1_deg = torch.rad2deg(rot_err_top1_rad)

        print(f"📊 旋转误差 (Top-1):")
        print(f"   平均: {rot_err_top1_deg.mean().item():6.2f}° ± {rot_err_top1_deg.std().item():.2f}°")
        print(f"   中位数: {rot_err_top1_deg.median().item():6.2f}°")

        print("\n" + "-"*80)
        print("步骤7: 计算尺度定位误差")
        print("-"*80)

        scale_err_top1 = torch.abs(torch.log(coords_gallery_topK[:, 0, 3] / coords_uav[:, 3].cpu()))
        norm_factor_scale = torch.log(torch.tensor(self.sat_dataset.satimgsize_scale_to_refm_boundary[1] / self.sat_dataset.satimgsize_scale_to_refm_boundary[0]))
        scale_err_top1_normed = scale_err_top1 / norm_factor_scale
        
        print(f"📊 尺度误差 (Top-1):")
        print(f"   归一化误差: {scale_err_top1_normed.mean().item():.5f} ± {scale_err_top1_normed.std().item():.5f}")
        print(f"   中位数: {scale_err_top1_normed.median().item():6.2f}")

        print("\n" + "="*80)
        print("✅ RC + 旋转 + 尺度 定位测试完成")
        print("="*80)

        return {
            'recall@k_rc': {k: r for k, r in zip(k_values, recalls_rc)},
            'error_rc_norm': dist_rc_topK[:, 0].mean().item(),
            'error_rot_deg': rot_err_top1_deg.mean().item(),
            'error_scale_normed': scale_err_top1_normed.mean().item(),
        }


    def _build_feature_gallery_rc_rot(self, overlap, delta_rot_deg, fixed_scale):
        """
        构建包含旋转维度的特征库（RC + Rotation）

        Args:
            overlap: 重叠度
            delta_rot_deg: 旋转角度间隔（度）
            fixed_scale: 固定尺度值

        Returns:
            dict: {
                'features': [N_total, feat_dim],  # N_total = N_rc * N_rot
                'coords': [N_total, 4],
                'gallery_shape': (H, W, n_rots)
            }
        """
        import numpy as np

        # 使用默认尺度
        if fixed_scale is None:
            fixed_scale = (self.sat_dataset.satimgsize_scale_to_refm_boundary[0] +
                          self.sat_dataset.satimgsize_scale_to_refm_boundary[1]) / 2

        # 计算裁剪尺寸
        satimgsize2crop = fixed_scale * self.sat_dataset.scale_ref_m / self.sat_dataset.geo_res_m

        # 调用数据集方法生成RC采样网格
        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=satimgsize2crop,
            overlap=overlap,
            only_nrcs=True
        )  # Shape: [H, W, 2]

        grid_shape_rc = nrcs_gallery.shape[:2]  # (H, W)

        # 展平RC坐标
        nrcs_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)  # [N_rc, 2]

        # 生成旋转角度列表
        rots_angle = [-180 + delta_rot_deg * i for i in range(360 // delta_rot_deg)]
        rots_rad = torch.tensor(np.deg2rad(np.array(rots_angle)), dtype=torch.float32)  # [N_rot]
        n_rots = len(rots_rad)

        print(f"   RC采样网格: {grid_shape_rc[0]} x {grid_shape_rc[1]} = {nrcs_flatten.shape[0]} 点")
        print(f"   旋转角度数: {n_rots} (间隔 {delta_rot_deg}°)")
        print(f"   总候选点数: {nrcs_flatten.shape[0] * n_rots}")
        print(f"   裁剪尺寸: {satimgsize2crop:.1f} 像素")

        # 扩展坐标：每个RC位置 × 每个旋转角度
        # [N_rc, 2] -> [N_rc, N_rot, 2]
        nrcs_expanded = nrcs_flatten.unsqueeze(1).expand(-1, n_rots, -1)

        # [N_rot] -> [N_rc, N_rot, 1]
        rots_expanded = rots_rad[None, :, None].expand(nrcs_flatten.shape[0], -1, 1)

        # 固定scale: [N_rc, N_rot, 1]
        scales_expanded = torch.full((nrcs_flatten.shape[0], n_rots, 1), fixed_scale)

        # 拼接4D坐标: [N_rc, N_rot, 4]
        coords_4d = torch.cat([nrcs_expanded, rots_expanded, scales_expanded], dim=-1)

        # 展平: [N_rc * N_rot, 4]
        coords_4d_flatten = coords_4d.flatten(start_dim=0, end_dim=1).to(self.device)

        # 从Grid提取特征（分块处理以节省内存）
        chunk_size = 512
        feat_gallery_list = []

        with torch.no_grad():
            coords_chunks = torch.split(coords_4d_flatten, chunk_size)

            for coords_chunk in coords_chunks:
                # 转换到6D空间
                coords_6d = self.coord_normer.raw_to_norm(coords_chunk, append_linear_rot=True)

                # Grid输入
                grid_coords_3d = torch.cat([coords_6d[:, 0:2], coords_6d[:, -1:]], dim=-1)
                feats_grid = self._get_feats_fm_grid(grid_coords_3d)

                # 位置编码
                coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])

                # Grid MLP调制
                feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
                feats_grid = torch.nn.functional.normalize(feats_grid, dim=-1)

                feat_gallery_list.append(feats_grid.detach().cpu())

        # 合并所有特征
        feats_gallery_flatten = torch.cat(feat_gallery_list, dim=0)

        return {
            'features': feats_gallery_flatten,
            'coords': coords_4d_flatten.cpu(),
            'gallery_shape': torch.Size([grid_shape_rc[0], grid_shape_rc[1], n_rots])
        }


    def _build_feature_gallery_rc_scale(self, overlap, n_scales):
        """
        构建包含尺度维度的特征库（RC + Scale）

        Args:
            overlap: 重叠度
            n_scales: 尺度数量

        Returns:
            dict: {
                'features': [N_total, feat_dim],
                'coords': [N_total, 4],
                'gallery_shape': (H, W, n_scales)
            }
        """
        import numpy as np

        # 生成尺度列表
        scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
        print(f"   尺度级别: {n_scales}")
        for i, (scale, size) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"     - Level {i}: scale={scale:.3f}, size={size:.1f}px")

        gallery_features = []
        gallery_coords = []
        
        fixed_rot = 0.0

        for scale_val, satimgsize2crop in zip(scale_list, satimgsize_list):
            print(f"   处理尺度: {scale_val:.3f}")
            
            nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap,
                only_nrcs=True
            )
            grid_shape_rc = nrcs_gallery.shape[:2]
            nrcs_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)

            coords_4d = torch.cat([
                nrcs_flatten,
                torch.full((nrcs_flatten.shape[0], 1), fixed_rot),
                torch.full((nrcs_flatten.shape[0], 1), scale_val)
            ], dim=-1).to(self.device)

            chunk_size = 512
            with torch.no_grad():
                coords_chunks = torch.split(coords_4d, chunk_size)
                for coords_chunk in coords_chunks:
                    coords_6d = self.coord_normer.raw_to_norm(coords_chunk, append_linear_rot=True)
                    grid_coords_3d = torch.cat([coords_6d[:, 0:2], coords_6d[:, -1:]], dim=-1)
                    feats_grid = self._get_feats_fm_grid(grid_coords_3d)
                    coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
                    feats_grid = torch.nn.functional.normalize(feats_grid, dim=-1)
                    gallery_features.append(feats_grid.detach().cpu())
                    gallery_coords.append(coords_chunk.cpu())

        feats_gallery_flatten = torch.cat(gallery_features, dim=0)
        coords_gallery_flatten = torch.cat(gallery_coords, dim=0)
        
        return {
            'features': feats_gallery_flatten,
            'coords': coords_gallery_flatten,
            'gallery_shape': torch.Size([grid_shape_rc[0], grid_shape_rc[1], n_scales])
        }

    def _build_feature_gallery_rc_rot_scale(self, overlap, delta_rot_deg, n_scales):
        """
        构建包含RC、旋转和尺度维度的4D特征库

        Args:
            overlap: 重叠度
            delta_rot_deg: 旋转角度间隔
            n_scales: 尺度数量

        Returns:
            dict: {'features': [N_total, feat_dim], 'coords': [N_total, 4]}
        """
        import numpy as np

        # 生成尺度和旋转列表
        scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
        rots_angle = [-180 + delta_rot_deg * i for i in range(360 // delta_rot_deg)]
        rots_rad = torch.tensor(np.deg2rad(np.array(rots_angle)), dtype=torch.float32)
        n_rots = len(rots_rad)

        print(f"   尺度级别: {n_scales}, 旋转角度数: {n_rots}")

        gallery_features = []
        gallery_coords = []

        # 遍历每个尺度
        for scale_val, satimgsize2crop in zip(scale_list, satimgsize_list):
            print(f"   处理尺度: {scale_val:.3f}")
            
            # 生成RC网格
            nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop, overlap=overlap, only_nrcs=True
            )
            nrcs_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)

            # 扩展坐标维度
            nrcs_expanded = nrcs_flatten.unsqueeze(1).expand(-1, n_rots, -1)
            rots_expanded = rots_rad[None, :, None].expand(nrcs_flatten.shape[0], -1, 1)
            scales_expanded = torch.full((nrcs_flatten.shape[0], n_rots, 1), scale_val)
            
            coords_4d = torch.cat([nrcs_expanded, rots_expanded, scales_expanded], dim=-1)
            coords_4d_flatten = coords_4d.flatten(start_dim=0, end_dim=1).to(self.device)

            # 提取特征
            chunk_size = 512
            with torch.no_grad():
                coords_chunks = torch.split(coords_4d_flatten, chunk_size)
                for coords_chunk in coords_chunks:
                    coords_6d = self.coord_normer.raw_to_norm(coords_chunk, append_linear_rot=True)
                    grid_coords_3d = torch.cat([coords_6d[:, 0:2], coords_6d[:, -1:]], dim=-1)
                    feats_grid = self._get_feats_fm_grid(grid_coords_3d)
                    coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
                    feats_grid = torch.nn.functional.normalize(feats_grid, dim=-1)
                    gallery_features.append(feats_grid.detach().cpu())
            
            gallery_coords.append(coords_4d_flatten.cpu())

        feats_gallery_flatten = torch.cat(gallery_features, dim=0)
        coords_gallery_flatten = torch.cat(gallery_coords, dim=0)
        
        return {
            'features': feats_gallery_flatten,
            'coords': coords_gallery_flatten,
        }





    def _build_feature_gallery_rc(self, overlap, fixed_rot, fixed_scale):
        """
        构建RC维度的特征库（固定rot和scale）
        复用数据集的crop_sat_unifrom方法来正确生成采样网格

        Args:
            overlap: 重叠度
            fixed_rot: 固定旋转角度（弧度）
            fixed_scale: 固定尺度值

        Returns:
            dict: {
                'features': [N, feat_dim],
                'coords': [N, 4],
                'grid_shape': (H, W)
            }
        """
        # 使用默认尺度
        if fixed_scale is None:
            fixed_scale = (self.sat_dataset.satimgsize_scale_to_refm_boundary[0] +
                          self.sat_dataset.satimgsize_scale_to_refm_boundary[1]) / 2

        # 使用数据集的crop_sat_unifrom方法生成正确的采样网格
        # 计算裁剪尺寸
        satimgsize2crop = fixed_scale * self.sat_dataset.scale_ref_m / self.sat_dataset.geo_res_m

        # 调用数据集方法生成采样网格（返回归一化坐标）
        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=satimgsize2crop,
            overlap=overlap,
            only_nrcs=True
        )  # Shape: [H, W, 2]

        grid_shape = nrcs_gallery.shape[:2]  # (H, W)

        # 展平为坐标列表
        nrcs_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)  # [N, 2]

        # 添加固定的rot和scale
        coords_4d = torch.cat([
            nrcs_flatten,
            torch.full((nrcs_flatten.shape[0], 1), fixed_rot),
            torch.full((nrcs_flatten.shape[0], 1), fixed_scale)
        ], dim=-1).to(self.device)  # [N, 4]

        print(f"   采样网格: {grid_shape[0]} x {grid_shape[1]} = {coords_4d.shape[0]} 点")
        print(f"   裁剪尺寸: {satimgsize2crop:.1f} 像素")

        # 从Grid提取特征
        with torch.no_grad():
            # 转换到6D空间
            coords_6d = self.coord_normer.raw_to_norm(coords_4d, append_linear_rot=True)

            # Grid输入
            grid_coords_3d = torch.cat([coords_6d[:, 0:2], coords_6d[:, -1:]], dim=-1)
            feats_grid = self._get_feats_fm_grid(grid_coords_3d)

            # 位置编码
            coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])

            # Grid MLP调制
            feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
            feats_grid = torch.nn.functional.normalize(feats_grid, dim=-1)

        return {
            'features': feats_grid,
            'coords': coords_4d.cpu(),
            'grid_shape': grid_shape
        }


if __name__ == "__main__":
    import argparse

    # 添加 --test_only 参数
    parser = argparse.ArgumentParser(add_help=False) # add_help=False to avoid conflict with get_parse
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    args, remaining_argv = parser.parse_known_args()

    # test by manual modification
    args.test_only = True
    # 直接读取实验配置文件opts.yaml（包含所有参数，不再需要基础配置文件）
    # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage2_grid_hashfit.yaml']) #for trainging
    remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainers/.exps/stage2_grid_hashfit_1_hashz=rot_ep1k/opts.yaml']) #for testing

    sys.argv[1:] = remaining_argv # Pass remaining args to get_parse

    trainer = GridHashFitTrainer()
    
    if args.test_only:
        trainer.test()
    else:
        trainer.train()
