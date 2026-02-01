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

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainers.stage2_grid_hashfit import GridHashFitTrainer
from trainer_depends.base.components import NetworkComponents
from trainer_depends.utils.util_udf_computer import UDFComputer
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

        # 初始化MetricNet专用的坐标编码器（独立于stage2的pos_encoder）
        # 可以设置不同的频率以获得更适合距离预测的特征
        metric_multires_rc = getattr(self.opt, 'metric_multires_rc', 6)  # 默认6，可在配置中调整
        metric_multires_rot = getattr(self.opt, 'metric_multires_rot', 4)  # 默认4
        metric_multires_scale = getattr(self.opt, 'metric_multires_scale', 3)  # 默认3

        self.pos_encoder_metric = components.create_coords_5d_encoder(
            multires_rc=metric_multires_rc,
            multires_rot=metric_multires_rot,
            multires_scale=metric_multires_scale
        )

        print(f"MetricNet坐标编码器配置:")
        print(f"  - multires_rc: {metric_multires_rc}")
        print(f"  - multires_rot: {metric_multires_rot}")
        print(f"  - multires_scale: {metric_multires_scale}")
        print(f"  - 输出维度: {self.pos_encoder_metric.out_dim}")
        print(f"对比Stage2坐标编码器输出维度: {self.pos_encoder.out_dim}\n")

        self.metric_net = components.create_metric_net(
            feat_dim=self.feat_q_dim,
            coord_dim=self.pos_encoder_metric.out_dim,  # 使用新的编码器维度
            branch_hidden_dim=768,
            branch_output_dim=512,
            resblock_hidden_dim=384,
            resblock_output_dim=256,
            dropout=0.1,
            init_weights=True,
            output_activation=None  # 不在模型内部应用激活，在train/test中手动控制
        )

        # 初始化Softplus激活函数（用于距离预测的输出激活）
        self.softplus = torch.nn.Softplus(beta=1)
        self.use_softplus = True  # 设为False可以调试原始输出

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
            print("可训练: metric_net")
            print("冻结:   vis_encoder, vis_aggregator, grid, grid_mlp\n")

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


    def _compute_eikonal_loss(self, query_feat, n_samples=1024):
        """
        计算Eikonal正则化损失

        目标：约束距离场的梯度范数接近1，使MetricNet学习到平滑的距离场

        Args:
            query_feat: 查询特征 [1, feat_dim]，用于固定query
            n_samples: 采样点数量

        Returns:
            loss_eikonal: Eikonal损失标量
        """
        # 1. 采样随机坐标点（4D原始坐标）
        eikonal_points_4d = self.sat_dataset.mk_rand_coords_4d(
            n_rand=n_samples,
            return_tensor=True
        ).to(self.device)  # [n_samples, 4]

        # 2. 转换为6D归一化坐标（带线性旋转）
        eikonal_points_6d = self.coord_normer.raw_to_norm(
            eikonal_points_4d,
            append_linear_rot=True
        )  # [n_samples, 6]

        # 3. 提取5D坐标并设置requires_grad
        eikonal_coords_5d = eikonal_points_6d[:, :5].clone()
        eikonal_coords_5d.requires_grad = True

        # 4. 冻结网络，获取grid特征（特征不需要梯度）
        with torch.no_grad():
            # Grid输入：拼接 [row, col] + [scale]
            grid_input = torch.cat([
                eikonal_points_6d[:, :2],  # row, col
                eikonal_points_6d[:, -1:]   # linear_rot (用作scale)
            ], dim=-1)  # [n_samples, 3]

            feats_grid_raw = self._get_feats_fm_grid(grid_input)

            # 位置编码（从5D坐标）
            coords_encoded_frozen = self.pos_encoder(eikonal_coords_5d.detach())

            # Grid MLP调制
            feats_grid = self.grid_mlp(
                inputs=feats_grid_raw,
                condition_features=coords_encoded_frozen
            )
            feats_grid = TF.normalize(feats_grid, dim=-1)
            feats_grid_exp = feats_grid.unsqueeze(0)  # [1, n_samples, feat_dim]

        # 5. 非冻结的坐标编码（需要梯度流）- 使用MetricNet专用编码器
        coords_encoded = self.pos_encoder_metric(eikonal_coords_5d)
        coords_enc_exp = coords_encoded.unsqueeze(0)  # [1, n_samples, coord_dim]

        # 6. 扩展query特征
        query_feat_exp = query_feat.unsqueeze(1).expand(
            1, n_samples, -1
        )  # [1, n_samples, feat_dim]

        # 7. MetricNet前向传播（距离预测）
        dist_eikonal_raw = self.metric_net(
            query_feat_exp,
            feats_grid_exp,
            coords_enc_exp
        )  # [1, n_samples]

        # 应用激活函数（可调试控制）
        if self.use_softplus:
            dist_eikonal = self.softplus(dist_eikonal_raw)
        else:
            dist_eikonal = dist_eikonal_raw

        # 8. 计算对坐标的梯度
        grad_outputs = torch.ones_like(dist_eikonal)
        grad_coords = torch.autograd.grad(
            outputs=dist_eikonal,
            inputs=eikonal_coords_5d,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]  # [n_samples, 5]

        # 9. Eikonal约束：||∇d|| ≈ 1
        grad_norm = grad_coords.norm(dim=-1)  # [n_samples]
        loss_eikonal = ((grad_norm - 1.0) ** 2).mean()

        return loss_eikonal

    def visualize_udf_field_3d(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                               n_samples_per_dim=32, delta=0.1,
                               save_path="vis_results/udf_field_3d.html", show_plot=False):
        """
        可视化UDF场的空间分布（固定scale，在r,c,rot空间中均匀采样）
        使用plotly创建可交互的3D可视化，支持保存为HTML文件。

        Args:
            query_feat: 查询特征 [1, feat_dim]，如果为None则从测试集抽取
            gt_coord_4d: Ground truth坐标 [4] (row, col, rot, scale)，如果为None则从测试集抽取
            scale_fixed: 固定的scale值，如果为None则使用gt_coord_4d的scale
            n_samples_per_dim: 每个维度的采样点数（默认32）
            delta: 采样范围（相对于GT位置的偏移，归一化坐标）
            save_path: 可视化结果保存路径 (默认: "vis_results/udf_field_3d.html")
            show_plot: 是否直接弹出显示交互式图表 (默认: False)

        Returns:
            None
        """
        import numpy as np
        import os

        # 0. 确保所有模型处于评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        print(f"\n{'=' * 80}")
        print(f"可视化UDF距离场 (固定scale，在r,c,rot空间采样)")
        print(f"{'=' * 80}\n")

        # 1. 如果没有提供query_feat或gt_coord，从测试集抽取
        if query_feat is None or gt_coord_4d is None:
            # 注意：这里假设self.uav_dataloader_test是一个可以迭代的对象
            try:
                uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
            except StopIteration:
                # 如果dataloader耗尽，重新创建一个iterator (视具体实现而定)
                uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))

            uav_img = uav_img[0].to(self.device).unsqueeze(0)
            gt_coord_4d = uav_coords_4d[0].to(self.device)

            # 提取query特征
            query_feat = self._get_feats_fm_imgs(uav_img)  # [1, feat_dim]

        # 2. 确定固定的scale
        if scale_fixed is None:
            scale_fixed = gt_coord_4d[3].item()

        # 3. 创建评估网格（在GT周围采样）
        nr_center, nc_center, rot_center, scale_val = gt_coord_4d

        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)
        rot_range = torch.linspace(rot_center - torch.pi / 2, rot_center + torch.pi / 2, n_samples_per_dim,
                                   device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        coords_sampled_4d = torch.stack([
            grid_nr.flatten(),
            grid_nc.flatten(),
            grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_fixed)
        ], dim=-1)

        # 4. 转换为6D归一化坐标
        coords_sampled_6d = self.coord_normer.raw_to_norm(
            coords_sampled_4d,
            append_linear_rot=True
        )

        # 5. 提取特征并预测距离
        with torch.no_grad():
            # Grid输入
            grid_input = torch.cat([
                coords_sampled_6d[:, :2],  # row, col
                coords_sampled_6d[:, -1:]  # linear_rot
            ], dim=-1)

            feats_grid_raw = self._get_feats_fm_grid(grid_input)

            # Grid MLP使用stage2的位置编码
            coords_encoded_stage2 = self.pos_encoder(coords_sampled_6d[:, :5])

            # Grid MLP调制
            feats_grid = self.grid_mlp(
                inputs=feats_grid_raw,
                condition_features=coords_encoded_stage2
            )
            feats_grid = TF.normalize(feats_grid, dim=-1)
            feats_grid_exp = feats_grid.unsqueeze(0)  # [1, N, feat_dim]

            # MetricNet使用专用的位置编码
            coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])

            # 扩展query和坐标
            N = coords_sampled_4d.shape[0]
            query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
            coords_enc_exp = coords_encoded_metric.unsqueeze(0)

            # MetricNet预测距离
            dist_pred_raw = self.metric_net(
                query_feat_exp,
                feats_grid_exp,
                coords_enc_exp
            )  # [1, N]

            # 应用激活函数
            if self.use_softplus:
                dist_pred = self.softplus(dist_pred_raw)
            else:
                dist_pred = dist_pred_raw

            dist_pred = dist_pred.squeeze(0)  # [N]

            # 计算UDF Ground Truth
            gt_coord_6d = self.coord_normer.raw_to_norm(
                gt_coord_4d.unsqueeze(0).to(self.device),
                append_linear_rot=True
            )
            udf_gt = self.udf_compter_5d.compute_udf_matrix_from_norm(
                gt_coord_6d[:, :5],
                coords_sampled_6d[:, :5]
            )  # [1, N]
            udf_gt = udf_gt.squeeze(0)  # [N]

            # 计算误差
            error = torch.abs(dist_pred - udf_gt)

        # 6. 找到预测距离最小的点（预测的最佳位置）
        best_idx = torch.argmin(dist_pred)
        coord_pred_best = coords_sampled_4d[best_idx]
        dist_pred_best = dist_pred[best_idx]

        # 7. 可视化
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # 转换为numpy
            coords_np = coords_sampled_4d.cpu().numpy()
            dist_pred_np = dist_pred.cpu().numpy()
            udf_gt_np = udf_gt.cpu().numpy()
            error_np = error.cpu().numpy()

            # 创建子图：GT UDF、预测距离、误差（GT放最前面）
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Ground Truth UDF Field', 'Predicted Distance Field', 'Absolute Error'),
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
            )

            # 子图1: GT UDF场
            fig.add_trace(
                go.Scatter3d(
                    x=coords_np[:, 0],
                    y=coords_np[:, 1],
                    z=coords_np[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=udf_gt_np,
                        colorscale='Viridis',
                        colorbar=dict(title='UDF', x=0.3),
                        opacity=0.6
                    ),
                    name='GT UDF'
                ),
                row=1, col=1
            )

            # GT位置
            fig.add_trace(
                go.Scatter3d(
                    x=[nr_center.cpu().item()],
                    y=[nc_center.cpu().item()],
                    z=[rot_center.cpu().item()],
                    mode='markers+text',
                    marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
                    name='GT Position',
                    text=['GT'],
                    textposition="top left"
                ),
                row=1, col=1
            )

            # 子图2: 预测的距离场
            fig.add_trace(
                go.Scatter3d(
                    x=coords_np[:, 0],
                    y=coords_np[:, 1],
                    z=coords_np[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=dist_pred_np,
                        colorscale='Viridis',
                        colorbar=dict(title='Distance', x=0.65),
                        opacity=0.6
                    ),
                    name='Predicted Distance'
                ),
                row=1, col=2
            )

            # 添加GT位置（红色菱形）
            fig.add_trace(
                go.Scatter3d(
                    x=[nr_center.cpu().item()],
                    y=[nc_center.cpu().item()],
                    z=[rot_center.cpu().item()],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
                    name='GT',
                    showlegend=False
                ),
                row=1, col=2
            )

            # 添加预测最佳位置（青色叉号）
            fig.add_trace(
                go.Scatter3d(
                    x=[coord_pred_best[0].cpu().item()],
                    y=[coord_pred_best[1].cpu().item()],
                    z=[coord_pred_best[2].cpu().item()],
                    mode='markers+text',
                    marker=dict(size=12, color='cyan', symbol='cross', line=dict(width=2, color='blue')),
                    name='Predicted Best',
                    text=[f'{dist_pred_best:.3f}'],
                    textposition="top right"
                ),
                row=1, col=2
            )

            # 子图3: 误差分布
            fig.add_trace(
                go.Scatter3d(
                    x=coords_np[:, 0],
                    y=coords_np[:, 1],
                    z=coords_np[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=error_np,
                        colorscale='Hot',
                        colorbar=dict(title='|Error|', x=1.0),
                        opacity=0.6
                    ),
                    name='Absolute Error'
                ),
                row=1, col=3
            )

            # GT位置
            fig.add_trace(
                go.Scatter3d(
                    x=[nr_center.cpu().item()],
                    y=[nc_center.cpu().item()],
                    z=[rot_center.cpu().item()],
                    mode='markers',
                    marker=dict(size=10, color='blue', symbol='diamond', line=dict(width=2, color='white')),
                    name='GT',
                    showlegend=False
                ),
                row=1, col=3
            )

            # 更新布局
            fig.update_layout(
                title=f'UDF Field Visualization (scale={scale_fixed:.2f})',
                height=600,
                showlegend=True
            )

            # 更新所有子图的轴标签
            for i in range(1, 4):
                fig.update_scenes(
                    xaxis_title='Row',
                    yaxis_title='Col',
                    zaxis_title='Rotation (rad)',
                    row=1, col=i
                )

            # 打印统计信息
            print(f"\n=== UDF Field Statistics ===")
            print(
                f"Predicted Distance - Min: {dist_pred_np.min():.4f}, Max: {dist_pred_np.max():.4f}, Mean: {dist_pred_np.mean():.4f}")
            print(
                f"GT UDF Distance    - Min: {udf_gt_np.min():.4f}, Max: {udf_gt_np.max():.4f}, Mean: {udf_gt_np.mean():.4f}")
            print(
                f"Absolute Error     - Min: {error_np.min():.4f}, Max: {error_np.max():.4f}, Mean: {error_np.mean():.4f}")
            print(f"MSE: {(error_np ** 2).mean():.6f}")
            print(f"MAE: {error_np.mean():.6f}")
            print(f"\nGT Position: (r={nr_center:.4f}, c={nc_center:.4f}, rot={rot_center:.4f})")
            print(
                f"Predicted Best: (r={coord_pred_best[0]:.4f}, c={coord_pred_best[1]:.4f}, rot={coord_pred_best[2]:.4f})")
            print(f"{'=' * 80}\n")

            # --- 核心修改部分：保存文件逻辑 ---
            if save_path:
                # 确保保存目录存在
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                fig.write_html(save_path)
                print(f"✅ 交互式图表已保存至: {save_path}")

            # --- 核心修改部分：控制显示逻辑 ---
            if show_plot:
                fig.show()
            else:
                print(f"   (提示：设置 show_plot=True 可直接弹出交互式窗口，当前仅保存文件)")

        except ImportError:
            print("⚠️  需要安装plotly: pip install plotly")
            return None


    def visualize_udf_field(self, query_feat, gt_coord_4d=None, scale_fixed=None,
                           n_samples_per_dim=50, save_path=None):
        """
        可视化UDF场的空间分布（固定scale，在r,c,rot空间中均匀采样）

        Args:
            query_feat: 查询特征 [1, feat_dim]
            gt_coord_4d: Ground truth坐标 [4] (row, col, rot, scale)，用于标注真实位置
            scale_fixed: 固定的scale值，如果为None则使用gt_coord_4d的scale
            n_samples_per_dim: 每个维度的采样点数（默认50，会生成50x50的网格）
            save_path: 保存路径，如果为None则使用默认路径

        Returns:
            fig: matplotlib figure对象
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # 确定固定的scale
        if scale_fixed is None:
            if gt_coord_4d is not None:
                scale_fixed = gt_coord_4d[3].item()
            else:
                # 使用数据集的中间scale值
                scale_range = self.sat_dataset.get_scale_range()
                scale_fixed = (scale_range[0] + scale_range[1]) / 2

        # 获取数据集的坐标范围
        r_min, r_max = self.sat_dataset.get_row_range()
        c_min, c_max = self.sat_dataset.get_col_range()
        rot_min, rot_max = self.sat_dataset.get_rot_range()

        # 在r,c空间均匀采样（用于2D热图）
        r_samples = torch.linspace(r_min, r_max, n_samples_per_dim)
        c_samples = torch.linspace(c_min, c_max, n_samples_per_dim)

        # 固定rot为中间值（用于主热图）
        if gt_coord_4d is not None:
            rot_fixed = gt_coord_4d[2].item()
        else:
            rot_fixed = (rot_min + rot_max) / 2

        # 创建网格
        r_grid, c_grid = torch.meshgrid(r_samples, c_samples, indexing='ij')

        # 构造采样坐标 [n_samples_per_dim^2, 4]
        coords_sampled_4d = torch.stack([
            r_grid.flatten(),
            c_grid.flatten(),
            torch.full_like(r_grid.flatten(), rot_fixed),
            torch.full_like(r_grid.flatten(), scale_fixed)
        ], dim=-1).to(self.device)

        # 转换为6D归一化坐标
        coords_sampled_6d = self.coord_normer.raw_to_norm(
            coords_sampled_4d,
            append_linear_rot=True
        )

        # 提取特征
        with torch.no_grad():
            # Grid输入
            grid_input = torch.cat([
                coords_sampled_6d[:, :2],  # row, col
                coords_sampled_6d[:, -1:]   # linear_rot
            ], dim=-1)

            feats_grid_raw = self._get_feats_fm_grid(grid_input)

            # Grid MLP使用stage2的位置编码
            coords_encoded_stage2 = self.pos_encoder(coords_sampled_6d[:, :5])

            # Grid MLP调制
            feats_grid = self.grid_mlp(
                inputs=feats_grid_raw,
                condition_features=coords_encoded_stage2
            )
            feats_grid = TF.normalize(feats_grid, dim=-1)
            feats_grid_exp = feats_grid.unsqueeze(0)  # [1, N, feat_dim]

            # MetricNet使用专用的位置编码
            coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])

            # 扩展query和坐标
            N = coords_sampled_4d.shape[0]
            query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
            coords_enc_exp = coords_encoded_metric.unsqueeze(0)

            # MetricNet预测距离
            dist_pred_raw = self.metric_net(
                query_feat_exp,
                feats_grid_exp,
                coords_enc_exp
            )  # [1, N]

            # 应用激活函数
            if self.use_softplus:
                dist_pred = self.softplus(dist_pred_raw)
            else:
                dist_pred = dist_pred_raw

            dist_pred = dist_pred.squeeze(0).cpu().numpy()  # [N]

            # 计算UDF Ground Truth（如果提供了gt坐标）
            if gt_coord_4d is not None:
                gt_coord_6d = self.coord_normer.raw_to_norm(
                    gt_coord_4d.unsqueeze(0).to(self.device),
                    append_linear_rot=True
                )
                udf_gt = self.udf_compter_5d.compute_udf_matrix_from_norm(
                    gt_coord_6d[:, :5],
                    coords_sampled_6d[:, :5]
                )  # [1, N]
                udf_gt = udf_gt.squeeze(0).cpu().numpy()
            else:
                udf_gt = None

        # 重塑为2D网格
        dist_pred_grid = dist_pred.reshape(n_samples_per_dim, n_samples_per_dim)
        if udf_gt is not None:
            udf_gt_grid = udf_gt.reshape(n_samples_per_dim, n_samples_per_dim)

        # 创建可视化
        if udf_gt is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]

        # 转换为numpy用于绘图
        r_grid_np = r_grid.cpu().numpy()
        c_grid_np = c_grid.cpu().numpy()

        # 1. 预测的距离场
        im0 = axes[0].contourf(c_grid_np, r_grid_np, dist_pred_grid, levels=50, cmap='viridis')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        axes[0].set_title(f'Predicted Distance Field\n(rot={rot_fixed:.2f}, scale={scale_fixed:.2f})')
        plt.colorbar(im0, ax=axes[0], label='Distance')

        # 标注GT位置
        if gt_coord_4d is not None:
            gt_r, gt_c = gt_coord_4d[0].item(), gt_coord_4d[1].item()
            axes[0].scatter(gt_c, gt_r, c='red', s=200, marker='*',
                          edgecolors='white', linewidths=2, label='GT Position')
            axes[0].legend()

        if udf_gt is not None:
            # 2. Ground Truth UDF场
            im1 = axes[1].contourf(c_grid_np, r_grid_np, udf_gt_grid, levels=50, cmap='viridis')
            axes[1].set_xlabel('Column')
            axes[1].set_ylabel('Row')
            axes[1].set_title(f'Ground Truth UDF Field\n(rot={rot_fixed:.2f}, scale={scale_fixed:.2f})')
            plt.colorbar(im1, ax=axes[1], label='Distance')
            axes[1].scatter(gt_c, gt_r, c='red', s=200, marker='*',
                          edgecolors='white', linewidths=2, label='GT Position')
            axes[1].legend()

            # 3. 误差分布
            error_grid = np.abs(dist_pred_grid - udf_gt_grid)
            im2 = axes[2].contourf(c_grid_np, r_grid_np, error_grid, levels=50, cmap='hot')
            axes[2].set_xlabel('Column')
            axes[2].set_ylabel('Row')
            axes[2].set_title(f'Absolute Error\n(rot={rot_fixed:.2f}, scale={scale_fixed:.2f})')
            plt.colorbar(im2, ax=axes[2], label='|Pred - GT|')
            axes[2].scatter(gt_c, gt_r, c='blue', s=200, marker='*',
                          edgecolors='white', linewidths=2, label='GT Position')
            axes[2].legend()

            # 打印统计信息
            print(f"\n=== UDF Field Statistics ===")
            print(f"Predicted Distance - Min: {dist_pred.min():.4f}, Max: {dist_pred.max():.4f}, Mean: {dist_pred.mean():.4f}")
            print(f"GT UDF Distance    - Min: {udf_gt.min():.4f}, Max: {udf_gt.max():.4f}, Mean: {udf_gt.mean():.4f}")
            print(f"Absolute Error     - Min: {error_grid.min():.4f}, Max: {error_grid.max():.4f}, Mean: {error_grid.mean():.4f}")
            print(f"MSE: {(error_grid**2).mean():.6f}")
            print(f"MAE: {error_grid.mean():.6f}")

        plt.tight_layout()

        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.exp_dir2save, 'udf_field_visualization.png')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ UDF场可视化已保存到: {save_path}")

        return fig


    def train(self):
        """Stage 3训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 3 训练: MetricNet")
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
        # 4.5 初始化4d坐标归一化器
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)

        # 4.6 初始化分层坐标采样器（使用新的Span-Ratio策略）
        from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
        self.coord_sampler = create_hierarchical_sampler_from_dataset(
            sat_dataset=self.sat_dataset,
            bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
            num_uniform_samples=getattr(opt, 'sampler_num_uniform', 256),
            device=self.device
        )

        # 5. 配置Loss
        # from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        loss_mse = torch.nn.MSELoss(reduction='mean')
        from trainer_depends.utils.util_weight_annealing import SigmoidWeightScheduler
        eikonal_weight_scheduler = SigmoidWeightScheduler(max_steps=1000,max_weight=0.0001,min_weight=0,center_step=500,warmup_steps=1000)

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

                # 使用分层采样器生成负样本坐标
                coords_rand_hierarchical = self.coord_sampler.sample(coords_gt)  # [2B, N_samples, 4]
                # 将batch维度和sample维度合并
                coords_rand = coords_rand_hierarchical.view(-1, 4)  # [2B*N_samples, 4]

                # 构建所有坐标
                coords_all = torch.cat([coords_gt, coords_rand], dim=0)  # [2B + 2B*N_samples, 4]
                coords_all_6d = self.coord_normer.raw_to_norm(coords_all,append_linear_rot=True)

                # 从Grid提取特征
                feats_all_grid = self._get_feats_fm_grid(torch.concatenate([coords_all_6d[:,:2],coords_all_6d[:,-1:]],dim=-1))  # [2B, feat_dim]
                # 位置编码
                coords_all_encoded = self.pos_encoder(
                    coords_all_6d[:,:5],
                )  # [2B, coord_encoded_dim]
                # Grid MLP调制
                feats_all_grid = self.grid_mlp(
                    inputs=feats_all_grid,
                    condition_features=coords_all_encoded
                )  # [2B, feat_dim]
                feats_all_grid = TF.normalize(feats_all_grid, dim=-1)

                # === MetricNet距离预测 ===
                # 使用MetricNet专用的坐标编码器（不同于grid_mlp的编码器）
                coords_all_encoded_metric = self.pos_encoder_metric(coords_all_6d[:, :5])  # [N, coord_dim_metric]

                B_vis = feats_vis.shape[0]  # 2B
                N_grid = feats_all_grid.shape[0]  # 2B+1024
                # 扩展为MetricNet输入格式 [B_vis, N_grid, C]
                feats_vis_expanded = feats_vis.unsqueeze(1).expand(B_vis, N_grid, -1)
                feats_grid_expanded = feats_all_grid.unsqueeze(0).expand(B_vis, N_grid, -1)
                coords_all_encoded_metric_expanded = coords_all_encoded_metric.unsqueeze(0).expand(B_vis, N_grid, -1)
                # MetricNet前向
                metric_dist_mat_raw = self.metric_net(
                    feats_vis_expanded,
                    feats_grid_expanded,
                    coords_all_encoded_metric_expanded  # 使用metric专用编码
                )  # [B_vis, N_grid]

                # 应用激活函数（可调试控制）
                if self.use_softplus:
                    metric_dist_mat = self.softplus(metric_dist_mat_raw)
                else:
                    metric_dist_mat = metric_dist_mat_raw

                # === 计算UDF Loss ===
                udf_gtdist_mat = self.udf_compter_5d.compute_udf_matrix_from_norm(
                    coords_all_6d[:coords_gt.shape[0], :5],
                    coords_all_6d[:, :5]
                )

                # 整体矩阵loss
                # loss = loss_mse(metric_dist_mat, udf_gtdist_mat)
                udf_delta = torch.abs(metric_dist_mat - udf_gtdist_mat)
                loss_per_query = (udf_delta**2).mean(dim=-1)
                loss_mean = loss_per_query.mean()
                # 提取对角线元素（视觉特征对应GT坐标的距离）
                metric_dist_diag = torch.diagonal(metric_dist_mat[:, :coords_gt.shape[0] ])  # [2B]
                loss_matched = metric_dist_diag.mean()
                loss = loss_mean + loss_matched

                #debug
                # self.visualize_udf_field_3d(query_feat=feats_vis[:1],gt_coord_4d=coords_gt[0],delta=0.1)

                # 保存UDF loss用于日志记录
                loss_udf = loss.item()

                # === Eikonal正则化 ===
                lambda_eikonal = eikonal_weight_scheduler.get_weight(current_step=step)
                if lambda_eikonal > 0.0:
                    query_feat_fixed = feats_vis[0:1].detach()
                    loss_eikonal = self._compute_eikonal_loss(
                        query_feat=query_feat_fixed,
                        n_samples=1024
                    )
                    loss_eikonal_weighted = loss_eikonal * lambda_eikonal
                    loss = loss + loss_eikonal_weighted

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

                    # 记录两种loss的数值
                    self.logger.info(f'Iter {it}: loss_udf={loss_udf:.6f}' )
                    if lambda_eikonal>0:
                        self.logger.info(f'loss_eikonal={loss_eikonal_weighted.item():.6f}')
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


    def test(self, use_train_uav=False, use_augmentation=False):
        """
        Stage 3测试函数

        测试目标：
        1. 验证MetricNet预测的距离矩阵与UDF Ground Truth的一致性
        2. 评估距离预测的精度（MSE、MAE、相关系数等）
        3. 可视化距离预测的误差分布

        Args:
            use_train_uav: 是否使用训练集的UAV图像（默认False，使用测试集）
            use_augmentation: 是否使用数据增强（默认False，不使用）
        """
        print("\n" + "🧪"*40)
        print("开始 Stage 3 测试: MetricNet距离预测评估")
        print("🧪"*40 + "\n")

        # 1. 初始化数据集和坐标归一化器
        self._init_datasets(create_train_loader=False)

        # 如果需要，重新创建UAV数据集以控制数据增强
        if use_train_uav or not use_augmentation:
            print(f"测试配置: use_train_uav={use_train_uav}, use_augmentation={use_augmentation}")

            from trainer_depends.datasets.dataset_wingtra_4d import UAVDataset
            opt = self.opt
            scene = opt.scenes_setting['scenes'][0]  # 使用第一个场景

            # 重新创建UAV数据集
            stage = 'train' if use_train_uav else 'test'
            self.uav_dataset_for_test = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                trans_georc2nrc_func=self.sat_dataset.transfrom_georc_to_nrc,
                geo_res_m=0.3,
                stage=stage,
                use_augmentation=use_augmentation,  # 控制是否使用数据增强
            )
        else:
            # 使用默认的测试集（无数据增强）
            self.uav_dataset_for_test = self.uav_dataset_test

        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)

        # 创建测试DataLoader
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_for_test,  # 使用配置后的数据集
            batch_size=self.opt.batchsize_uav,
            num_workers=self.opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        self.sat_dataloader = torch.utils.data.DataLoader(
            self.sat_dataset,
            batch_size=self.opt.batchsize_sat,
            num_workers=self.opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        # 2. 加载checkpoint
        self._load_checkpoints_for_test()

        # 3. 设置为评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 4. 测试距离预测精度
        print("\n" + "="*80)
        print("测试1: MetricNet距离预测精度评估")
        print("="*80)
        self._test_distance_prediction_accuracy()

        # 5. 测试距离场平滑性（Eikonal约束）
        print("\n" + "="*80)
        print("测试2: 距离场梯度范数评估（Eikonal约束）")
        print("="*80)
        self._test_eikonal_constraint()

        print("\n" + "🧪"*40)
        print("✅ Stage 3 测试完成！")
        print("🧪"*40 + "\n")


    def _load_checkpoints_for_test(self):
        """
        测试时加载checkpoint的统一方法

        加载逻辑：
        1. Stage 3自身的checkpoint (metric_net)
        2. Stage 2的checkpoint (grid, grid_mlp)
        3. Stage 1的预训练模型 (vis_encoder, vis_aggregator)
        """
        import yaml

        print("\n" + "="*80)
        print("加载测试用的checkpoint")
        print("="*80)

        # --- 1. 加载Stage 3的checkpoint (当前stage) ---
        stage3_ckpt_path = self._get_stage3_checkpoint_path()

        if stage3_ckpt_path:
            print(f"\n📦 Stage 3 checkpoint: {stage3_ckpt_path}")
            self._load_checkpoint(
                stage3_ckpt_path,
                {'metric_net': self.metric_net},
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 3的checkpoint，无法进行测试。")

        # --- 2. 加载Stage 2的checkpoint (依赖的预训练模型) ---
        stage2_ckpt_path = self._get_stage2_checkpoint_path(stage3_ckpt_path)

        if stage2_ckpt_path:
            print(f"\n📦 Stage 2 checkpoint: {stage2_ckpt_path}")
            self._load_checkpoint(
                stage2_ckpt_path,
                {'grid': self.grid, 'grid_mlp': self.grid_mlp},
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 2的checkpoint，无法进行测试。")

        # --- 3. 加载Stage 1的checkpoint (依赖的预训练模型) ---
        stage1_ckpt_path = self._get_stage1_checkpoint_path(stage2_ckpt_path)

        if stage1_ckpt_path:
            print(f"\n📦 Stage 1 checkpoint: {stage1_ckpt_path}")
            self._load_checkpoint(
                stage1_ckpt_path,
                {'vis_encoder': self.vis_encoder, 'vis_aggregator': self.vis_aggregator},
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 1的checkpoint，无法进行测试。")

        print("\n" + "="*80)
        print("✅ 所有checkpoint加载完成")
        print("="*80 + "\n")


    def _get_stage3_checkpoint_path(self):
        """获取Stage 3的checkpoint路径"""
        import os

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

        print(f"⚠️  未找到Stage 3 checkpoint:")
        print(f"   opt.load2test = {getattr(self.opt, 'load2test', 'NOT SET')}")
        print(f"   exp_dir2save = {self.exp_dir2save}")
        return None


    def _get_stage2_checkpoint_path(self, stage3_ckpt_path):
        """
        获取Stage 2的checkpoint路径

        优先级：
        1. 命令行参数 (opt.load_stage2_ckpt)
        2. Stage 3实验目录中的opts.yaml
        """
        import yaml
        import os

        # 优先使用命令行参数
        if hasattr(self.opt, 'load_stage2_ckpt') and self.opt.load_stage2_ckpt:
            return self.opt.load_stage2_ckpt

        # 从Stage 3的opts.yaml中读取
        if stage3_ckpt_path:
            stage3_exp_dir = os.path.dirname(stage3_ckpt_path)
            stage3_opts_path = os.path.join(stage3_exp_dir, 'opts.yaml')

            if os.path.exists(stage3_opts_path):
                try:
                    with open(stage3_opts_path, 'r') as f:
                        stage3_opts = yaml.safe_load(f)

                    if 'exp_setting' in stage3_opts:
                        stage2_path = stage3_opts['exp_setting'].get('load_stage2_ckpt')
                        if stage2_path:
                            print(f"从opts.yaml读取Stage 2路径: {stage3_opts_path}")
                            return stage2_path
                except Exception as e:
                    print(f"⚠️  读取opts.yaml失败: {e}")

        return None


    def _get_stage1_checkpoint_path(self, stage2_ckpt_path):
        """
        获取Stage 1的checkpoint路径

        优先级：
        1. 命令行参数 (opt.load_stage1_ckpt)
        2. Stage 2实验目录中的opts.yaml
        """
        import yaml
        import os

        # 优先使用命令行参数
        if hasattr(self.opt, 'load_stage1_ckpt') and self.opt.load_stage1_ckpt:
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


    def _test_distance_prediction_accuracy(self, n_samples=10):
        """
        测试MetricNet距离预测的精度

        Args:
            n_samples: 测试样本数量
        """
        import numpy as np

        print(f"\n测试样本数: {n_samples}")
        print(f"每个样本包含: GT坐标 + 随机采样坐标")

        all_errors = []
        all_metric_dists = []
        all_udf_dists = []

        with torch.no_grad():
            for sample_idx in range(n_samples):
                # 获取一个batch的UAV和SAT数据
                batch_uav = next(iter(self.uav_dataloader_test))
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)

                batch_sat = next(iter(self.sat_dataloader))
                satimgs = batch_sat[0].to(self.device)
                coords_sat = batch_sat[1].to(self.device)

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([uavimgs, satimgs], dim=0)
                )

                # Ground truth坐标
                coords_gt = torch.cat([coords_uav, coords_sat], dim=0)

                # 随机采样负样本坐标
                coords_rand = self.sat_dataset.mk_rand_coords_4d(
                    n_rand=512,
                    return_tensor=True
                ).to(self.device)

                # 所有坐标
                coords_all = torch.cat([coords_gt, coords_rand], dim=0)
                coords_all_6d = self.coord_normer.raw_to_norm(coords_all, append_linear_rot=True)

                # 从Grid提取特征
                feats_all_grid = self._get_feats_fm_grid(
                    torch.cat([coords_all_6d[:, :2], coords_all_6d[:, -1:]], dim=-1)
                )
                # Grid MLP使用stage2的编码器
                coords_all_encoded_stage2 = self.pos_encoder(coords_all_6d[:, :5])
                feats_all_grid = self.grid_mlp(
                    inputs=feats_all_grid,
                    condition_features=coords_all_encoded_stage2
                )
                feats_all_grid = TF.normalize(feats_all_grid, dim=-1)

                # MetricNet使用专用编码器
                coords_all_encoded_metric = self.pos_encoder_metric(coords_all_6d[:, :5])

                # MetricNet预测距离
                B_vis = feats_vis.shape[0]
                N_grid = feats_all_grid.shape[0]
                feats_vis_expanded = feats_vis.unsqueeze(1).expand(B_vis, N_grid, -1)
                feats_grid_expanded = feats_all_grid.unsqueeze(0).expand(B_vis, N_grid, -1)
                coords_all_encoded_metric_expanded = coords_all_encoded_metric.unsqueeze(0).expand(B_vis, N_grid, -1)

                metric_dist_mat_raw = self.metric_net(
                    feats_vis_expanded,
                    feats_grid_expanded,
                    coords_all_encoded_metric_expanded
                )

                # 应用激活函数（可调试控制）
                if self.use_softplus:
                    metric_dist_mat = self.softplus(metric_dist_mat_raw)
                else:
                    metric_dist_mat = metric_dist_mat_raw

                # 计算UDF Ground Truth
                udf_gtdist_mat = self.udf_compter_5d.compute_udf_matrix_from_norm(
                    coords_all_6d[:coords_gt.shape[0], :5],
                    coords_all_6d[:, :5]
                )

                # 收集统计数据
                all_metric_dists.append(metric_dist_mat.cpu().flatten())
                all_udf_dists.append(udf_gtdist_mat.cpu().flatten())
                all_errors.append((metric_dist_mat.cpu() - udf_gtdist_mat.cpu()).flatten())

        # 合并所有样本的数据
        all_metric_dists = torch.cat(all_metric_dists)
        all_udf_dists = torch.cat(all_udf_dists)
        all_errors = torch.cat(all_errors)

        # 计算评估指标
        mse = (all_errors ** 2).mean().item()
        mae = all_errors.abs().mean().item()
        rmse = np.sqrt(mse)

        # 计算相关系数
        corr = np.corrcoef(all_metric_dists.numpy(), all_udf_dists.numpy())[0, 1]

        # 打印结果
        print("\n" + "-"*80)
        print("距离预测精度评估结果:")
        print("-"*80)
        print(f"MSE (均方误差):     {mse:.6f}")
        print(f"MAE (平均绝对误差): {mae:.6f}")
        print(f"RMSE (均方根误差):  {rmse:.6f}")
        print(f"相关系数:           {corr:.6f}")
        print(f"\nMetricNet预测距离范围: [{all_metric_dists.min().item():.4f}, {all_metric_dists.max().item():.4f}]")
        print(f"UDF Ground Truth范围:  [{all_udf_dists.min().item():.4f}, {all_udf_dists.max().item():.4f}]")
        print("-"*80)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': corr
        }


    def _test_eikonal_constraint(self, n_samples=5, n_points_per_sample=1024):
        """
        测试距离场的Eikonal约束（梯度范数应接近1）

        Args:
            n_samples: 测试样本数量
            n_points_per_sample: 每个样本采样的点数
        """
        print(f"\n测试样本数: {n_samples}")
        print(f"每样本采样点数: {n_points_per_sample}")

        all_grad_norms = []

        with torch.no_grad():
            # 获取一个固定的query特征
            batch_uav = next(iter(self.uav_dataloader_test))
            uavimgs = batch_uav[0][:1].to(self.device)  # 只取一个
            query_feat = self._get_feats_fm_imgs(uavimgs)

        for sample_idx in range(n_samples):
            # 采样随机坐标点
            eikonal_points_4d = self.sat_dataset.mk_rand_coords_4d(
                n_rand=n_points_per_sample,
                return_tensor=True
            ).to(self.device)

            # 转换为6D归一化坐标
            eikonal_points_6d = self.coord_normer.raw_to_norm(
                eikonal_points_4d,
                append_linear_rot=True
            )

            # 提取5D坐标并设置requires_grad
            eikonal_coords_5d = eikonal_points_6d[:, :5].clone()
            eikonal_coords_5d.requires_grad = True

            # 获取grid特征（冻结）
            with torch.no_grad():
                grid_input = torch.cat([
                    eikonal_points_6d[:, :2],
                    eikonal_points_6d[:, -1:]
                ], dim=-1)

                feats_grid_raw = self._get_feats_fm_grid(grid_input)
                coords_encoded_frozen = self.pos_encoder(eikonal_coords_5d.detach())
                feats_grid = self.grid_mlp(
                    inputs=feats_grid_raw,
                    condition_features=coords_encoded_frozen
                )
                feats_grid = TF.normalize(feats_grid, dim=-1)
                feats_grid_exp = feats_grid.unsqueeze(0)

            # 非冻结的坐标编码 - 使用MetricNet专用编码器
            coords_encoded = self.pos_encoder_metric(eikonal_coords_5d)
            coords_enc_exp = coords_encoded.unsqueeze(0)

            # 扩展query特征
            query_feat_exp = query_feat.unsqueeze(1).expand(1, n_points_per_sample, -1)

            # MetricNet前向
            dist_eikonal_raw = self.metric_net(
                query_feat_exp,
                feats_grid_exp,
                coords_enc_exp
            )

            # 应用激活函数（可调试控制）
            if self.use_softplus:
                dist_eikonal = self.softplus(dist_eikonal_raw)
            else:
                dist_eikonal = dist_eikonal_raw

            # 计算梯度
            grad_outputs = torch.ones_like(dist_eikonal)
            grad_coords = torch.autograd.grad(
                outputs=dist_eikonal,
                inputs=eikonal_coords_5d,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]

            # 计算梯度范数
            grad_norm = grad_coords.norm(dim=-1)
            all_grad_norms.append(grad_norm.detach().cpu())

        # 合并所有样本
        all_grad_norms = torch.cat(all_grad_norms)

        # 计算统计量
        mean_norm = all_grad_norms.mean().item()
        std_norm = all_grad_norms.std().item()
        median_norm = all_grad_norms.median().item()
        deviation_from_1 = (all_grad_norms - 1.0).abs().mean().item()

        # 打印结果
        print("\n" + "-"*80)
        print("Eikonal约束评估结果 (||∇d|| ≈ 1):")
        print("-"*80)
        print(f"梯度范数均值:     {mean_norm:.6f}")
        print(f"梯度范数标准差:   {std_norm:.6f}")
        print(f"梯度范数中位数:   {median_norm:.6f}")
        print(f"与1的平均偏差:    {deviation_from_1:.6f}")
        print(f"范围:             [{all_grad_norms.min().item():.4f}, {all_grad_norms.max().item():.4f}]")
        print("-"*80)

        return {
            'mean_grad_norm': mean_norm,
            'std_grad_norm': std_norm,
            'median_grad_norm': median_norm,
            'deviation_from_1': deviation_from_1
        }


if __name__ == "__main__":
    import argparse
    import sys

    # 添加 --test_only 参数
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid conflict with get_parse
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    args, remaining_argv = parser.parse_known_args()

    # test by manual modification
    # args.test_only = True
    # 直接读取实验配置文件opts.yaml（包含所有参数，不再需要基础配置文件）
    # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_metric_net.yaml'])  # for training
    # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainers/.exps/stage3_metric_net_7/opts.yaml'])  # for testing

    # 如果没有指定配置文件，使用 stage3 的默认配置
    if '--p_yaml' not in ' '.join(remaining_argv):
        remaining_argv.extend(['--p_yaml', 'trainer_depends/configs/stage3_metric_net.yaml'])

    sys.argv[1:] = remaining_argv  # Pass remaining args to get_parse

    trainer = MetricNetTrainer()

    if args.test_only:
        trainer.test(use_train_uav=True)
    else:
        trainer.train()
