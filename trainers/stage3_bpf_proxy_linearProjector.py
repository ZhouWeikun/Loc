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
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainers.stage2_INGP import GridHashFitTrainer
# from trainer_depends.base.components import NetworkComponents
import torch.nn.functional as F


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
        self._init_projector()
        self.energy_temperature = 0.05
        self.loss_fn_state = None
        self.loss_fn_beta = None

        # 重新设置可训练参数
        self._setup_trainable_params_stage3()

        # 子空间采样器配置
        self.n_coarse = getattr(opt, 'n_coarse', (40, 30, 36, 1))  # 2304 类
        self.n_fine_per_coarse = getattr(opt, 'n_fine_per_coarse', (1, 1, 1, 1))  # 8 细分格子

        # 邻域半径定义, 2simag=核衰减宽度=有效权重半径
        # 相对长度定义，一个子空间对应一个单位长度，后续会和子空间单位长度相乘
        # self.sigma_nrc_bin = 0.5
        # self.sigma_radrot_bin = 0.5
        # 定义物理 Sigma网,绝对长度
        self.gs_sigma_nrc = 0.65 / 40.  #n_bins=40,即单位格长=1/40,sigma_bin=0.5
        self.gs_sigma_radrot = 0.65 * torch.pi / 18  #2sigma_abs=10度
        self.gs_sigma_logscale = 0.1
        # 定义物理 Sigma,绝对长度
        # self.phy_sigma_abs = {
        #     'gs_sigma_nrc': 0.5/40. , #n_bins=40,即单位网格长=1/40,sigma_bin=0.5
        #     'gs_sigma_radrot':0.5 * torch.pi/18,  #2sigma_abs=10度
        #     'gs_sigma_logscale': 0.1 #(Log单位，约 10% 缩放误差)
        # }

    def _init_projector(self):
        from models.projector_mlp_sample import Projector
        """初始化Projector"""
        print("\n" + "=" * 80)
        print("初始化 Projector ")
        print("=" * 80)

        # self.projector = Projector(
        #     input_dim = 1024,
        #     hidden_dims = [256,256],
        #     output_dim = 128
        # ).to(self.device)

        from models.projector_mlp_res import AdvancedProjector
        self.projector = AdvancedProjector(
            input_dim=1024,
            hidden_dim=1024,  # 保持宽通道
            expansion_factor=2,  # 内部升维到 2048
            num_res_blocks=3,  # 深度
            output_dim=128,
            use_spectral_norm=True,  # 关键：开启谱归一化
        ).to(self.device)

        print("✅ Projector 初始化完成")
        print("=" * 80 + "\n")

    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        for param in self.grid.parameters():
            param.requires_grad = False
        for param in self.grid_mlp.parameters():
            param.requires_grad = False

        self.param2optimize = {
            'projector': self.projector,
        }

        # 始终冻结Stage 1组件和Grid
        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
            'vis_aggregator': self.vis_aggregator,
            'grid': self.grid,
            'grid_mlp': self.grid_mlp,
        }

        # 动态生成参数配置信息
        freeze_grid_status = "freeze_grid=True" if getattr(self.opt, 'freeze_grid', True) else "freeze_grid=False"
        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys())

        print(f"参数配置 ({freeze_grid_status}):")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")

    def _load_checkpoints_for_test(self):
        """
        测试时加载checkpoint的统一方法

        加载逻辑：
        1. Stage 3自身的checkpoint (metric_net)
        2. Stage 2的checkpoint (grid, grid_mlp)
        3. Stage 1的预训练模型 (vis_encoder, vis_aggregator)
        """
        import yaml

        print("\n" + "=" * 80)
        print("加载测试用的checkpoint")
        print("=" * 80)

        # --- 1. 加载Stage 3的checkpoint (当前stage) ---
        stage3_ckpt_path = self._get_stage3_checkpoint_path()

        if stage3_ckpt_path:
            print(f"\n📦 Stage 3 checkpoint: {stage3_ckpt_path}")
            self._load_checkpoint(
                stage3_ckpt_path,
                self.param2optimize,
                mode='test'
            )
            self._load_loss_fn_temperature_from_ckpt(stage3_ckpt_path)
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

        print("\n" + "=" * 80)
        print("✅ 所有checkpoint加载完成")
        print("=" * 80 + "\n")

    def _load_loss_fn_temperature_from_ckpt(self, ckpt_path):
        """
        从checkpoint中读取loss_fn的beta作为测试温度参数。
        """
        if not ckpt_path:
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'loss_fn' not in ckpt:
            print("⚠️  checkpoint中未找到loss_fn，保持默认temperature")
            return

        loss_state = ckpt['loss_fn']
        self.loss_fn_state = loss_state

        beta = None
        if isinstance(loss_state, dict):
            if 'log_beta' in loss_state:
                beta = torch.exp(loss_state['log_beta'])
            elif 'fixed_beta' in loss_state:
                beta = loss_state['fixed_beta']

        if beta is None:
            print("⚠️  loss_fn中未找到beta，保持默认temperature")
            return

        beta_val = float(beta.detach().cpu().item())
        self.loss_fn_beta = beta_val
        self.energy_temperature = beta_val
        print(f"✅ 从loss_fn加载temperature(beta): {self.energy_temperature:.5f}")

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
                ckpts.sort(key=lambda x: int(x.replace('epoch', '').split('.')[0]))
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

    def _run_epoch_evaluation(self, epoch, run_visualization=False, n_test_samples=256):
        """
        在每个epoch结束时运行评估

        Args:
            epoch: 当前epoch编号
            run_visualization: 是否运行可视化（默认False）
            n_test_samples: 测试样本数量
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Epoch {epoch} 评估开始")
        self.logger.info(f"{'=' * 60}")

        # 切换为eval模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 运行3D分类测试
        results_3d = self._test_3d_classification_accuracy(
            n_samples=n_test_samples,
            use_train_uav=False,
            temperature=self.energy_temperature,
        )

        # 记录测试结果到日志
        self.logger.info(f"\n[Epoch {epoch}] 3D分类测试结果:")
        self.logger.info(f"  样本数: {results_3d['n_samples']}")
        self.logger.info(f"  总cell数: {results_3d['n_total_cells']}")
        self.logger.info(f"  Top-1准确率: {results_3d['top1_acc']:.2f}%")
        self.logger.info(f"  Top-8准确率: {results_3d['top8_acc']:.2f}%")
        self.logger.info(f"  Top-27准确率: {results_3d['top27_acc']:.2f}%")
        self.logger.info(f"  Top-64准确率: {results_3d['top64_acc']:.2f}%")
        self.logger.info(f"  Top-256准确率: {results_3d['top256_acc']:.2f}%")
        self.logger.info(f"  Top-512准确率: {results_3d['top512_acc']:.2f}%")
        self.logger.info(f"  平均排名: {results_3d['mean_rank']:.2f}")
        self.logger.info(f"  中位数排名: {results_3d['median_rank']:.2f}")
        self.logger.info(f"  2D位置误差 - 平均: {results_3d['mean_dist_error_2d']:.4f}")
        self.logger.info(f"  2D位置误差 - 中位数: {results_3d['median_dist_error_2d']:.4f}")
        self.logger.info(f"  旋转误差 - 平均: {results_3d['mean_rot_error_deg']:.2f}°")
        self.logger.info(f"  旋转误差 - 中位数: {results_3d['median_rot_error_deg']:.2f}°")
        self.logger.info(f"  固定位置旋转Top-1准确率: {results_3d['rot_only_top1_acc']:.2f}%")
        self.logger.info(f"  固定位置旋转Top-2准确率: {results_3d['rot_only_top2_acc']:.2f}%")
        self.logger.info(f"  固定位置旋转Top-3准确率: {results_3d['rot_only_top3_acc']:.2f}%")
        self.logger.info(f"{'=' * 60}\n")

        # 记录到TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('test/top1_acc', results_3d['top1_acc'], epoch)
            self.writer.add_scalar('test/top8_acc', results_3d['top8_acc'], epoch)
            self.writer.add_scalar('test/top27_acc', results_3d['top27_acc'], epoch)
            self.writer.add_scalar('test/top64_acc', results_3d['top64_acc'], epoch)
            self.writer.add_scalar('test/top256_acc', results_3d['top256_acc'], epoch)
            self.writer.add_scalar('test/top512_acc', results_3d['top512_acc'], epoch)
            self.writer.add_scalar('test/mean_rank', results_3d['mean_rank'], epoch)
            self.writer.add_scalar('test/mean_dist_error_2d', results_3d['mean_dist_error_2d'], epoch)
            self.writer.add_scalar('test/mean_rot_error_deg', results_3d['mean_rot_error_deg'], epoch)
            self.writer.add_scalar('test/rot_only_top1_acc', results_3d['rot_only_top1_acc'], epoch)

        # 恢复为train模式
        for model in self.param2optimize.values():
            model.train()

        self.logger.info(f"Epoch {epoch} 评估完成\n")
        return results_3d

    def _test_2d_classification_accuracy(self, n_samples=256, use_train_uav=False, temperature=0.5):
        """
        测试2D平面分类正确性

        Args:
            n_samples: 测试样本数量
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数

        Returns:
            dict: 包含各种准确率指标的字典
        """
        print(f"\n{'=' * 60}")
        print(f"2D平面分类测试")
        print(f"测试样本数: {n_samples}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, n_samples),
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_net(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

        # 获取每个2D grid cell的中心坐标（用于计算距离误差）
        # coords_candidates shape: [1, N_total, 4], 需要reshape成 [NR, NC, Rot, Scale, 4]
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        # 取每个2D cell的中心（对Rot和Scale取均值）
        cell_centers_2d = coords_reshaped[:, :, 0, 0, :2]  # [NR, NC, 2] 取第一个rot/scale的坐标

        # 统计容器
        all_ranks = []
        all_dist_errors = []
        all_rot_ranks_weighted = []  # 加权聚合后的旋转排名
        all_rot_ranks_marginalized = []  # 直接边缘化后的旋转排名
        processed = 0

        with torch.no_grad():
            for batch in test_loader:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.device)
                coords_gt = batch[1].to(self.device)  # [B, 4]
                batch_size = imgs.shape[0]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

                # 计算能量
                energys = self.projector.compute_energy(
                    feats_vis, feats_grid_all, metric='euclidean'
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                energys_reshaped = energys.reshape(batch_size, *n_coarse)

                # 边缘化Rot和Scale维度 (使用logsumexp)
                neg_dists = -energys_reshaped
                scaled_logits = neg_dists / temperature
                logits_2d = torch.logsumexp(scaled_logits, dim=[-2, -1])  # [B, NR, NC]

                # 获取预测的2D索引
                logits_flat = logits_2d.view(batch_size, -1)  # [B, NR*NC]
                pred_indices = logits_flat.argmax(dim=-1)  # [B]
                pred_nr = pred_indices // n_coarse_2d[1]
                pred_nc = pred_indices % n_coarse_2d[1]

                # 计算GT的2D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_flat_idx = gt_nr * n_coarse_2d[1] + gt_nc

                # 计算排名
                sorted_indices = logits_flat.argsort(dim=-1, descending=True)
                for i in range(batch_size):
                    rank = (sorted_indices[i] == gt_flat_idx[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_ranks.append(rank)

                # 计算距离误差（预测cell中心到GT坐标的距离）
                pred_centers = cell_centers_2d[pred_nr, pred_nc]  # [B, 2]
                gt_coords_2d = coords_gt[:, :2]  # [B, 2]
                dist_errors = torch.norm(pred_centers - gt_coords_2d, dim=-1, p=2)
                all_dist_errors.extend(dist_errors.cpu().numpy().tolist())

                # ========== 计算加权旋转分类精度（类似train的逻辑） ==========
                # 1. 获取4D logits: [B, NR, NC, Rot] (边缘化Scale维度)
                logits_4d = torch.logsumexp(scaled_logits, dim=-1)  # [B, NR, NC, Rot]

                # 2. 计算2D空间概率分布（作为权重）
                probs_2d = torch.softmax(logits_2d.view(batch_size, -1) / temperature, dim=-1)  # [B, NR*NC]
                probs_2d = probs_2d.reshape(batch_size, n_coarse[0], n_coarse[1])  # [B, NR, NC]

                # 3. 计算每个2D位置的旋转概率分布
                probs_rot_local = torch.softmax(logits_4d / temperature, dim=-1)  # [B, NR, NC, Rot]

                # 4. 用2D空间权重对旋转分布进行加权聚合
                spatial_weights = probs_2d.unsqueeze(-1)  # [B, NR, NC, 1]
                probs_rot_aggregated = (probs_rot_local * spatial_weights).sum(dim=[1, 2])  # [B, Rot]

                # 5. 计算GT旋转的排名
                gt_rot = gt_indices_multi[:, 2]  # [B]
                for i in range(batch_size):
                    # 对聚合后的旋转分布排序
                    sorted_rot_indices = probs_rot_aggregated[i].argsort(descending=True)
                    # 找到GT旋转的排名
                    rot_rank_weighted = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks_weighted.append(rot_rank_weighted)

                # ========== 计算边缘化旋转分类精度（直接对除Rot外的维度求和） ==========
                # 1. 计算概率体积: [B, NR, NC, Rot, Scale]
                probs_5d = torch.softmax(scaled_logits.view(batch_size, -1) / temperature, dim=-1)
                probs_5d = probs_5d.reshape(batch_size, *n_coarse)  # [B, NR, NC, Rot, Scale]

                # 2. 边缘化除Rot外的所有维度（求和）
                probs_rot_marginalized = probs_5d.sum(dim=[1, 2, 4])  # [B, Rot]

                # 3. 计算GT旋转的排名
                for i in range(batch_size):
                    # 对边缘化后的旋转分布排序
                    sorted_rot_indices = probs_rot_marginalized[i].argsort(descending=True)
                    # 找到GT旋转的排名
                    rot_rank_marg = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks_marginalized.append(rot_rank_marg)

                processed += batch_size

        # 计算指标
        all_ranks = np.array(all_ranks)
        all_dist_errors = np.array(all_dist_errors)
        all_rot_ranks_weighted = np.array(all_rot_ranks_weighted)
        all_rot_ranks_marginalized = np.array(all_rot_ranks_marginalized)
        n_total_cells = n_coarse_2d[0] * n_coarse_2d[1]

        results = {
            'n_samples': len(all_ranks),
            'n_total_cells': n_total_cells,
            'top1_acc': (all_ranks == 1).mean() * 100,
            'top4_acc': (all_ranks <= 4).mean() * 100,
            'top9_acc': (all_ranks <= 9).mean() * 100,
            'top16_acc': (all_ranks <= 16).mean() * 100,
            'mean_rank': all_ranks.mean(),
            'median_rank': np.median(all_ranks),
            'mean_dist_error': all_dist_errors.mean(),
            'median_dist_error': np.median(all_dist_errors),
            'dist_error_std': all_dist_errors.std(),
            # 加权旋转分类准确率
            'rot_weighted_top1_acc': (all_rot_ranks_weighted == 1).mean() * 100,
            'rot_weighted_top2_acc': (all_rot_ranks_weighted <= 2).mean() * 100,
            'rot_weighted_top3_acc': (all_rot_ranks_weighted <= 3).mean() * 100,
            'rot_weighted_mean_rank': all_rot_ranks_weighted.mean(),
            'rot_weighted_median_rank': np.median(all_rot_ranks_weighted),
            # 边缘化旋转分类准确率
            'rot_marginalized_top1_acc': (all_rot_ranks_marginalized == 1).mean() * 100,
            'rot_marginalized_top2_acc': (all_rot_ranks_marginalized <= 2).mean() * 100,
            'rot_marginalized_top3_acc': (all_rot_ranks_marginalized <= 3).mean() * 100,
            'rot_marginalized_mean_rank': all_rot_ranks_marginalized.mean(),
            'rot_marginalized_median_rank': np.median(all_rot_ranks_marginalized),
        }

        # 打印结果
        print(f"\n{'=' * 60}")
        print(f"2D分类测试结果 (共{n_total_cells}个cell)")
        print(f"{'=' * 60}")
        print(f"Top-1  准确率: {results['top1_acc']:.2f}%")
        print(f"Top-4  准确率: {results['top4_acc']:.2f}%")
        print(f"Top-9  准确率: {results['top9_acc']:.2f}%")
        print(f"Top-16 准确率: {results['top16_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"距离误差统计:")
        print(f"  平均误差: {results['mean_dist_error']:.4f}")
        print(f"  中位数误差: {results['median_dist_error']:.4f}")
        print(f"  标准差: {results['dist_error_std']:.4f}")
        print(f"{'=' * 60}")
        print(f"加权旋转分类准确率 (共{n_coarse[2]}个rot, 使用2D预测概率加权):")
        print(f"  Top-1 准确率: {results['rot_weighted_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_weighted_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_weighted_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_weighted_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_weighted_median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"边缘化旋转分类准确率 (共{n_coarse[2]}个rot, 直接对除Rot外维度求和):")
        print(f"  Top-1 准确率: {results['rot_marginalized_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_marginalized_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_marginalized_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_marginalized_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_marginalized_median_rank']:.2f}")
        print(f"{'=' * 60}\n")

        return results

    def _test_2d_sequence_localization_accuracy(
            self,
            n_samples=None,
            use_train_uav=False,
            temperature=0.5,
            seq_window_len=5,
            len_neighbors=2,
            shuffle=False,
            save_pred_pdf=True
    ):
        """
        测试2D平面序列定位精度（支持序列聚合）

        这个函数是 _test_2d_classification_accuracy 的升级版，支持：
        1. 单帧定位测试
        2. 序列聚合定位测试
        3. n×n邻域聚合定位测试

        Args:
            n_samples: 测试样本数量（None表示使用全部数据）
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数
            seq_window_len: 序列聚合窗口长度
            len_neighbors: 邻域大小（2表示2×2=4个邻域）
            shuffle: 是否打乱数据顺序（序列测试时应该为False）
            save_pred_pdf: 是否保存预测概率分布到checkpoint文件夹（默认True）

        Returns:
            dict: 包含各种准确率指标的字典
        """
        from trainer_depends.datasets.util_core_loc_in_girds import (
            agg_seq_pdf,
            compute_agged_pred_nneighbors_id
        )

        print(f"\n{'=' * 80}")
        print(f"2D序列定位测试")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"序列聚合窗口: {seq_window_len}")
        print(f"邻域大小: {len_neighbors}×{len_neighbors} = {len_neighbors ** 2}")
        print(f"{'=' * 80}\n")

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 确定测试样本数
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        # 创建DataLoader（序列测试通常不打乱）
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # ==================== 预计算候选点特征 ====================
        print("预计算所有候选点的特征...")

        coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]
        n_grid_h, n_grid_w = n_coarse_2d[0], n_coarse_2d[1]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_net(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

        # 获取每个2D grid cell的中心坐标
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        cell_centers_2d = coords_reshaped[:, :, 0, 0, :2]  # [NR, NC, 2]

        print(f"✅ 候选点特征预计算完成，网格大小: {n_grid_h}×{n_grid_w}\n")

        # ==================== 收集所有预测和GT ====================
        print("开始推理并收集预测概率...")

        pred_pdf_list = []  # 收集所有预测的概率分布
        q_label_list = []  # 收集所有GT标签
        coords_gt_list = []  # 收集所有GT坐标
        processed = 0

        with torch.no_grad():
            for batch in test_loader:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.device)
                coords_gt = batch[1].to(self.device)  # [B, 4]
                batch_size = imgs.shape[0]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

                # 计算能量
                energys = self.projector.compute_energy(
                    feats_vis, feats_grid_all, metric='euclidean'
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                energys_reshaped = energys.reshape(batch_size, *n_coarse)

                # 边缘化Rot和Scale维度 (使用logsumexp)
                neg_dists = -energys_reshaped
                scaled_logits = neg_dists / temperature
                logits_2d = torch.logsumexp(scaled_logits, dim=[-2, -1])  # [B, NR, NC]

                # 转换为概率分布
                pred_pdf = torch.softmax(logits_2d.view(batch_size, -1), dim=-1)  # [B, H*W]
                pred_pdf_list.append(pred_pdf.cpu())

                # 计算GT标签
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_flat_idx = gt_nr * n_coarse_2d[1] + gt_nc
                q_label_list.append(gt_flat_idx.cpu())
                coords_gt_list.append(coords_gt.cpu())

                processed += batch_size

        # 拼接所有结果
        pred_pdf_all = torch.cat(pred_pdf_list, dim=0)  # [N, H*W]
        q_label_all = torch.cat(q_label_list, dim=0)  # [N]
        coords_gt_all = torch.cat(coords_gt_list, dim=0)  # [N, 4]
        n_total_samples = pred_pdf_all.shape[0]

        print(f"✅ 收集完成，共 {n_total_samples} 个样本\n")

        # ==================== 保存预测概率分布 ====================
        if save_pred_pdf:
            import os

            # 获取checkpoint路径
            stage3_ckpt_path = self._get_stage3_checkpoint_path()
            if stage3_ckpt_path:
                # 保存到checkpoint同目录
                ckpt_dir = os.path.dirname(stage3_ckpt_path)
                save_dir = os.path.join(ckpt_dir, 'loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 准备保存数据
                save_data = {
                    'pred_pdf_all': pred_pdf_all.numpy(),  # [N, H*W]
                    'q_label_all': q_label_all.numpy(),  # [N]
                    'coords_gt_all': coords_gt_all.numpy(),  # [N, 4]
                    'n_grid_h': n_grid_h,
                    'n_grid_w': n_grid_w,
                    'n_samples': n_total_samples,
                    'cell_centers_2d': cell_centers_2d.cpu().numpy(),  # [H, W, 2]
                    'test_config': {
                        'use_train_uav': use_train_uav,
                        'temperature': temperature,
                    }
                }

                # 保存为npz文件
                dataset_name = 'train' if use_train_uav else 'test'
                save_path = os.path.join(save_dir, f'pred_pdf_{dataset_name}_n{n_total_samples}.npz')
                np.savez_compressed(save_path, **save_data)

                print(f"💾 已保存预测概率分布: {save_path}")
                print(f"   - pred_pdf_all shape: {pred_pdf_all.shape}")
                print(f"   - q_label_all shape: {q_label_all.shape}")
                print(f"   - coords_gt_all shape: {coords_gt_all.shape}")
                print(f"   - cell_centers_2d shape: {cell_centers_2d.shape}\n")
            else:
                print("⚠️  未找到checkpoint路径，跳过保存预测概率分布\n")

        # ==================== 单帧定位测试 ====================
        print("=" * 80)
        print("1. 单帧定位测试")
        print("=" * 80)

        results_single = self._compute_2d_loc_metrics(
            pred_pdf_all,
            q_label_all,
            coords_gt_all,
            cell_centers_2d,
            n_grid_w,
            title="单帧定位"
        )

        # ==================== 序列聚合定位测试 ====================
        print("\n" + "=" * 80)
        print(f"2. 序列聚合定位测试 (窗口长度={seq_window_len})")
        print("=" * 80)

        #debug2vis:
        if True:  # 设置为False可关闭可视化
            import matplotlib.pyplot as plt
            import os

            k_samples = 16  # 可视化前k个样本
            n_cols = 4  # 每行显示4个
            n_rows = (k_samples + n_cols - 1) // n_cols

            # Reshape概率分布 [k, H*W] -> [k, H, W]
            pred_pdf_vis = pred_pdf_all[:k_samples].reshape(k_samples, n_grid_h, n_grid_w).numpy()
            q_labels_vis = q_label_all[:k_samples].numpy()

            # 创建子图
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), dpi=100)
            if n_rows == 1:
                axes = axes[None, :]

            for idx in range(k_samples):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                # 绘制热力图
                im = ax.imshow(pred_pdf_vis[idx], cmap='hot', interpolation='bilinear')

                # 标记GT位置
                gt_label = q_labels_vis[idx]
                gt_nr = gt_label // n_grid_w
                gt_nc = gt_label % n_grid_w
                ax.plot(gt_nc, gt_nr, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)

                # 标记预测位置
                pred_label = pred_pdf_vis[idx].argmax()
                pred_nr = pred_label // n_grid_w
                pred_nc = pred_label % n_grid_w
                ax.plot(pred_nc, pred_nr, 'bx', markersize=12, markeredgewidth=2)

                # 标题
                is_correct = (pred_label == gt_label)
                title_color = 'green' if is_correct else 'red'
                ax.set_title(f'Sample {idx}\n{"✓" if is_correct else "✗"}',
                             fontsize=10, color=title_color)

                # 添加颜色条
                plt.colorbar(im, ax=ax, fraction=0.046)

                ax.axis('off')

            # 隐藏多余的子图
            for idx in range(k_samples, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')

            plt.tight_layout()

            # 保存可视化
            vis_save_dir = os.path.dirname(self.opt.load2test)
            epoch_id = os.path.basename(self.opt.load2test)[-7:-4]
            vis_save_path = os.path.join(vis_save_dir, f'pred_pdf_samples_' + epoch_id + '.png')
            plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"🎨 已保存概率分布可视化: {vis_save_path}")
            print(f"   - 绿色星号(*) = GT位置")
            print(f"   - 蓝色叉号(×) = 预测位置")
            print(f"   - 绿色标题 = 预测正确，红色标题 = 预测错误\n")

        # 进行序列聚合
        pred_pdf_agged = agg_seq_pdf(
            pred_pdf_all.to(self.device),
            window_len=seq_window_len,
            padding=False
        ).cpu()  # [N-window_len+1, H*W]

        # GT也需要截取对应的部分
        q_label_agged = q_label_all[seq_window_len - 1:]
        coords_gt_agged = coords_gt_all[seq_window_len - 1:]

        results_seq = self._compute_2d_loc_metrics(
            pred_pdf_agged,
            q_label_agged,
            coords_gt_agged,
            cell_centers_2d,
            n_grid_w,
            title=f"序列聚合(窗口={seq_window_len})"
        )

        # ==================== 邻域聚合测试 ====================
        print("\n" + "=" * 80)
        print(f"3. {len_neighbors}×{len_neighbors}邻域聚合测试")
        print("=" * 80)

        # 计算单帧的邻域聚合
        id_neighbors_1d, id_neighbors_2d = compute_agged_pred_nneighbors_id(
            pred_pdf_all.reshape(-1, n_grid_h, n_grid_w).to(self.device),
            len_neighbors,
            ret_2d=True
        )  # [N, n*n], [N, n*n, 2]

        k_values = list(range(1, len_neighbors ** 2 + 1))
        results_neighbors = self._compute_neighbors_recall(
            q_label_all.numpy(),
            id_neighbors_1d.cpu().numpy(),
            k_values,
            title=f"单帧+{len_neighbors ** 2}邻域"
        )

        # 计算序列聚合+邻域聚合
        id_neighbors_seq_1d, id_neighbors_seq_2d = compute_agged_pred_nneighbors_id(
            pred_pdf_agged.reshape(-1, n_grid_h, n_grid_w).to(self.device),
            len_neighbors,
            ret_2d=True
        )

        results_seq_neighbors = self._compute_neighbors_recall(
            q_label_agged.numpy(),
            id_neighbors_seq_1d.cpu().numpy(),
            k_values,
            title=f"序列聚合+{len_neighbors ** 2}邻域"
        )

        # ==================== 汇总结果 ====================
        results = {
            # 基本信息
            'n_samples': n_total_samples,
            'n_grid_cells': n_grid_h * n_grid_w,
            'n_grid_h': n_grid_h,
            'n_grid_w': n_grid_w,
            'seq_window_len': seq_window_len,
            'len_neighbors': len_neighbors,

            # 单帧结果
            'single_frame': results_single,

            # 序列聚合结果
            'sequence_agg': results_seq,

            # 邻域结果
            'neighbors': results_neighbors,
            'seq_neighbors': results_seq_neighbors,
        }

        # ==================== 打印总结 ====================
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        print(f"总样本数: {n_total_samples}")
        print(f"网格大小: {n_grid_h}×{n_grid_w} = {n_grid_h * n_grid_w} cells")
        print(f"\n【单帧定位】")
        print(f"  Top-1: {results_single['top1_acc']:.2f}%")
        print(f"  Top-4: {results_single['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_single['mean_dist_error']:.4f}")
        print(f"\n【序列聚合 (窗口={seq_window_len})】")
        print(f"  Top-1: {results_seq['top1_acc']:.2f}%")
        print(f"  Top-4: {results_seq['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_seq['mean_dist_error']:.4f}")
        print(f"\n【{len_neighbors ** 2}邻域聚合】")
        print(f"  单帧@1: {results_neighbors['recall@1']:.2f}%")
        print(f"  单帧@{len_neighbors ** 2}: {results_neighbors[f'recall@{len_neighbors ** 2}']:.2f}%")
        print(f"  序列@1: {results_seq_neighbors['recall@1']:.2f}%")
        print(f"  序列@{len_neighbors ** 2}: {results_seq_neighbors[f'recall@{len_neighbors ** 2}']:.2f}%")
        print("=" * 80 + "\n")

        return results

    def _compute_2d_loc_metrics(
            self,
            pred_pdf,
            q_labels,
            coords_gt,
            cell_centers_2d,
            n_grid_w,
            title="定位测试"
    ):
        """
        计算2D定位指标

        Args:
            pred_pdf: [N, H*W] 预测概率
            q_labels: [N] GT标签
            coords_gt: [N, 4] GT坐标
            cell_centers_2d: [H, W, 2] 每个cell的中心坐标
            n_grid_h, n_grid_w: 网格大小
            title: 标题
        """
        # 预测的cell索引
        pred_indices = pred_pdf.argmax(dim=-1)  # [N]
        pred_nr = pred_indices // n_grid_w
        pred_nc = pred_indices % n_grid_w

        # GT的cell索引
        gt_nr = q_labels // n_grid_w
        gt_nc = q_labels % n_grid_w

        # 计算排名
        sorted_indices = pred_pdf.argsort(dim=-1, descending=True)
        ranks = []
        for i in range(len(q_labels)):
            rank = (sorted_indices[i] == q_labels[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        ranks = np.array(ranks)

        # 计算距离误差
        pred_centers = cell_centers_2d[pred_nr, pred_nc]  # [N, 2]
        gt_coords_2d = coords_gt[:, :2]  # [N, 2]
        dist_errors = torch.norm(pred_centers - gt_coords_2d.to(pred_centers.device), dim=-1, p=2)

        # 汇总指标
        results = {
            'n_samples': len(ranks),
            'top1_acc': (ranks == 1).mean() * 100,
            'top2_acc': (ranks <= 2).mean() * 100,
            'top3_acc': (ranks <= 3).mean() * 100,
            'top4_acc': (ranks <= 4).mean() * 100,
            'top9_acc': (ranks <= 9).mean() * 100,
            'top16_acc': (ranks <= 16).mean() * 100,
            'mean_rank': ranks.mean(),
            'median_rank': np.median(ranks),
            'mean_dist_error': dist_errors.mean().item(),
            'median_dist_error': torch.median(dist_errors).item(),
            'dist_error_std': dist_errors.std().item(),
        }

        # 打印结果
        print(f"{title}结果:")
        print(f"  样本数: {results['n_samples']}")
        print(f"  Top-1: {results['top1_acc']:.2f}%")
        print(f"  Top-2: {results['top2_acc']:.2f}%")
        print(f"  Top-3: {results['top3_acc']:.2f}%")
        print(f"  Top-4: {results['top4_acc']:.2f}%")
        print(f"  Top-16: {results['top16_acc']:.2f}%")
        print(f"  平均排名: {results['mean_rank']:.2f}")
        print(f"  中位数排名: {results['median_rank']:.2f}")
        print(f"  距离误差 - 平均: {results['mean_dist_error']:.4f}")
        print(f"  距离误差 - 中位数: {results['median_dist_error']:.4f}")
        print(f"  距离误差 - 标准差: {results['dist_error_std']:.4f}")

        return results

    def _compute_neighbors_recall(self, q_labels, id_neighbors, k_values, title="邻域Recall"):
        """
        计算邻域recall

        Args:
            q_labels: [N] GT标签
            id_neighbors: [N, K] 预测的K个邻域cell索引
            k_values: list of int，要计算的k值
            title: 标题
        """
        recall_dict = {}

        # 尝试使用现成的函数
        try:
            import sys
            sys.path.insert(0, '/home/data/zwk/pyproj_pylib_zwk')
            from uavloc_utils.eval_recall_fm_salad import compute_recall_by_label
            recall_dict_raw = compute_recall_by_label(q_labels, id_neighbors, k_values, title=title)

            # 确保键格式为 'recall@k'
            for k in k_values:
                # 尝试不同的键格式
                if f'recall@{k}' in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[f'recall@{k}']
                elif k in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[k]
                elif f'Recall@{k}' in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[f'Recall@{k}']

        except Exception as e:
            # 如果导入失败，手动计算
            print(f"⚠️  无法导入compute_recall_by_label ({str(e)})，使用简化版计算")
            print(f"{title}:")
            for k in k_values:
                # 检查GT是否在top-k中
                correct = np.any(id_neighbors[:, :k] == q_labels[:, None], axis=1)
                recall = correct.mean() * 100
                recall_dict[f'recall@{k}'] = recall
                print(f"  Recall@{k}: {recall:.2f}%")

        return recall_dict

    def _get_dir2save(self, ret_epoch=False):
        if hasattr(self, 'exp_dir2save') and self.exp_dir2save and os.path.exists(self.exp_dir2save):
            # 训练模式：保存到当前实验目录
            checkpoint_dir = self.exp_dir2save[1:]
            save_dir = os.path.join(checkpoint_dir, 'loc_results')
            os.makedirs(save_dir, exist_ok=True)

            # 从实验目录找到最新的epoch号
            ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch') and f.endswith('.pth')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace('epoch', '').replace('.pth', '')))
                epoch_num = ckpts[-1].replace('epoch', '').replace('.pth', '')
            else:
                epoch_num = 'current'
        else:
            # 测试模式：使用 load2test 指定的路径
            stage3_ckpt_path = self._get_stage3_checkpoint_path()
            if stage3_ckpt_path:
                checkpoint_dir = os.path.dirname(stage3_ckpt_path)
                save_dir = os.path.join(checkpoint_dir, 'loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 从checkpoint文件名提取epoch号
                epoch_num = os.path.basename(stage3_ckpt_path).replace('epoch', '').replace('.pth', '')

        if ret_epoch:
            return save_dir, epoch_num
        else:
            return save_dir

    def _save_pred_pdf_3d(
        self,
        pred_pdf_3d_all,
        q_label_3d_all,
        coords_gt_all,
        n_coarse_3d,
        cell_centers_3d,
        temperature,
        data_type,
        tag=""
    ):
        save_dir, epoch_num = self._get_dir2save(ret_epoch=True)
        tag_part = f"_{tag}" if tag else ""
        save_filename = (
            f'pred_3d{tag_part}_ep{epoch_num}_nr{n_coarse_3d[0]}_nc{n_coarse_3d[1]}_nrot{n_coarse_3d[2]}.npz'
        )
        save_path = os.path.join(save_dir, save_filename)

        if isinstance(pred_pdf_3d_all, torch.Tensor):
            pred_pdf_3d_all = pred_pdf_3d_all.detach().cpu().numpy()
        if isinstance(q_label_3d_all, torch.Tensor):
            q_label_3d_all = q_label_3d_all.detach().cpu().numpy()
        if isinstance(coords_gt_all, torch.Tensor):
            coords_gt_all = coords_gt_all.detach().cpu().numpy()
        if isinstance(cell_centers_3d, torch.Tensor):
            cell_centers_3d = cell_centers_3d.detach().cpu().numpy()

        save_data = {
            'pred_pdf_3d_all': pred_pdf_3d_all,
            'q_label_3d_all': q_label_3d_all,
            'coords_gt_all': coords_gt_all,
            'n_coarse_3d': n_coarse_3d,
            'cell_centers_3d': cell_centers_3d,
            'temperature': temperature,
            'n_samples': pred_pdf_3d_all.shape[0],
            'data_type': data_type,
        }

        np.savez(save_path, **save_data)
        print(f"\n✓ 已保存3D预测概率分布到: {save_path}")
        print(f"  - pred_pdf_3d_all: {pred_pdf_3d_all.shape}")
        print(f"  - q_label_3d_all: {q_label_3d_all.shape}")
        print(f"  - coords_gt_all: {coords_gt_all.shape}")
        print(f"  - n_coarse_3d: {n_coarse_3d}")
        print(f"  - cell_centers_3d: {cell_centers_3d.shape}")

    def _compute_metric_from_query_and_points(
        self,
        query_feats,
        ref_points,
        temperature=10.0,
        metric='possibility',
        coord_space='raw',
        chunk_size=2048,
        feat_type='projector',
    ):
        """
        计算 query_feats 与 ref_points 的距离或概率。

        Args:
            query_feats: [B, C]
            ref_points: [N, 4] or [B, N, 4]
            temperature: softmax 温度（仅在 metric='possibility' 时使用）
            metric: 'dist' 返回距离；'possibility' 返回 softmax 概率。
                    兼容旧写法: 'euclidean'/'l2' 等同于 'possibility'
            coord_space: 'raw' (nr,nc,rot,scale) 或 'linear'
            chunk_size: 候选分块大小
            feat_type: 'ingp' 使用 INGP 特征距离；'projector' 使用 Projector 输出特征距离
        Returns:
            dist 或 prob: [B, N]
        """
        metric = metric.lower()
        if metric in ('euclidean', 'l2'):
            metric = 'possibility'
        if metric not in ('dist', 'possibility'):
            raise ValueError(f"metric must be 'dist' or 'possibility', got {metric}")

        feat_type = feat_type.lower()
        if feat_type not in ('ingp', 'projector'):
            raise ValueError(f"feat_type must be 'ingp' or 'projector', got {feat_type}")

        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # 如果是形如 [N, 1, 4] 的输入，视为单batch展开
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])
        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points shape not supported after reshape: {ref_points.shape}")

        # 归一化坐标
        if coord_space == 'linear':
            ref_points_raw = self.coord_normer.linear_to_raw(ref_points.reshape(-1, 4))
        else:
            ref_points_raw = ref_points.reshape(-1, 4)

        # chunk_size 控制候选维度分块，避免 grid_mlp 激活过大
        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points_raw.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        # 预处理 query 特征
        if feat_type == 'ingp':
            query_emb = TF.normalize(query_feats, dim=-1)
        else:
            with torch.no_grad():
                query_emb = self.projector(query_feats)

        dist_chunks = []
        if ref_points.dim() == 2:
            for start in range(0, ref_points_raw.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points_raw.shape[0])
                coords_chunk = ref_points_raw[start:end]
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1)
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)
        else:
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coords_chunk_flat = coords_chunk.reshape(-1, 4)
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk_flat, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1).view(B, -1, feats_ref.shape[-1])
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk_flat, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                    feats_ref = feats_ref.view(B, -1, feats_ref.shape[-1])
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)

        dist = torch.cat(dist_chunks, dim=1)

        if metric == 'dist':
            return dist

        logit = -temperature * dist
        prob = F.softmax(logit, dim=-1)
        return prob

    def _compute_metric_from_ingp(
        self,
        query_feats,
        ref_points,
        coord_space='raw',
        chunk_size=4096,
        metric='sim',
    ):
        """
        使用 INGP 特征与视觉特征计算相似度/距离，支持分块避免 OOM。

        Returns:
            metric_out: [B, N] 相似度或距离
        """
        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # 展平不匹配 batch 的 [N,1,4] 等情况
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])

        q_norm = TF.normalize(query_feats, dim=-1)
        q_norm = query_feats
        metric = metric.lower()
        if metric not in ('sim', 'dist'):
            raise ValueError(f"metric must be 'sim' or 'dist', got {metric}")

        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        metric_chunks = []
        if ref_points.dim() == 2:
            # 公用候选 [N, 4]
            for start in range(0, ref_points.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points.shape[0])
                coords_chunk = ref_points[start:end]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk, coord_mode=coord_mode)
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    # [B, C] x [C, chunk] -> [B, chunk]
                    metric_chunk = torch.matmul(q_norm, feats_chunk.t())
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        elif ref_points.dim() == 3:
            # 每个样本各自的候选 [B, N, 4]
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk.reshape(-1, 4), coord_mode=coord_mode)
                feats_chunk = feats_chunk.view(B, -1, feats_chunk.shape[-1])
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    metric_chunk = torch.sum(q_norm.unsqueeze(1) * feats_chunk, dim=-1)  # [B, chunk]
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        else:
            raise ValueError(f"ref_points shape not supported: {ref_points.shape}")

        metric_out = torch.cat(metric_chunks, dim=1)
        return metric_out

    def _opt_coords_topN(self, coords_topN, feat_q, n_step=200,lr=1e-5):
        """
        对TopN候选坐标进行梯度优化重排序

        Args:
            coords_topN: [B, N, 4] - TopN候选坐标 (nr, nc, rot, scale)
            feat_q: [B, C] - query特征
            n_step: int - 优化步数

        Returns:
            coords_sorted: [B, N, 4] - 按优化后的loss排序的坐标
        """
        import math

        # Detach feat_q，因为我们只优化坐标，不需要对query特征求导
        feat_q = feat_q.detach().to(self.device)
        coords_opted_topN = []
        loss_topN = []

        # 临时设置为train模式以启用梯度计算图构建（即使参数是冻结的）
        # 这是必需的，因为eval()模式可能会阻止梯度流动到输入
        grid_was_training = self.grid.training
        grid_mlp_was_training = self.grid_mlp.training
        pos_encoder_was_training = self.pos_encoder_grid.training

        self.grid.train()
        self.grid_mlp.train()
        self.pos_encoder_grid.train()

        for id in range(coords_topN.shape[1]):
            coords_init = coords_topN[:, id, :].clone().detach()

            # 关键改进：分别创建参数并设置不同学习率
            xy_param = coords_init[:, :2].clone().requires_grad_(True)
            rot_param = coords_init[:, 2:3].clone().requires_grad_(True)
            scale_param = coords_init[:, 3:4].clone().requires_grad_(True)

            # 根据各维度的数值范围设置学习率
            optimizer = torch.optim.Adam([
                {'params': [xy_param], 'lr': lr},
                {'params': [rot_param], 'lr': lr*0.5},
                {'params': [scale_param], 'lr': lr}
            ], lr=1e-4)  # 默认学习率（会被参数组覆盖）

            # 可选：学习率调度
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_step, eta_min=1e-6
            )

            for i in range(n_step):
                optimizer.zero_grad()

                # 关键修改：将 enable_grad 提早开启，覆盖参数的使用起点
                with torch.enable_grad():
                    # 1. 组合坐标 (必须在 enable_grad 下进行，才能追踪到 xy_param)
                    coords2opt = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                    # 2. 转换为6D坐标
                    coords_6d = self.coord_normer.raw_to_net(coords2opt, append_linear_rot=True)

                    # 3. 提取特征与前向传播
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feat_ref_raw = self._get_feats_fm_grid(grid_input)

                    coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                    feat_ref = self.grid_mlp(feat_ref_raw, coords_encoded)
                    feat_ref = TF.normalize(feat_ref, dim=-1, p=2)

                    # 4. 计算 Loss
                    loss = TF.mse_loss(feat_q, feat_ref)

                    # 5. 反向传播
                    loss.backward()
                # 优化步 (可以在 enable_grad 外，也可以在内，通常在外)
                # 可选：梯度裁剪
                torch.nn.utils.clip_grad_norm_([xy_param, rot_param, scale_param], max_norm=1.0)

                optimizer.step()
                scheduler.step()

                if i % 50 == 0:  # 减少打印频率
                    print(f"  候选 {id + 1}/{coords_topN.shape[1]}, Step {i}/{n_step}, Loss: {loss.item():.5f}")

            # 保存最终结果
            with torch.no_grad():
                coords_final = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                # 计算最终loss
                coords_6d_final = self.coord_normer.raw_to_net(coords_final, append_linear_rot=True)
                grid_input_final = torch.cat([coords_6d_final[:, :2], coords_6d_final[:, -1:]], dim=-1)
                feat_ref_final_raw = self._get_feats_fm_grid(grid_input_final)
                coords_encoded_final = self.pos_encoder_grid(coords_6d_final[:, :5])
                feat_ref_final = self.grid_mlp(feat_ref_final_raw, coords_encoded_final)
                feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)

                final_loss = TF.mse_loss(
                    feat_q, feat_ref_final, reduction='none'
                ).mean(dim=-1)

                loss_topN.append(final_loss)
                coords_opted_topN.append(coords_final.detach())

        coords_opted_topN = torch.stack(coords_opted_topN, dim=1)
        loss_topN = torch.stack(loss_topN, dim=1)

        # 候选重排序
        sorted_indices = loss_topN.argsort(dim=1)
        coords_sorted = coords_opted_topN[
            torch.arange(coords_opted_topN.shape[0]).unsqueeze(1),
            sorted_indices
        ]

        # 恢复模型的原始状态
        if not grid_was_training:
            self.grid.eval()
        if not grid_mlp_was_training:
            self.grid_mlp.eval()
        if not pos_encoder_was_training:
            self.pos_encoder_grid.eval()

        return coords_sorted

    def _test_3d_classification_accuracy(
        self,
        n_samples=256,
        use_train_uav=False,
        shuffle=False,
        temperature=10,
        energy_fn=None,
        save_pred_pdf=True,
        chunk_size=2048,
    ):
        """
        测试3D分类正确性 (NR, NC, Rot)，只边缘化Scale维度

        Args:
            n_samples: 测试样本数量（None表示使用全部数据）
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数
            shuffle: 是否打乱数据顺序（序列测试时应该为False）
            save_pred_pdf: 是否保存预测概率分布到checkpoint文件夹（默认True）

        Returns:
            dict: 包含各种准确率指标的字典
        """
        temperature = temperature if temperature is not None else self.energy_temperature
        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 确定测试样本数
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False, rand_offset=False,
        )
        coords_candidates_flat = coords_candidates.view(-1, 4)
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]

        if energy_fn is None:
            energy_fn = self._compute_metric_from_query_and_points

        # 获取每个3D grid cell的中心坐标
        # coords_candidates shape: [1, N_total, 4], reshape成 [NR, NC, Rot, Scale, 4]
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        # 取每个3D cell的中心（对Scale取第一个值）
        cell_centers_3d = coords_reshaped[:, :, :, 0, :3]  # [NR, NC, Rot, 3] -> (nr, nc, rot)

        # 统计容器
        all_ranks = []
        all_dist_errors_2d = []  # 2D位置误差
        all_rot_errors = []  # 旋转误差
        all_rot_ranks = []  # 固定位置时的旋转排名
        processed = 0

        # 用于保存的容器
        pred_pdf_3d_list = []  # 收集3D概率分布
        q_label_3d_list = []  # 收集GT的3D标签
        coords_gt_list = []  # 收集GT坐标

        with torch.no_grad():
            for batch in test_loader:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.device)
                coords_gt = batch[1].to(self.device)  # [B, 4]
                batch_size = imgs.shape[0]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

                # 计算能量
                possibilities = energy_fn(
                    feats_vis,
                    coords_candidates_flat,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                possibilities_reshaped = possibilities.reshape(batch_size, *n_coarse)

                # 只边缘化Scale维度
                logits_3d = possibilities_reshaped.sum(dim=-1)  # [B, NR, NC, Rot]

                # 获取预测的3D索引
                logits_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]
                pred_indices = logits_flat.argmax(dim=-1)  # [B]

                # 解析3D索引
                pred_rot = pred_indices % n_coarse_3d[2]
                pred_nc = (pred_indices // n_coarse_3d[2]) % n_coarse_3d[1]
                pred_nr = pred_indices // (n_coarse_3d[1] * n_coarse_3d[2])

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 计算排名
                sorted_indices = logits_flat.argsort(dim=-1, descending=True)
                for i in range(batch_size):
                    rank = (sorted_indices[i] == gt_flat_idx[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_ranks.append(rank)

                # 计算2D距离误差
                pred_centers = cell_centers_3d[pred_nr, pred_nc, pred_rot]  # [B, 3]
                gt_coords_2d = coords_gt[:, :2]  # [B, 2]
                dist_errors_2d = torch.norm(pred_centers[:, :2] - gt_coords_2d, dim=-1, p=2)
                all_dist_errors_2d.extend(dist_errors_2d.cpu().numpy().tolist())

                # 计算旋转误差（弧度）
                pred_rot_val = pred_centers[:, 2]  # [B]
                gt_rot_val = coords_gt[:, 2]  # [B]
                # 处理角度环绕 [-pi, pi]
                rot_diff = pred_rot_val - gt_rot_val
                rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))  # 归一化到[-pi, pi]
                rot_errors = torch.abs(rot_diff)
                all_rot_errors.extend(rot_errors.cpu().numpy().tolist())

                # 计算固定位置时的旋转排名（只在正确的nr, nc位置上比较rot）
                for i in range(batch_size):
                    # 提取GT位置(nr, nc)处所有旋转的logits
                    logits_at_gt_pos = logits_3d[i, gt_nr[i], gt_nc[i], :]  # [n_rot]
                    # 排序找到GT旋转的排名
                    sorted_rot_indices = logits_at_gt_pos.argsort(descending=True)
                    rot_rank = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks.append(rot_rank)

                # 收集用于保存的数据
                # if save_pred_pdf:
                # 将logits转换为概率分布
                pred_pdf_3d = logits_flat  # [B, NR*NC*Rot]
                pred_pdf_3d_list.append(pred_pdf_3d.cpu())
                q_label_3d_list.append(gt_flat_idx.cpu())
                coords_gt_list.append(coords_gt.cpu())

                processed += batch_size

            # 保存预测结果到checkpoint文件夹
            if save_pred_pdf and len(pred_pdf_3d_list) > 0:
                # 拼接所有批次的数据
                pred_pdf_3d_all = torch.cat(pred_pdf_3d_list, dim=0)  # [N, NR*NC*Rot]
                q_label_3d_all = torch.cat(q_label_3d_list, dim=0)  # [N]
                coords_gt_all = torch.cat(coords_gt_list, dim=0)  # [N, 4]
                data_type = 'train' if use_train_uav else 'test'
                self._save_pred_pdf_3d(
                    pred_pdf_3d_all=pred_pdf_3d_all,
                    q_label_3d_all=q_label_3d_all,
                    coords_gt_all=coords_gt_all,
                    n_coarse_3d=n_coarse_3d,
                    cell_centers_3d=cell_centers_3d,
                    temperature=temperature,
                    data_type=data_type,
                )

        # 计算指标
        all_ranks = np.array(all_ranks)
        all_dist_errors_2d = np.array(all_dist_errors_2d)
        all_rot_errors = np.array(all_rot_errors)
        all_rot_errors_deg = np.rad2deg(all_rot_errors)  # 转为度
        all_rot_ranks = np.array(all_rot_ranks)
        n_total_cells = n_coarse_3d[0] * n_coarse_3d[1] * n_coarse_3d[2]

        results = {
            'n_samples': len(all_ranks),
            'n_total_cells': n_total_cells,
            'n_coarse_3d': n_coarse_3d,
            'top1_acc': (all_ranks == 1).mean() * 100,
            'top8_acc': (all_ranks <= 8).mean() * 100,
            'top27_acc': (all_ranks <= 27).mean() * 100,
            'top64_acc': (all_ranks <= 64).mean() * 100,
            'top256_acc': (all_ranks <= 256).mean() * 100,
            'top512_acc': (all_ranks <= 512).mean() * 100,
            'mean_rank': all_ranks.mean(),
            'median_rank': np.median(all_ranks),
            # 2D位置误差
            'mean_dist_error_2d': all_dist_errors_2d.mean(),
            'median_dist_error_2d': np.median(all_dist_errors_2d),
            # 旋转误差（度）
            'mean_rot_error_deg': all_rot_errors_deg.mean(),
            'median_rot_error_deg': np.median(all_rot_errors_deg),
            # 固定位置时的旋转准确率
            'rot_only_top1_acc': (all_rot_ranks == 1).mean() * 100,
            'rot_only_top2_acc': (all_rot_ranks <= 2).mean() * 100,
            'rot_only_top3_acc': (all_rot_ranks <= 3).mean() * 100,
            'rot_only_mean_rank': all_rot_ranks.mean(),
            'rot_only_median_rank': np.median(all_rot_ranks),
        }

        # 打印结果
        print(f"\n{'=' * 60}")
        print(f"3D分类测试结果 (共{n_total_cells}个cell: {n_coarse_3d[0]}x{n_coarse_3d[1]}x{n_coarse_3d[2]})")
        print(f"{'=' * 60}")
        print(f"Top-1   准确率: {results['top1_acc']:.2f}%")
        print(f"Top-8   准确率: {results['top8_acc']:.2f}%")
        print(f"Top-27  准确率: {results['top27_acc']:.2f}%")
        print(f"Top-64  准确率: {results['top64_acc']:.2f}%")
        print(f"Top-256 准确率: {results['top256_acc']:.2f}%")
        print(f"Top-512 准确率: {results['top512_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"2D位置误差统计:")
        print(f"  平均误差: {results['mean_dist_error_2d']:.4f}")
        print(f"  中位数误差: {results['median_dist_error_2d']:.4f}")
        print(f"{'=' * 60}")
        print(f"旋转误差统计 (度):")
        print(f"  平均误差: {results['mean_rot_error_deg']:.2f}°")
        print(f"  中位数误差: {results['median_rot_error_deg']:.2f}°")
        print(f"{'=' * 60}")
        print(f"固定位置(NR,NC)时的旋转准确率 (共{n_coarse_3d[2]}个rot):")
        print(f"  Top-1 准确率: {results['rot_only_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_only_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_only_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_only_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_only_median_rank']:.2f}")
        print(f"{'=' * 60}\n")

        return results

    def _get_feats_fm_INGP(self, coords, coord_mode='raw'):
        """
        [通用原子操作] 输入任意格式坐标，输出归一化特征

        修改为与 stage3_project_integrateRot_classify.py 一致的实现方式

        Args:
            coords: 坐标张量，形状可以是 [N, 4] 或 [N, 5] 或 [N, 6]
            coord_mode: 输入坐标的类型
                - 'raw':     [N, 4] 物理坐标 [r, c, theta, s] (Dataset输出)
                - 'linear':  [N, 4] 线性坐标 [r_n, c_n, t_lin, s_n] (Sampler输出)
                - 'net_5d':  [N, 5] 网络坐标 [r_n, c_n, cos, sin, s_n] (直接透传)
                - 'net_6d':  [N, 6] 网络坐标+线性角度 (Processor输出的中间态)

        Returns:
            feat_norm: [N, C] L2归一化后的特征
        """

        # =========================================================
        # 1. 统一转换层：转换为 6D 格式 (与 stage3_project_integrateRot_classify 一致)
        # =========================================================
        if coord_mode == 'raw':
            # 直接使用 raw_to_net 并追加 linear_rot
            coords_6d = self.coord_normer.raw_to_net(coords, append_linear_rot=True)
        elif coord_mode == 'linear':
            coords_net = self.coord_normer.linear_to_net(coords)
            theta_lin = coords[..., 2:3]  # linear 空间的 theta 已经是归一化的
            coords_6d = torch.cat([coords_net, theta_lin], dim=-1)
        elif coord_mode == 'net_5d':
            theta_lin = torch.atan2(coords[..., 3:4], coords[..., 2:3]) / torch.pi
            coords_6d = torch.cat([coords, theta_lin], dim=-1)
        elif coord_mode == 'net_6d':
            coords_6d = coords
        else:
            raise ValueError(f"Unknown coord_mode: {coord_mode}")

        # =========================================================
        # 2. 构造子模块输入 (与 stage3_project_integrateRot_classify 一致)
        # =========================================================
        grid_input = torch.cat([coords_6d[..., :2], coords_6d[..., -1:]], dim=-1)  # [nr, nc, theta_lin]
        mlp_input = coords_6d[..., :5]  # [nr, nc, cos, sin, log_s]

        # =========================================================
        # 3. 前向传播
        # =========================================================

        # (A) Query HashGrid (Backbone)
        feat_raw = self._get_feats_fm_grid(grid_input)

        # (B) Positional Encoding (用于 MLP 条件)
        pos_enc = self.pos_encoder_grid(mlp_input)

        # (C) Tiny MLP (Decoder)
        feat_out = self.grid_mlp(feat_raw, pos_enc)

        # 4. L2 Normalize (Metric Learning 标准操作)
        return torch.nn.functional.normalize(feat_out, dim=-1, p=2)

    def _sample_around_candidates(self, coords_centers, grid_dims, space_size=None):
        """
        逻辑2：在给定的中心点周围进行网格采样。

        Args:
            coords_centers: [B, N, 4] 中心点坐标
            grid_dims: tuple (nr, nc, rot, scale) 网格维度
            space_size: list/tuple (Optional) 物理空间大小

        Returns:
            coords_new: [B, N * Points_Per_Center, 4] 采样后的新坐标点
        """
        B, N, _ = coords_centers.shape

        # 1. 展平以适配 Sampler 接口 [B*N, 4]
        centers_flat = coords_centers.reshape(-1, 4)

        # 2. 调用 Sampler
        # 输出: [B*N, Points_Per_Center, 4]
        coords_sampled = self.subspace_sampler.sample_grid_around_coords_gpu(
            center_coords=centers_flat,
            grid_dims=grid_dims,
            sample_space_size=space_size,
            device=self.device
        )

        points_per_center = coords_sampled.shape[1]

        # 3. 恢复 Batch 维度
        # [B*N, Points, 4] -> [B, N*Points, 4]
        # 这里的 reshape 实际上把每个中心点生成的点铺平了
        coords_new = coords_sampled.reshape(B, N * points_per_center, 4)

        return coords_new

    def _sample_indices(self, scores, n_samples, strategy='sampling', temperature=1.0, alpha=0.8):
        """
        辅助函数：根据分数选择索引
        :param scores: [B, N] 原始分数或概率
        :param n_samples: 需要选择的数量
        :param strategy: 'topk' (原有逻辑), 'sampling' (纯概率采样), 'hybrid' (混合策略)
        :param temperature: 温度系数，越小越趋近于argmax，越大越均匀
        :param alpha: 混合策略中，保留 TopK 的比例 (0.0 - 1.0)
        :return: sampled_indices [B, n_samples], sampled_scores [B, n_samples]
        """
        B, N = scores.shape
        n_samples = min(n_samples, N)

        # 预处理：如果是logits则softmax，如果是概率则归一化
        # 这里假设输入是正值概率或相似度，先进行归一化
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        # 应用温度系数 (注意：如果原本已经是概率，需要先取log再除温度再softmax，或者直接幂次)
        # 这里采用幂次调节法调节由于温度带来的分布平滑度
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if strategy == 'topk':
            return torch.topk(probs, k=n_samples, dim=-1, largest=True)

        elif strategy == 'sampling':
            # 纯概率采样 (无放回)
            indices = torch.multinomial(probs, num_samples=n_samples, replacement=False)
            # 为了后续处理方便，通常还是把采样出来的点按概率从大到小排个序
            batch_indices = torch.arange(B, device=scores.device).unsqueeze(-1)
            selected_probs = scores[batch_indices, indices]
            sorted_idx = torch.argsort(selected_probs, dim=-1, descending=True)
            final_indices = torch.gather(indices, 1, sorted_idx)
            final_scores = torch.gather(selected_probs, 1, sorted_idx)
            return final_scores, final_indices

        elif strategy == 'hybrid':
            # 混合策略：Top (alpha * N) 确定性保留 + 剩余部分随机采样
            # 这种方法既能保住最可能的点（保Top1），又能探索长尾（保Top64）
            n_deterministic = int(n_samples * alpha)
            n_stochastic = n_samples - n_deterministic

            # 1. 先取 Top K
            top_vals, top_inds = torch.topk(probs, k=n_deterministic, dim=-1, largest=True)

            # 2. 将 Top K 的概率置 0，防止重复采样
            probs_clone = probs.clone()
            probs_clone.scatter_(1, top_inds, 0)
            probs_clone = probs_clone / (probs_clone.sum(dim=-1, keepdim=True) + 1e-6)  # 重新归一化

            # 3. 对剩余部分采样
            if n_stochastic > 0:
                stochastic_inds = torch.multinomial(probs_clone, num_samples=n_stochastic, replacement=False)
                final_indices = torch.cat([top_inds, stochastic_inds], dim=1)
            else:
                final_indices = top_inds

            # 同样按原始分数排序输出
            batch_indices = torch.arange(B, device=scores.device).unsqueeze(-1)
            final_raw_scores = scores[batch_indices, final_indices]
            sorted_idx = torch.argsort(final_raw_scores, dim=-1, descending=True)

            final_indices = torch.gather(final_indices, 1, sorted_idx)
            final_scores = torch.gather(final_raw_scores, 1, sorted_idx)

            return final_scores, final_indices


    def _test_3d_fine_accuracy_v1(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=True,
            topN_refine=True,
            filter_topN_by_gt=False,
            enable_filter=True,
            chunk_size=2048,
            use_ingp_similarity=False,
            ingp_metric='sim',
    ):
        def _evaluate_and_report(coords_pred, coords_gt_source, tag="Eval",
                                 dist_lambda=1.0, rot_th=10.0, scale_th=None):
            """
            通用评估函数：计算并打印 Top-N 准确率
            Args:
                coords_pred:      [N, K, 4] 或 [N, 4] 预测坐标
                coords_gt_source: [N, 4] GT坐标 tensor 或 list of tensors
                tag:              str 打印标题前缀
                dist_lambda:      float 距离阈值缩放系数
                rot_th:           float 角度阈值 (度)
                scale_th:         float 或 None. 尺度阈值. 如果为 None 则忽略尺度评估.
            """
            from scripts.analysis.util_stage3_analyze_pred3d import (
                compute_topN_acc_given_threshold,
                print_topN_acc_results,
            )

            # 1. 处理 GT 数据 (支持传入 list 或 tensor)
            if isinstance(coords_gt_source, list):
                if len(coords_gt_source) > 0:
                    coords_gt_all = torch.cat(coords_gt_source, dim=0).to(coords_pred.device)
                else:
                    # 防止空列表报错，虽然理论上不该发生
                    print(f"[{tag}] Warning: GT list is empty!")
                    return None
            else:
                coords_gt_all = coords_gt_source.to(coords_pred.device)

            # 2. 配置阈值
            thresh_cfg = {
                'norm_dist': self.sat_dataset.halfimg_radius_nrc * dist_lambda,
                'rot': rot_th,
                'scale': scale_th  # [修改] 使用传入的 scale 参数
            }

            # 3. 计算指标
            print(f"\n>>> [{tag}] 评估结果:")
            target_k_values = [1, 5, 10, 16, 32, 64]

            # 确保 pred 和 gt 长度对齐
            min_len = min(len(coords_pred), len(coords_gt_all))
            if min_len == 0:
                print("No data to evaluate.")
                return None

            coords_pred = coords_pred[:min_len]
            coords_gt_all = coords_gt_all[:min_len]

            acc_metrics, err_stats = compute_topN_acc_given_threshold(
                coords_pred=coords_pred,
                coords_gt=coords_gt_all,
                dist_th=thresh_cfg['norm_dist'],
                rot_th_deg=thresh_cfg['rot'],
                scale_th=thresh_cfg['scale'],
                k_values=target_k_values
            )

            # 4. 打印
            print_topN_acc_results(acc_metrics, err_stats, thresh_cfg)

            return acc_metrics

        # ==============================================================================
        # [新增] 内部辅助函数：混合采样逻辑封装
        # ==============================================================================
        def _execute_hybrid_sampling(coords_candidates, scores, target_topN,
                                     alpha=0.5, threshold=0.0, tag="Level-X"):
            """
            通用混合采样函数：结合 Top-K (利用) 和 Multinomial (探索)
            Args:
                coords_candidates: [B, N, 4] 当前阶段所有的候选点坐标
                scores:            [B, N]    对应的分数 (不需要归一化，函数内处理)
                target_topN:       int       希望保留的点数
                alpha:             float     混合比例 (0.5 表示一半TopK，一半随机)
                threshold:         float     绝对阈值过滤
            Returns:
                sampled_coords:    [B, target_topN, 4]
            """
            B, N = scores.shape
            K = min(target_topN, N)  # 防止候选点不够

            # 1. 预处理：阈值过滤 & 归一化
            # 为了数值稳定性，先过一遍阈值
            if threshold > 0:
                mask = scores > threshold
                # 这是一个软处理：不直接丢弃，而是把低分置为0，保证维度不变
                scores = scores * mask.float()

            # 转换为概率分布 (Sum=1)
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

            # 2. 计算配额
            n_det = int(K * alpha)  # 确定性数量
            n_sto = K - n_det  # 随机性数量

            # 3. 确定性采样 (Top-K)
            # values, indices
            _, idx_det = torch.topk(probs, k=n_det, dim=-1)

            # 4. 随机性采样 (Multinomial)
            if n_sto > 0:
                # 克隆概率，将已被 Top-K 选中的位置置零，避免重复
                probs_remain = probs.clone()
                # scatter_ 的 dim=1
                probs_remain.scatter_(1, idx_det, 0.0)

                # 重新归一化剩余部分 (防止概率和为0导致报错，加个极小值兜底)
                probs_remain = probs_remain / (probs_remain.sum(dim=-1, keepdim=True) + 1e-10)

                # 带放回采样 (replacement=True)，防止有效点少于 n_sto 时报错
                # 如果点非常少，重复也没关系，反正都在那个峰值附近
                idx_sto = torch.multinomial(probs_remain, num_samples=n_sto, replacement=True)

                # 5. 合并索引
                final_indices = torch.cat([idx_det, idx_sto], dim=1)
            else:
                final_indices = idx_det

            # 6. Gather 坐标
            # [B, K, 1] -> [B, K, 4]
            idx_exp = final_indices.unsqueeze(-1).expand(-1, -1, 4)
            sampled_coords = torch.gather(coords_candidates, 1, idx_exp)

            return sampled_coords

        # ---------- 概率计算封装 ----------
        def _compute_probs(coords_batch, mode="ingp"):
            """
            统一的候选概率计算:
            mode ∈ {'ingp', 'projector', 'product'}
            product = P_ingp * P_projector 后再归一化
            """
            mode = mode.lower()
            # INGP 距离 → 相似度 (未归一化)
            if mode in ("ingp", "product"):
                dist_ingp = self._compute_metric_from_ingp(
                    feats_vis_all,
                    coords_batch,
                    coord_space='raw',
                    chunk_size=chunk_size,
                    metric='dist',
                )
                prob_ingp = (2 - dist_ingp).clamp(min=0) / 2
            else:
                prob_ingp = None

            # Projector 概率 (已 softmax 归一化)
            if mode in ("projector", "product"):
                dist_proj = self._compute_metric_from_query_and_points(
                    feats_vis_all,
                    coords_batch,
                    metric='dist',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, M]
                prob_proj = (2 - dist_proj).clamp(min=0) / 2
            else:
                prob_proj = None

            if mode == "ingp":
                prob = prob_ingp
            elif mode == "projector":
                prob = prob_proj
            elif mode == "product":
                # 若 projector 结果为概率，乘积后需重新归一化
                prob = prob_ingp * prob_proj
            else:
                raise ValueError(f"Unknown prob mode: {mode}")

            # 统一归一化，避免数值范围差异
            prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-8)
            return prob

        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")
        from scripts.analysis.util_stage3_analyze_pred3d import (
            compute_top_k_accuracy,
            print_accuracy_results,
            convert_3d_to_2d_predictions,
        )

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 确定测试样本数
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, _ = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False, rand_offset=False,
        )
        coords_candidates_flat = coords_candidates.view(-1, 4)
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]

        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        cell_centers_3d = coords_reshaped[:, :, :, 0, :3]  # [NR, NC, Rot, 3]

        # 统计容器
        processed = 0

        # 用于保存的容器
        pred_pdf_3d_list = []  # 收集3D概率分布
        q_label_3d_list = []  # 收集GT的3D标签
        coords_gt_list = []  # 收集GT坐标
        feats_vis_list = []

        with torch.no_grad():
            for batch in test_loader:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.device)
                coords_gt = batch[1].to(self.device)  # [B, 4]
                batch_size = imgs.shape[0]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

                # 通过统一概率函数计算 (与 _test_3d_classification_accuracy 一致)
                possibilities = self._compute_metric_from_query_and_points(
                    feats_vis,
                    coords_candidates_flat,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                possibilities_reshaped = possibilities.reshape(batch_size, *n_coarse)

                # 只边缘化Scale维度
                logits_3d = possibilities_reshaped.sum(dim=-1)  # [B, NR, NC, Rot]
                logits_3d = logits_3d/logits_3d.sum()

                # 获取预测的3D索引
                prob_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 收集用于保存的数据
                # if save_pred_pdf:
                # probabilities already computed
                pred_pdf_3d = prob_flat  # [B, NR*NC*Rot]
                pred_pdf_3d_list.append(pred_pdf_3d.cpu())
                q_label_3d_list.append(gt_flat_idx.cpu())
                coords_gt_list.append(coords_gt.cpu())
                feats_vis_list.append(feats_vis.cpu())  # 保存视觉特征用于topN精细检索

                processed += batch_size

            # 拼接所有批次的数据
            pred_pdf_3d_all = torch.cat(pred_pdf_3d_list, dim=0)  # [N, NR*NC*Rot]
            q_label_3d_all = torch.cat(q_label_3d_list, dim=0)  # [N]
            coords_gt_all = torch.cat(coords_gt_list, dim=0).to(self.device)  # [N, 4] 移到GPU
            feats_vis_all = torch.cat(feats_vis_list, dim=0).to(self.device)  # [N, C] 拼接视觉特征

            if save_pred_pdf:
                data_type = 'train' if use_train_uav else 'test'
                self._save_pred_pdf_3d(
                    pred_pdf_3d_all=pred_pdf_3d_all,
                    q_label_3d_all=q_label_3d_all,
                    coords_gt_all=coords_gt_all,
                    n_coarse_3d=n_coarse_3d,
                    cell_centers_3d=cell_centers_3d,
                    temperature=temperature,
                    data_type=data_type,
                    tag="prefilter",
                )

            # 显示滤波前分类精度
            single_frame_results = compute_top_k_accuracy(
                pred_pdf_3d_all.cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(single_frame_results, title="loc_res_3d_single")

            # 对pred_3d_pdf进行直方图滤波（可选）
            pred_pdf_3d_shaped = pred_pdf_3d_all.reshape(-1, *n_coarse_3d)
            if enable_filter:
                raw_diff = torch.diff(coords_gt_all[:, 2])
                diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi
                from util_core_histogram_filter_3d import HistogramFilter3D
                histfilter = HistogramFilter3D(H=n_coarse_3d[0], W=n_coarse_3d[1], O=n_coarse_3d[2],
                                               device=pred_pdf_3d_shaped.device)
                pred_pdf_3d_hist = pred_pdf_3d_shaped.permute(0, 3, 1, 2)
                preds_filtered = []
                histfilter.belief = histfilter.belief * pred_pdf_3d_hist[0:1]
                preds_filtered.append(histfilter.belief.clone())

                for i in range(diff_rot_rad.shape[0]):
                    if i == diff_rot_rad.shape[0]:
                        break
                    histfilter.predict(move_rot=diff_rot_rad[i], noise_std_rot=30 / 180 * torch.pi, direction_aware=False,
                                       noise_std_xy=0.65, xy_k_size=5)
                    histfilter.update(pred_pdf_3d_hist[i + 1:i + 2], alpha=0.25)
                    preds_filtered.append(histfilter.belief.clone())
                preds_filtered = torch.cat(preds_filtered)
                preds_filtered = preds_filtered.permute(0, 2, 3, 1)

                if save_pred_pdf:
                    data_type = 'train' if use_train_uav else 'test'
                    self._save_pred_pdf_3d(
                        pred_pdf_3d_all=preds_filtered,
                        q_label_3d_all=q_label_3d_all,
                        coords_gt_all=coords_gt_all,
                        n_coarse_3d=n_coarse_3d,
                        cell_centers_3d=cell_centers_3d,
                        temperature=temperature,
                        data_type=data_type,
                        tag="postfilter",
                    )

                # 显示滤波后分类精度
                filtered_frame_results = compute_top_k_accuracy(
                    preds_filtered.reshape(preds_filtered.shape[0], -1).cpu().numpy(),
                    q_label_3d_all,
                    k_values=[1, 8, 27, 64, 128, 256, 512],
                    dim_order='HWO'
                )
                print_accuracy_results(filtered_frame_results, title="loc_res_3d_filtered")
            else:
                preds_filtered = pred_pdf_3d_shaped

            # 使用平滑滤波（可选）
            enable_smoothing = False
            smooth_sigma = 0.75 # 适当的平滑半径，太大可能会模糊掉尖锐的定位
            smooth_kernel = 3  # 3x3x3 的核
            if enable_smoothing:
                from trainer_depends.utils.util_refine_sampling import _get_gaussian_kernel_3d,_apply_gaussian_smoothing_3d
                preds_filtered = _apply_gaussian_smoothing_3d(preds_filtered, kernel_size=smooth_kernel, sigma=smooth_sigma)

            # ==================== 阶段 1: 进行粗筛选Level0 ====================
            # --- 1. 参数配置 (手工设定) ---
            topN_l0 = 1024  # 粗筛阶段保留的粒子总数
            prob_thresh = 0.  # 【阈值过滤】绝对概率阈值，低于此值的噪声直接剔除
            sample_mode = 'hybrid'  # 【采样策略】'hybrid'(推荐) 或 'multinomial'
            hybrid_alpha_l0 = 0.8  # 混合策略中保留确定性 Top-K 的比例 (0.5表示一半TopK，一半随机)

            # --- 2. 概率分布预处理 ---
            B = preds_filtered.shape[0]
            # [B, H, W, O] -> [B, N_total] (Flatten)
            probs_flat = preds_filtered.reshape(B, -1)

            # [Filter] 阈值过滤：将低置信度区域的概率置为 0
            if prob_thresh > 0:
                mask_valid = probs_flat > prob_thresh
                probs_flat = probs_flat * mask_valid.float()
            # [Normalize] 重新归一化，确保和为 1 (torch.multinomial 的要求)
            probs_sum = probs_flat.sum(dim=-1, keepdim=True) + 1e-8
            probs_norm = probs_flat / probs_sum

            # --- 3. 执行采样 (Hybrid Strategy) ---
            # 目标：获取 sampled_indices [B, topN_l0]
            if sample_mode == 'hybrid':
                # >>> 混合策略：既保住最强峰值，又探索次优峰值 <<<
                n_deterministic = int(topN_l0 * hybrid_alpha_l0)
                n_stochastic = topN_l0 - n_deterministic

                # A. 确定性部分 (Top-K): 稳住基本盘
                _, indices_top = torch.topk(probs_norm, k=n_deterministic, dim=-1)

                # B. 随机性部分 (Sampling): 探索长尾
                # 先把已经被 Top-K 选走的位置概率置 0，避免重复采样
                probs_remain = probs_norm.clone()
                probs_remain.scatter_(1, indices_top, 0)
                # 再次归一化剩余部分
                probs_remain = probs_remain / (probs_remain.sum(dim=-1, keepdim=True) + 1e-8)

                # 带放回采样 (Replacement=True 保证即使有效点不足也不会报错)
                indices_sto = torch.multinomial(probs_remain, num_samples=n_stochastic, replacement=True)

                # 合并索引
                sampled_indices = torch.cat([indices_top, indices_sto], dim=1)
            else:
                # >>> 纯蒙特卡洛采样 <<<
                # 完全基于概率密度随机游走
                sampled_indices = torch.multinomial(probs_norm, num_samples=topN_l0, replacement=True)

            # 确保索引与后续查表张量在同一设备
            sampled_indices = sampled_indices.to(coords_candidates_flat.device)

            # --- 4. 索引回溯 (Indices -> 3D Coordinates) ---
            # 我们拿到了 flat index，需要转换回 (NR, NC, Rot) 索引，再映射到真实物理坐标
            H, W, O = n_coarse_3d  # [NR, NC, Rot]

            # 解算 3D 索引
            idx_rot = sampled_indices % O
            idx_nc = (sampled_indices // O) % W
            idx_nr = (sampled_indices // (W * O))

            # 构建 4D 索引用于查表 (NR, NC, Rot, Scale)
            # 注意：Level 0 边缘化了 Scale，我们这里默认取 Scale=0 的位置作为中心点
            # Level 1 的 refine 会在这个中心点周围覆盖所有 Scale 的搜索范围
            idx_scale = torch.zeros_like(idx_nr)

            # 堆叠成 [B, topN_l0, 4] 的多维索引
            indices_4d = torch.stack([idx_nr, idx_nc, idx_rot, idx_scale], dim=-1)

            # 计算在 coords_candidates (包含 Scale 维度) 中的 Flat Index
            n_scale = n_coarse[3]
            # Strides calculation: [NC*Rot*Scale, Rot*Scale, Scale, 1]
            _dev = sampled_indices.device
            strides = torch.tensor([
                n_coarse[1] * n_coarse[2] * n_scale,
                n_coarse[2] * n_scale,
                n_scale,
                1
            ], device=_dev)

            # [B, topN_l0] -> 最终的全局 Flat Index
            flat_indices_4d = (indices_4d * strides).sum(dim=-1)

            # --- 5. 获取候选坐标 (Gather Coordinates) ---
            # 从预计算好的 candidate 库中取出真实的 (x, y, z, s) 坐标
            # coords_candidates_flat: [Total_Points, 4]
            # 这一步后，coords_topN_l0 就是准备喂给 Level 1 的初始粒子群了
            coords_topN_l0_lefted = coords_candidates_flat[flat_indices_4d]  # [B, topN_l0, 4]

            # ==================== 阶段 2: 空间再采样 Level1  ====================
            # 以第level0阶段得到的TopN个中心点周围生成更密集的点
            resample_dims = (4, 4, 2, 1)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_topN_l0_lefted,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes'] / 0.75,
            )  # -> [B, topN * 16, 4]
            # 评估新点 (返回全部新点及其能量)
            coords2eval_l1 = torch.cat([coords_resampled, coords_topN_l0_lefted], dim=1)
            # prob_mode = "ingp"  # 可选: ingp / projector / product
            prob_l1 = _compute_probs(coords2eval_l1, mode= "product")
            # 截取重采样结果
            topN_l1_lefted = 128
            coords_resorted_l1 = _execute_hybrid_sampling(
                coords_candidates=coords2eval_l1,
                scores=prob_l1,
                target_topN=topN_l1_lefted,
                alpha=1.0,  # Level 1 还可以保持 50% 的探索率
                threshold=0.00,  # 过滤极低分
                tag="Level-1"
            )

            # ==================== 阶段 3: 空间再采样 (Level 2) ====================
            # ... (采样代码不变) ...
            resample_dims = (5, 5, 2, 1)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_resorted_l1,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes'] / 0.85,
            )  # -> [B, topN * 16, 4]
            # ... (计算 prob_l2) ...
            coords2eval_l2 = torch.cat([coords_resampled, coords_resorted_l1], dim=1)
            prob_l2 = _compute_probs(coords2eval_l2, mode="ingp")
            # 最后一层通常建议主要看 Top-K，但保留一点点随机性防止只拿到一个点
            topN_l2_lefted = 128
            coords_resorted_l2 = _execute_hybrid_sampling(
                coords_candidates=coords2eval_l2,
                scores=prob_l2,
                target_topN=topN_l2_lefted,
                alpha=1.0,  # Level 2 接近收敛，可以提高 Top-K 比例 (70% 确定性)
                threshold=0.000,
                tag="Level-2"
            )

            # ==================== 阶段 4: 空间再采样 (Level 3) ====================
            # 更细层级重采样
            resample_dims = (4, 4, 2, 5)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_resorted_l2,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes'],
            )  # -> [B, topN * 16, 4]
            coords2eval_l3 = torch.cat([coords_resampled, coords_resorted_l2], dim=1)
            # 计算概率
            prob_l3 = _compute_probs(coords2eval_l3, mode= "ingp")
            #截取重采样结果
            topN_l3_lefted = 128
            coords_resorted_l3 = _execute_hybrid_sampling(
                coords_candidates=coords2eval_l3,
                scores=prob_l3,
                target_topN=topN_l3_lefted,
                alpha=1.,  # Level 3 90% 相信最高分，只留 10% 给周围
                threshold=0.000,
                tag="Level-3"
            )
        _evaluate_and_report(coords_resorted_l3,coords_gt_all,dist_lambda=1.1,rot_th=11,scale_th=None)


    def _test_3d_fine_accuracy(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=True,
            topN_refine=True,
            filter_topN_by_gt=False,
            enable_filter=True,
            chunk_size=2048,
            use_ingp_similarity=False,
            ingp_metric='sim',
    ):
    #todo: 基于概率分布进行细粒度采样
        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")
        from scripts.analysis.util_stage3_analyze_pred3d import (
            compute_top_k_accuracy,
            print_accuracy_results,
            convert_3d_to_2d_predictions,
        )

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 确定测试样本数
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, _ = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False, rand_offset=False,
        )
        coords_candidates_flat = coords_candidates.view(-1, 4)
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]

        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        cell_centers_3d = coords_reshaped[:, :, :, 0, :3]  # [NR, NC, Rot, 3]

        # 统计容器
        processed = 0

        # 用于保存的容器
        pred_pdf_3d_list = []  # 收集3D概率分布
        q_label_3d_list = []  # 收集GT的3D标签
        coords_gt_list = []  # 收集GT坐标
        feats_vis_list = []

        with torch.no_grad():
            for batch in test_loader:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.device)
                coords_gt = batch[1].to(self.device)  # [B, 4]
                batch_size = imgs.shape[0]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

                # 通过统一概率函数计算 (与 _test_3d_classification_accuracy 一致)
                possibilities = self._compute_metric_from_query_and_points(
                    feats_vis,
                    coords_candidates_flat,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                possibilities_reshaped = possibilities.reshape(batch_size, *n_coarse)

                # 只边缘化Scale维度
                logits_3d = possibilities_reshaped.sum(dim=-1)  # [B, NR, NC, Rot]
                logits_3d = logits_3d/logits_3d.sum()

                # 获取预测的3D索引
                prob_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 收集用于保存的数据
                # if save_pred_pdf:
                # probabilities already computed
                pred_pdf_3d = prob_flat  # [B, NR*NC*Rot]
                pred_pdf_3d_list.append(pred_pdf_3d.cpu())
                q_label_3d_list.append(gt_flat_idx.cpu())
                coords_gt_list.append(coords_gt.cpu())
                feats_vis_list.append(feats_vis.cpu())  # 保存视觉特征用于topN精细检索

                processed += batch_size

            # 拼接所有批次的数据
            pred_pdf_3d_all = torch.cat(pred_pdf_3d_list, dim=0)  # [N, NR*NC*Rot]
            q_label_3d_all = torch.cat(q_label_3d_list, dim=0)  # [N]
            coords_gt_all = torch.cat(coords_gt_list, dim=0).to(self.device)  # [N, 4] 移到GPU
            feats_vis_all = torch.cat(feats_vis_list, dim=0).to(self.device)  # [N, C] 拼接视觉特征

            if save_pred_pdf:
                data_type = 'train' if use_train_uav else 'test'
                self._save_pred_pdf_3d(
                    pred_pdf_3d_all=pred_pdf_3d_all,
                    q_label_3d_all=q_label_3d_all,
                    coords_gt_all=coords_gt_all,
                    n_coarse_3d=n_coarse_3d,
                    cell_centers_3d=cell_centers_3d,
                    temperature=temperature,
                    data_type=data_type,
                    tag="prefilter",
                )

            # 显示滤波前分类精度
            single_frame_results = compute_top_k_accuracy(
                pred_pdf_3d_all.cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(single_frame_results, title="loc_res_3d_single")

            # 对pred_3d_pdf进行直方图滤波（可选）
            pred_pdf_3d_shaped = pred_pdf_3d_all.reshape(-1, *n_coarse_3d)
            if enable_filter:
                raw_diff = torch.diff(coords_gt_all[:, 2])
                diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi
                from util_core_histogram_filter_3d import HistogramFilter3D
                histfilter = HistogramFilter3D(H=n_coarse_3d[0], W=n_coarse_3d[1], O=n_coarse_3d[2],
                                               device=pred_pdf_3d_shaped.device)
                pred_pdf_3d_hist = pred_pdf_3d_shaped.permute(0, 3, 1, 2)
                preds_filtered = []
                histfilter.belief = histfilter.belief * pred_pdf_3d_hist[0:1]
                preds_filtered.append(histfilter.belief.clone())

                for i in range(diff_rot_rad.shape[0]):
                    if i == diff_rot_rad.shape[0]:
                        break
                    histfilter.predict(move_rot=diff_rot_rad[i], noise_std_rot=30 / 180 * torch.pi, direction_aware=False,
                                       noise_std_xy=0.65, xy_k_size=5)
                    histfilter.update(pred_pdf_3d_hist[i + 1:i + 2], alpha=0.25)
                    preds_filtered.append(histfilter.belief.clone())
                preds_filtered = torch.cat(preds_filtered)
                preds_filtered = preds_filtered.permute(0, 2, 3, 1)

                if save_pred_pdf:
                    data_type = 'train' if use_train_uav else 'test'
                    self._save_pred_pdf_3d(
                        pred_pdf_3d_all=preds_filtered,
                        q_label_3d_all=q_label_3d_all,
                        coords_gt_all=coords_gt_all,
                        n_coarse_3d=n_coarse_3d,
                        cell_centers_3d=cell_centers_3d,
                        temperature=temperature,
                        data_type=data_type,
                        tag="postfilter",
                    )

                # 显示滤波后分类精度
                filtered_frame_results = compute_top_k_accuracy(
                    preds_filtered.reshape(preds_filtered.shape[0], -1).cpu().numpy(),
                    q_label_3d_all,
                    k_values=[1, 8, 27, 64, 128, 256, 512],
                    dim_order='HWO'
                )
                print_accuracy_results(filtered_frame_results, title="loc_res_3d_filtered")
            else:
                preds_filtered = pred_pdf_3d_shaped

            # from trainer_depends.utils.util_refine_sampling import get_batch_adaptive_indices
            # selected_indices, selected_mask, selected_probs, lengths=get_batch_adaptive_indices(scores_flat=preds_filtered.view(preds_filtered.shape[0],-1),top_p=0.1)

            # ==================== 阶段 1: 初始评估与筛选 ====================
            # 对初始的粗粒度候选点进行计算和排序，选出 TopK 用于再采样
            topN_l0_lefted = 1024  # 选择前N个候选位置（level0过滤后剩余数量）
            print(f"\n{'=' * 60}")
            print(f"使用3D直接采样策略 (topN={topN_l0_lefted})")
            print(f"{'=' * 60}")

            # preds_filtered shape: [N_samples, H, W, Rot]
            N_samples = preds_filtered.shape[0]
            H, W, Rot = preds_filtered.shape[1], preds_filtered.shape[2], preds_filtered.shape[3]

            # 1. Flatten 3D概率体为 [N_samples, H*W*Rot]
            pred_3d_flat = preds_filtered.reshape(N_samples, -1)  # [N_samples, H*W*Rot]

            # 2. 对每个样本选择topN个最高概率的3D位置
            sorted_indices_3d = torch.argsort(pred_3d_flat, dim=-1, descending=True)  # [N_samples, H*W*Rot]
            topN_indices_flat = sorted_indices_3d[:, :topN_l0_lefted]  # [N_samples, topN]

            # 3. 将flat索引转换为3D索引 (nr, nc, rot)
            # flat_idx = nr * (W * Rot) + nc * Rot + rot
            topN_rot = topN_indices_flat % Rot
            topN_nc = (topN_indices_flat // Rot) % W
            topN_nr = topN_indices_flat // (W * Rot)
            topN_3d_indices = torch.stack([topN_nr, topN_nc, topN_rot], dim=-1)  # [N_samples, topN, 3]

            # 4. 获取这些3D位置的中心坐标
            # 使用coarse_indices_to_multi获取坐标
            # 首先将3D multi索引转换为4D multi索引（添加scale维度，默认为0）
            topN_4d_indices = torch.cat([
                topN_3d_indices,  # [N_samples, topN, 3]
                torch.zeros(N_samples, topN_l0_lefted, 1, dtype=torch.long, device=topN_3d_indices.device)
                # scale=0
            ], dim=-1)  # [N_samples, topN, 4]

            # 预计算的候选坐标拉平后直接索引，避免双层for循环
            factors = torch.tensor(
                [
                    n_coarse[1] * n_coarse[2] * n_coarse[3],
                    n_coarse[2] * n_coarse[3],
                    n_coarse[3],
                    1
                ],
                device=topN_3d_indices.device,
                dtype=topN_3d_indices.dtype
            )
            flat_indices = (topN_4d_indices * factors).sum(dim=-1)  # [N_samples, topN]
            coords_flat_all = coords_candidates.view(-1, 4)  # [NR*NC*Rot*Scale, 4]
            coords_topN_l0_lefted = coords_flat_all[flat_indices]  # [N_samples, topN, 4]

            # ==================== 阶段 2: 空间再采样 ====================
            # 以第level0阶段得到的TopN个中心点周围生成更密集的点
            resample_dims = (4, 4, 2, 1)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_topN_l0_lefted,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes']/0.75,
            )  # -> [B, topN * 16, 4]

            # 3.1 评估新点 (返回全部新点及其能量)
            coords2eval_l1 = torch.cat([coords_resampled, coords_topN_l0_lefted], dim=1)
            use_ingp_similarity=True
            if use_ingp_similarity:
                ingp_metric = 'dist'
                metric_out_l1 = self._compute_metric_from_ingp(
                    feats_vis_all,
                    coords2eval_l1,
                    coord_space='raw',
                    chunk_size=chunk_size,
                    metric=ingp_metric,
                )
                if ingp_metric == 'sim':
                    prob_l1 = (metric_out_l1 + 1) / 2
                else:
                    prob_l1 = (2 - metric_out_l1).clamp(min=0) / 2
            else:
                prob_l1 = self._compute_metric_from_query_and_points(
                    feats_vis_all,
                    coords2eval_l1,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, M]
            topN_l1_lefted = 128
            top_prob_l1, top_idx_l1 = torch.topk(prob_l1, k=min(topN_l1_lefted, prob_l1.shape[1]), dim=-1, largest=True)
            top_idx_l1_exp = top_idx_l1.unsqueeze(-1).expand(-1, -1, 4)
            coords_resorted_l1 = torch.gather(coords2eval_l1, 1, top_idx_l1_exp)


            # ==================== 阶段 3: 空间再采样 ====================
            # 以第1个fine阶段得到的TopN个中心点周围生成更密集的点
            resample_dims = (5, 5, 2, 1)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_resorted_l1,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes']/0.85,
            )  # -> [B, topN * 16, 4]
            coords2eval_l2 = torch.cat([coords_resampled, coords_resorted_l1], dim=1)

            use_ingp_similarity=True
            if use_ingp_similarity:
                ingp_metric = 'dist'
                metric_out_l2 = self._compute_metric_from_ingp(
                    feats_vis_all,
                    coords2eval_l2,
                    coord_space='raw',
                    chunk_size=chunk_size,
                    metric=ingp_metric,
                )
                if ingp_metric == 'sim':
                    prob_l2 = (metric_out_l2 + 1) / 2
                else:
                    prob_l2 = (2 - metric_out_l2).clamp(min=0) / 2
            else:
                prob_l2 = self._compute_metric_from_query_and_points(
                    feats_vis_all,
                    coords2eval_l2,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, M]
            topN_l2_lefted = 128
            top_prob_l2, top_idx_l2 = torch.topk(prob_l2, k=min(topN_l2_lefted, prob_l2.shape[1]), dim=-1,largest=True)
            top_idx_l2_exp = top_idx_l2.unsqueeze(-1).expand(-1, -1, 4)
            coords_resorted_l2 = torch.gather(coords2eval_l2, 1, top_idx_l2_exp)

            # ==================== 阶段 4: 空间再采样 ====================
            # 以第1个fine阶段得到的TopN个中心点周围生成更密集的点
            resample_dims = (4, 4, 2, 5)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_resorted_l2,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes'],
            )  # -> [B, topN * 16, 4]
            coords2eval_l3 = torch.cat([coords_resampled, coords_resorted_l2], dim=1)

            use_ingp_similarity=True
            if use_ingp_similarity:
                ingp_metric = 'dist'
                metric_out_l3 = self._compute_metric_from_ingp(
                    feats_vis_all,
                    coords2eval_l3,
                    coord_space='raw',
                    chunk_size=chunk_size,
                    metric=ingp_metric,
                )
                if ingp_metric == 'sim':
                    prob_l3 = (metric_out_l3 + 1) / 2
                else:
                    prob_l3 = (2 - metric_out_l3).clamp(min=0) / 2
            else:
                prob_l3 = self._compute_metric_from_query_and_points(
                    feats_vis_all,
                    coords2eval_l3,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, M]
            topN_l3_lefted = 128
            top_prob_l3, top_idx_l3 = torch.topk(prob_l3, k=min(topN_l3_lefted, prob_l3.shape[1]), dim=-1,largest=True)
            top_idx_l3_exp = top_idx_l3.unsqueeze(-1).expand(-1, -1, 4)
            coords_resorted_l3 = torch.gather(coords2eval_l3, 1, top_idx_l3_exp)

            # ==================== 可选阶段 4: 反向传播优化 ====================
            opt = False
            if opt:
                coords_resorted_last_opted = self._opt_coords_topN(
                    coords_resorted_l3[:, :64],
                    feats_vis_all,
                    n_step=200,
                    lr=1e-5,
                )  # [N, K, 4] - 返回已经按优化后的loss排序的坐标
                # 将优化后的TopK替换回原coords_sorted
                coords_resorted_last = torch.cat([
                    coords_resorted_last_opted,  # [N, K, 4] 优化后的TopK
                    coords_resorted_l3  # [N, M-K, 4] 剩余的未优化候选
                ], dim=1)  # [N, M, 4]

            #评估&输出
            from scripts.analysis.util_stage3_analyze_pred3d import (
                compute_topN_acc_given_threshold,
                print_topN_acc_results,
            )
            dist_lambda = 1.0
            thresh_cfg = {
                'norm_dist': self.sat_dataset.halfimg_radius_nrc * dist_lambda,  # 例如 3米 或 3个Grid单位
                'rot': 15.0,  # 15度
                'scale':None # 不评估尺度
            }
            target_k_values = [1, 5, 10, 16, 32, 64]
            acc_metrics, err_stats = compute_topN_acc_given_threshold(
                coords_pred=coords_resorted_l3,
                coords_gt=torch.concatenate(coords_gt_list,dim=0),  # [B, 4] 你的GT坐标
                dist_th=thresh_cfg['norm_dist'],
                rot_th_deg=thresh_cfg['rot'],
                scale_th=thresh_cfg['scale'],
                k_values=target_k_values
            )
            print_topN_acc_results(acc_metrics, err_stats, thresh_cfg)


    def visualize_energy_of_coords(
            self,
            coords_samples,
            energys,
            coord_gt,
            coord_pred=None,
            save_path=None,
            mode='min'  # 新增参数: 'min' 或 'max'
    ):
        """
        生成极简版可交互3D可视化 (HTML)

        Args:
            coords_samples: [M, 4] 采样点坐标
            energys: [M] 采样点对应的值
            coord_gt: [4] GT坐标
            coord_pred: [4] (可选) 预测坐标
            save_path: (str) 保存路径
            mode: (str) 'min' 表示寻找能量最小值，'max' 表示寻找最大值 (当 coord_pred 为 None 时生效)
        """
        import plotly.graph_objects as go
        import numpy as np
        import os
        import torch
        import time

        # 1. 数据转换
        def to_numpy(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        coords_np = to_numpy(coords_samples)
        energys_np = to_numpy(energys)
        gt_np = to_numpy(coord_gt)

        # 2. 准备坐标
        X = coords_np[:, 1]  # NC
        Y = coords_np[:, 0]  # NR
        Z = np.rad2deg(coords_np[:, 2])  # Rot

        # 3. 创建 Figure
        fig = go.Figure()

        # --- Layer 1: 所有采样点 ---
        # 根据 mode 调整 colorscale 可能视觉效果更好，这里保持 Hot_r
        # 如果是 'max' 模式 (如相似度)，通常数值越大越好，可能更适合 'Viridis'
        colorscale = 'Hot_r' if mode == 'min' else 'Viridis'

        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(
                size=3,
                color=energys_np,
                colorscale=colorscale,
                opacity=0.6,
                colorbar=dict(title='Value', thickness=20)
            ),
            text=[f"Val: {e:.4f}" for e in energys_np],
            name='Samples'
        ))

        # --- Layer 2: GT ---
        fig.add_trace(go.Scatter3d(
            x=[gt_np[1]], y=[gt_np[0]], z=[np.rad2deg(gt_np[2])],
            mode='markers+text',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='GT',
            text=['GT'],
            textposition="top center"
        ))

        # --- Layer 3: Pred (或 Best Candidate) ---
        if coord_pred is not None:
            pred_np = to_numpy(coord_pred)
            label = "Pred"
        else:
            # === 这里增加了 mode 判断逻辑 ===
            if mode == 'max':
                best_idx = energys_np.argmax()
                label = "Max Value"
            else:
                best_idx = energys_np.argmin()
                label = "Min Energy"

            pred_np = coords_np[best_idx]

        fig.add_trace(go.Scatter3d(
            x=[pred_np[1]], y=[pred_np[0]], z=[np.rad2deg(pred_np[2])],
            mode='markers+text',
            marker=dict(size=6, color='red', symbol='x'),
            name=label,
            text=[f'{label}'],
            textposition="top center"
        ))

        # 4. 设置布局
        err_2d = np.linalg.norm(pred_np[:2] - gt_np[:2])

        fig.update_layout(
            title=f"3D Distribution ({label} Error 2D: {err_2d:.2f})",
            width=1000, height=800,
            scene=dict(
                xaxis_title='NC (Col)',
                yaxis_title='NR (Row)',
                zaxis_title='Rotation (deg)',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 5. 保存
        if save_path:
            final_path = save_path
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
        else:
            save_dir = self._get_dir2save() if hasattr(self, '_get_dir2save') else "vis_results"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = f"{time.time():.5f}".replace('.', '_')
            final_path = os.path.join(save_dir, f'vis_3d_{timestamp}.html')

        fig.write_html(final_path)
        print(f"✅ 3D可视化(HTML)已保存: {final_path}")

    def _compute_energy_field_local(
            self,
            query_feat,
            gt_coord_4d,
            scale_fixed=None,
            n_samples_per_dim=32,
            delta=0.1,
            rot_span=torch.pi,
            show_grad_field=True,
            adaptive_z_scale=False,
            argmode='min',
            energy_backend='ingp',
            chunk_size=4096,
    ):
        """
        生成局部能量/概率场，返回绘制所需的张量与元信息。
        """
        import torch.nn.functional as TF

        nr_center, nc_center, rot_center, _ = gt_coord_4d
        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)
        rot_range = torch.linspace(rot_center - rot_span / 2, rot_center + rot_span / 2, n_samples_per_dim,
                                   device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        coords_sampled_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_fixed)
        ], dim=-1)

        if show_grad_field:
            coords_sampled_4d.requires_grad_(True)

        energy_backend = energy_backend.lower()
        if energy_backend not in ('ingp', 'projector'):
            raise ValueError(f"energy_backend must be 'ingp' or 'projector', got {energy_backend}")

        if energy_backend == 'ingp':
            metric_out = self._compute_metric_from_ingp(
                query_feats=query_feat,
                ref_points=coords_sampled_4d,
                coord_space='raw',
                chunk_size=chunk_size,
                metric='dist',
            ).squeeze(0)  # [N]
        else:
            # 使用 projector 路径计算概率，能量定义为 1 - prob
            prob_chunks = []
            total = coords_sampled_4d.shape[0]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                coords_chunk = coords_sampled_4d[start:end]
                prob_chunk = self._compute_metric_from_query_and_points(
                    query_feats=query_feat,
                    ref_points=coords_chunk,
                    temperature=self.energy_temperature,
                    metric='possibility',
                    coord_space='raw',
                    chunk_size=None,  # 这里按外层 chunk 手动控制
                    feat_type='projector',
                )  # [1, chunk]
                prob_chunks.append(prob_chunk)
            prob_all = torch.cat(prob_chunks, dim=1).squeeze(0)  # [N]
            metric_out = 1.0 - prob_all  # 概率越大，能量越小

        argmode = argmode.lower()
        if argmode not in ('min', 'max'):
            raise ValueError(f"argmode must be 'min' or 'max', got {argmode}")

        if argmode == 'min':
            energy_pred = metric_out
            prob_pred = (2 - metric_out).clamp(min=0) / 2  # 小距离 → 大概率
            best_reducer = torch.argmin
            grad_dir = -1.0
        else:
            energy_pred = -metric_out  # 取负后“越近越大”，便于 argmax
            e_min, e_max = energy_pred.min(), energy_pred.max()
            prob_pred = (energy_pred - e_min) / (e_max - e_min + 1e-8)
            best_reducer = torch.argmax
            grad_dir = 1.0

        z_amplification = 1.0
        grad_vec_norm = None
        if show_grad_field:
            grad_outputs = torch.ones_like(energy_pred)
            gradients = torch.autograd.grad(energy_pred, coords_sampled_4d, grad_outputs, create_graph=False)[0]

            grad_r = grad_dir * gradients[:, 0]
            grad_c = grad_dir * gradients[:, 1]
            grad_rot = grad_dir * gradients[:, 2]

            grad_rc_mean = (grad_r.abs().mean() + grad_c.abs().mean()) / 2
            grad_rot_mean = grad_rot.abs().mean()
            if adaptive_z_scale:
                if grad_rot_mean < 1e-9:
                    z_amplification = 100.0
                else:
                    z_amplification = (grad_rc_mean / grad_rot_mean).item()
                grad_rot_amplified = grad_rot * z_amplification
            else:
                grad_rot_amplified = grad_rot
                z_amplification = 1.0

            grad_vec_vis = torch.stack([grad_r, grad_c, grad_rot_amplified], dim=1)
            grad_vec_norm = TF.normalize(grad_vec_vis, dim=1)

        best_idx = best_reducer(energy_pred).item()

        return {
            "coords_sampled_4d": coords_sampled_4d.detach(),
            "energy_pred": energy_pred.detach(),
            "prob_pred": prob_pred.detach(),
            "grad_vec_norm": grad_vec_norm.detach() if grad_vec_norm is not None else None,
            "z_amplification": z_amplification,
            "best_idx": best_idx,
            "centers": (nr_center.item(), nc_center.item(), rot_center.item()),
            "scale_fixed": scale_fixed,
        }

    def _render_energy_field_local(
            self,
            field_data,
            n_samples_per_dim,
            surface_min_ratio,
            save_path,
            show_plot,
            argmode,
    ):
        import numpy as np
        import os
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        coords_sampled_4d = field_data["coords_sampled_4d"]
        energy_pred = field_data["energy_pred"]
        grad_vec_norm = field_data["grad_vec_norm"]
        z_amplification = field_data["z_amplification"]
        best_idx = field_data["best_idx"]
        gt_r, gt_c, gt_rot = field_data["centers"]
        scale_fixed = field_data["scale_fixed"]

        coords_np = coords_sampled_4d.cpu().numpy()
        X = coords_np[:, 0]
        Y = coords_np[:, 1]
        Z = coords_np[:, 2]

        V = energy_pred.cpu().numpy()
        G_vec = grad_vec_norm.cpu().numpy() if grad_vec_norm is not None else None

        min_v, max_v = V.min(), V.max()
        print(f"📊 统计: Min Energy={min_v:.5f}, Max Energy={max_v:.5f}, Z-Scale={z_amplification:.1f}x (argmode={argmode})")

        pred_r, pred_c, pred_rot = X[best_idx], Y[best_idx], Z[best_idx]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f'场值散点 (Min={min_v:.2f})', '梯度流场 (Descent)', '等值面结构 (Geometry)'),
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.02
        )

        step = 1 if n_samples_per_dim <= 32 else 2
        mask = slice(None, None, step)

        fig.add_trace(go.Scatter3d(
            x=X[mask], y=Y[mask], z=Z[mask],
            mode='markers',
            marker=dict(size=3, opacity=0.4, color=V[mask], colorscale='Viridis',
                        colorbar=dict(title='Energy', x=0.28, len=0.5)),
            hovertemplate='Energy: %{marker.color:.4f}<extra></extra>',
            name='Energy Cloud'
        ), row=1, col=1)

        if grad_vec_norm is not None:
            step_r = (X.max() - X.min()) / n_samples_per_dim
            cone_scale = step_r * 8.0

            fig.add_trace(go.Cone(
                x=X[mask], y=Y[mask], z=Z[mask],
                u=G_vec[mask, 0], v=G_vec[mask, 1], w=G_vec[mask, 2],
                sizemode="absolute", sizeref=cone_scale, anchor="tail",
                colorscale='Jet', showscale=False, opacity=0.7,
                name='Gradients'
            ), row=1, col=2)

        val_range = max_v - min_v
        split_val = min_v + val_range * surface_min_ratio

        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=min_v,
            isomax=split_val,
            surface_count=2,
            colorscale='Plasma',
            opacity=0.2,
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            showscale=False,
            name='Inner Core',
            hovertemplate='Core Energy: %{value:.4f}<extra></extra>'
        ), row=1, col=3)

        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=split_val,
            isomax=max_v,
            surface_count=3,
            colorscale='Viridis',
            opacity=0.15,
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            colorbar=dict(title='Energy Level', x=1.0, len=0.5),
            name='Outer Shell',
            hovertemplate='Global Energy: %{value:.4f}<extra></extra>'
        ), row=1, col=3)

        for col in [1, 2, 3]:
            fig.add_trace(go.Scatter3d(
                x=[gt_r], y=[gt_c], z=[gt_rot],
                mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
                showlegend=(col == 1), name='GT'
            ), row=1, col=col)
            fig.add_trace(go.Scatter3d(
                x=[pred_r], y=[pred_c], z=[pred_rot],
                mode='markers', marker=dict(size=6, color='yellow', symbol='x'),
                showlegend=(col == 1), name='Best'
            ), row=1, col=col)

        scene_layout = dict(
            xaxis_title='NR', yaxis_title='NC', zaxis_title='Rot',
            aspectmode='cube'
        )

        fig.update_layout(
            title=f'Energy Field Comprehensive Analysis (Scale={scale_fixed:.2f})',
            height=600, width=1600,
            scene1=scene_layout,
            scene2=scene_layout,
            scene3=scene_layout,
            margin=dict(l=10, r=10, b=10, t=60)
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 能量场可视化已保存: {save_path}")

        if show_plot:
            fig.show()

    def visualize_energy_field_local(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                                     n_samples_per_dim=32, delta=0.1, rot_span=torch.pi,
                                     surface_min_ratio=0.2, adaptive_z_scale=False,
                                     save_path="vis_results/energy_field.html", show_plot=False,
                                     use_train_uav=False, show_grad_field=True, argmode='min',
                                     energy_backend='ingp', chunk_size=4096):
        """
        能量场综合可视化：拆分为生成场与绘制两个子步骤。
        """
        import torch

        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        if query_feat is None or gt_coord_4d is None:
            if use_train_uav and hasattr(self, 'uav_dataloader_train'):
                dataloader = self.uav_dataloader_train
                tag = "Train"
            elif hasattr(self, 'uav_dataloader_test'):
                dataloader = self.uav_dataloader_test
                tag = "Test"
            else:
                raise AttributeError("未找到可用的UAV数据加载器。")

            try:
                uav_img, uav_coords_4d = next(iter(dataloader))
            except StopIteration:
                uav_img, uav_coords_4d = next(iter(dataloader))

            uav_img = uav_img[0].to(self.device).unsqueeze(0)
            gt_coord_4d = uav_coords_4d[0].to(self.device)
            query_feat = self._get_feats_fm_imgs(uav_img)
            print(f"可视化数据来源: {tag} Set")

        if scale_fixed is None:
            scale_fixed = gt_coord_4d[3].item()

        field_data = self._compute_energy_field_local(
            query_feat=query_feat,
            gt_coord_4d=gt_coord_4d,
            scale_fixed=scale_fixed,
            n_samples_per_dim=n_samples_per_dim,
            delta=delta,
            rot_span=rot_span,
            show_grad_field=show_grad_field,
            adaptive_z_scale=adaptive_z_scale,
            argmode=argmode,
            energy_backend=energy_backend,
            chunk_size=chunk_size,
        )

        self._render_energy_field_local(
            field_data=field_data,
            n_samples_per_dim=n_samples_per_dim,
            surface_min_ratio=surface_min_ratio,
            save_path=save_path,
            show_plot=show_plot,
            argmode=argmode,
        )

    def analyze_feat_freq_band(self, n_points_per_subspace=1, use_fine=False,vis=False):
        """
        获取当前网格下的所有采样点（用于后续特征频域分析）。

        Args:
            n_points_per_subspace: 每个粗子空间采样的点数
            use_fine: 是否展开细粒度采样

        Returns:
            coords_all: [1, N, 4] 物理坐标
            candidate_labels: [1, N, 4] 对应索引标签
        """
        if not hasattr(self, "subspace_sampler"):
            raise AttributeError("subspace_sampler 未初始化，先确保已运行 _init_datasets/_load_checkpoints 创建采样器。")

        coords_all, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=n_points_per_subspace,
            use_fine=use_fine,
            rand_offset=False,
        )
        # 恢复为多维 shape: [NR, NC, Rot, Scale, P, 4] / [NR, NC, Rot, Scale, P]
        n_coarse = tuple(self.subspace_sampler.n_coarse.tolist())
        coords_multi = coords_all.view(*n_coarse, n_points_per_subspace, 4)
        labels_multi = candidate_labels.view(*n_coarse, n_points_per_subspace)
        print(f"采样完成: flat coords={coords_all.shape}, multi coords={coords_multi.shape}")

        rot_id = min(18, n_coarse[2] - 1)
        coords_2d = coords_multi[:, :, rot_id, 0, 0, :]

        # 2D网格细采样（固定 rot/scale），在每个 coarse cell 内生成 fine_grid×fine_grid 细分点
        fine_grid = 2
        delta_nr, delta_nc = self.subspace_sampler.coarse_bin_sizes[:2] * 0.5
        nr_lin = torch.linspace(-delta_nr, delta_nr, fine_grid, device=coords_2d.device)
        nc_lin = torch.linspace(-delta_nc, delta_nc, fine_grid, device=coords_2d.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing='ij')  # [g, g]
        nr_grid = nr_grid.unsqueeze(0).unsqueeze(0)
        nc_grid = nc_grid.unsqueeze(0).unsqueeze(0)
        nr_fine = coords_2d[..., 0:1, None, None] + nr_grid
        nc_fine = coords_2d[..., 1:2, None, None] + nc_grid
        rot_fine = torch.ones_like(nr_fine) * coords_2d[..., 2:3, None, None]
        scale_fine = torch.ones_like(nr_fine) * coords_2d[..., 3:4, None, None]
        # coords_2d_fine = torch.stack([nr_fine, nc_fine, rot_fine, scale_fine], dim=-1)  # [NR,NC,g,g,4]
        g = fine_grid
        coords_2d_fine = torch.stack([nr_fine, nc_fine, rot_fine, scale_fine], dim=-1)  # [NR,NC,1,g,g,4]
        coords_2d_fine = coords_2d_fine.squeeze(2)  # [NR,NC,g,g,4]
        coords_2d_fine = coords_2d_fine.permute(0, 2, 1, 3, 4).contiguous()  # [NR,g,NC,g,4]
        coords_2d_fine = coords_2d_fine.view(coords_2d.shape[0] * g, coords_2d.shape[1] * g, 4)  # [NR*g,NC*g,4]

        with torch.no_grad():
            feats_ingpg_2d = self._get_feats_fm_INGP(coords_2d_fine.view(-1, 4), coord_mode='raw').reshape(
                *coords_2d_fine.shape[:2], -1)
            feats_projector_2d = self.projector(
                feats_ingpg_2d.view(-1, feats_ingpg_2d.shape[-1])
            ).reshape(*coords_2d_fine.shape[:2], -1)

        from scripts.analysis.util_fft_analyse import analyse_feature_frequency
        res = analyse_feature_frequency(
            feats_ingpg_2d, feats_projector_2d,
            cdf_tau=0.95, hf_frac=0.33, eps=1e-12,
            norm="ortho", channel_norm=True, return_radial=True
        )
        # 打印（保持原格式）
        mF, mZ, d = res["metrics_F"], res["metrics_Z"], res["delta"]
        print(f"[INGP feat space ] fc={mF['fc']:.3f}, f95={mF['f95']:.1f}, hf_ratio={mF['hf_ratio']:.5f} (f0 bin={mF['f0_bin']})")
        print(f"[Proj feat space ] fc={mZ['fc']:.3f}, f95={mZ['f95']:.1f}, hf_ratio={mZ['hf_ratio']:.5f} (f0 bin={mZ['f0_bin']})")
        print(f"Delta fc   : {d['fc']:+.3f}")
        print(f"Delta f95  : {d['f95']:+.1f}")
        print(f"HF ratio Z/F: {d['hf_ratio_Z_over_F']:.3f}")


        if vis:
            res = analyse_feature_frequency(
                feats_ingpg_2d, feats_projector_2d,
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12,
                norm="ortho", channel_norm=True, return_radial=True,wo_DC=False
            )
            P_F, P_Z = res["P_F"], res["P_Z"]
            eps = 1e-12
            dir2save,epoch = self._get_dir2save(ret_epoch=True)

            import matplotlib.pyplot as plt
            from matplotlib import cm
            P_F_np = torch.fft.fftshift(P_F).cpu().numpy()
            P_Z_np = torch.fft.fftshift(P_Z).cpu().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            vmin = min(np.log(P_F_np + eps).min(), np.log(P_Z_np + eps).min())
            vmax = max(np.log(P_F_np + eps).max(), np.log(P_Z_np + eps).max())
            im0 = axs[0].imshow(np.log(P_F_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[0].set_title("P_F (INGP) log spectrum")
            im1 = axs[1].imshow(np.log(P_Z_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[1].set_title("P_Z (Projector) log spectrum")
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(os.path.join(dir2save,f'feat_space_fft_w_dc_ep{epoch}.png'))
            print(f'已保存'+os.path.join(dir2save,f'feat_space_fft_w_dc_ep{epoch}.png'))
            # plt.show()


    def analyze_energy_field(self, n_nr=128, n_nc=128, use_train_uav=True, local_zoom_wh=None, vis=False,
                                       use_vis_ref=False, chunk_size_vis=1024, analyse_fft=False, query_id = 20):
        """
        随机取一帧，固定 rot/scale 与 coord_q 一致，在 nr/nc 平面均匀采样，计算 INGP 相似度/距离场。
        """

        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
        if dataset is None or len(dataset) == 0:
            raise ValueError("数据集为空或未初始化，无法抽样进行分析。")

        # idx = torch.randint(0, len(dataset), (1,)).item()
        idx = query_id
        img, coord_q = dataset[idx]
        img = img.unsqueeze(0).to(self.device)
        coord_q = coord_q.to(self.device)

        feat_vis = self._get_feats_fm_imgs(img)  # [1, C]

        # 1. 获取全局物理边界
        global_nr_min = float(self.sat_dataset.nr2sample_min)
        global_nr_max = float(self.sat_dataset.nr2sample_max)
        global_nc_min = float(self.sat_dataset.nc2sample_min)
        global_nc_max = float(self.sat_dataset.nc2sample_max)

        # local_zoom_wh=(0.2, 0.2)
        # 2. 确定采样范围 (逻辑分支)
        if local_zoom_wh is not None:
            # --- 局部采样模式 (Local Zoom) ---
            w_ratio_nr, w_ratio_nc = local_zoom_wh

            # 计算总跨度
            span_nr = global_nr_max - global_nr_min
            span_nc = global_nc_max - global_nc_min

            # 计算半宽
            half_nr = (span_nr * w_ratio_nr) / 2
            half_nc = (span_nc * w_ratio_nc) / 2

            # 确定中心 (GT)
            center_nr = coord_q[0].item()
            center_nc = coord_q[1].item()

            # 计算局部边界 (并截断防止越界)
            start_nr = max(global_nr_min, center_nr - half_nr)
            end_nr = min(global_nr_max, center_nr + half_nr)

            start_nc = max(global_nc_min, center_nc - half_nc)
            end_nc = min(global_nc_max, center_nc + half_nc)

            if vis:
                print(f"[Analyze] Local Zoom Enabled: Center=({center_nr:.2f}, {center_nc:.2f})")
                print(f"          Range NR: [{start_nr:.2f}, {end_nr:.2f}], NC: [{start_nc:.2f}, {end_nc:.2f}]")
        else:
            # --- 全局采样模式 (Global) ---
            start_nr, end_nr = global_nr_min, global_nr_max
            start_nc, end_nc = global_nc_min, global_nc_max
        local_zoom_wh='global' if local_zoom_wh == None else local_zoom_wh

        # 3. 生成网格
        nr_lin = torch.linspace(start_nr, end_nr, n_nr, device=self.device)
        nc_lin = torch.linspace(start_nc, end_nc, n_nc, device=self.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing='ij')  # [n_nr, n_nc]

        # 固定 rot 和 scale 为真值
        rot_grid = torch.full_like(nr_grid, coord_q[2])
        scale_grid = torch.full_like(nr_grid, coord_q[3])

        coords_grid = torch.stack([nr_grid, nc_grid, rot_grid, scale_grid], dim=-1)  # [n_nr, n_nc, 4]
        coords_flat = coords_grid.view(-1, 4)

        energy_visencoder=None
        with torch.no_grad():
            if use_vis_ref:
                # 使用视觉编码器直接提取参考特征，避免 INGP 路径；分块以避免显存爆炸
                feat_q_vis = TF.normalize(feat_vis, dim=-1)

                dist_chunks = []
                # 分块裁剪 + 编码，避免一次性生成全部裁剪导致显存峰值
                for start in range(0, coords_flat.shape[0], chunk_size_vis):
                    end = min(start + chunk_size_vis, coords_flat.shape[0])
                    satimgs_refs_chunk = self.sat_dataset.crop_satimg_by_4d_coords(coords_flat[start:end].cpu()).to(self.device)
                    feats_ref_chunk = TF.normalize(
                        self._get_feats_fm_imgs(satimgs_refs_chunk), dim=-1
                    )
                    dist_chunk = torch.norm(feats_ref_chunk - feat_q_vis, dim=-1)
                    dist_chunks.append(dist_chunk)

                dist_visencoder = torch.cat(dist_chunks, dim=0).reshape(*coords_grid.shape[:2])
                energy_visencoder = torch.exp(-dist_visencoder)
            else:
                dist_ingp  = self._compute_metric_from_query_and_points(
                    metric='dist',feat_type='ingp',
                    query_feats=feat_vis,ref_points=coords_flat,
                    temperature=self.energy_temperature).reshape(*coords_grid.shape[:2])
                energy_ingp = torch.exp(-dist_ingp)
                dist_projector  = self._compute_metric_from_query_and_points(
                    metric='dist',feat_type='projector',
                    query_feats=feat_vis,ref_points=coords_flat,
                    temperature=self.energy_temperature).reshape(*coords_grid.shape[:2])
                energy_projector = torch.exp(-dist_projector)

        if vis:
            set="train" if use_train_uav else "test"
            dir2save,epoch = self._get_dir2save(ret_epoch=True)
            suffix = f'hw{local_zoom_wh[0]:.2f}' if local_zoom_wh != 'global' else local_zoom_wh

            from vis_featmap import vis_girddata_in_3d_surface,vis_griddata_in_3d_surface_interactive
            projector_path2save = os.path.join(dir2save,f'ep{epoch}_energy_projector_{set}_id{idx}_ns{n_nr}_{suffix}.html')
            vis_griddata_in_3d_surface_interactive(energy_projector,p2save=projector_path2save,colorscale='RdBu_r',show_axis_info=True)
            print(f'已保存'+projector_path2save)
            # vis_griddata_in_3d_surface_interactive(dist_projector,p2save=projector_path2save.replace('energy','dist'),colorscale='RdBu_r')

            ingp_path2save = os.path.join(dir2save, f'ep{epoch}_energy_ingp_{set}_id{idx}_nr{n_nr}_{suffix}.html')
            vis_griddata_in_3d_surface_interactive(energy_ingp, p2save=ingp_path2save,colorscale='RdBu_r',show_axis_info=True)
            print(f'已保存'+ingp_path2save)
            # vis_griddata_in_3d_surface_interactive(dist_ingp,p2save=ingp_path2save.replace('energy','dist'),colorscale='RdBu_r')

            if energy_visencoder is not None:
                visencoder_path2save =os.path.join(dir2save,f'ep{epoch}_energy_visencoder_{set}_id{idx}_ns{n_nr}_{suffix}.html')
                # vis_griddata_in_3d_surface_interactive(dist_visencoder, p2save=visencoder_path2save, colorscale='RdBu_r')
                vis_griddata_in_3d_surface_interactive(energy_visencoder, p2save=visencoder_path2save, colorscale='RdBu_r',show_axis_info=False)
                print(f'已保存' + visencoder_path2save)

            from vis_featmap import plot_contour
            contour_p2save =os.path.join(dir2save,f'contour_{set}_id{idx}_ns{n_nr}_{suffix}_visencoder&ingp.png')
            plot_contour(dist_ingp=energy_ingp.detach().cpu(),dist_proj=energy_projector.detach().cpu(),
                                  gt_coords=(n_nr//2,n_nc//2),crop_size=0,with_flow=False,
                                  flow_mode='ascent',unified_scale=False,
                                    save_path=os.path.join(dir2save,f'contour_{set}_id{idx}_ns{n_nr}_{suffix}_ingp&projector.png'))
            print(f'已保存' + contour_p2save)

        if analyse_fft:
            from scripts.analysis.util_fft_analyse import analyse_feature_frequency
            res = analyse_feature_frequency(
                energy_ingp[...,None], energy_projector[...,None],
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12,wo_DC=True,
                norm="ortho", channel_norm=True, return_radial=True
            )
            # 打印（保持原格式）
            mF, mZ, d = res["metrics_F"], res["metrics_Z"], res["delta"]
            print(
                f"[INGP energy space] fc={mF['fc']:.3f}, f95={mF['f95']:.1f}, hf_ratio={mF['hf_ratio']:.5f} (f0 bin={mF['f0_bin']})")
            print(
                f"[Proj energy space] fc={mZ['fc']:.3f}, f95={mZ['f95']:.1f}, hf_ratio={mZ['hf_ratio']:.5f} (f0 bin={mZ['f0_bin']})")
            print(f"Delta fc   : {d['fc']:+.3f}")
            print(f"Delta f95  : {d['f95']:+.1f}")
            print(f"HF ratio Z/F: {d['hf_ratio_Z_over_F']:.3f}")

            res = analyse_feature_frequency(
                energy_ingp[...,None], energy_projector[...,None],
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12,wo_DC=False,
                norm="ortho", channel_norm=True, return_radial=True
            )
            dir2save, epoch = self._get_dir2save(ret_epoch=True)
            P_F, P_Z = res["P_F"], res["P_Z"]
            eps = 1e-12

            import matplotlib.pyplot as plt
            from matplotlib import cm
            P_F_np = torch.fft.fftshift(P_F).cpu().numpy()
            P_Z_np = torch.fft.fftshift(P_Z).cpu().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            vmin = min(np.log(P_F_np + eps).min(), np.log(P_Z_np + eps).min())
            vmax = max(np.log(P_F_np + eps).max(), np.log(P_Z_np + eps).max())
            im0 = axs[0].imshow(np.log(P_F_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[0].set_title("P_F (INGP) log spectrum")
            im1 = axs[1].imshow(np.log(P_Z_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[1].set_title("P_Z (Projector) log spectrum")
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(os.path.join(dir2save,f'energy_space_fft_w_dc_ep{epoch}.png'))
            print(f'已保存'+os.path.join(dir2save,f'energy_space_fft_w_dc_ep{epoch}.png'))
            # plt.show()

    def test(self, use_train_uav=False):
        """
        Stage 3测试函数
        """
        print("\n" + "🧪" * 40)
        print("开始 Stage 3 测试: Projector")
        print("🧪" * 40 + "\n")

        # 1. 初始化数据集
        self._init_datasets(create_train_loader=False)

        # 2. 加载checkpoint（自适应）
        self._load_checkpoints_for_test()

        # 3. 设置为评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 4. 初始化必要的工具
        from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)

        from trainer_depends.datasets.util_core_subspace_sampler import SubspaceSampler
        self.subspace_sampler = SubspaceSampler(
            sat_dataset=self.sat_dataset,
            n_coarse=self.n_coarse,
            n_fine_per_coarse=self.n_fine_per_coarse
        )

        # 创建测试用的dataloader（用于可视化）
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        if use_train_uav:
            self.uav_dataloader_train = torch.utils.data.DataLoader(
                self.uav_dataset_train,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                drop_last=False,
                pin_memory=True
            )

        # self.analyze_feat_freq_band(vis=False)
        self.analyze_energy_field(vis=True,query_id=40)

        # 6. 运行3D分类测试 (NR, NC, Rot)
        # results_3d = self._test_3d_classification_accuracy(
        #     n_samples=256,
        #     use_train_uav=use_train_uav,
        #     temperature=self.energy_temperature,
        #     save_pred_pdf=True,
        # )

        #todo:_test_3d_fine_accuracy_v1和_test_3d_fine_accuracy输出不一致？
        results_3d = self._test_3d_fine_accuracy_v1(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature,
            save_pred_pdf=False,
            enable_filter=False,
        )

        results_3d = self._test_3d_fine_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature,
            save_pred_pdf=False,
            enable_filter=False,
        )

        # 5. 运行2D分类测试
        results_2d = self._test_2d_classification_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature,
        )

        # 7. 构造可视化保存路径（基于checkpoint信息）
        import os
        import re
        # 获取checkpoint路径
        stage3_ckpt_path = self._get_stage3_checkpoint_path()
        if stage3_ckpt_path:
            # 提取实验目录名（如 stage3_metric_net_31）
            exp_dir = os.path.dirname(stage3_ckpt_path)
            exp_name = os.path.basename(exp_dir)

            # 提取epoch号（如 epoch99.pth -> 99）
            ckpt_filename = os.path.basename(stage3_ckpt_path)
            epoch_match = re.search(r'epoch(\d+)', ckpt_filename)
            epoch_num = epoch_match.group(1) if epoch_match else 'unknown'

            # 构造保存路径
            dataset_type = 'train' if use_train_uav else 'test'
            viz_save_path = os.path.join(exp_dir,
                                         f"energy_field_{exp_name}_epoch{epoch_num}_{dataset_type}.html")
        else:
            # 如果没有找到checkpoint路径，使用默认名称
            dataset_type = 'train' if use_train_uav else 'test'
            viz_dir = self.log_dir2save or os.path.join(
                getattr(self.opt, "dir2save_log", "logs"),
                self.opt.exp_name,
            )
            os.makedirs(viz_dir, exist_ok=True)
            viz_save_path = os.path.join(viz_dir, f"energy_field_{dataset_type}.html")

        print(f"\n📊 生成能量场可视化: {viz_save_path}")
        self.visualize_energy_field_local(
            save_path=viz_save_path,
            use_train_uav=use_train_uav,
            show_grad_field=False,
            delta=0.1,
            argmode='min',
            energy_backend='ingp',
        )

        print("✅ Stage 3 测试完成！")
        # return {'2d': results_2d, '3d': results_3d}

    def train(self):
        """Stage 3训练主循环 (DualStream + Listwise Ranking)"""
        opt = self.opt

        print("\n" + "🚀" * 40)
        print("开始 Stage 3 训练: MetricNet (DualStream + Listwise Ranking)")
        print("🚀" * 40 + "\n")

        # 0. 初始化GradScaler
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple

        from losses.CL_losses_w_weight import pairLoss_singleEdge_weightedHardest,pairLoss_multiEdge_logSum,WeightedDirichletEnergyLoss
        # self.sms_loss = pairLoss_multiEdge_logSum(beta=10.0, margin=0.1, learnable_beta=True).to(self.device)
        self.sms_loss = pairLoss_singleEdge_weightedHardest(beta=10.0, margin=0., learnable_beta=True).to(self.device)
        self.param2optimize['loss_fn'] = self.sms_loss
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam', opt=self.opt)
        # 其他候选loss：
        # self.wde_loss = WeightedDirichletEnergyLoss(apply_log=True)
        # from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        # self.sms_loss = SWTLoss_fm_mat(decoupling=False)


        # 2. 加载checkpoint
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

        # 同时创建测试集的数据加载器（用于训练时的可视化和测试）
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=opt.batchsize_uav,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        # 4.5 初始化4d坐标处理工具
        # from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)

        # 初始化高斯邻域重要性采样器
        from trainer_depends.utils.util_gaussian_importance_sampler import NormalizedGaussianSampler
        # self.phy_sigmas = {'gs_sigma_nrc': self.gs_sigma_nrc, 'gs_sigma_radrot': self.gs_sigma_radrot, 'gs_sigma_scale': self.gs_sigma_logscale}
        # self.norm_sigmas = self.coord_normer.get_linear_sigmas(**self.phy_sigmas)
        self.normed_sigmas = self.coord_normer.get_linear_sigmas(self.gs_sigma_nrc, self.gs_sigma_radrot,
                                                                 self.gs_sigma_logscale)
        self.gs_sampler = NormalizedGaussianSampler(self.normed_sigmas, device=self.device)

        # 初始化子空间采样器（替代原来的 coord_sampler 和 udf_computer）
        from trainer_depends.datasets.util_core_subspace_sampler import SubspaceSampler
        self.n_points_per_subspace = getattr(opt, 'n_points_per_subspace', 1)
        self.subspace_sampler = SubspaceSampler(
            sat_dataset=self.sat_dataset,
            n_coarse=self.n_coarse,
            n_fine_per_coarse=self.n_fine_per_coarse
        )

        # 6. 训练循环
        sms_loss = self.sms_loss
        num_epochs = opt.num_epochs
        since = time.time()
        cfg_saved = False
        step = 0
        # begin_epoch = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")
        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch_uav in tqdm.tqdm(enumerate(self.uav_dataloader_train)):
                # =================== 1. 数据准备 ===================
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)  # [B, 4]

                batch_sat = next(iter(self.sat_dataloader))
                satimgs = batch_sat[0].to(self.device)
                coords_sat = batch_sat[1].to(self.device)  # [B, 4]

                # 提取视觉特征
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([uavimgs, satimgs], dim=0)
                )  # [2B, C]

                # Ground truth 坐标
                coords_gt = torch.cat([coords_uav, coords_sat], dim=0)  # [2B, 4]
                coords_gt_linear = self.coord_normer.raw_to_linear(coords_gt)

                #采样 for de_loss
                n_neighbors = 32
                coords_neighbor_linear = self.gs_sampler.sample_importance(coords_gt_linear.to(self.device),
                                                                           num_samples=n_neighbors, include_center=True)
                #采样 for neg_cl_loss
                coords_all_grid, coords_all_grid_labels = self.subspace_sampler.sample_all_subspaces_gpu(
                    n_points_per_subspace=1, use_fine=False)
                n_rand = int(coords_all_grid.shape[0] / 4)
                n_candidates = coords_all_grid.shape[0]
                perm = torch.randperm(n_candidates, device=coords_all_grid.device)
                perm = perm[:n_rand]
                coords_rand = coords_all_grid[perm]
                coords_rand_grid_id = coords_all_grid_labels[perm]
                coords_rand_linear = self.coord_normer.raw_to_linear(coords_rand)

                #compute weight
                coords_ref_linear = torch.concatenate([coords_neighbor_linear,
                                                       coords_rand_linear.permute(1, 0, 2).expand(
                                                           coords_neighbor_linear.shape[0], -1, -1)], dim=1)
                weights_ref = self.coord_normer.compute_weight_matrix_linear(coords_gt_linear.unsqueeze(1),
                                                                             coords_ref_linear, self.normed_sigmas,
                                                                             ignore_dim=[3]).squeeze()

                # compute distance metric with chunking to avoid large intermediate tensors
                bsz = coords_gt_linear.shape[0]
                feat_dist_chunks = []

                coords_neighbor_linear_flat = coords_neighbor_linear.reshape(-1, 4)
                coords_neighbor_net = self.coord_normer.linear_to_net(coords_neighbor_linear_flat)
                coords_neighbor_6d = torch.cat(
                    [coords_neighbor_net, coords_neighbor_linear_flat[:, 2:3]],
                    dim=-1
                )
                grid_input_neighbor = torch.cat(
                    [coords_neighbor_6d[:, :2], coords_neighbor_6d[:, -1:]],
                    dim=-1
                )
                with torch.no_grad():
                    feats_grid_raw = self._get_feats_fm_grid(grid_input_neighbor)
                    coords_encoded = self.pos_encoder_grid(coords_neighbor_6d[:, :5])
                    feats_grid_neighbor = self.grid_mlp(
                        inputs=feats_grid_raw,
                        condition_features=coords_encoded
                    )
                    feats_grid_neighbor = TF.normalize(feats_grid_neighbor, dim=-1)
                feats_grid_neighbor = feats_grid_neighbor.view(
                    bsz, coords_neighbor_linear.shape[1], -1
                )
                dist_neighbor = self.projector.compute_energy(
                    feats_vis, feats_grid_neighbor, metric='euclidean'
                )
                feat_dist_chunks.append(dist_neighbor)

                coords_rand_flat = coords_rand_linear.view(-1, 4)
                rand_chunk = 4096
                for start_idx in range(0, coords_rand_flat.shape[0], rand_chunk):
                    end_idx = min(start_idx + rand_chunk, coords_rand_flat.shape[0])
                    coords_chunk = coords_rand_flat[start_idx:end_idx]
                    coords_chunk_net = self.coord_normer.linear_to_net(coords_chunk)
                    coords_chunk_6d = torch.cat(
                        [coords_chunk_net, coords_chunk[:, 2:3]],
                        dim=-1
                    )
                    grid_input_chunk = torch.cat(
                        [coords_chunk_6d[:, :2], coords_chunk_6d[:, -1:]],
                        dim=-1
                    )
                    with torch.no_grad():
                        feats_grid_raw = self._get_feats_fm_grid(grid_input_chunk)
                        coords_encoded = self.pos_encoder_grid(coords_chunk_6d[:, :5])
                        feats_grid_chunk = self.grid_mlp(
                            inputs=feats_grid_raw,
                            condition_features=coords_encoded
                        )
                        feats_grid_chunk = TF.normalize(feats_grid_chunk, dim=-1)
                    dist_chunk = self.projector.compute_energy(
                        feats_vis, feats_grid_chunk, metric='euclidean'
                    )
                    feat_dist_chunks.append(dist_chunk)

                feat_dist = torch.cat(feat_dist_chunks, dim=1)

                loss_pos, loss_neg = sms_loss(feat_dist, weights_ref, 1 - weights_ref)
                loss = loss_pos + loss_neg
                # loss_de = self.wde_loss(feat_dist, weights_ref,1-weights_ref)*0.1
                # loss = loss_de+loss
                # =================== 7. 反向传播 ===================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 日志记录
                if it % 10 == 0:
                    self.logger.info(
                        f'Iter {it}: Loss={loss.item():.6f} | '
                        # f'Pos={loss_pos.item():.6f} | '
                        # f'Neg={loss_neg.item():.6f}'
                    )

                    if self.writer is not None:
                        self.writer.add_scalar('loss/total', loss.item(), step)
                        # self.writer.add_scalar('loss/pos', loss_pos.item(), step)
                        # self.writer.add_scalar('loss/neg', loss_neg.item(), step)

                    # 运行评估
                    self._run_epoch_evaluation(
                        epoch=epoch,
                        run_visualization=False,
                        n_test_samples=256
                    )
                if it % 20 == 0:
                    self.analyze_feat_freq_band()

                step += 1

            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f'epoch {epoch} 完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            self.logger.info('-' * 50)

            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss, epoch)

            # 备份代码
            if not cfg_saved:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(self.exp_dir2save, self.opt)
                cfg_saved = True

            # 每个epoch结束后保存checkpoint
            self._save_checkpoint(
                epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )

            # 每个epoch结束后运行评估
            self._run_epoch_evaluation(
                epoch=epoch,
                run_visualization=False,
                n_test_samples=256
            )

        # 保存最后一个epoch的checkpoint（避免重复保存）
        final_epoch = num_epochs - 1
        if not ((final_epoch % 5 == 0) and (final_epoch > 0)):
            self.logger.info(f"保存最后一个epoch的checkpoint: epoch_{final_epoch}")
            self._save_checkpoint(
                final_epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )

        self.logger.info("✅ Stage 3 训练完成！")


if __name__ == "__main__":
    import argparse
    import sys

    # 添加 --test_only 参数
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid conflict with get_parse
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    args, remaining_argv = parser.parse_known_args()

    # test by manual modification
    args.test_only = True
    # 直接读取实验配置文件opts.yaml（包含所有参数，不再需要基础配置文件）
    # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_metric_net.yaml'])  # for training
    # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainers/.exps/stage3_metric_net_31/opts.yaml'])  # for testing

    # 如果没有指定配置文件，使用 stage3 的默认配置
    if '--p_yaml' not in ' '.join(remaining_argv):
        remaining_argv.extend(['--p_yaml', 'trainer_depends/configs/stage3_metric_net.yaml'])

    sys.argv[1:] = remaining_argv  # Pass remaining args to get_parse

    trainer = MetricNetTrainer()

    if args.test_only:
        trainer.test(use_train_uav=False)
    else:
        trainer.train()
