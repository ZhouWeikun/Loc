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

from trainers.stage2_grid_hashfit import GridHashFitTrainer
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
        self.energy_temperature=0.05

        # 重新设置可训练参数
        self._setup_trainable_params_stage3()

        # 子空间采样器配置
        self.n_coarse = getattr(opt, 'n_coarse', (40, 30, 36, 1))  # 2304 类
        self.n_fine_per_coarse = getattr(opt, 'n_fine_per_coarse', (1, 1, 1, 1))  # 8 细分格子


    def _init_projector(self):
        from models.projector_mlp_sample import Projector
        """初始化Projector"""
        print("\n" + "="*80)
        print("初始化 Projector ")
        print("="*80)

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
            num_res_blocks=2,  # 深度
            output_dim=128,
            use_spectral_norm=False  # 关键：开启谱归一化
        ).to(self.device)

        print("✅ Projector 初始化完成")
        print("="*80 + "\n")


    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        for param in self.grid.parameters():
            param.requires_grad = False
        for param in self.grid_mlp.parameters():
            param.requires_grad = False

        self.param2optimize = {
            # 'metric_net': self.metric_net,
            'projector':self.projector,
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

        print("\n" + "="*80)
        print("加载测试用的checkpoint")
        print("="*80)

        # --- 1. 加载Stage 3的checkpoint (当前stage) ---
        stage3_ckpt_path = self._get_stage3_checkpoint_path()

        if stage3_ckpt_path:
            print(f"\n📦 Stage 3 checkpoint: {stage3_ckpt_path}")
            self._load_checkpoint(
                stage3_ckpt_path,
                self.param2optimize,
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


    def _compute_infonce_loss(
            self,
            log_energy: torch.Tensor,
            candidate_labels: torch.Tensor,
            anchor_labels: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:
        """
        计算 InfoNCE Loss

        Args:
            log_energy: [B, N_candidates] 预测的对数能量值
            candidate_labels: [B, N_candidates] 候选点的子空间标签
            anchor_labels: [B] anchor 的子空间标签
            temperature: 温度参数

        Returns:
            loss: 标量
        """
        # B, N = log_energy.shape

        # 正样本 mask: candidate_label == anchor_label
        pos_mask = (candidate_labels == anchor_labels.unsqueeze(1)).float()  # [B, N]#距离加权可以在这里加权

        # 检查是否每个样本都有正样本
        n_pos_per_sample = pos_mask.sum(dim=1)  # [B]
        if (n_pos_per_sample == 0).any():
            print(f"警告: 有 {(n_pos_per_sample == 0).sum()} 个样本没有正样本")

        # 转换为 log probability: log_prob = -log_energy / temperature
        # 能量越低，概率越高
        log_prob = -log_energy / temperature  # [B, N]

        # LogSumExp 归一化
        log_sum_exp_all = torch.logsumexp(log_prob, dim=1, keepdim=True)  # [B, 1]

        # 正样本的 log probability
        # 处理 mask: 将负样本位置设为 -inf
        log_prob_masked = log_prob.clone()
        log_prob_masked[pos_mask == 0] = float('-inf')
        log_sum_exp_pos = torch.logsumexp(log_prob_masked, dim=1)  # [B]

        # InfoNCE Loss = -log( sum(exp(log_prob_pos)) / sum(exp(log_prob_all)) )
        #              = log_sum_exp_all - log_sum_exp_pos
        loss_per_sample = log_sum_exp_all.squeeze(1) - log_sum_exp_pos  # [B]

        # 只对有正样本的样本计算 loss
        valid_mask = (n_pos_per_sample > 0)
        if valid_mask.sum() > 0:
            loss = loss_per_sample[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=log_energy.device)

        return loss


    def _run_epoch_evaluation(self, epoch, run_visualization=False, n_test_samples=256):
        """
        在每个epoch结束时运行评估

        Args:
            epoch: 当前epoch编号
            run_visualization: 是否运行可视化（默认False）
            n_test_samples: 测试样本数量
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch} 评估开始")
        self.logger.info(f"{'='*60}")

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
        self.logger.info(f"{'='*60}\n")

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
        print(f"\n{'='*60}")
        print(f"2D平面分类测试")
        print(f"测试样本数: {n_samples}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'='*60}\n")

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
        coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
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
        print(f"\n{'='*60}")
        print(f"2D分类测试结果 (共{n_total_cells}个cell)")
        print(f"{'='*60}")
        print(f"Top-1  准确率: {results['top1_acc']:.2f}%")
        print(f"Top-4  准确率: {results['top4_acc']:.2f}%")
        print(f"Top-9  准确率: {results['top9_acc']:.2f}%")
        print(f"Top-16 准确率: {results['top16_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'='*60}")
        print(f"距离误差统计:")
        print(f"  平均误差: {results['mean_dist_error']:.4f}")
        print(f"  中位数误差: {results['median_dist_error']:.4f}")
        print(f"  标准差: {results['dist_error_std']:.4f}")
        print(f"{'='*60}")
        print(f"加权旋转分类准确率 (共{n_coarse[2]}个rot, 使用2D预测概率加权):")
        print(f"  Top-1 准确率: {results['rot_weighted_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_weighted_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_weighted_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_weighted_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_weighted_median_rank']:.2f}")
        print(f"{'='*60}")
        print(f"边缘化旋转分类准确率 (共{n_coarse[2]}个rot, 直接对除Rot外维度求和):")
        print(f"  Top-1 准确率: {results['rot_marginalized_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_marginalized_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_marginalized_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_marginalized_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_marginalized_median_rank']:.2f}")
        print(f"{'='*60}\n")

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
        from trainer_depends.datasets.util_loc_in_girds import (
            agg_seq_pdf,
            compute_agged_pred_nneighbors_id
        )

        print(f"\n{'='*80}")
        print(f"2D序列定位测试")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"序列聚合窗口: {seq_window_len}")
        print(f"邻域大小: {len_neighbors}×{len_neighbors} = {len_neighbors**2}")
        print(f"{'='*80}\n")

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
        coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
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
        q_label_list = []   # 收集所有GT标签
        coords_gt_list = [] # 收集所有GT坐标
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
        q_label_all = torch.cat(q_label_list, dim=0)    # [N]
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
                save_dir = os.path.join(ckpt_dir, 'seq_loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 准备保存数据
                save_data = {
                    'pred_pdf_all': pred_pdf_all.numpy(),  # [N, H*W]
                    'q_label_all': q_label_all.numpy(),    # [N]
                    'coords_gt_all': coords_gt_all.numpy(), # [N, 4]
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
        print("="*80)
        print("1. 单帧定位测试")
        print("="*80)

        results_single = self._compute_2d_loc_metrics(
            pred_pdf_all,
            q_label_all,
            coords_gt_all,
            cell_centers_2d,
            n_grid_w,
            title="单帧定位"
        )

        # ==================== 序列聚合定位测试 ====================
        print("\n" + "="*80)
        print(f"2. 序列聚合定位测试 (窗口长度={seq_window_len})")
        print("="*80)

        #debug2vis:
        if True:  # 设置为False可关闭可视化
            import matplotlib.pyplot as plt
            import os

            k_samples = 16  # 可视化前k个样本
            n_cols = 4      # 每行显示4个
            n_rows = (k_samples + n_cols - 1) // n_cols

            # Reshape概率分布 [k, H*W] -> [k, H, W]
            pred_pdf_vis = pred_pdf_all[:k_samples].reshape(k_samples, n_grid_h, n_grid_w).numpy()
            q_labels_vis = q_label_all[:k_samples].numpy()

            # 创建子图
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3), dpi=100)
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
            vis_save_path = os.path.join(vis_save_dir, f'pred_pdf_samples_'+epoch_id+'.png')
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
        q_label_agged = q_label_all[seq_window_len-1:]
        coords_gt_agged = coords_gt_all[seq_window_len-1:]

        results_seq = self._compute_2d_loc_metrics(
            pred_pdf_agged,
            q_label_agged,
            coords_gt_agged,
            cell_centers_2d,
            n_grid_w,
            title=f"序列聚合(窗口={seq_window_len})"
        )

        # ==================== 邻域聚合测试 ====================
        print("\n" + "="*80)
        print(f"3. {len_neighbors}×{len_neighbors}邻域聚合测试")
        print("="*80)

        # 计算单帧的邻域聚合
        id_neighbors_1d, id_neighbors_2d = compute_agged_pred_nneighbors_id(
            pred_pdf_all.reshape(-1, n_grid_h, n_grid_w).to(self.device),
            len_neighbors,
            ret_2d=True
        )  # [N, n*n], [N, n*n, 2]

        k_values = list(range(1, len_neighbors**2 + 1))
        results_neighbors = self._compute_neighbors_recall(
            q_label_all.numpy(),
            id_neighbors_1d.cpu().numpy(),
            k_values,
            title=f"单帧+{len_neighbors**2}邻域"
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
            title=f"序列聚合+{len_neighbors**2}邻域"
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
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print(f"总样本数: {n_total_samples}")
        print(f"网格大小: {n_grid_h}×{n_grid_w} = {n_grid_h*n_grid_w} cells")
        print(f"\n【单帧定位】")
        print(f"  Top-1: {results_single['top1_acc']:.2f}%")
        print(f"  Top-4: {results_single['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_single['mean_dist_error']:.4f}")
        print(f"\n【序列聚合 (窗口={seq_window_len})】")
        print(f"  Top-1: {results_seq['top1_acc']:.2f}%")
        print(f"  Top-4: {results_seq['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_seq['mean_dist_error']:.4f}")
        print(f"\n【{len_neighbors**2}邻域聚合】")
        print(f"  单帧@1: {results_neighbors['recall@1']:.2f}%")
        print(f"  单帧@{len_neighbors**2}: {results_neighbors[f'recall@{len_neighbors**2}']:.2f}%")
        print(f"  序列@1: {results_seq_neighbors['recall@1']:.2f}%")
        print(f"  序列@{len_neighbors**2}: {results_seq_neighbors[f'recall@{len_neighbors**2}']:.2f}%")
        print("="*80 + "\n")

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

    def _get_dir2save(self,ret_epoch=False):
        if hasattr(self, 'exp_dir2save') and self.exp_dir2save and os.path.exists(self.exp_dir2save):
            # 训练模式：保存到当前实验目录
            checkpoint_dir = self.exp_dir2save
            save_dir = os.path.join(checkpoint_dir, 'seq_loc_results')
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
                save_dir = os.path.join(checkpoint_dir, 'seq_loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 从checkpoint文件名提取epoch号
                epoch_num = os.path.basename(stage3_ckpt_path).replace('epoch', '').replace('.pth', '')

        if ret_epoch:
            return save_dir, epoch_num
        else:
            return save_dir

    def _test_3d_classification_accuracy(
        self,
        n_samples=256,
        use_train_uav=False,
        temperature=0.5,
        shuffle=False,
        save_pred_pdf=True
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
        print(f"\n{'='*60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'='*60}\n")

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
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

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
        q_label_3d_list = []   # 收集GT的3D标签
        coords_gt_list = []    # 收集GT坐标

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

                # 只边缘化Scale维度 (使用logsumexp)
                neg_dists = -energys_reshaped
                scaled_logits = neg_dists / temperature
                logits_3d = torch.logsumexp(scaled_logits, dim=-1)  # [B, NR, NC, Rot]

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
                if save_pred_pdf:
                    # 将logits转换为概率分布
                    pred_pdf_3d = torch.softmax(logits_flat / temperature, dim=-1)  # [B, NR*NC*Rot]
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

                # 构建保存路径
                # 区分训练和测试模式：
                # - 训练时：使用 self.exp_dir2save（当前实验目录）
                # - 测试时：使用 opt.load2test 指定的checkpoint目录
                save_dir,epoch_num = self._get_dir2save(ret_epoch=True)

                # 生成文件名
                if save_pred_pdf:
                    data_type = 'train' if use_train_uav else 'test'
                    save_filename = f'pred_3d_ep{epoch_num}_nr{n_coarse[0]}_nc{n_coarse[1]}_nrot{n_coarse[2]}_Ttrain{self.energy_temperature}_Ttest{temperature}.npz'
                    save_path = os.path.join(save_dir, save_filename)

            # 准备保存的数据
            save_data = {
                'pred_pdf_3d_all': pred_pdf_3d_all.numpy(),  # [N, NR*NC*Rot]
                'q_label_3d_all': q_label_3d_all.numpy(),  # [N]
                'coords_gt_all': coords_gt_all.numpy(),  # [N, 4]
                'n_coarse_3d': n_coarse_3d,  # [NR, NC, Rot]
                'cell_centers_3d': cell_centers_3d.cpu().numpy(),  # [NR, NC, Rot, 3]
                'temperature': temperature,
                'n_samples': pred_pdf_3d_all.shape[0],
                'data_type': data_type,
            }

            # 保存
            np.savez(save_path, **save_data)
            print(f"\n✓ 已保存3D预测概率分布到: {save_path}")
            print(f"  - pred_pdf_3d_all: {pred_pdf_3d_all.shape}")
            print(f"  - q_label_3d_all: {q_label_3d_all.shape}")
            print(f"  - coords_gt_all: {coords_gt_all.shape}")
            print(f"  - n_coarse_3d: {n_coarse_3d}")
            print(f"  - cell_centers_3d: {cell_centers_3d.shape}")

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
        print(f"\n{'='*60}")
        print(f"3D分类测试结果 (共{n_total_cells}个cell: {n_coarse_3d[0]}x{n_coarse_3d[1]}x{n_coarse_3d[2]})")
        print(f"{'='*60}")
        print(f"Top-1   准确率: {results['top1_acc']:.2f}%")
        print(f"Top-8   准确率: {results['top8_acc']:.2f}%")
        print(f"Top-27  准确率: {results['top27_acc']:.2f}%")
        print(f"Top-64  准确率: {results['top64_acc']:.2f}%")
        print(f"Top-256 准确率: {results['top256_acc']:.2f}%")
        print(f"Top-512 准确率: {results['top512_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'='*60}")
        print(f"2D位置误差统计:")
        print(f"  平均误差: {results['mean_dist_error_2d']:.4f}")
        print(f"  中位数误差: {results['median_dist_error_2d']:.4f}")
        print(f"{'='*60}")
        print(f"旋转误差统计 (度):")
        print(f"  平均误差: {results['mean_rot_error_deg']:.2f}°")
        print(f"  中位数误差: {results['median_rot_error_deg']:.2f}°")
        print(f"{'='*60}")
        print(f"固定位置(NR,NC)时的旋转准确率 (共{n_coarse_3d[2]}个rot):")
        print(f"  Top-1 准确率: {results['rot_only_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_only_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_only_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_only_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_only_median_rank']:.2f}")
        print(f"{'='*60}\n")

        return results

    def _get_feats_fm_INGP(self, coords_raw):
        """
        [原子操作] 输入原始坐标 [N, 4]，输出归一化特征 [N, C]
        """
        # 1. Raw -> Norm (6D)
        coords_6d = self.coord_normer.raw_to_norm(coords_raw, append_linear_rot=True)

        # 2. 构造网络输入
        grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)

        # 3. 前向传播
        # 注意：这里假设处于 no_grad 上下文，或者由外部控制
        feat_raw = self._get_feats_fm_grid(grid_input)
        pos_enc = self.pos_encoder_grid(coords_6d[:, :5])
        feat_out = self.grid_mlp(feat_raw, pos_enc)

        return TF.normalize(feat_out, dim=-1, p=2)

    def _evaluate_and_rank_candidates(self, coords, feat_q, topN=None, chunk_size=4096):
        """
        逻辑1（内存优化版）：分块计算候选点能量，排序，并返回TopN结果。
        通过分块处理避免 N 很大时 [B, N, C] 张量导致的 OOM。

        Args:
            coords: [B, N, 4] 候选点坐标
            feat_q: [B, C] Query视觉特征
            topN: int (Optional) 返回前N个点。如果为None，则返回全部排序后的点。
            chunk_size: int 每次送入MLP处理的候选点数量（针对 N 维度分块）

        Returns:
            coords_sorted: [B, K, 4] 排序并截取后的坐标
            energys_sorted: [B, K] 对应的能量值
        """
        B, N, _ = coords.shape

        # 用于收集所有分块计算的能量
        all_energys_list = []

        # 1. 分块循环 (Chunk Loop)
        # 我们在 N (候选点) 维度上进行切片
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            # 1.1 获取当前块的坐标 [B, current_chunk, 4]
            coords_chunk = coords[:, start_idx:end_idx, :]
            current_chunk_size = coords_chunk.shape[1]

            # 1.2 展平以便输入 MLP [B*chunk, 4]
            # 这样不仅减少了显存，还保证了 MLP 输入不会过大
            coords_chunk_flat = coords_chunk.reshape(-1, 4)

            # 1.3 计算特征 [B*chunk, C] -> [B, chunk, C]
            # _compute_feats_from_coords 是我们之前定义的原子函数
            with torch.no_grad():
                feats_chunk_flat = self._get_feats_fm_INGP(coords_chunk_flat)
            feats_chunk = feats_chunk_flat.reshape(B, current_chunk_size, -1)

            # 1.4 立即计算能量并释放特征显存 [B, chunk]
            # feat_q: [B, 1, C] - feats_chunk: [B, chunk, C]
            # 这一步计算完后，庞大的 feats_chunk 就会被释放
            energys_chunk = torch.norm(feat_q.unsqueeze(1) - feats_chunk, p=2, dim=-1)

            all_energys_list.append(energys_chunk)

        # 2. 拼接所有能量 [B, N]
        # 能量只是标量，占用显存极小
        energys_all = torch.cat(all_energys_list, dim=1)

        # 3. 排序 (升序，能量小的在前)
        # sorted_indices: [B, N]
        sorted_energys_all, sorted_indices = torch.sort(energys_all, dim=-1, descending=False)

        # 4. 截取 TopN
        K = N
        if topN is not None:
            K = min(topN, N)

        # 截取索引和能量
        topK_indices = sorted_indices[:, :K]  # [B, K]
        energys_sorted = sorted_energys_all[:, :K]  # [B, K]

        # 5. Gather 对应的坐标
        # 注意：这里我们只 Gather 最终需要的 K 个坐标，而不是 N 个
        # 这样避免了对整个 huge coords tensor 进行 shuffle
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, K)

        # coords 本身 [B, N, 4] 如果 N 很大，直接索引也不会太占显存（只复制 K 个）
        coords_sorted = coords[batch_indices, topK_indices]  # [B, K, 4]

        return coords_sorted, energys_sorted

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

    def _test_3d_fine_accuracy(
        self,
        n_samples=256,
        use_train_uav=False,
        temperature=0.5,
        shuffle=False,
        save_pred_pdf=True,
        topN_refine=True,
        filter_topN_by_gt=False,
    ):
        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")
        from util_analyze_pred3d import compute_top_k_accuracy, print_accuracy_results, convert_3d_to_2d_predictions

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
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]
        n_grid_h, n_grid_w = n_coarse_2d[0], n_coarse_2d[1]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

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

                # 计算能量
                energys = self.projector.compute_energy(
                    feats_vis, feats_grid_all, metric='euclidean'
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                energys_reshaped = energys.reshape(batch_size, *n_coarse)

                # 只边缘化Scale维度 (使用logsumexp)
                neg_dists = -energys_reshaped
                scaled_logits = neg_dists / temperature
                logits_3d = torch.logsumexp(scaled_logits, dim=-1)  # [B, NR, NC, Rot]

                # 获取预测的3D索引
                logits_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 收集用于保存的数据
                if save_pred_pdf:
                    # 将logits转换为概率分布
                    pred_pdf_3d = torch.softmax(logits_flat / temperature, dim=-1)  # [B, NR*NC*Rot]
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

            # 显示滤波前分类精度
            single_frame_results = compute_top_k_accuracy(
                pred_pdf_3d_all.cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(single_frame_results, title="loc_res_3d_filtered")

            # 对pred_3d_pdf进行直方图滤波：
            raw_diff = torch.diff(coords_gt_all[:, 2])
            diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi
            pred_pdf_3d_shaped = pred_pdf_3d_all.reshape(-1, *n_coarse_3d)
            from util_histogram_filter_3d import HistogramFilter3D
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

            # 显示滤波后分类精度
            filtered_frame_results = compute_top_k_accuracy(
                preds_filtered.reshape(preds_filtered.shape[0], -1).cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(filtered_frame_results, title="loc_res_3d_filtered")

            # ==================== 阶段 1: 初始评估与筛选 ====================
            # 对初始的粗粒度候选点进行计算和排序，选出 TopK 用于再采样
            topN_l0 = 512  # 选择前N个候选位置
            print(f"\n{'=' * 60}")
            print(f"使用3D直接采样策略 (topN={topN_l0})")
            print(f"{'=' * 60}")

            # preds_filtered shape: [N_samples, H, W, Rot]
            N_samples = preds_filtered.shape[0]
            H, W, Rot = preds_filtered.shape[1], preds_filtered.shape[2], preds_filtered.shape[3]

            # 1. Flatten 3D概率体为 [N_samples, H*W*Rot]
            pred_3d_flat = preds_filtered.reshape(N_samples, -1)  # [N_samples, H*W*Rot]

            # 2. 对每个样本选择topN个最高概率的3D位置
            sorted_indices_3d = torch.argsort(pred_3d_flat, dim=-1, descending=True)  # [N_samples, H*W*Rot]
            topN_indices_flat = sorted_indices_3d[:, :topN_l0]  # [N_samples, topN]

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
                torch.zeros(N_samples, topN_l0, 1, dtype=torch.long, device=topN_3d_indices.device)
                # scale=0
            ], dim=-1)  # [N_samples, topN, 4]

            # 获取这些位置的坐标（从预计算的coords_candidates中提取）
            coords_reshaped_full = coords_candidates.squeeze(0).reshape(*n_coarse, 4)  # [NR, NC, Rot, Scale, 4]

            # 批量提取中心坐标
            coords_topN_centers = []
            for sample_idx in range(N_samples):
                sample_coords = []
                for k in range(topN_l0):
                    nr, nc, rot, scale = topN_4d_indices[sample_idx, k]
                    center_coord = coords_reshaped_full[nr, nc, rot, scale]  # [4]
                    sample_coords.append(center_coord)
                coords_topN_centers.append(torch.stack(sample_coords))  # [topN, 4]
            coords_topN_centers = torch.stack(coords_topN_centers)  # [N_samples, topN, 4]

            # ==================== 阶段 2: 空间再采样 ====================
            # 以第coarse个阶段得到的TopN个中心点周围生成更密集的点
            resample_dims = (2, 2, 2, 2)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_topN_centers,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes']/0.75,
            )  # -> [B, topN * 16, 4]

            # 3.1 评估新点 (返回全部新点及其能量)
            topN_l1 = 256
            coords2eval_l1 = torch.cat([coords_resampled, coords_topN_centers], dim=1)
            coords_resorted_l1, energys_resorted_l1 = self._evaluate_and_rank_candidates(
                coords=coords2eval_l1,
                feat_q=feats_vis_all,
                topN=topN_l1  # 不截断，保留所有新点
            )

            # ==================== 阶段 3: 空间再采样 ====================
            # 以第1个fine阶段得到的TopN个中心点周围生成更密集的点
            topN_l2 = 64
            resample_dims = (3, 3, 3, 2)  # (nr, nc, rot, scale)
            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_resorted_l1[:,:topN_l2],  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes']/0.75,
            )  # -> [B, topN * 16, 4]

            # ==================== 阶段 3: 空间再采样 ====================
            coords2eval_l2 = torch.cat([coords_resampled, coords_resorted_l1], dim=1)
            coords_resorted_l2, energys_resorted_l2 = self._evaluate_and_rank_candidates(
                coords=coords2eval_l2,
                feat_q=feats_vis_all,
                topN=64  # 不截断，保留所有新点
            )

            # ==================== 可选阶段 4: 反向传播优化 ====================
            opt=False
            if opt:
                coords_resorted_l2_opted = self._opt_coords_topN(
                    coords_resorted_l2[:,:32],
                    feats_vis_all,
                    n_step=200,
                )  # [N, K, 4] - 返回已经按优化后的loss排序的坐标

                # 将优化后的TopK替换回原coords_sorted
                coords_resorted_l2 = torch.cat([
                    coords_resorted_l2_opted,  # [N, K, 4] 优化后的TopK
                    coords_resorted_l2  # [N, M-K, 4] 剩余的未优化候选
                ], dim=1)  # [N, M, 4]

            #评估&输出
            from util_analyze_pred3d import compute_topN_acc_given_threshold,print_topN_acc_results
            dist_lambda=1.0
            thresh_cfg = {
                'norm_dist': self.sat_dataset.halfimg_radius_nrc*dist_lambda,  # 例如 3米 或 3个Grid单位
                'rot': 10.0,  # 15度
                'scale': None  # 不评估尺度
            }
            target_k_values = [1, 5, 10, 16, 32, 64]
            acc_metrics, err_stats = compute_topN_acc_given_threshold(
                coords_pred=coords_resorted_l2,
                coords_gt=torch.concatenate(coords_gt_list,dim=0),  # [B, 4] 你的GT坐标
                dist_th=thresh_cfg['norm_dist'],
                rot_th_deg=thresh_cfg['rot'],
                scale_th=thresh_cfg['scale'],
                k_values=target_k_values
            )
            print_topN_acc_results(acc_metrics, err_stats, thresh_cfg)


    def _test_3d_fine_accuracy_debug(
        self,
        n_samples=256,
        use_train_uav=False,
        temperature=0.5,
        shuffle=False,
        save_pred_pdf=True,
        topN_refine=True,
        filter_topN_by_gt=False,
    ):
        """
        测试3D分类正确性 (NR, NC, Rot)，只边缘化Scale维度

        Args:
            n_samples: 测试样本数量（None表示使用全部数据）
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数
            shuffle: 是否打乱数据顺序（序列测试时应该为False）
            save_pred_pdf: 是否保存预测概率分布到checkpoint文件夹（默认True）
            topN_refine: 是否进行topN精细检索（默认True）

        Returns:
            dict: 包含各种准确率指标的字典
        """
        print(f"\n{'='*60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'='*60}\n")
        from util_analyze_pred3d import compute_top_k_accuracy,print_accuracy_results,convert_3d_to_2d_predictions

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
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]
        n_grid_h, n_grid_w = n_coarse_2d[0], n_coarse_2d[1]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

        # 统计容器
        processed = 0

        # 用于保存的容器
        pred_pdf_3d_list = []  # 收集3D概率分布
        q_label_3d_list = []   # 收集GT的3D标签
        coords_gt_list = []    # 收集GT坐标
        feats_vis_list =[]

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

                # 只边缘化Scale维度 (使用logsumexp)
                neg_dists = -energys_reshaped
                scaled_logits = neg_dists / temperature
                logits_3d = torch.logsumexp(scaled_logits, dim=-1)  # [B, NR, NC, Rot]

                # 获取预测的3D索引
                logits_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 收集用于保存的数据
                if save_pred_pdf:
                    # 将logits转换为概率分布
                    pred_pdf_3d = torch.softmax(logits_flat / temperature, dim=-1)  # [B, NR*NC*Rot]
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

            # 显示滤波前分类精度
            single_frame_results = compute_top_k_accuracy(
                pred_pdf_3d_all.cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(single_frame_results, title="single_frame_results")

            # 对pred_3d_pdf进行直方图滤波：
            raw_diff = torch.diff(coords_gt_all[:, 2])
            diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi
            pred_pdf_3d_shaped = pred_pdf_3d_all.reshape(-1,*n_coarse_3d)
            from util_histogram_filter_3d import HistogramFilter3D
            histfilter = HistogramFilter3D(H=n_coarse_3d[0], W=n_coarse_3d[1], O=n_coarse_3d[2],device=pred_pdf_3d_shaped.device)
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

            # 显示滤波后分类精度
            # from analyze_pred3d import convert_3d_to_2d_predictions,compute_2d_plane_accuracy,print_accuracy_results,compute_top_k_accuracy
            # loc_res_2d_filtered = compute_2d_plane_accuracy(pred_3d=preds_filtered,gt_labels_3d=q_label_3d_all,dim_order='HWO')
            # print_accuracy_results(loc_res_2d_filtered,title='loc_res_2d_filtered')
            filtered_frame_results = compute_top_k_accuracy(
                preds_filtered.reshape(preds_filtered.shape[0], -1).cpu().numpy(),
                q_label_3d_all,
                k_values=[1, 8, 27, 64, 128, 256, 512],
                dim_order='HWO'
            )
            print_accuracy_results(filtered_frame_results, title="loc_res_3d_filtered")

            # ==================== 采样策略分支选择 ====================
            use_3d_direct_sampling = True  # True: 直接3D采样; False: 2D边缘化后采样
            topN_candidates = 256  # 选择前N个候选位置

            if use_3d_direct_sampling:
                # ==================== 新分支：直接从3D概率体选择topN ====================
                print(f"\n{'='*60}")
                print(f"使用3D直接采样策略 (topN={topN_candidates})")
                print(f"{'='*60}")

                # preds_filtered shape: [N_samples, H, W, Rot]
                N_samples = preds_filtered.shape[0]
                H, W, Rot = preds_filtered.shape[1], preds_filtered.shape[2], preds_filtered.shape[3]

                # 1. Flatten 3D概率体为 [N_samples, H*W*Rot]
                pred_3d_flat = preds_filtered.reshape(N_samples, -1)  # [N_samples, H*W*Rot]

                # 2. 对每个样本选择topN个最高概率的3D位置
                sorted_indices_3d = torch.argsort(pred_3d_flat, dim=-1, descending=True)  # [N_samples, H*W*Rot]
                topN_indices_flat = sorted_indices_3d[:, :topN_candidates]  # [N_samples, topN]

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
                    torch.zeros(N_samples, topN_candidates, 1, dtype=torch.long, device=topN_3d_indices.device)  # scale=0
                ], dim=-1)  # [N_samples, topN, 4]

                # 将4D multi索引转换为flat索引
                topN_4d_flat = (topN_4d_indices[:, :, 0] * (n_coarse[1] * n_coarse[2] * n_coarse[3]) +
                               topN_4d_indices[:, :, 1] * (n_coarse[2] * n_coarse[3]) +
                               topN_4d_indices[:, :, 2] * n_coarse[3] +
                               topN_4d_indices[:, :, 3])  # [N_samples, topN]

                # 获取这些位置的坐标（从预计算的coords_candidates中提取）
                coords_reshaped_full = coords_candidates.squeeze(0).reshape(*n_coarse, 4)  # [NR, NC, Rot, Scale, 4]

                # 批量提取中心坐标
                coords_topN_centers = []
                for sample_idx in range(N_samples):
                    sample_coords = []
                    for k in range(topN_candidates):
                        nr, nc, rot, scale = topN_4d_indices[sample_idx, k]
                        center_coord = coords_reshaped_full[nr, nc, rot, scale]  # [4]
                        sample_coords.append(center_coord)
                    coords_topN_centers.append(torch.stack(sample_coords))  # [topN, 4]
                coords_topN_centers = torch.stack(coords_topN_centers)  # [N_samples, topN, 4]

                # 5. 在这些3D中心周围采样
                grid_dims = (2, 2, 2, 2)  # (nr, nc, rot, scale)
                coords_topN_centers_flat = coords_topN_centers.reshape(-1, 4)  # [N_samples*topN, 4]

                coords_in_topN = self.subspace_sampler.sample_grid_around_coords_gpu(
                    center_coords=coords_topN_centers_flat,
                    grid_dims=grid_dims,
                    sample_space_size=None,  # 使用默认的coarse_bin_sizes
                    device=self.device
                )  # [N_samples*topN, samples_per_grid, 4]

                samples_per_space = coords_in_topN.shape[1]
                print(f"  3D采样完成:")
                print(f"    - 每个样本选择topN={topN_candidates}个3D位置")
                print(f"    - 每个3D位置采样grid={grid_dims}")
                print(f"    - 每个位置生成{samples_per_space}个采样点")
                print(f"    - 总采样点数: {N_samples} × {topN_candidates} × {samples_per_space}")
                print(f"{'='*60}\n")

            else:
                # ==================== 原分支：2D边缘化后采样 ====================
                print(f"\n{'='*60}")
                print(f"使用2D边缘化采样策略 (topN={topN_candidates})")
                print(f"{'='*60}")

                # 收集2D平面的topN定位索引，为search in fine space进行准备：
                pred_2d, gt_labels_2d, (H, W) = convert_3d_to_2d_predictions(
                    preds_filtered, q_label_3d_all, dim_order='HWO')
                pred_2d_flat = pred_2d.reshape(-1, H * W)  # [N, H*W]

                sorted_indices_2d = torch.argsort(pred_2d_flat, dim=-1, descending=True)  # [N, H*W] 降序
                pred_2d_nr = sorted_indices_2d // n_grid_w
                pred_2d_nc = sorted_indices_2d % n_grid_w
                pred_2d = torch.stack([pred_2d_nr,pred_2d_nc],dim=-1)

                grid_dims = torch.tensor([2, 2, 1, 1])
                samples_per_space = torch.prod(grid_dims)
                coords_in_topN, labels_in_topN = self.subspace_sampler.sample_grid_at_2d_indices_gpu(
                    pred_2d[:,:topN_candidates,:].reshape(-1,2),
                    grid_dims=grid_dims
                )  # coords: [N*topN_2d, K_rot*K_scale, samples_per_space, 4], labels: [N*topN_2d, K_rot*K_scale]

                print(f"  2D采样完成:")
                print(f"    - 每个样本选择topN={topN_candidates}个2D位置")
                print(f"    - 每个2D位置采样所有rot×scale组合")
                print(f"{'='*60}\n")

            # 可选过滤，只保留GT在topN中的样本：
            N_total_samples = feats_vis_all.shape[0]
            if filter_topN_by_gt:
                if use_3d_direct_sampling:
                    # ==================== 3D直接采样：检查GT的3D位置是否在topN中 ====================
                    # q_label_3d_all: [N] - GT在3D网格中的flat索引
                    # topN_3d_indices: [N, topN, 3] - topN预测的3D网格索引 [nr, nc, rot]

                    # 将GT的3D flat索引转换为multi索引
                    gt_3d_flat = q_label_3d_all  # [N]
                    gt_3d_rot = gt_3d_flat % Rot
                    gt_3d_nc = (gt_3d_flat // Rot) % W
                    gt_3d_nr = gt_3d_flat // (W * Rot)
                    gt_3d_coords = torch.stack([gt_3d_nr, gt_3d_nc, gt_3d_rot], dim=-1)  # [N, 3]

                    # 检查GT的3D坐标是否在topN预测中
                    # topN_3d_indices: [N, topN, 3], gt_3d_coords: [N, 1, 3]
                    matches = (topN_3d_indices == gt_3d_coords.unsqueeze(1)).all(dim=-1)  # [N, topN]
                    mask_gt_in_topN = matches.any(dim=-1)  # [N] - 每个样本是否有GT在topN中

                    n_gt_in_topN = mask_gt_in_topN.sum().item()
                    n_total = mask_gt_in_topN.shape[0]
                else:
                    # ==================== 2D边缘化采样：检查GT的2D位置是否在topN中 ====================
                    # gt_labels_2d: [N] - GT在2D网格中的flat索引
                    # pred_2d: [N, topN_2d, 2] - topN预测的2D网格索引 [nr, nc]
                    n_grid_h, n_grid_w = H, W
                    gt_2d_nr = torch.from_numpy(gt_labels_2d // n_grid_w)  # [N]
                    gt_2d_nc = torch.from_numpy(gt_labels_2d % n_grid_w)   # [N]
                    gt_2d_coords = torch.stack([gt_2d_nr, gt_2d_nc], dim=-1)  # [N, 2]

                    # 检查GT坐标是否在topN预测中
                    # pred_2d: [N, topN_2d, 2], gt_2d_coords: [N, 1, 2]
                    matches = (pred_2d[:, :topN_candidates, :] == gt_2d_coords.unsqueeze(1)).all(dim=-1)  # [N, topN_2d]
                    mask_gt_in_topN = matches.any(dim=-1)  # [N] - 每个样本是否有GT在topN中

                    n_gt_in_topN = mask_gt_in_topN.sum().item()
                    n_total = mask_gt_in_topN.shape[0]
                print(f"\n{'='*60}")
                print(f"TopN候选过滤：{n_gt_in_topN}/{n_total} 样本的GT在Top{topN_candidates}中 ({100*n_gt_in_topN/n_total:.1f}%)")
                print(f"{'='*60}")

                # 应用mask过滤数据
                feats_vis_for_refine = feats_vis_all[mask_gt_in_topN]  # [N_filtered, C]
                coords_gt_for_refine = coords_gt_all[mask_gt_in_topN]  # [N_filtered, 4]

                # coords_in_topN需要特殊处理：它是 [N*topN_2d, K_rot*K_scale, samples_per, 4]
                # 需要先reshape为 [N, topN_2d, K_rot*K_scale, samples_per, 4]，然后过滤
                coords_in_topN_for_refine = coords_in_topN.reshape(
                    N_total_samples, topN_candidates,
                    grid_dims[0] * grid_dims[1],  # nr_dim * nc_dim = 4
                    grid_dims[2] * grid_dims[3],  # rot_dim * scale_dim = 4
                    4  # The 4 dimensions of the coordinate itself (nr, nc, rot, scale)
                )
                coords_in_topN_for_refine = coords_in_topN_reshaped_full[mask_gt_in_topN]  # [N_filtered, topN_2d, K_rot*K_scale, samples_per, 4]

                print(f"过滤后样本数: {feats_vis_for_refine.shape[0]}")
            else:
                # 不过滤，使用全部样本
                print(f"\n{'='*60}")
                print(f"跳过GT过滤，使用全部 {N_total_samples} 个样本进行topN精细检索")
                print(f"{'='*60}")

                feats_vis_for_refine = feats_vis_all
                coords_gt_for_refine = coords_gt_all

                # coords_in_topN需要reshape为 [N, topN_2d, K_rot*K_scale, samples_per, 4]
                coords_in_topN_for_refine = coords_in_topN.reshape(
                    N_total_samples, topN_candidates,
                    grid_dims[0] * grid_dims[1],  # nr_dim * nc_dim = 4
                    grid_dims[2] * grid_dims[3],  # rot_dim * scale_dim = 4
                    4  # The 4 dimensions of the coordinate itself (nr, nc, rot, scale)
                )

            # ==================== fine loc in topN from Projector====================
            if topN_refine and feats_vis_for_refine.shape[0] > 0:
                print("\n" + "="*60)
                print("TopN精细检索：在候选区域内进行精细定位")
                print("="*60)

                N_samples = feats_vis_for_refine.shape[0]
                # 使用coords_in_topN_for_refine: [N_samples, topN_2d, K_rot*K_scale, samples_per, 4]
                # K_rot_scale = coords_in_topN_for_refine.shape[2]    # K_rot * K_scale
                # samples_per = int(samples_per_space.item())

                # 将所有topN候选点的坐标flatten为 [N_samples, topN_2d*K_rot_scale*samples_per, 4]
                coords_topN_flat = coords_in_topN_for_refine.reshape(N_samples, -1, 4).to(self.device)  # [N, M, 4] 移到GPU
                M_candidates = coords_topN_flat.shape[1]

                # 转换为6D坐标
                coords_topN_6d = self.coord_normer.raw_to_norm(
                    coords_topN_flat.reshape(-1, 4),
                    append_linear_rot=True
                )  # [N*M, 6]

                # 提取grid特征
                grid_input_topN = torch.cat([
                    coords_topN_6d[:, :2],
                    coords_topN_6d[:, -1:]
                ], dim=-1)

                with torch.no_grad():
                    feats_grid_topN_raw = self._get_feats_fm_grid(grid_input_topN)
                    coords_encoded_topN = self.pos_encoder_grid(coords_topN_6d[:, :5])
                    feats_grid_topN = self.grid_mlp(feats_grid_topN_raw, coords_encoded_topN)
                    feats_grid_topN = TF.normalize(feats_grid_topN, dim=-1)  # [N*M, C]

                # Reshape为 [N, M, C]
                feats_grid_topN = feats_grid_topN.reshape(N_samples, M_candidates, -1)

                # ========== 分支选择：使用Projector或直接计算欧氏距离 ==========
                use_projector_in_topN = False  # 设置为False则直接使用INGP输出计算距离
                if use_projector_in_topN:
                    # 方法1: 使用Projector计算能量
                    print("  使用方法: Projector (Metric Learning)")
                    energys_topN = self.projector.compute_energy(
                        feats_vis_for_refine,  # [N_samples, C]
                        feats_grid_topN,       # [N_samples, M, C]
                        metric='euclidean'
                    )  # [N_samples, M]
                else:
                    # 方法2: 直接计算INGP特征的欧氏距离
                    print("  使用方法: 直接欧氏距离 (INGP输出)")
                    # feats_vis_for_refine: [N_samples, C]
                    # feats_grid_topN: [N_samples, M, C]
                    # 计算欧氏距离: ||feats_vis - feats_grid||_2
                    feats_vis_expanded = feats_vis_for_refine.unsqueeze(1)  # [N_samples, 1, C]
                    energys_topN = torch.norm(
                        feats_vis_expanded - feats_grid_topN,
                        p=2,
                        dim=-1
                    )  # [N_samples, M]

                #todo:topN精度输出

                # ==================== 基于TopK中心点的空间再采样 ====================
                enable_resample_around_topK = True  # 是否启用再采样
                topK_for_resampling = 16  # 选择前K个能量最小的点作为中心
                resample_grid_dims = (3, 3, 3, 1)  # 再采样的网格密度 (nr, nc, rot, scale)
                resample_space_size = None  # None表示使用默认coarse_bin_sizes

                if enable_resample_around_topK:
                    print(f"\n{'='*60}")
                    print(f"基于TopK中心点的空间再采样")
                    print(f"{'='*60}")
                    print(f"  TopK中心点数量: {topK_for_resampling}")
                    print(f"  再采样网格密度: {resample_grid_dims}")
                    print(f"  每个中心点采样数: {np.prod(resample_grid_dims)}")

                    # 1. 选择TopK个能量最小的点作为中心
                    K_resample = min(topK_for_resampling, M_candidates)
                    topK_energys, topK_indices = torch.topk(
                        energys_topN,
                        k=K_resample,
                        dim=-1,
                        largest=False  # 选择最小的K个
                    )  # topK_energys: [N_samples, K], topK_indices: [N_samples, K]

                    # 2. 提取TopK中心点的坐标
                    # coords_topN_flat: [N_samples, M, 4]
                    batch_indices_resample = torch.arange(N_samples, device=self.device).unsqueeze(1).expand(-1, K_resample)
                    coords_centers = coords_topN_flat[batch_indices_resample, topK_indices]  # [N_samples, K, 4]

                    print(f"  选择的TopK中心点形状: {coords_centers.shape}")

                    # 3. 对每个样本的K个中心点进行再采样
                    # 将 [N_samples, K, 4] reshape 为 [N_samples*K, 4] 以便批量处理
                    coords_centers_flat = coords_centers.reshape(-1, 4)  # [N_samples*K, 4]

                    # 调用 sample_grid_around_coords_gpu 进行再采样
                    coords_resampled = self.subspace_sampler.sample_grid_around_coords_gpu(
                        center_coords=coords_centers_flat,
                        grid_dims=resample_grid_dims,
                        sample_space_size=resample_space_size,
                        device=self.device
                    )  # [N_samples*K, Total_Points, 4]

                    n_points_per_center = coords_resampled.shape[1]
                    print(f"  再采样点数/中心: {n_points_per_center}")
                    print(f"  总再采样点数: {coords_resampled.shape[0] * coords_resampled.shape[1]}")

                    # 4. 计算再采样点的特征和能量
                    # Reshape: [N_samples*K, Total_Points, 4] -> [N_samples*K*Total_Points, 4]
                    coords_resampled_flat = coords_resampled.reshape(-1, 4)

                    # 转换为6D坐标
                    coords_resampled_6d = self.coord_normer.raw_to_norm(
                        coords_resampled_flat,
                        append_linear_rot=True
                    )  # [N_samples*K*Total_Points, 6]

                    # 提取grid特征
                    grid_input_resampled = torch.cat([
                        coords_resampled_6d[:, :2],
                        coords_resampled_6d[:, -1:]
                    ], dim=-1)

                    with torch.no_grad():
                        feats_grid_resampled_raw = self._get_feats_fm_grid(grid_input_resampled)
                        coords_encoded_resampled = self.pos_encoder_grid(coords_resampled_6d[:, :5])
                        feats_grid_resampled = self.grid_mlp(feats_grid_resampled_raw, coords_encoded_resampled)
                        feats_grid_resampled = TF.normalize(feats_grid_resampled, dim=-1)  # [N_samples*K*Total_Points, C]

                    # Reshape回 [N_samples, K*Total_Points, C]
                    feats_grid_resampled = feats_grid_resampled.reshape(
                        N_samples, K_resample * n_points_per_center, -1
                    )

                    # 直接计算欧氏距离
                    feats_vis_expanded_resample = feats_vis_for_refine.unsqueeze(1)  # [N_samples, 1, C]
                    energys_resampled = torch.norm(
                        feats_vis_expanded_resample - feats_grid_resampled,
                        p=2,
                        dim=-1
                    )  # [N_samples, K*Total_Points]

                    # Reshape坐标: [N_samples*K, Total_Points, 4] -> [N_samples, K*Total_Points, 4]
                    coords_resampled_reshaped = coords_resampled.reshape(
                        N_samples, K_resample * n_points_per_center, 4
                    )

                    # 5. 合并原有候选点和再采样点
                    # 坐标合并
                    coords_topN_flat = torch.cat([
                        coords_topN_flat,  # [N_samples, M, 4] 原有候选点
                        coords_resampled_reshaped  # [N_samples, K*Total_Points, 4] 新采样点
                    ], dim=1)  # [N_samples, M + K*Total_Points, 4]

                    # 能量合并
                    energys_topN = torch.cat([
                        energys_topN,  # [N_samples, M] 原有能量
                        energys_resampled  # [N_samples, K*Total_Points] 新采样点能量
                    ], dim=1)  # [N_samples, M + K*Total_Points]

                    # 更新候选点数量
                    M_candidates_new = coords_topN_flat.shape[1]

                    print(f"\n  合并后候选点数量: {M_candidates_new} (原始: {M_candidates}, 新增: {K_resample * n_points_per_center})")
                    print(f"  能量最小值: {energys_topN.min().item():.6f}")
                    print(f"  能量最大值: {energys_topN.max().item():.6f}")
                    print(f"{'='*60}\n")

                    # 更新M_candidates为新的候选点数量
                    M_candidates = M_candidates_new

                ################################################################################
                # 1. 排序：获取所有候选点的排序索引 (Energy越小越好，升序)
                sorted_indices = torch.argsort(energys_topN, dim=-1, descending=False)  # [N, M]
                # 2. 重排坐标：根据排序索引提取排序后的坐标
                # coords_topN_flat: [N, M, 4]
                # 使用 gather 或者高级索引进行重排
                batch_indices = torch.arange(N_samples, device=self.device).unsqueeze(1).expand(-1, M_candidates)
                coords_sorted = coords_topN_flat[batch_indices, sorted_indices]  # [N, M, 4]

                # ==================== 可选：对TopK候选进行梯度优化重排序 ====================
                enable_gradient_refinement = False  # 设置为True启用梯度优化
                topK_for_refinement = 20  # 对前K个候选进行优化
                n_optimization_steps = 200  # 优化步数
                if enable_gradient_refinement:
                    print(f"\n{'='*60}")
                    print(f"对Top{topK_for_refinement}候选进行梯度优化重排序 (优化步数={n_optimization_steps})")
                    print(f"{'='*60}")

                    # 提取Top-K候选
                    K = min(topK_for_refinement, M_candidates)
                    coords_topK = coords_sorted[:, :K, :]  # [N, K, 4]

                    # 调用优化函数
                    coords_topK_optimized = self._opt_coords_topN(
                        coords_topK,
                        feats_vis_for_refine,
                        n_step=200,
                    )  # [N, K, 4] - 返回已经按优化后的loss排序的坐标

                    # 将优化后的TopK替换回原coords_sorted
                    coords_sorted = torch.cat([
                        coords_topK_optimized,  # [N, K, 4] 优化后的TopK
                        coords_sorted[:, K:, :]  # [N, M-K, 4] 剩余的未优化候选
                    ], dim=1)  # [N, M, 4]

                    print(f"✅ 梯度优化完成，Top{K}候选已重新排序")
                    print(f"{'='*60}\n")

                # 3. 计算所有排序后候选点的误差
                # 扩展 GT 以便广播: [N, 4] -> [N, 1, 4]
                coords_gt_expanded = coords_gt_for_refine.unsqueeze(1)

                # 3.1 2D 位置误差 [N, M]
                dist_errors_all = torch.norm(
                    coords_sorted[..., :2] - coords_gt_expanded[..., :2],
                    dim=-1, p=2
                )

                # 3.2 旋转误差 (弧度) [N, M]
                rot_diff_all = torch.abs(coords_sorted[..., 2] - coords_gt_expanded[..., 2])
                rot_errors_all = torch.min(rot_diff_all, 2 * np.pi - rot_diff_all)
                rot_errors_deg_all = torch.rad2deg(rot_errors_all)

                # ==================== 分析结果 ====================
                # 原有的 "Best Match" (Top-1) 统计保持不变，直接取第0个即可
                best_dist_err = dist_errors_all[:, 0]  # [N]
                best_rot_err_deg = rot_errors_deg_all[:, 0]  # [N]

                print(f"\nTopN精细检索结果 (Best Match / Top-1):")
                print(f"  候选点总数(M): {M_candidates}")
                print(f"  2D位置误差 - 平均: {best_dist_err.mean().item():.4f}")
                print(f"  2D位置误差 - 中位数: {torch.median(best_dist_err).item():.4f}")
                print(f"  旋转误差 - 平均: {best_rot_err_deg.mean().item():.2f}°")
                print(f"  旋转误差 - 中位数: {torch.median(best_rot_err_deg).item():.2f}°")

                # ==================== 新增：Top-K Recall 分析 ====================
                # 定义成功的阈值 (你可以作为函数参数传入)
                dist_lambda = 1.0
                SUCCESS_DIST_THRESH = self.sat_dataset.halfimg_radius_nrc*dist_lambda # 例如 3米 (根据你的归一化坐标尺度调整)
                SUCCESS_ROT_THRESH = 10.0  # 例如 30度

                # 判定每个候选点是否合格 [N, M] (Bool)
                is_hit = (dist_errors_all <= SUCCESS_DIST_THRESH) & (rot_errors_deg_all <= SUCCESS_ROT_THRESH)

                # 检查 Top-K 中是否有任意一个合格 (Recall@K / Hit Rate@K)
                ks_to_check = [1, 5, 10, 20]  # 检查前1, 5, 10, 20个
                ks_to_check = [k for k in ks_to_check if k <= M_candidates]  # 确保不越界

                print(f"\nTop-K 召回率分析 (阈值: NormalizedDist<={SUCCESS_DIST_THRESH:.3f},GeoDist<={self.sat_dataset.halfimg_radius_meter:.2f}m,Rot<={SUCCESS_ROT_THRESH}°):")
                print(f"{'K':<5} | {'Recall (%)':<10} | {'Mean Dist (of Hits)':<20}")
                print("-" * 45)

                for k in ks_to_check:
                    # 取前k列，检查每行是否有True
                    # any(dim=1) -> [N]
                    has_hit_in_k = is_hit[:, :k].any(dim=1)
                    recall_k = has_hit_in_k.float().mean().item() * 100

                    # 统计命中样本的平均距离误差 (只统计命中的)
                    if has_hit_in_k.sum() > 0:
                        # 这里取 Top-K 中 *最好的* 那个满足条件的点的误差，或者简单起见，取 Top-1 的误差
                        # 更严谨的做法：如果命中了，通常取 Top-K 中第一个命中的点的误差
                        # 这里为了简单展示，打印 Recall 即可
                        pass

                    print(f"{k:<5} | {recall_k:<10.2f}")

                print("=" * 60 + "\n")
                # 可视化topN精细检索（调试用）
                if False:  # 设置为False可关闭可视化
                    vis_idx=1
                    from trainer_depends.utils.util_vis_filelds import visualize_coarse_partition_interactive
                    p2save = os.path.join(self._get_dir2save(),f'coarse_partition_3d_{vis_idx}.html')
                    visualize_coarse_partition_interactive(
                        self.subspace_sampler.coord_ranges,
                        self.subspace_sampler.n_coarse,
                        coord_gt=coords_gt_all[vis_idx],
                        candidates=coords_topN_flat[vis_idx],
                        energies=energys_topN[vis_idx],
                        save_path=p2save,
                    )

    def _opt_coords_topN(self, coords_topN, feat_q, n_step=500,lr=1e-5):
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
                    coords_6d = self.coord_normer.raw_to_norm(coords2opt, append_linear_rot=True)

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
                    print(f"  候选 {id + 1}/{coords_topN.shape[1]}, Step {i}/{n_step}, Loss: {loss.item():.6f}")

            # 保存最终结果
            with torch.no_grad():
                coords_final = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                # 计算最终loss
                coords_6d_final = self.coord_normer.raw_to_norm(coords_final, append_linear_rot=True)
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

    def _opt_coords_topN_per_batch(self, coords_topN, feat_q, n_step=100):
        """
        对TopN候选坐标进行梯度优化 (逐样本 Sample-wise 版本)
        逻辑：外层循环遍历 Batch，每次优化单独一个样本的所有候选点。

        Args:
            coords_topN: [B, N, 4] - TopN候选坐标
            feat_q: [B, C] - query特征
            n_step: int - 优化步数

        Returns:
            coords_sorted: [B, N, 4] - 按优化后的loss排序的坐标
        """
        B, N, D = coords_topN.shape

        # 1. 准备数据
        feat_q = feat_q.detach().to(self.device)

        # 结果容器
        coords_sorted_list = []

        # 2. 临时开启训练模式 (为了构建计算图)
        grid_was_training = self.grid.training
        grid_mlp_was_training = self.grid_mlp.training
        pos_encoder_was_training = self.pos_encoder_grid.training
        self.grid.train()
        self.grid_mlp.train()
        self.pos_encoder_grid.train()

        print(f"  开始梯度优化 (Mode: Sample-wise Loop, Batch={B}, TopN={N})...")

        # =======================================================
        # 外层循环：遍历 Batch 中的每一个样本 (i from 0 to B-1)
        # =======================================================
        for b in range(B):
            # --- A. 准备当前样本的数据 ---
            # 取出第 b 个样本的所有候选点: [N, 4]
            coords_init = coords_topN[b].clone().detach()

            # 取出第 b 个样本的目标特征: [C] -> 扩展为 [N, C] 以匹配候选点数量
            target_feat = feat_q[b].unsqueeze(0).expand(N, -1)

            # --- B. 初始化参数 (针对这 N 个点) ---
            xy_param = coords_init[:, :2].clone().requires_grad_(True)
            rot_param = coords_init[:, 2:3].clone().requires_grad_(True)
            scale_param = coords_init[:, 3:4].clone().requires_grad_(True)

            # --- C. 优化器 ---
            optimizer = torch.optim.Adam([
                {'params': [xy_param], 'lr': 1e-4},
                {'params': [rot_param], 'lr': 1e-3},
                {'params': [scale_param], 'lr': 5e-4}
            ], lr=1e-3)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_step, eta_min=1e-5
            )

            # --- D. 优化循环 ---
            for step in range(n_step):
                optimizer.zero_grad()

                with torch.enable_grad():
                    # 组合参数 [N, 4]
                    coords2opt = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                    # 归一化 & 6D转换
                    coords_6d = self.coord_normer.raw_to_norm(coords2opt, append_linear_rot=True)

                    # 准备网络输入
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)

                    # 前向传播 (此时 Batch Size = N)
                    feat_ref_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                    feat_ref = self.grid_mlp(feat_ref_raw, coords_encoded)
                    feat_ref = TF.normalize(feat_ref, dim=-1, p=2)

                    # 计算 Loss [N, C] vs [N, C] -> scalar
                    loss = TF.mse_loss(target_feat, feat_ref)

                    loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_([xy_param, rot_param, scale_param], max_norm=1.0)

                optimizer.step()
                scheduler.step()

            # --- E. 整理当前样本的结果 ---
            with torch.no_grad():
                # 获取最终坐标 [N, 4]
                coords_final = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                # 重新计算逐点 Loss 用于排序
                coords_6d_final = self.coord_normer.raw_to_norm(coords_final, append_linear_rot=True)
                grid_input_final = torch.cat([coords_6d_final[:, :2], coords_6d_final[:, -1:]], dim=-1)
                feat_ref_final = self.grid_mlp(
                    self._get_feats_fm_grid(grid_input_final),
                    self.pos_encoder_grid(coords_6d_final[:, :5])
                )
                feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)

                # 计算 MSE [N]
                final_loss_sample = TF.mse_loss(target_feat, feat_ref_final, reduction='none').mean(dim=-1)

                # 排序
                sorted_indices = final_loss_sample.argsort()  # [N]
                coords_sorted_sample = coords_final[sorted_indices]  # [N, 4]

                coords_sorted_list.append(coords_sorted_sample)

            if (b + 1) % 10 == 0:
                print(f"  已完成样本优化 {b + 1}/{B}")

        # 3. 恢复模型状态
        if not grid_was_training: self.grid.eval()
        if not grid_mlp_was_training: self.grid_mlp.eval()
        if not pos_encoder_was_training: self.pos_encoder_grid.eval()

        # 4. 堆叠结果 [B, N, 4]
        coords_sorted = torch.stack(coords_sorted_list, dim=0)

        return coords_sorted

    def _opt_coords_topN_fast(self, coords_topN, feat_q, n_step=200):
        """
        对TopN候选坐标进行梯度优化重排序 (全并行版)

        Args:
            coords_topN: [B, N, 4] - TopN候选坐标 (nr, nc, rot, scale)
            feat_q: [B, C] - query特征
            n_step: int - 优化步数

        Returns:
            coords_sorted: [B, N, 4] - 按优化后的loss排序的坐标
        """
        import math

        B, N, D = coords_topN.shape
        C = feat_q.shape[-1]

        # 1. 数据准备：全部展平，利用GPU并行计算
        # feat_q: [B, C] -> [B, N, C] -> [B*N, C]
        # 这里的逻辑是：第 b 个样本的所有 N 个候选点，都应该去匹配第 b 个 query 特征
        feat_q_expanded = feat_q.unsqueeze(1).expand(B, N, C).reshape(-1, C)
        feat_q_expanded = feat_q_expanded.detach().to(self.device)

        # coords_topN: [B, N, 4] -> [B*N, 4]
        coords_init_flat = coords_topN.reshape(-1, 4).clone().detach()

        # 2. 临时开启训练模式 (为了构建计算图)
        grid_was_training = self.grid.training
        grid_mlp_was_training = self.grid_mlp.training
        pos_encoder_was_training = self.pos_encoder_grid.training
        self.grid.train()
        self.grid_mlp.train()
        self.pos_encoder_grid.train()

        # 3. 初始化参数 (针对 B*N 个点同时优化)
        # 分别创建参数以便设置不同学习率
        xy_param = coords_init_flat[:, :2].clone().requires_grad_(True)
        rot_param = coords_init_flat[:, 2:3].clone().requires_grad_(True)
        scale_param = coords_init_flat[:, 3:4].clone().requires_grad_(True)

        # 优化器设置
        optimizer = torch.optim.Adam([
            {'params': [xy_param], 'lr': 1e-4},
            {'params': [rot_param], 'lr': 1e-3},
            {'params': [scale_param], 'lr': 5e-4}
        ], lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_step, eta_min=1e-5
        )

        # 4. 优化循环 (只跑一次循环，并行处理所有 B*N 个点)
        for i in range(n_step):
            optimizer.zero_grad()

            with torch.enable_grad():
                # 组合参数 [B*N, 4]
                coords2opt = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                # 归一化坐标 & 增加线性旋转 [B*N, 6]
                coords_6d = self.coord_normer.raw_to_norm(coords2opt, append_linear_rot=True)

                # 准备网络输入
                grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)

                # --- 前向传播 ---
                # 注意：这里调用的所有函数输入都是 [B*N, ...]，Batch大小变大了N倍，但对网络来说只是更大的Batch而已
                feat_ref_raw = self._get_feats_fm_grid(grid_input)
                coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                feat_ref = self.grid_mlp(feat_ref_raw, coords_encoded)
                feat_ref = TF.normalize(feat_ref, dim=-1, p=2)

                # 计算 Loss [B*N]
                # feat_q_expanded 也是 [B*N, C]
                loss = TF.mse_loss(feat_q_expanded, feat_ref)

                loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([xy_param, rot_param, scale_param], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            if i % 50 == 0:
                print(f"  Gradient Refine Step {i}/{n_step}, Avg Loss: {loss.item():.6f}")

        # 5. 整理结果
        with torch.no_grad():
            # 获取最终坐标 [B*N, 4]
            coords_final_flat = torch.cat([xy_param, rot_param, scale_param], dim=-1)

            # 重新计算一次最终的 Loss 用于排序 (逐点计算)
            # 重复前向传播逻辑
            coords_6d_final = self.coord_normer.raw_to_norm(coords_final_flat, append_linear_rot=True)
            grid_input_final = torch.cat([coords_6d_final[:, :2], coords_6d_final[:, -1:]], dim=-1)
            feat_ref_final_raw = self._get_feats_fm_grid(grid_input_final)
            coords_encoded_final = self.pos_encoder_grid(coords_6d_final[:, :5])
            feat_ref_final = self.grid_mlp(feat_ref_final_raw, coords_encoded_final)
            feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)

            # 计算每个点的 MSE: [B*N]
            final_loss_flat = TF.mse_loss(feat_q_expanded, feat_ref_final, reduction='none').mean(dim=-1)

            # Reshape 回 [B, N]
            coords_opted_topN = coords_final_flat.view(B, N, 4)
            loss_topN = final_loss_flat.view(B, N)

            # 恢复模型状态
            if not grid_was_training: self.grid.eval()
            if not grid_mlp_was_training: self.grid_mlp.eval()
            if not pos_encoder_was_training: self.pos_encoder_grid.eval()

            # 6. 候选重排序
            # 对每个样本内部的 N 个候选点进行排序
            sorted_indices = loss_topN.argsort(dim=1)  # [B, N]

            # gather 排序后的坐标
            # 需要扩展索引维度以匹配坐标维度
            # batch_indices: [B, N] -> [[0,0...], [1,1...]]
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, N)
            coords_sorted = coords_opted_topN[batch_indices, sorted_indices]

            return coords_sorted

    def _visualize_single_sample_refinement(
        self,
        coords_samples,
        energys,
        coord_gt,
        coord_pred,
        sample_idx=None
    ):
        """
        可视化单个样本的topN精细检索过程

        Args:
            coords_samples: [M, 4] 所有采样点的坐标
            energys: [M] 采样点对应的能量值
            coord_gt: [4] GT坐标
            coord_pred: [4] 预测坐标
            sample_idx: (可选) 样本索引，用于文件名
        """
        import matplotlib.pyplot as plt
        import os
        import time

        # 转numpy
        coords_np = coords_samples.cpu().numpy()  # [M, 4]
        energys_np = energys.cpu().numpy()  # [M]
        gt_np = coord_gt.cpu().numpy()  # [4]
        pred_np = coord_pred.cpu().numpy()  # [4]

        # 创建figure (1行3列)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ===== 子图1: 2D空间视图 =====
        ax1 = axes[0]

        # 绘制采样点，用能量着色
        scatter = ax1.scatter(
            coords_np[:, 1], coords_np[:, 0],  # nc, nr
            c=energys_np, s=40, cmap='hot_r', alpha=0.6,
            edgecolors='black', linewidths=0.3,
            vmin=energys_np.min(), vmax=energys_np.max()
        )
        plt.colorbar(scatter, ax=ax1, label='Energy', fraction=0.046)

        # 标记GT位置
        ax1.plot(gt_np[1], gt_np[0], 'g*', markersize=20,
                markeredgecolor='white', markeredgewidth=2,
                label='GT', zorder=10)

        # 标记预测位置
        ax1.plot(pred_np[1], pred_np[0], 'rx', markersize=15,
                markeredgewidth=3, label='Pred (min E)', zorder=10)

        # 计算误差
        dist_error = np.linalg.norm(pred_np[:2] - gt_np[:2])
        title_prefix = f'Sample {sample_idx}' if sample_idx is not None else 'Sample'
        ax1.set_title(f'{title_prefix} - 2D Spatial View\n2D Error: {dist_error:.4f}')
        ax1.set_xlabel('NC (column)')
        ax1.set_ylabel('NR (row)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # ===== 子图2: 能量-距离关系 =====
        ax2 = axes[1]

        # 计算距离
        distances = np.linalg.norm(coords_np[:, :2] - gt_np[:2], axis=1)

        # 绘制散点
        scatter2 = ax2.scatter(
            distances, energys_np,
            c=energys_np, s=50, cmap='hot_r', alpha=0.6,
            edgecolors='black', linewidth=0.5
        )

        # 标记最小距离点
        min_dist_idx = distances.argmin()
        ax2.plot(distances[min_dist_idx], energys_np[min_dist_idx],
                'g*', markersize=15, markeredgecolor='white',
                markeredgewidth=1.5,
                label=f'Closest to GT (E={energys_np[min_dist_idx]:.3f})')

        # 标记最小能量点
        min_energy_idx = energys_np.argmin()
        ax2.plot(distances[min_energy_idx], energys_np[min_energy_idx],
                'rx', markersize=12, markeredgewidth=2.5,
                label=f'Min Energy (dist={distances[min_energy_idx]:.3f})')

        # 计算相关系数
        correlation = np.corrcoef(distances, energys_np)[0, 1]

        ax2.set_xlabel('Distance to GT')
        ax2.set_ylabel('Energy')
        ax2.set_title(f'Energy vs Distance\nCorr: {correlation:.3f}')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ===== 子图3: 能量-旋转误差关系 =====
        ax3 = axes[2]

        # 计算旋转误差
        rot_diff = np.abs(coords_np[:, 2] - gt_np[2])
        rot_errors = np.minimum(rot_diff, 2 * np.pi - rot_diff)  # 周期化
        rot_errors_deg = np.rad2deg(rot_errors)

        # 绘制散点
        scatter3 = ax3.scatter(
            rot_errors_deg, energys_np,
            c=energys_np, s=50, cmap='hot_r', alpha=0.6,
            edgecolors='black', linewidth=0.5
        )

        # 标记最小旋转误差点
        min_rot_idx = rot_errors.argmin()
        ax3.plot(rot_errors_deg[min_rot_idx], energys_np[min_rot_idx],
                'g*', markersize=15, markeredgecolor='white',
                markeredgewidth=1.5,
                label=f'Best Rot (E={energys_np[min_rot_idx]:.3f})')

        # 标记最小能量点的旋转误差
        pred_rot_error = rot_errors_deg[min_energy_idx]
        ax3.plot(pred_rot_error, energys_np[min_energy_idx],
                'rx', markersize=12, markeredgewidth=2.5,
                label=f'Min Energy (rot={pred_rot_error:.1f}°)')

        # 计算相关系数
        rot_correlation = np.corrcoef(rot_errors, energys_np)[0, 1]

        ax3.set_xlabel('Rotation Error (degrees)')
        ax3.set_ylabel('Energy')
        ax3.set_title(f'Energy vs Rotation\nCorr: {rot_correlation:.3f}')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存到checkpoint文件夹
        # 区分训练和测试模式：
        # - 训练时：使用 self.exp_dir2save（当前实验目录）
        # - 测试时：使用 opt.load2test 指定的checkpoint目录
        save_dir = self._get_dir2save()

        # 生成文件名
        if sample_idx is not None:
            filename = f'topN_sample_{sample_idx}.png'
        else:
            timestamp = int(time.time() * 1000) % 10000
            filename = f'topN_sample_{timestamp}.png'

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if sample_idx is not None:
            print(f"   Sample {sample_idx}: {save_path}")
        else:
            print(f"   Saved: {save_path}")

    def _run_epoch_test_evaluation(self, epoch, n_samples=256, use_train_uav=False):
        """
        在每个epoch结束后运行测试评估并记录结果

        Args:
            epoch: 当前epoch编号
            n_samples: 测试样本数量
            use_train_uav: 是否使用训练集UAV数据
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch} 测试评估")
        self.logger.info(f"{'='*60}")

        # 切换到评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 运行3D分类测试
        results_3d = self._test_3d_classification_accuracy(
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            temperature=self.energy_temperature
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
        self.logger.info(f"{'='*60}\n")

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

        # 恢复训练模式
        for model in self.param2optimize.values():
            model.train()

        return results_3d

    def visualize_energy_field_local(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                                n_samples_per_dim=32, delta=0.2, rot_span=torch.pi,
                                surface_min_ratio=0.2, adaptive_z_scale=False,
                                save_path="vis_results/energy_field.html", show_plot=False,
                                use_train_uav=False):
        """
        能量场综合可视化（改造自 visualize_comprehensive_udf）

        将能量场视为UDF场进行可视化，在一个 HTML 文件中并排显示三个子图：
        1. Energy 场值散点图 (Scatter): 查看采样点的数值分布
        2. Energy 梯度流场 (Cone): 查看梯度下降的方向
        3. Energy 等值面 (Isosurface): 查看场的几何拓扑结构

        Args:
            query_feat: 查询特征 [1, C]
            gt_coord_4d: ground truth 4D坐标 [4]
            scale_fixed: 固定的scale值
            n_samples_per_dim: 每个维度的采样点数
            delta: NR和NC的采样范围（相对于GT的偏移）
            rot_span: Rotation的采样范围
            surface_min_ratio: 等值面分界比例
            adaptive_z_scale: 是否自适应缩放Z轴（旋转维度）
            save_path: 保存路径
            show_plot: 是否显示图形
            use_train_uav: 是否使用训练集数据
        """
        import numpy as np
        import os
        import torch
        import torch.nn.functional as TF
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 0. 模式切换
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 1. 数据准备
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

        # 2. 构建采样网格 (Meshgrid)
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

        # 3. 推理与梯度计算 (一次性完成)
        # 为了计算梯度，必须开启 requires_grad
        coords_sampled_4d.requires_grad_(True)
        coords_sampled_6d = self.coord_normer.raw_to_norm(coords_sampled_4d, append_linear_rot=True)

        # --- Forward: 提取Grid特征 ---
        grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
        feats_grid_raw = self._get_feats_fm_grid(grid_input)
        coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
        feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
        feats_grid = TF.normalize(feats_grid, dim=-1)  # [N, C]

        # --- 计算能量场（使用Projector） ---
        # projector.compute_energy(query_feats, ref_feats) -> [B, N]
        # query_feat: [1, C], feats_grid: [N, C]
        energy_pred = self.projector.compute_energy(
            query_feat, feats_grid, metric='euclidean'
        ).squeeze(0)  # [N]

        # --- Backward (Gradient) ---
        grad_outputs = torch.ones_like(energy_pred)
        gradients = torch.autograd.grad(energy_pred, coords_sampled_4d, grad_outputs, create_graph=False)[0]

        grad_r = -gradients[:, 0]
        grad_c = -gradients[:, 1]
        grad_rot = -gradients[:, 2]

        # 梯度自适应放大 (仅用于显示)
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

        # 4. 转 Numpy
        coords_np = coords_sampled_4d.detach().cpu().numpy()
        X = coords_np[:, 0]
        Y = coords_np[:, 1]
        Z = coords_np[:, 2]

        V = energy_pred.detach().cpu().numpy()
        G_vec = grad_vec_norm.detach().cpu().numpy()

        min_v, max_v = V.min(), V.max()
        print(f"📊 统计: Min Energy={min_v:.6f}, Max Energy={max_v:.6f}, Z-Scale={z_amplification:.1f}x")

        # 关键点
        gt_r, gt_c, gt_rot = nr_center.item(), nc_center.item(), rot_center.item()
        best_idx = np.argmin(V)
        pred_r, pred_c, pred_rot = X[best_idx], Y[best_idx], Z[best_idx]

        # 5. 创建子图画布 (1行3列)
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f'场值散点 (Min={min_v:.2f})', '梯度流场 (Descent)', '等值面结构 (Geometry)'),
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.02
        )

        # === 子图 1: Scatter (散点场值) ===
        # 降采样一下散点图，防止浏览器卡死
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

        # === 子图 2: Cone (梯度流场) ===
        # 计算合适的 Cone 大小
        step_r = (X.max() - X.min()) / n_samples_per_dim
        cone_scale = step_r * 8.0

        fig.add_trace(go.Cone(
            x=X[mask], y=Y[mask], z=Z[mask],
            u=G_vec[mask, 0], v=G_vec[mask, 1], w=G_vec[mask, 2],
            sizemode="absolute", sizeref=cone_scale, anchor="tail",
            colorscale='Jet', showscale=False, opacity=0.7,
            name='Gradients'
        ), row=1, col=2)

        # === 子图 3: Isosurface (等值面) ===
        # 设定分界线
        val_range = max_v - min_v
        split_val = min_v + val_range * surface_min_ratio

        # 第一层：核心细节（低能量区）
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

        # 第二层：全局概览（高能量区）
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

        # === 通用标记: GT 和 Pred Min ===
        for col in [1, 2, 3]:
            # GT
            fig.add_trace(go.Scatter3d(
                x=[gt_r], y=[gt_c], z=[gt_rot],
                mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
                showlegend=(col == 1), name='GT'
            ), row=1, col=col)
            # Pred Min
            fig.add_trace(go.Scatter3d(
                x=[pred_r], y=[pred_c], z=[pred_rot],
                mode='markers', marker=dict(size=6, color='yellow', symbol='x'),
                showlegend=(col == 1), name='Min'
            ), row=1, col=col)

        # === 统一 Layout 设置 ===
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


    def test(self,use_train_uav=False):
        """
        Stage 3测试函数
        """
        print("\n" + "🧪"*40)
        print("开始 Stage 3 测试: Projector")
        print("🧪"*40 + "\n")

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
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)

        from trainer_depends.datasets.util_subspace_sampler import SubspaceSampler
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

        results_3d = self._test_3d_fine_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature= self.energy_temperature,
            # temperature=0.5,
            save_pred_pdf=True,
        )

        # 6. 运行3D分类测试 (NR, NC, Rot)
        results_3d = self._test_3d_classification_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature= self.energy_temperature,
            # temperature=0.5,
            save_pred_pdf=True,
        )

        # 5. 运行2D分类测试
        results_2d = self._test_2d_classification_accuracy(
            n_samples=256,
            use_train_uav=use_train_uav,
            temperature= self.energy_temperature,
        )



        # results_seq = self._test_2d_sequence_localization_accuracy(
        #     n_samples=None,          # 使用全部测试数据
        #     use_train_uav=use_train_uav,
        #     temperature=self.energy_temperature,
        #     seq_window_len=4,        # 序列聚合窗口长度
        #     len_neighbors=3,         # 2×2邻域
        #     shuffle=False,            # 序列测试不打乱顺序
        #     save_pred_pdf=False,
        # )

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
            viz_save_path = f"vis_results/energy_field_{exp_name}_epoch{epoch_num}_{dataset_type}.html"
        else:
            # 如果没有找到checkpoint路径，使用默认名称
            dataset_type = 'train' if use_train_uav else 'test'
            viz_save_path = f"vis_results/energy_field_{dataset_type}.html"

        print(f"\n📊 生成能量场可视化: {viz_save_path}")
        self.visualize_energy_field_local(
            save_path=viz_save_path,
            use_train_uav=use_train_uav
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
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam')

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
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)

        # 初始化子空间采样器（替代原来的 coord_sampler 和 udf_computer）
        from trainer_depends.datasets.util_subspace_sampler import SubspaceSampler
        self.n_subspaces_to_sample = getattr(opt, 'n_subspaces_to_sample', 1024)
        self.n_points_per_subspace = getattr(opt, 'n_points_per_subspace', 1)
        self.infonce_temperature = getattr(opt, 'infonce_temperature', 1.0)
        self.subspace_sampler = SubspaceSampler(
            sat_dataset=self.sat_dataset,
            n_coarse=self.n_coarse,
            n_fine_per_coarse=self.n_fine_per_coarse
        )
        print(f"子空间采样器初始化完成: {self.subspace_sampler}")

        # 6. 训练循环
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
                BatchSize = coords_gt.shape[0]

                # =================== 2. 获取子空间标签 ===================
                # anchor_labels = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [2B]
                #v2：sample all
                coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(n_points_per_subspace=1,use_fine=False)
                # subspade_info = self.subspace_sampler.get_subspace_info()

                # =================== 4. 特征提取 ===================
                coords_flat = coords_candidates.view(-1, 4)  # [2B * N, 4]

                # 坐标归一化
                coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)

                # Grid 特征
                grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)
                feats_grid_raw = self._get_feats_fm_grid(grid_input)

                # Grid MLP
                coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
                feats_grid_flat = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                feats_grid_flat = TF.normalize(feats_grid_flat, dim=-1)

                # 计算预测能量分布
                shape_config = [-1, *self.subspace_sampler.n_coarse]
                energys = self.projector.compute_energy(feats_vis,feats_grid_flat,metric='euclidean')
                energys_5d = energys.reshape(shape_config)
                neg_dists = -energys_5d
                temperature = self.energy_temperature
                scaled_logits_5d = neg_dists / temperature #这是 Logits（未归一化的对数概率）, $-E/T$ 本身就是概率的对数形式
                # 在 Scale 维度 (dim=-1) 做 LogSumExp -> 得到 [B, H, W, R];物理含义：只要有一个 Scale 对得上，这个 (Pos, Rot) 组合就是好的
                logits_4d = torch.logsumexp(scaled_logits_5d, dim=-1) #利用负能量直接作为对数概率

                # =======================================================
                # 步骤 B: 空间 Loss (Spatial Loss) - 保持你原来的逻辑
                # =======================================================
                # 1. 边缘化 Rot 维度 -> 得到 [B, H, W]
                logits_spatial = torch.logsumexp(logits_4d, dim=-1)
                # 2. 计算空间 Log Prob
                B, H, W = logits_spatial.shape
                log_probs_2d = F.log_softmax(logits_spatial.view(B, -1), dim=-1).reshape(B, H, W)
                # 3. 计算空间 Target (和你原来的一样)
                distmat_q2ref = torch.norm(coords_gt.unsqueeze(1)[..., :2] - coords_candidates.reshape(1, coords_candidates.shape[0], 4)[...,:2], dim=-1, p=2)
                # 注意：这里 reshape 要和 logits_4d 对应，取 mean 是把 R, S 维度平均掉
                distmat_q2ref_2d = distmat_q2ref.reshape(shape_config).mean(dim=[-2, -1])
                sigma_2d = 0.5*0.65 * (self.subspace_sampler.coarse_bin_sizes[0] + self.subspace_sampler.coarse_bin_sizes[1])
                target_probs_2d = torch.exp(- (distmat_q2ref_2d ** 2) / (2 * sigma_2d ** 2))
                # 归一化空间 Target
                target_probs_2d = target_probs_2d / (target_probs_2d.view(B, -1).sum(dim=-1).view(B, 1, 1) + 1e-8)

                loss_spatial = F.kl_div(log_probs_2d, target_probs_2d, reduction='batchmean')

                # from matplotlib import pyplot as plt
                # fig, axs = plt.subplots(1, 2)  # 1行2列的子图布局
                # axs[0].imshow(torch.exp(log_probs_2d[0]).detach().cpu().numpy())
                # axs[1].imshow(target_probs_2d[0].detach().cpu().numpy())
                # plt.show()
                KL_first=False
                if KL_first:
                    # =======================================================
                    # 步骤 C (新版): 旋转 Loss (Rot Loss) - 局部 KL 加权求和
                    # =======================================================

                    # 1. 准备局部 Log 概率: [B, H, W, R]
                    # 在 R 维度上做 LogSoftmax，得到每个 grid 的局部旋转分布
                    log_probs_rot_local = F.log_softmax(logits_4d, dim=-1)

                    # 2. 准备局部 Target 分布: [B, H, W, R]
                    # (这段代码复用之前的逻辑，计算每个 grid 的理想旋转分布)
                    coords_5d = coords_candidates.reshape(*shape_config, 4)
                    cand_rot_map = coords_5d[..., 0, 2]  # [B, H, W, R]
                    gt_rot = coords_gt[:, 2].view(B, 1, 1, 1)

                    rot_diff = torch.abs(gt_rot - cand_rot_map)
                    circular_dist_map = torch.min(rot_diff, 2 * np.pi - rot_diff)

                    rot_bin_width_rad = torch.tensor(360.0 / self.subspace_sampler.n_coarse[2] * np.pi / 180.0,
                                                     dtype=torch.float32, device=self.device)
                    sigma_rot = 0.65 * rot_bin_width_rad

                    # 计算未归一化的高斯目标
                    target_probs_rot_local_raw = torch.exp(-(circular_dist_map ** 2) / (2 * sigma_rot ** 2))
                    # 归一化 Target (在 R 维度归一化)
                    target_probs_rot_local = target_probs_rot_local_raw / (
                                target_probs_rot_local_raw.sum(dim=-1, keepdim=True) + 1e-8)

                    # 3. 计算“逐点” KL 散度: [B, H, W]
                    # PyTorch 的 kl_div 默认是 sum 或 mean，我们要保留空间维度，所以手动实现或者设置 reduction='none'
                    # KL(Q || P) = sum(Q * (log Q - log P))，通常 P 是预测(log输入)，Q 是 Target
                    # F.kl_div(input=log_probs, target=probs, reduction='none') 返回同形状的 [B, H, W, R]
                    kl_map_per_bin = F.kl_div(log_probs_rot_local, target_probs_rot_local, reduction='none')

                    # 在 R 维度求和，得到每个空间点的 KL 值: [B, H, W]
                    kl_map_spatial = kl_map_per_bin.sum(dim=-1)

                    # 4. 空间加权求和
                    # 使用 target_probs_2d (空间 GT) 作为权重
                    # target_probs_2d: [B, H, W], 已经在空间维度归一化了 sum(HW)=1

                    # 我们希望 Loss 只关注 GT 附近区域的旋转是否正确
                    # 如果某处 target_probs_2d 接近 0 (远离 GT)，那里的旋转预测错了也没关系
                    loss_rot = (kl_map_spatial * target_probs_2d).sum(dim=[-1, -2]).mean()  # 先在空间求和，再在 Batch 求平均
                else:
                    # =======================================================
                    # 步骤 C: 旋转 Loss (Rot Loss) - 聚合与归一化
                    # =======================================================
                    # 方法：用 target_probs_2d (空间 GT) 对 logits_4d 进行加权求和
                    # 先计算每个 grid 局部的旋转概率 P(rot | h, w) -> [B, H, W, R]
                    probs_rot_local = F.softmax(logits_4d, dim=-1)
                    # 准备权重: [B, H, W, 1]
                    # target_probs_2d 已经在空间上归一化了(sum=1)，直接用作权重即可
                    spatial_weights = target_probs_2d.unsqueeze(-1)
                    # 加权聚合: sum_{h,w} ( P(rot|h,w) * Weight(h,w) ) -> [B, R]
                    # 物理含义：只有在 GT 位置附近的格子，才有资格对旋转进行投票
                    probs_rot_aggregated = (probs_rot_local * spatial_weights).sum(dim=[1, 2])
                    # 转回 Log 域 (为了算 KL) -> [B, R]
                    log_prob_rot_final = torch.log(probs_rot_aggregated + 1e-8)

                    # 2. 构造目标的 Rot 分布
                    # 坐标重塑 -> [B, H, W, R, S, 4]
                    coords_5d = coords_candidates.reshape(*shape_config, 4)
                    cand_rot_map = coords_5d[..., 0, 2]
                    gt_rot = coords_gt[:, 2].view(B, 1, 1, 1)
                    # 3. 计算“逐点”的圆周距离
                    rot_diff = torch.abs(gt_rot - cand_rot_map)
                    circular_dist_map = torch.min(rot_diff, 2 * np.pi - rot_diff)  # [B, H, W, R]
                    # 4. 计算“逐点”的 GT 概率 (Local GT Likelihood)
                    rot_bin_width_rad = torch.tensor(360.0 / self.subspace_sampler.n_coarse[2] * np.pi / 180.0, dtype=torch.float32, device=self.device)  # 30度转弧度 ≈ 0.52
                    sigma_rot = 0.65 * rot_bin_width_rad
                    target_probs_rot_local = torch.exp(-(circular_dist_map ** 2) / (2 * sigma_rot ** 2))
                    # 5. 空间聚合 (Aggregation)
                    # 这一步是点睛之笔：我们用同样的 spatial_weights 来聚合 Target
                    target_probs_rot_aggregated = (target_probs_rot_local * spatial_weights).sum(dim=[1, 2])
                    target_probs_rot = target_probs_rot_aggregated / (target_probs_rot_aggregated.sum(dim=-1, keepdim=True) + 1e-8)

                    loss_rot = F.kl_div(log_prob_rot_final, target_probs_rot, reduction='batchmean')

                # 总损失
                loss = loss_spatial + loss_rot

                # energys_reshaped = energys_reshaped.sum(dim=-1)  #积分消除scale尺度
                # neg_dists = -energys_reshaped  # 取负号 (距离转相似度)
                # temperature = 0.5  # 除以温度 (缩放)
                # scaled_logits = neg_dists / temperature
                #
                # logits_2d = torch.logsumexp(scaled_logits, dim=[-1]) # LogSumExp (真正的边缘化：只要有一个好，整体就好)
                # log_probs_2d = F.log_softmax(logits_2d.view(N_candidates, -1), dim=-1).reshape(*logits_2d.shape)
                #
                # distmat_q2ref = torch.norm(coords_gt.unsqueeze(1)[...,:2]-coords_candidates.reshape(1,coords_candidates.shape[0],4)[...,:2],dim=-1,p=2)
                # distmat_q2ref_2d = distmat_q2ref.reshape(-1,*self.subspace_sampler.n_coarse).mean([-1])
                # sigma_2d = 0.5*(self.subspace_sampler.coarse_bin_sizes[0]+self.subspace_sampler.coarse_bin_sizes[1])  # 这个值非常关键！单位要和 distmat 一致（比如米）
                # target_probs_2d = torch.exp(- (distmat_q2ref_2d ** 2) / (2 * sigma_2d ** 2))
                # target_probs_2d = target_probs_2d / (target_probs_2d.sum(dim=[-1], keepdim=True) + 1e-8)
                #
                # loss = F.kl_div(log_probs_2d, target_probs_2d, reduction='batchmean')
                #
                # rot_diff_mat = coords_gt[:,2].unsqueeze(1)-coords_candidates[:,0,2].unsqueeze(0)
                # abs_diff = torch.abs(rot_diff_mat)
                # circular_dist = torch.min(abs_diff, 2 * np.pi - abs_diff)
                # rot_bin_width_rad = torch.tensor(360.0 / self.subspace_sampler.n_coarse[2] * np.pi / 180.0,dtype=torch.float32, device=self.device)  # 30度转弧度 ≈ 0.52
                # sigma_rot = 0.65 * rot_bin_width_rad
                # target_probs_rot = torch.exp(-(circular_dist ** 2) / (2 * sigma_rot ** 2))
                # target_probs_rot = target_probs_rot.reshape(*log_prob_rot.shape)
                #
                # log_prob_rot = F.log_softmax(scaled_logits.sum(dim=-1), dim=-1) # 2. 网络预测的旋转分布 (在 R 维度做 Softmax)


                # =================== 7. 反向传播 ===================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 日志记录
                if it % 10 == 0:
                    self.logger.info(
                        f'Iter {it}: Loss={loss.item():.6f} | '
                        f'Spatial={loss_spatial.item():.6f} | '
                        f'Rot={loss_rot.item():.6f}'
                    )
                    if self.writer is not None:
                        self.writer.add_scalar('loss/total', loss.item(), step)
                        self.writer.add_scalar('loss/spatial', loss_spatial.item(), step)
                        self.writer.add_scalar('loss/rot', loss_rot.item(), step)

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
