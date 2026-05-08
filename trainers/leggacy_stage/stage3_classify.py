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

from trainers.stage2_INGP import GridHashFitTrainer
from trainer_depends.base.components import NetworkComponents


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

        # 子空间采样器配置
        self.n_coarse = getattr(opt, 'n_coarse', (40, 30, 12, 1))  # 2304 类
        self.n_fine_per_coarse = getattr(opt, 'n_fine_per_coarse', (1, 1, 1, 1))  # 8 细分格子


    def _init_metric_net(self):
        """初始化DualStreamMetricNet"""
        print("\n" + "="*80)
        print("初始化 DualStreamMetricNet (双流架构)")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        # 初始化MetricNet专用的坐标编码器（独立于stage2的pos_encoder）
        # 可以设置不同的频率以获得更适合距离预测的特征
        metric_multires_rc = getattr(self.opt, 'metric_multires_rc', 6)  # 默认6，可在配置中调整
        metric_multires_rot = getattr(self.opt, 'metric_multires_rot', 4)  # 默认4
        metric_multires_scale = getattr(self.opt, 'metric_multires_scale', 3)  # 默认3
        self.pos_encoder_663 = components.create_coords_5d_encoder(
            multires_rc=6,
            multires_rot=6,
            multires_scale=3
        )
        self.pos_encoder_metric = self.pos_encoder_grid
        # self.pos_encoder_metric = self.pos_encoder_663

        print(f"MetricNet坐标编码器配置:")
        print(f"  - multires_rc: {metric_multires_rc}")
        print(f"  - multires_rot: {metric_multires_rot}")
        print(f"  - multires_scale: {metric_multires_scale}")
        print(f"  - 输出维度: {self.pos_encoder_metric.out_dim}")
        print(f"对比Stage2坐标编码器输出维度: {self.pos_encoder_grid.out_dim}\n")

        # 直接实例化 DualStreamMetricNet
        from models.metric_net_dual import DualStreamMetricNet
        self.metric_net = DualStreamMetricNet(
            feat_dim=self.feat_q_dim,
            coord_dim=self.pos_encoder_metric.out_dim,  # 使用新的编码器维度
            branch_hidden_dim=512,
            branch_output_dim=256,
            resblock_hidden_dim=256,
            resblock_output_dim=256,
            dropout=0.,
            init_weights=True,
            output_activation=None  # 不在模型内部应用激活，在train/test中手动控制
        ).to(self.device)

        # 初始化Softplus激活函数（用于距离预测的输出激活）
        self.softplus = torch.nn.Softplus(beta=1)
        self.use_softplus = True  # 设为False可以调试原始输出

        print("✅ DualStreamMetricNet 初始化完成")
        print("="*80 + "\n")


    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        for param in self.grid.parameters():
            param.requires_grad = False
        for param in self.grid_mlp.parameters():
            param.requires_grad = False

        self.param2optimize = {
            'metric_net': self.metric_net
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


    def _run_epoch_evaluation(self, epoch, run_visualization=True, n_test_samples=256):
        """
        在每个epoch结束时运行评估

        Args:
            epoch: 当前epoch编号
            run_visualization: 是否运行可视化（默认True）
            n_test_samples: 测试样本数量（默认64）
        """
        # 可视化相关输出只显示在控制台，不记录到日志
        print("\n" + "🔍" * 40)
        print(f"Epoch {epoch} 评估开始")
        print("🔍" * 40)

        # 1. 可视化（可选）
        if run_visualization:
            print("\n>>> 步骤1: UDF场可视化")
            use_train_uav = False
            name = 'train' if use_train_uav else 'test'
            viz_save_path = f"/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results/udf_epoch{epoch}_uav_{name}.html"

            # 保持训练模式进行可视化（以保持dropout等训练行为一致性）
            self.visualize_comprehensive_udf(
                save_path=viz_save_path,
                n_samples_per_dim=32,  # 建议30-40，太高会生成很慢且网页卡顿
                delta=0.15,  # 视野范围
                # surface_count=6,  # 等值面层数
                # opacity=0.2  # 透明度
                adaptive_z_scale=False,
            )
            print(f"✅ 可视化已保存: {viz_save_path}")

        # 2. 距离预测精度测试
        print("\n>>> 步骤2: 距离预测精度测试")
        self.logger.info("="*80)

        # 切换为eval模式进行测试
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 运行测试
        test_results = self._test_distance_prediction_accuracy_batch(
            n_samples=n_test_samples,
            n_candidates=2048,
            use_hierarchical_sampler=True,
            use_train_uav=use_train_uav
        )

        # 记录详细的测试结果到日志
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"【Epoch {epoch} 距离预测精度测试结果】(样本数: {n_test_samples})")
        self.logger.info(f"{'='*80}")

        # UDF值统计
        self.logger.info(f"\n【UDF值统计】")
        udf_diff_mean = test_results['gt_udf_mean'] - test_results['min_udf_mean']
        self.logger.info(f"  预测最小UDF:  均值={test_results['min_udf_mean']:.6f}, 中位数={test_results['min_udf_median']:.6f}")
        self.logger.info(f"  GT位置UDF:    均值={test_results['gt_udf_mean']:.6f}, 中位数={test_results['gt_udf_median']:.6f}")
        self.logger.info(f"  UDF差异:      {udf_diff_mean:.6f} (GT - Min, 理想值应接近0)")

        # GT相对位置分析
        self.logger.info(f"\n【GT位置质量评估】")
        self.logger.info(f"  GT相对位置比例: 均值={test_results['gt_udf_ratio_mean']:.4f}, 中位数={test_results['gt_udf_ratio_median']:.4f}")
        self.logger.info(f"                   (0=完美/最小值, 1=最差/最大值)")
        self.logger.info(f"  平均排名:        {test_results['gt_rank_mean']:.1f} (理想值=1)")

        # Top-K准确率
        self.logger.info(f"\n***候选点总数：{test_results['n_candidates']}***")
        self.logger.info(f"\n【Top-K准确率】(GT位置在排序中的排名)")
        self.logger.info(f"  Top-1:   {test_results['top1_accuracy']:6.2f}%")
        self.logger.info(f"  Top-10:  {test_results['top10_accuracy']:6.2f}%")
        self.logger.info(f"  Top-100: {test_results['top100_accuracy']:6.2f}%")

        # 性能评估
        top1_acc = test_results['top1_accuracy']
        if top1_acc >= 50:
            performance = "优秀 ✓✓✓"
        elif top1_acc >= 30:
            performance = "良好 ✓✓"
        elif top1_acc >= 10:
            performance = "一般 ✓"
        else:
            performance = "较差 ✗"
        self.logger.info(f"\n【整体性能评估】{performance}")

        self.logger.info(f"{'='*80}\n")

        # 3. UAV定位精度测试
        print("\n>>> 步骤3: UAV定位精度测试")
        self.logger.info("="*80)

        # 运行定位精度测试
        loc_results = self._test_localization_accuracy_batch(
            n_samples=n_test_samples,
            n_candidates=2048,
            use_hierarchical_sampler=True,
            use_train_uav=use_train_uav
        )

        # 记录详细的定位测试结果到日志
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"【Epoch {epoch} UAV定位精度测试结果】(样本数: {n_test_samples})")
        self.logger.info(f"{'='*80}")

        # RC定位误差统计
        self.logger.info(f"\n【RC定位误差】")
        self.logger.info(f"  归一化误差(NRC):  {loc_results['rc_norm_mean']:.4f}")
        self.logger.info(f"  物理误差(米):     {loc_results['rc_m_mean']:.2f} m")

        # 旋转误差统计
        self.logger.info(f"\n【旋转误差】")
        self.logger.info(f"  角度误差(度):     {loc_results['rot_deg_mean']:.2f}°")

        # Scale误差统计
        self.logger.info(f"\n【Scale误差】")
        self.logger.info(f"  Scale误差:        {loc_results['scale_error_mean']:.4f}")

        # 综合成功率
        self.logger.info(f"\n【综合成功率】")
        self.logger.info(f"  双指标成功率(RC+Rot): {loc_results['success_rate']:.2f}%")

        # 性能评估
        success_rate = loc_results['success_rate']
        if success_rate >= 80:
            loc_performance = "优秀 ✓✓✓"
        elif success_rate >= 60:
            loc_performance = "良好 ✓✓"
        elif success_rate >= 40:
            loc_performance = "一般 ✓"
        else:
            loc_performance = "较差 ✗"
        self.logger.info(f"\n【定位性能评估】{loc_performance}")

        self.logger.info(f"{'='*80}\n")

        # 恢复为train模式
        for model in self.param2optimize.values():
            model.train()

        print("\n" + "🔍" * 40)
        print(f"Epoch {epoch} 评估完成")
        print("🔍" * 40 + "\n")

        # 合并返回结果
        test_results['localization'] = loc_results
        return test_results


    def _test_distance_prediction_accuracy_batch(self, n_samples=128, n_candidates=2048, use_hierarchical_sampler=False,
                                           use_train_uav=False, test_batch_size=16):
        """
        [优化版] 测试MetricNet的UDF预测质量 (支持 Batch 推理以大幅提升速度)

        Args:
            test_batch_size: 批处理大小。用内存换速度的关键参数。
                             建议设为 16 或 32 (取决于显存)。
        """
        import numpy as np

        print(f"\n🚀 开始高效评估 (Batch Size: {test_batch_size})")
        print(f"测试样本数: {n_samples}")
        print(f"每个样本的候选位置数: {n_candidates}")
        print(f"采样策略: {'分层采样' if use_hierarchical_sampler else '均匀随机采样'}")

        # 1. 准备高效的数据加载器
        # 我们创建一个临时的 DataLoader，专门用于测试，使用较大的 batch_size
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 使用单进程避免多进程清理问题
        temp_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=0,  # 使用单进程避免多进程清理问题
            drop_last=False,
            pin_memory=True
        )

        # 初始化采样器
        if use_hierarchical_sampler:
            from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
            coord_sampler = create_hierarchical_sampler_from_dataset(
                sat_dataset=self.sat_dataset,
                bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
                num_uniform_samples=n_candidates,
                device=self.device
            )

        # 统计容器
        stats = {
            'min_udf': [], 'gt_udf': [], 'gt_rank': [], 'gt_ratio': []
        }

        processed_count = 0
        data_iter = iter(temp_loader)

        with torch.no_grad():
            # 使用 tqdm 显示进度
            pbar = tqdm.tqdm(total=n_samples, desc="Evaluated Samples")

            while processed_count < n_samples:
                # A. 获取 Batch 数据
                try:
                    batch_uav = next(data_iter)
                except StopIteration:
                    data_iter = iter(temp_loader)
                    batch_uav = next(data_iter)

                # 截断多余的数据 (如果只剩几个就满 n_samples 了)
                current_batch_size = batch_uav[0].shape[0]
                if processed_count + current_batch_size > n_samples:
                    current_batch_size = n_samples - processed_count
                    batch_uav[0] = batch_uav[0][:current_batch_size]
                    batch_uav[1] = batch_uav[1][:current_batch_size]

                uavimgs = batch_uav[0].to(self.device)  # [B, C, H, W]
                gt_coords_4d = batch_uav[1].to(self.device)  # [B, 4]
                B = uavimgs.shape[0]

                # B. 提取视觉特征 (Batch)
                # [B, feat_dim]
                query_feats = self._get_feats_fm_imgs(uavimgs)

                # C. 生成候选位置 (Batch)
                # 目标形状: [B, N_candidates, 4]
                if use_hierarchical_sampler:
                    # 采样器通常支持广播，输入 [B, 4] -> 输出 [B, N, 4]
                    coords_candidates = coord_sampler.sample(gt_coords_4d)
                else:
                    # 均匀采样: 生成一次，复制 B 份; 或者生成 B*N 个随机点
                    # 这里为了简单，生成 B*N 个点然后 reshape
                    flat_candidates = self.sat_dataset.mk_rand_coords_4d(
                        n_rand=n_candidates * B, return_tensor=True
                    ).to(self.device)
                    coords_candidates = flat_candidates.view(B, n_candidates, 4)

                # D. 拼接 GT + Candidates
                # GT: [B, 4] -> [B, 1, 4]
                # Result: [B, 1+N, 4]
                coords_all = torch.cat([gt_coords_4d.unsqueeze(1), coords_candidates], dim=1)
                N_total = coords_all.shape[1]

                # E. 展平以进行 Grid/MLP 处理
                # [B * (1+N), 4]
                coords_flat = coords_all.view(-1, 4)

                # --- 坐标处理流水线 (Flattened) ---
                # 1. Norm
                coords_flat_6d = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)

                # 2. Grid Feats
                grid_input = torch.cat([coords_flat_6d[:, :2], coords_flat_6d[:, -1:]], dim=-1)
                feats_grid_raw = self._get_feats_fm_grid(grid_input)

                # 3. Grid MLP
                coords_encoded_stage2 = self.pos_encoder_grid(coords_flat_6d[:, :5])
                feats_grid_flat = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                feats_grid_flat = TF.normalize(feats_grid_flat, dim=-1)

                # 4. Metric Encoding
                coords_encoded_metric_flat = self.pos_encoder_metric(coords_flat_6d[:, :5])

                # --- 重塑回 Batch 维度 ---
                # Feats Grid: [B, 1+N, feat_dim]
                feats_grid_batch = feats_grid_flat.view(B, N_total, -1)
                # Coords Enc: [B, 1+N, coord_dim]
                coords_enc_batch = coords_encoded_metric_flat.view(B, N_total, -1)

                # Query Feats: [B, feat_dim] -> [B, 1+N, feat_dim]
                query_feats_exp = query_feats.unsqueeze(1).expand(-1, N_total, -1)

                # F. MetricNet 前向 (Batch)
                # 输出: [B, 1+N]
                dist_pred_raw = self.metric_net(query_feats_exp, feats_grid_batch, coords_enc_batch)

                if self.use_softplus:
                    dist_pred = self.softplus(dist_pred_raw)
                else:
                    dist_pred = dist_pred_raw

                # G. 统计指标 (向量化计算)
                # GT 也就是 index 0
                gt_udfs = dist_pred[:, 0]  # [B]
                min_udfs, _ = dist_pred.min(dim=1)  # [B]
                max_udfs, _ = dist_pred.max(dim=1)  # [B]

                # Rank: 统计有多少个点比 GT 小
                # dist_pred < gt_udfs.unsqueeze(1) 会得到一个 bool 矩阵
                # sum(dim=1) 得到比 GT 小的点的数量，+1 就是排名
                ranks = (dist_pred < gt_udfs.unsqueeze(1)).sum(dim=1) + 1  # [B]

                # Ratio
                denominators = max_udfs - min_udfs
                denominators[denominators < 1e-8] = 1.0  # 避免除零
                ratios = (gt_udfs - min_udfs) / denominators

                # 收集结果
                stats['min_udf'].extend(min_udfs.cpu().numpy().tolist())
                stats['gt_udf'].extend(gt_udfs.cpu().numpy().tolist())
                stats['gt_rank'].extend(ranks.cpu().numpy().tolist())
                stats['gt_ratio'].extend(ratios.cpu().numpy().tolist())

                processed_count += B
                pbar.update(B)

            pbar.close()

        # 清理DataLoader以避免多进程警告
        del data_iter
        del temp_loader

        # 转换为 numpy 并打印报告 (与原函数保持一致)
        min_udf_values = np.array(stats['min_udf'])
        gt_udf_values = np.array(stats['gt_udf'])
        min_udf_ranks = np.array(stats['gt_rank'])
        gt_udf_ratios = np.array(stats['gt_ratio'])

        # 打印结果
        print("\n" + "-"*80)
        print("MetricNet UDF预测质量评估结果:")
        print("-"*80)

        print(f"\n【预测的最小UDF值】（越小越好，说明模型有信心找到正确位置）")
        print(f"  均值:     {min_udf_values.mean():.6f}")
        print(f"  中位数:   {np.median(min_udf_values):.6f}")
        print(f"  标准差:   {min_udf_values.std():.6f}")
        print(f"  范围:     [{min_udf_values.min():.6f}, {min_udf_values.max():.6f}]")

        print(f"\n【GT位置的预测UDF值】（理想情况下应该接近0）")
        print(f"  均值:     {gt_udf_values.mean():.6f}")
        print(f"  中位数:   {np.median(gt_udf_values):.6f}")
        print(f"  标准差:   {gt_udf_values.std():.6f}")
        print(f"  范围:     [{gt_udf_values.min():.6f}, {gt_udf_values.max():.6f}]")

        print(f"\n【GT UDF的相对位置比例】（0=最小值/完美，1=最大值/最差）")
        print(f"  均值:     {gt_udf_ratios.mean():.4f}")
        print(f"  中位数:   {np.median(gt_udf_ratios):.4f}")
        print(f"  标准差:   {gt_udf_ratios.std():.4f}")
        print(f"  范围:     [{gt_udf_ratios.min():.4f}, {gt_udf_ratios.max():.4f}]")
        print(f"  <0.1的比例 (接近最小): {(gt_udf_ratios < 0.1).mean() * 100:.1f}%")
        print(f"  <0.2的比例 : {(gt_udf_ratios < 0.2).mean() * 100:.1f}%")

        print(f"***候选点总数：{dist_pred.shape[1]}***")
        print(f"\n【GT位置在排序中的排名】（理想情况下应该是1）")
        print(f"  均值排名: {min_udf_ranks.mean():.1f}")
        print(f"  中位数:   {np.median(min_udf_ranks):.0f}")
        print(f"  Top-1准确率 (GT是最小UDF): {(min_udf_ranks == 1).mean() * 100:.1f}%")
        print(f"  Top-10准确率: {(min_udf_ranks <= 10).mean() * 100:.1f}%")
        print(f"  Top-100准确率: {(min_udf_ranks <= 100).mean() * 100:.1f}%")
        print(f"  Top-200准确率: {(min_udf_ranks <= 200).mean() * 100:.1f}%")

        print(f"\n【UDF值差异】（GT位置 vs 最小值）")
        udf_diff = gt_udf_values - min_udf_values
        print(f"  均值差异: {udf_diff.mean():.6f}")
        print(f"  中位数:   {np.median(udf_diff):.6f}")
        print(f"  GT=Min的比例: {(udf_diff == 0).mean() * 100:.1f}%")

        print("-"*80)

        return {
            'min_udf_mean': min_udf_values.mean(),
            'min_udf_median': np.median(min_udf_values),
            'gt_udf_mean': gt_udf_values.mean(),
            'gt_udf_median': np.median(gt_udf_values),
            'gt_udf_ratio_mean': gt_udf_ratios.mean(),
            'gt_udf_ratio_median': np.median(gt_udf_ratios),
            'gt_rank_mean': min_udf_ranks.mean(),
            'top1_accuracy': (min_udf_ranks == 1).mean() * 100,
            'top10_accuracy': (min_udf_ranks <= 10).mean() * 100,
            'top100_accuracy': (min_udf_ranks <= 100).mean() * 100,
            'top200_accuracy': (min_udf_ranks <= 200).mean() * 100,
            'n_candidates':dist_pred.shape[1],
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
                coords_encoded_frozen = self.pos_encoder_grid(eikonal_coords_5d.detach())
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

    def _test_localization_accuracy_batch(self, n_samples=128, n_candidates=2048, use_hierarchical_sampler=True,
                                    rc_threshold_m=10., rot_threshold_deg=10., scale_threshold=0.1,
                                    use_train_uav=False, test_batch_size=64):
        """
        [终极版] 测试UAV定位精度
        特性：
        1. Batch推理 (高速)
        2. 多维度误差统计 (NRC归一化误差 + 物理误差 + 详细Scale误差)
        """
        import numpy as np
        import tqdm

        print(f"\n🚀 开始UAV定位精度评估 (Batch Size: {test_batch_size})")
        print(f"测试样本数: {n_samples}")
        print(f"候选位置数: {n_candidates}")
        print(f"数据源: {'训练集' if use_train_uav else '测试集'}")

        # 1. 准备高效的数据加载器
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
        temp_loader = torch.utils.data.DataLoader(
            dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=2, drop_last=False, pin_memory=True  # 使用单进程避免多进程清理问题
        )

        # 初始化采样器
        if use_hierarchical_sampler:
            from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
            coord_sampler = create_hierarchical_sampler_from_dataset(
                sat_dataset=self.sat_dataset,
                bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
                num_uniform_samples=n_candidates,
                device=self.device
            )

        # 统计容器
        stats = {
            'rc_norm': [],  # 归一化平面误差
            'rc_m': [],  # 物理平面误差
            'rot_deg': [],  # 角度误差
            'scale': [],  # Scale绝对误差
            'min_udf': []  # 预测的UDF最小值
        }

        # 计算米/归一化单位的换算系数
        # 假设 NRC=1.0 对应卫星图的最大边长
        m_per_nrc = self.sat_dataset.satmap_hw_max * self.sat_dataset.geo_res_m

        processed_count = 0
        data_iter = iter(temp_loader)

        pred_rot_list,gt_rot_list = [],[]
        with torch.no_grad():
            pbar = tqdm.tqdm(total=n_samples, desc="Loc Test")

            while processed_count < n_samples:
                # --- A. 获取 Batch 数据 ---
                try:
                    batch_uav = next(data_iter)
                except StopIteration:
                    data_iter = iter(temp_loader)
                    batch_uav = next(data_iter)

                # 截断多余数据
                current_bs = batch_uav[0].shape[0]
                if processed_count + current_bs > n_samples:
                    current_bs = n_samples - processed_count
                    batch_uav[0] = batch_uav[0][:current_bs]
                    batch_uav[1] = batch_uav[1][:current_bs]

                uavimgs = batch_uav[0].to(self.device)
                gt_coords = batch_uav[1].to(self.device)
                B = uavimgs.shape[0]

                # --- B. 准备特征与候选点 ---
                query_feats = self._get_feats_fm_imgs(uavimgs)  # [B, C]

                if use_hierarchical_sampler:
                    coords_candidates = coord_sampler.sample(gt_coords)  # [B, N, 4]
                else:
                    flat_candidates = self.sat_dataset.mk_rand_coords_4d(
                        n_rand=n_candidates * B, return_tensor=True
                    ).to(self.device)
                    coords_candidates = flat_candidates.view(B, n_candidates, 4)

                N = coords_candidates.shape[1]

                # --- C. 批量推理 (Flatten -> Norm -> Net -> Reshape) ---
                coords_flat = coords_candidates.view(-1, 4)

                # 坐标归一化 & 编码
                coords_flat_6d = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)

                # Grid 特征
                grid_in = torch.cat([coords_flat_6d[:, :2], coords_flat_6d[:, -1:]], dim=-1)
                feats_grid_raw = self._get_feats_fm_grid(grid_in)

                # Grid MLP
                enc_stage2 = self.pos_encoder_grid(coords_flat_6d[:, :5])
                feats_grid = self.grid_mlp(feats_grid_raw, enc_stage2)
                feats_grid = TF.normalize(feats_grid, dim=-1)

                # Metric Encoding
                enc_metric = self.pos_encoder_metric(coords_flat_6d[:, :5])

                # 重塑回 Batch 维度
                feats_grid_batch = feats_grid.view(B, N, -1)
                coords_enc_batch = enc_metric.view(B, N, -1)
                query_feats_exp = query_feats.unsqueeze(1).expand(-1, N, -1)

                # MetricNet Forward
                dist_pred = self.metric_net(query_feats_exp, feats_grid_batch, coords_enc_batch)
                if self.use_softplus:
                    dist_pred = self.softplus(dist_pred)

                # --- D. 寻找最小值 & 计算误差 ---
                # 找到每个样本预测距离最小的索引
                min_indices = torch.argmin(dist_pred, dim=1)  # [B]

                # 取出对应的预测坐标 (高级索引)
                batch_indices = torch.arange(B, device=self.device)
                pred_coords = coords_candidates[batch_indices, min_indices]  # [B, 4]
                min_vals = dist_pred[batch_indices, min_indices]

                # 1. RC 误差 (核心计算)
                diff_rc = pred_coords[:, :2] - gt_coords[:, :2]

                # 归一化误差 (NRC)
                rc_err_norm = diff_rc.norm(dim=1)  # [B]

                # 物理误差 (Meter)
                rc_err_m = rc_err_norm * m_per_nrc

                # 2. Rot 误差
                diff_rot = torch.abs(pred_coords[:, 2] - gt_coords[:, 2])
                diff_rot = torch.min(diff_rot, 2 * torch.pi - diff_rot)
                rot_err_deg = diff_rot * 180 / torch.pi
                #debug
                pred_rot_list.append(pred_coords[:,2]*180/torch.pi)
                gt_rot_list.append(gt_coords[:,2]*180/torch.pi)

                # 3. Scale 误差
                scale_err = torch.abs(pred_coords[:, 3] - gt_coords[:, 3])

                # 收集
                stats['rc_norm'].extend(rc_err_norm.cpu().numpy().tolist())
                stats['rc_m'].extend(rc_err_m.cpu().numpy().tolist())
                stats['rot_deg'].extend(rot_err_deg.cpu().numpy().tolist())
                stats['scale'].extend(scale_err.cpu().numpy().tolist())
                stats['min_udf'].extend(min_vals.cpu().numpy().tolist())

                processed_count += B
                pbar.update(B)

            pbar.close()

        # 清理DataLoader以避免多进程警告
        del data_iter
        del temp_loader

        # --- E. 统计报告 ---
        arrays = {k: np.array(v) for k, v in stats.items()}

        # 计算各类成功率
        succ_rc = (arrays['rc_m'] < rc_threshold_m).mean() * 100
        succ_rot = (arrays['rot_deg'] < rot_threshold_deg).mean() * 100
        succ_scale = (arrays['scale'] < scale_threshold).mean() * 100

        # 综合成功率
        succ_comb = ((arrays['rc_m'] < rc_threshold_m) & (arrays['rot_deg'] < rot_threshold_deg)).mean() * 100
        succ_full = ((arrays['rc_m'] < rc_threshold_m) & (arrays['rot_deg'] < rot_threshold_deg) & (
                    arrays['scale'] < scale_threshold)).mean() * 100

        print("\n" + "=" * 80)
        print("UAV定位精度评估结果 (Batch高效版)")
        print("=" * 80)

        # 1. 归一化 RC 误差 (新增板块)
        print(f"\n【RC定位误差 - 归一化 (NRC 0~1)】")
        print(f"  均值:     {arrays['rc_norm'].mean():.4f}")
        print(f"  中位数:   {np.median(arrays['rc_norm']):.4f}")
        print(f"  标准差:   {arrays['rc_norm'].std():.4f}")
        print(f"  范围:     [{arrays['rc_norm'].min():.4f}, {arrays['rc_norm'].max():.4f}]")

        # 2. 物理 RC 误差
        print(f"\n【RC定位误差 - 物理距离 (米)】")
        print(f"  均值:     {arrays['rc_m'].mean():.2f} m")
        print(f"  中位数:   {np.median(arrays['rc_m']):.2f} m")
        print(f"  标准差:   {arrays['rc_m'].std():.2f} m")
        print(f"  范围:     [{arrays['rc_m'].min():.2f}, {arrays['rc_m'].max():.2f}] m")
        print(f"  成功率 (<{rc_threshold_m}m): {succ_rc:.1f}%")

        # 3. 旋转误差
        print(f"\n【旋转误差 (度)】")
        print(f"  均值:     {arrays['rot_deg'].mean():.2f}°")
        print(f"  中位数:   {np.median(arrays['rot_deg']):.2f}°")
        print(f"  标准差:   {arrays['rot_deg'].std():.2f}°")
        print(f"  范围:     [{arrays['rot_deg'].min():.2f}, {arrays['rot_deg'].max():.2f}]°")
        print(f"  成功率 (<{rot_threshold_deg}°): {succ_rot:.1f}%")

        # 4. Scale 误差 (详细版)
        print(f"\n【Scale估计误差 (归一化单位)】")
        print(f"  均值:     {arrays['scale'].mean():.4f}")
        print(f"  中位数:   {np.median(arrays['scale']):.4f}")
        print(f"  标准差:   {arrays['scale'].std():.4f}")
        print(f"  范围:     [{arrays['scale'].min():.4f}, {arrays['scale'].max():.4f}]")
        print(f"  成功率 (<{scale_threshold}): {succ_scale:.1f}%")

        # 5. 综合
        print(f"\n【综合指标】")
        print(f"  预测UDF均值: {arrays['min_udf'].mean():.4f}")
        print(f"  双指标成功率 (RC+Rot): {succ_comb:.1f}%")
        print(f"  全指标成功率 (RC+Rot+Scale): {succ_full:.1f}%")
        print("=" * 80)

        return {
            'rc_norm_mean': arrays['rc_norm'].mean(),
            'rc_m_mean': arrays['rc_m'].mean(),
            'rot_deg_mean': arrays['rot_deg'].mean(),
            'scale_error_mean': arrays['scale'].mean(),
            'success_rate': succ_comb
        }

    def _test_coarse_partition_accuracy(self, n_samples=256, n_candidates=2048,
                                         use_hierarchical_sampler=True, use_train_uav=False,
                                         test_batch_size=32):
        """
        测试粗分区（Coarse Partition）的定位准确度

        评估指标：
        1. 子空间匹配率: 预测最小UDF位置与GT是否在同一子空间
        2. 相邻子空间匹配率: 预测位置是否在GT的相邻子空间内
        3. 各维度的分区误差统计

        Args:
            n_samples: 测试样本数
            n_candidates: 每个样本的候选点数
            use_hierarchical_sampler: 是否使用分层采样
            use_train_uav: 是否使用训练集
            test_batch_size: 批处理大小
        """
        import numpy as np

        print(f"\n🎯 开始粗分区准确度评估")
        print(f"测试样本数: {n_samples}")
        print(f"子空间配置: n_coarse={self.subspace_sampler.n_coarse.tolist()}")
        print(f"子空间总数: {self.subspace_sampler.n_subspaces}")

        # 准备数据加载器
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
        temp_loader = torch.utils.data.DataLoader(
            dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=0, drop_last=False, pin_memory=True
        )

        # 初始化采样器
        if use_hierarchical_sampler:
            from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
            coord_sampler = create_hierarchical_sampler_from_dataset(
                sat_dataset=self.sat_dataset,
                bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
                num_uniform_samples=n_candidates,
                device=self.device
            )

        # 统计容器
        stats = {
            'exact_match': [],      # 完全匹配（同一子空间）
            'adjacent_match': [],   # 相邻匹配
            'gt_subspace': [],      # GT 子空间标签
            'pred_subspace': [],    # 预测子空间标签
            'subspace_dist': [],    # 子空间索引距离（多维）
        }

        processed_count = 0
        data_iter = iter(temp_loader)

        with torch.no_grad():
            pbar = tqdm.tqdm(total=n_samples, desc="Coarse Partition Test")

            while processed_count < n_samples:
                try:
                    batch_uav = next(data_iter)
                except StopIteration:
                    data_iter = iter(temp_loader)
                    batch_uav = next(data_iter)

                # 截断数据
                current_bs = batch_uav[0].shape[0]
                if processed_count + current_bs > n_samples:
                    current_bs = n_samples - processed_count
                    batch_uav[0] = batch_uav[0][:current_bs]
                    batch_uav[1] = batch_uav[1][:current_bs]

                uavimgs = batch_uav[0].to(self.device)
                gt_coords = batch_uav[1].to(self.device)
                B = uavimgs.shape[0]

                # 提取特征
                query_feats = self._get_feats_fm_imgs(uavimgs)

                # 生成候选点
                if use_hierarchical_sampler:
                    coords_candidates = coord_sampler.sample(gt_coords)
                else:
                    flat_candidates = self.sat_dataset.mk_rand_coords_4d(
                        n_rand=n_candidates * B, return_tensor=True
                    ).to(self.device)
                    coords_candidates = flat_candidates.view(B, n_candidates, 4)

                N = coords_candidates.shape[1]

                # 批量推理
                coords_flat = coords_candidates.view(-1, 4)
                coords_flat_6d = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)

                grid_in = torch.cat([coords_flat_6d[:, :2], coords_flat_6d[:, -1:]], dim=-1)
                feats_grid_raw = self._get_feats_fm_grid(grid_in)

                enc_stage2 = self.pos_encoder_grid(coords_flat_6d[:, :5])
                feats_grid = self.grid_mlp(feats_grid_raw, enc_stage2)
                feats_grid = TF.normalize(feats_grid, dim=-1)

                enc_metric = self.pos_encoder_metric(coords_flat_6d[:, :5])

                feats_grid_batch = feats_grid.view(B, N, -1)
                coords_enc_batch = enc_metric.view(B, N, -1)
                query_feats_exp = query_feats.unsqueeze(1).expand(-1, N, -1)

                dist_pred = self.metric_net(query_feats_exp, feats_grid_batch, coords_enc_batch)
                if self.use_softplus:
                    dist_pred = self.softplus(dist_pred)

                # 找到预测最小值位置
                min_indices = torch.argmin(dist_pred, dim=1)
                batch_indices = torch.arange(B, device=self.device)
                pred_coords = coords_candidates[batch_indices, min_indices]

                # 计算子空间标签
                gt_subspace_labels = self.subspace_sampler.coords_to_coarse_indices(gt_coords)
                pred_subspace_labels = self.subspace_sampler.coords_to_coarse_indices(pred_coords)

                # 计算匹配情况
                exact_match = (gt_subspace_labels == pred_subspace_labels)

                # 计算相邻匹配（需要逐个检查）
                gt_labels_np = gt_subspace_labels.cpu().numpy()
                pred_labels_np = pred_subspace_labels.cpu().numpy()

                for i in range(B):
                    gt_label = int(gt_labels_np[i])
                    pred_label = int(pred_labels_np[i])

                    # 检查是否相邻
                    adjacent_subspaces = self.subspace_sampler.get_adjacent_subspaces(gt_label, include_diagonal=True)
                    is_adjacent = pred_label in adjacent_subspaces or pred_label == gt_label

                    stats['exact_match'].append(exact_match[i].item())
                    stats['adjacent_match'].append(is_adjacent)
                    stats['gt_subspace'].append(gt_label)
                    stats['pred_subspace'].append(pred_label)

                    # 计算多维索引距离
                    gt_multi = self.subspace_sampler._flat_to_multi_index(
                        np.array([gt_label]), self.subspace_sampler.n_coarse
                    )[0]
                    pred_multi = self.subspace_sampler._flat_to_multi_index(
                        np.array([pred_label]), self.subspace_sampler.n_coarse
                    )[0]

                    # 处理旋转维度的周期性
                    dim_dist = np.abs(gt_multi - pred_multi)
                    n_rot = self.subspace_sampler.n_coarse[2]
                    dim_dist[2] = min(dim_dist[2], n_rot - dim_dist[2])

                    stats['subspace_dist'].append(dim_dist)

                processed_count += B
                pbar.update(B)

            pbar.close()

        del data_iter
        del temp_loader

        # 统计报告
        exact_match_arr = np.array(stats['exact_match'])
        adjacent_match_arr = np.array(stats['adjacent_match'])
        subspace_dist_arr = np.array(stats['subspace_dist'])  # [N, 4]

        print("\n" + "=" * 80)
        print("粗分区准确度评估结果")
        print("=" * 80)

        print(f"\n【子空间匹配率】")
        print(f"  完全匹配率 (Same Subspace):     {exact_match_arr.mean() * 100:.2f}%")
        print(f"  相邻匹配率 (Adjacent Subspace): {adjacent_match_arr.mean() * 100:.2f}%")

        print(f"\n【各维度分区误差】(单位: 子空间格子数)")
        dim_names = ['NR', 'NC', 'Rot', 'Scale']
        for i, name in enumerate(dim_names):
            dim_err = subspace_dist_arr[:, i]
            print(f"  {name:6s}: 均值={dim_err.mean():.2f}, "
                  f"中位数={np.median(dim_err):.1f}, "
                  f"最大={dim_err.max():.0f}, "
                  f"=0占比={((dim_err == 0).mean() * 100):.1f}%")

        # 综合L1距离
        l1_dist = subspace_dist_arr.sum(axis=1)
        print(f"\n【综合分区误差】(L1距离，所有维度之和)")
        print(f"  均值: {l1_dist.mean():.2f}")
        print(f"  中位数: {np.median(l1_dist):.1f}")
        print(f"  L1<=1 占比: {((l1_dist <= 1).mean() * 100):.1f}%")
        print(f"  L1<=2 占比: {((l1_dist <= 2).mean() * 100):.1f}%")

        print("=" * 80)

        return {
            'exact_match_rate': exact_match_arr.mean() * 100,
            'adjacent_match_rate': adjacent_match_arr.mean() * 100,
            'dim_errors': {
                name: subspace_dist_arr[:, i].mean()
                for i, name in enumerate(dim_names)
            },
            'l1_dist_mean': l1_dist.mean(),
            'l1_le1_rate': (l1_dist <= 1).mean() * 100,
        }

    def _analyze_udf_similarity_distribution(self, n_samples=32, n_candidates=2048,
                                             save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results',
                                             use_train_uav=False, batch_size=16):
        """
        [分析专用] UDF分布 vs 相似度分布 分析
        用于验证：
        1. 学习到的UDF是否比原始相似度更平滑/单峰？
        2. GT点在UDF和相似度中的排名情况
        3. 相似度与UDF的相关性（是否高度负相关？）
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import torch.nn.functional as TF
        import os

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n🔬 开始 UDF vs Similarity 分布分析 (Samples: {n_samples})")

        # 1. 数据准备
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
        temp_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=False, pin_memory=True
        )

        # 统计容器
        metrics = {
            'pearson_sim_dist': [],  # 相似度与物理距离的相关性
            'pearson_udf_dist': [],  # UDF与物理距离的相关性
            'gt_rank_sim': [],  # GT在相似度中的排名 (越小越好)
            'gt_rank_udf': []  # GT在UDF中的排名 (越小越好)
        }

        data_iter = iter(temp_loader)
        processed_count = 0

        with torch.no_grad():
            while processed_count < n_samples:
                try:
                    batch_uav = next(data_iter)
                except StopIteration:
                    break

                # --- A. 数据获取 & 截断 ---
                current_bs = batch_uav[0].shape[0]
                if processed_count + current_bs > n_samples:
                    current_bs = n_samples - processed_count
                    batch_uav[0] = batch_uav[0][:current_bs]
                    batch_uav[1] = batch_uav[1][:current_bs]

                uavimgs = batch_uav[0].to(self.device)  # [B, C, H, W]
                gt_coords = batch_uav[1].to(self.device)  # [B, 4]
                B = uavimgs.shape[0]

                # --- B. 特征提取 ---
                query_feats = self._get_feats_fm_imgs(uavimgs)  # [B, C]
                query_feats = TF.normalize(query_feats, dim=-1)  # 确保归一化

                # --- C. 构建采样点 (GT + Random) ---
                # 1. 生成随机候选点
                flat_candidates = self.sat_dataset.mk_rand_coords_4d(
                    n_rand=n_candidates * B, return_tensor=True
                ).to(self.device)
                coords_rand = flat_candidates.view(B, n_candidates, 4)

                # 2. 将 GT 插入到第一个位置，方便追踪
                # coords_all shape: [B, N+1, 4]
                coords_all = torch.cat([gt_coords.unsqueeze(1), coords_rand], dim=1)
                N_total = coords_all.shape[1]

                # --- D. 计算 UDF 和 Similarity ---
                # Flatten
                coords_flat = coords_all.view(-1, 4)

                # 坐标编码
                coords_flat_6d = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)

                # 1. 获取 Grid 特征 (用于计算相似度)
                grid_in = torch.cat([coords_flat_6d[:, :2], coords_flat_6d[:, -1:]], dim=-1)
                feats_grid_raw = self._get_feats_fm_grid(grid_in)

                enc_stage2 = self.pos_encoder_grid(coords_flat_6d[:, :5])
                feats_grid = self.grid_mlp(feats_grid_raw, enc_stage2)
                feats_grid = TF.normalize(feats_grid, dim=-1)  # [B*(N+1), C]

                # Reshape back
                feats_grid_batch = feats_grid.view(B, N_total, -1)  # [B, N+1, C]

                # 2. 计算余弦相似度 (Cosine Similarity)
                # query: [B, 1, C], grid: [B, N+1, C] -> dot -> [B, N+1]
                sim_pred = (query_feats.unsqueeze(1) * feats_grid_batch).sum(dim=-1)

                # 3. 计算 UDF (MetricNet Prediction)
                enc_metric = self.pos_encoder_metric(coords_flat_6d[:, :5])
                coords_enc_batch = enc_metric.view(B, N_total, -1)
                query_feats_exp = query_feats.unsqueeze(1).expand(-1, N_total, -1)

                udf_pred = self.metric_net(query_feats_exp, feats_grid_batch, coords_enc_batch)
                if self.use_softplus:
                    udf_pred = self.softplus(udf_pred)
                udf_pred = udf_pred.squeeze(-1)  # [B, N+1]

                # --- E. 计算物理距离 (Ground Truth Distance) ---
                # 仅计算 RC 距离用于分析 (忽略 Rot/Scale 简化图表)
                diff_rc = coords_all[..., :2] - gt_coords.unsqueeze(1)[..., :2]
                # 转换为米
                m_per_nrc = self.sat_dataset.satmap_hw_max * self.sat_dataset.geo_res_m
                dist_m = diff_rc.norm(dim=-1) * m_per_nrc  # [B, N+1]

                # --- F. 统计与可视化 ---
                for i in range(B):
                    # 获取当前样本的所有数据 (转为numpy)
                    d_m = dist_m[i].cpu().numpy()
                    udf = udf_pred[i].cpu().numpy()
                    sim = sim_pred[i].cpu().numpy()

                    # 1. 计算相关性 (Pearson)
                    # 我们期望: UDF 与 Distance 正相关 (越远 UDF 越大)
                    # 我们期望: Sim 与 Distance 负相关 (越远 Sim 越小)
                    p_udf = np.corrcoef(d_m, udf)[0, 1]
                    p_sim = np.corrcoef(d_m, sim)[0, 1]

                    # 2. 计算 GT 的排名 (GT 是索引 0)
                    # UDF: 越小越好 -> argsort 后的位置
                    rank_udf = np.where(np.argsort(udf) == 0)[0][0]
                    # Sim: 越大越好 -> argsort 倒序后的位置
                    rank_sim = np.where(np.argsort(sim)[::-1] == 0)[0][0]

                    metrics['pearson_udf_dist'].append(p_udf)
                    metrics['pearson_sim_dist'].append(p_sim)
                    metrics['gt_rank_udf'].append(rank_udf)
                    metrics['gt_rank_sim'].append(rank_sim)

                    # 3. 抽样画图 (每10个样本画一张)
                    if (processed_count + i) % 10 == 0:
                        self._plot_sample_analysis(
                            d_m, udf, sim,
                            rank_udf, rank_sim,
                            save_path=os.path.join(save_dir, f'sample_{processed_count + i}.png')
                        )

                processed_count += B

        # --- G. 输出综合报告 ---
        print("\n" + "=" * 60)
        print("📊 UDF vs Similarity 分析报告")
        print("=" * 60)
        print(f"平均相关性 (与物理距离):")
        print(f"  Sim Correlation (理想 -> -1.0): {np.mean(metrics['pearson_sim_dist']):.4f}")
        print(f"  UDF Correlation (理想 ->  1.0): {np.mean(metrics['pearson_udf_dist']):.4f}")
        print("-" * 30)
        print(f"GT 检索排名 (Rank 0 代表 Top-1, 越小越好):")
        print(f"  Sim Rank Mean: {np.mean(metrics['gt_rank_sim']):.1f} / {n_candidates}")
        print(f"  UDF Rank Mean: {np.mean(metrics['gt_rank_udf']):.1f} / {n_candidates}")
        print(f"  Sim Top-1 率:  {np.mean(np.array(metrics['gt_rank_sim']) == 0) * 100:.1f}%")
        print(f"  UDF Top-1 率:  {np.mean(np.array(metrics['gt_rank_udf']) == 0) * 100:.1f}%")
        print("=" * 60 + "\n")

    def _plot_sample_analysis(self, dists, udfs, sims, rank_udf, rank_sim, save_path):
        """辅助绘图函数"""
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制 UDF (左轴, 蓝色)
        ax1.set_xlabel('Physical Distance (m)')
        ax1.set_ylabel('Predicted UDF', color='tab:blue')
        # 绘制散点 (除去GT点以免覆盖，GT点单独画)
        ax1.scatter(dists[1:], udfs[1:], color='tab:blue', alpha=0.3, s=10, label='UDF Samples')
        # 绘制 GT 点
        ax1.scatter(dists[0], udfs[0], color='blue', marker='*', s=200, label=f'GT (Rank {rank_udf})')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 绘制 Similarity (右轴, 橙色)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Visual Similarity', color='tab:orange')
        ax2.scatter(dists[1:], sims[1:], color='tab:orange', alpha=0.3, s=10, label='Sim Samples')
        ax2.scatter(dists[0], sims[0], color='red', marker='*', s=200, label=f'GT (Rank {rank_sim})')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # 标题
        plt.title(f'Distribution Analysis\nUDF Rank: {rank_udf} | Sim Rank: {rank_sim}')
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def visualize_comprehensive_udf(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                                    n_samples_per_dim=40, delta=0.2, rot_span=torch.pi,
                                    surface_min_ratio=0.2, adaptive_z_scale=False,
                                    save_path="vis_results/udf_comprehensive.html", show_plot=False,
                                    use_train_uav=False):
        """
        [终极版] UDF 综合可视化仪表盘
        在一个 HTML 文件中并排显示三个子图：
        1. UDF 场值散点图 (Scatter): 查看采样点的数值分布
        2. UDF 梯度流场 (Cone): 查看梯度下降的方向
        3. UDF 等值面 (Isosurface): 查看场的几何拓扑结构

        特点：
        - 强制立方体视角 (aspectmode='cube')
        - 共享计算 (只推理一次)
        - 统一坐标系
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

        # --- Forward ---
        grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
        feats_grid_raw = self._get_feats_fm_grid(grid_input)
        coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
        feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
        feats_grid = TF.normalize(feats_grid, dim=-1)
        feats_grid_exp = feats_grid.unsqueeze(0)

        coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])
        N = coords_sampled_4d.shape[0]
        query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
        coords_enc_exp = coords_encoded_metric.unsqueeze(0)

        dist_pred_raw = self.metric_net(query_feat_exp, feats_grid_exp, coords_enc_exp)
        dist_pred = self.softplus(dist_pred_raw) if self.use_softplus else dist_pred_raw
        dist_pred = dist_pred.squeeze(0)  # [N]

        # --- Backward (Gradient) ---
        grad_outputs = torch.ones_like(dist_pred)
        gradients = torch.autograd.grad(dist_pred, coords_sampled_4d, grad_outputs, create_graph=False)[0]

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

        V = dist_pred.detach().cpu().numpy()
        G_vec = grad_vec_norm.detach().cpu().numpy()

        min_v, max_v = V.min(), V.max()
        print(f"📊 统计: Min UDF={min_v:.6f}, Max UDF={max_v:.6f}, Z-Scale={z_amplification:.1f}x")

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
        # 降采样一下散点图，防止浏览器卡死 (如果 n_samples 很大)
        step = 1 if n_samples_per_dim <= 32 else 2
        mask = slice(None, None, step)  # 如果点太多，隔点采样显示

        fig.add_trace(go.Scatter3d(
            x=X[mask], y=Y[mask], z=Z[mask],
            mode='markers',
            marker=dict(size=3, opacity=0.4, color=V[mask], colorscale='Viridis',
                        colorbar=dict(title='UDF', x=0.28, len=0.5)),  # 调整colorbar位置
            hovertemplate='UDF: %{marker.color:.4f}<extra></extra>',
            name='UDF Cloud'
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


        # 1. 设定分界线 (Threshold)
        # 比如：我们将前 10% 的数值范围定义为"核心区"，后 90% 定义为"全局背景"
        # split_ratio = 0.10
        val_range = max_v - min_v
        split_val = min_v + val_range * surface_min_ratio
        # 2. 第一层：核心细节 (关注最小值附近的微小变化)
        # 在 0% - 10% 的范围内切 4 刀，看清收敛结构
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=min_v,
            isomax=split_val,
            surface_count=4,  # 核心区切细一点
            colorscale='Plasma',  # 核心区用暖色调
            opacity=0.2,  # 稍微不透明一点，看清形状
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            showscale=False,  # 为了不搞乱 Colorbar，只显示一个或者都不显示
            name='Inner Core',
            hovertemplate='Core UDF: %{value:.4f}<extra></extra>'
        ), row=1, col=3)
        # 3. 第二层：全局概览 (关注整体势能场的走向)
        # 在 10% - 100% 的范围内切 3 刀，提供宏观参考
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=split_val,
            isomax=max_v,
            surface_count=4,  # 全局区稀疏一点
            colorscale='Viridis',  # 换个色系或者保持一致，Viridis 比较冷，适合做背景
            opacity=0.15,  # 很透明，不要挡住核心
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            colorbar=dict(title='UDF Level', x=1.0, len=0.5),  # 只保留这一个 Colorbar
            name='Outer Shell',
            hovertemplate='Global UDF: %{value:.4f}<extra></extra>'
        ), row=1, col=3)

        # === 通用标记: GT 和 Pred Min ===
        # 将这两个标记添加到所有三个子图中
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
        # 强制所有子图为立方体比例
        scene_layout = dict(
            xaxis_title='NR', yaxis_title='NC', zaxis_title='Rot',
            aspectmode='cube'  # 关键：固定体积为立方体
        )

        fig.update_layout(
            title=f'UDF Comprehensive Analysis (Scale={scale_fixed:.2f})',
            height=600, width=1600,  # 宽屏显示
            scene1=scene_layout,
            scene2=scene_layout,
            scene3=scene_layout,
            margin=dict(l=10, r=10, b=10, t=60)
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 综合可视化已保存: {save_path}")

        if show_plot:
            fig.show()


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

        # 初始化坐标归一化器和UDF计算器
        from trainer_depends.utils.util_core_coords_translater import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)

        # 同时创建训练集和测试集的数据加载器，以支持灵活选择
        from trainer_depends.datasets.dataset_wingtra_4d import UAVDataset
        opt = self.opt
        scene = opt.scenes_setting['scenes'][0]  # 使用第一个场景

        print(f"测试配置: use_train_uav={use_train_uav}, use_augmentation={use_augmentation}")

        # 创建训练集数据加载器（支持数据增强）
        uav_dataset_train = UAVDataset(
            p_uavinfo_json=scene['p_uavinfo_json'],
            trans_georc2nrc_func=self.sat_dataset.transfrom_georc_to_nrc,
            geo_res_m=0.3,
            stage='train',
            use_augmentation=use_augmentation,  # 控制是否使用数据增强
        )
        self.uav_dataloader_train = torch.utils.data.DataLoader(
            uav_dataset_train,
            batch_size=self.opt.batchsize_uav,
            num_workers=self.opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

        # 创建测试集数据加载器（支持数据增强）
        uav_dataset_test = UAVDataset(
            p_uavinfo_json=scene['p_uavinfo_json'],
            trans_georc2nrc_func=self.sat_dataset.transfrom_georc_to_nrc,
            geo_res_m=0.3,
            stage='test',
            use_augmentation=use_augmentation,  # 控制是否使用数据增强
        )
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            uav_dataset_test,
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


        from trainer_depends.utils.util_core_subspace_sampler import SubspaceSampler
        self.n_subspaces_to_sample = getattr(opt, 'n_subspaces_to_sample', 256)
        self.n_points_per_subspace = getattr(opt, 'n_points_per_subspace', 8)
        self.subspace_sampler = SubspaceSampler(
            sat_dataset=self.sat_dataset,
            n_coarse=self.n_coarse,
            n_fine_per_coarse=self.n_fine_per_coarse
        )
        results = self._test_coarse_partition_accuracy(
            n_samples=256,
            n_candidates=2048,
            use_hierarchical_sampler=True,
            use_train_uav=False
        )

        self._analyze_udf_similarity_distribution(n_samples=4, n_candidates=8192,use_train_uav=use_train_uav,save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results')

        # 4. 测试定位精度（UDF最小值回归）
        print("\n" + "="*80)
        print("测试3: UAV定位精度评估（UDF最小值回归）")
        print("="*80)
        # self._test_localization_accuracy(n_samples=128, n_candidates=8192,use_hierarchical_sampler=True,rc_threshold_m=50, use_train_uav=use_train_uav)
        self._test_localization_accuracy_batch(n_samples=256, n_candidates=8192,use_hierarchical_sampler=False,rc_threshold_m=50, use_train_uav=use_train_uav)

        # 5. 测试距离预测精度
        print("\n" + "="*80)
        print("测试1: MetricNet距离预测精度评估")
        print("="*80)
        # self._test_distance_prediction_accuracy(n_samples=256, n_candidates=2048, use_hierarchical_sampler=False, use_train_uav=use_train_uav)
        self._test_distance_prediction_accuracy_batch(n_samples=256, n_candidates=8192, use_hierarchical_sampler=False, use_train_uav=use_train_uav)

        # 6. 测试距离场平滑性（Eikonal约束）
        print("\n" + "="*80)
        print("测试2: 距离场梯度范数评估（Eikonal约束）")
        print("="*80)
        self._test_eikonal_constraint()

        # --- New Visualization Test Step ---
        import os # Ensure os is imported
        print("\n" + "="*80)
        print("测试4: 可视化UDF场（合并场值+梯度）")
        print("="*80)
        num_viz_samples = 5
        viz_output_dir = "/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results"
        os.makedirs(viz_output_dir, exist_ok=True)
        dataset_tag = "train" if use_train_uav else "test"
        print(f"可视化输出目录: {viz_output_dir}")
        print(f"数据集选择: {'训练集' if use_train_uav else '测试集'}")
        for i in range(num_viz_samples):
            try:
                udf_save_path = os.path.join(
                    viz_output_dir,
                    f"udf_{dataset_tag}_sample_{i:02d}.html"
                )
                self.visualize_comprehensive_udf(
                    save_path=udf_save_path,
                    n_samples_per_dim=32,  # 建议30-40，太高会生成很慢且网页卡顿
                    delta=0.3,  # 视野范围
                    surface_min_ratio=0.3,
                    adaptive_z_scale=False
                )
            except ImportError:
                print("⚠️ Plotly is not installed. Skipping visualization.")
                break # Stop if plotly is missing
        # --- End of New Visualization Test Step ---

        print("\n" + "🧪"*40)
        print("✅ Stage 3 测试完成！")
        print("🧪"*40 + "\n")


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
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam', opt=self.opt)

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
        from trainer_depends.utils.util_core_coords_translater import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)

        # 4.6 初始化分层坐标采样器
        from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
        self.coord_sampler = create_hierarchical_sampler_from_dataset(
            sat_dataset=self.sat_dataset,
            bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
            num_uniform_samples=getattr(opt, 'sampler_num_uniform', 1024),  # 每个query采样的点数
            device=self.device
        )

        # 初始化子空间采样器（替代原来的 coord_sampler 和 udf_computer）
        from trainer_depends.utils.util_core_subspace_sampler import SubspaceSampler
        self.n_subspaces_to_sample = getattr(opt, 'n_subspaces_to_sample', 256)
        self.n_points_per_subspace = getattr(opt, 'n_points_per_subspace', 8)
        self.infonce_temperature = getattr(opt, 'infonce_temperature', 1.0)
        self.subspace_sampler = SubspaceSampler(
            sat_dataset=self.sat_dataset,
            n_coarse=self.n_coarse,
            n_fine_per_coarse=self.n_fine_per_coarse
        )
        print(f"子空间采样器初始化完成: {self.subspace_sampler}")

        # 5. Loss无需预定义类，直接调用成员函数即可
        from trainer_depends.utils.util_weight_annealing import SigmoidWeightScheduler
        eikonal_weight_scheduler = SigmoidWeightScheduler(max_steps=10,max_weight=1,min_weight=0,center_step=5,warmup_steps=1)
        scale_weight_scheduler = SigmoidWeightScheduler(max_steps=10,max_weight=1,min_weight=0,center_step=5,warmup_steps=0)

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
                anchor_labels = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [2B]

                # =================== 3. 采样候选点 (极速GPU版本，使用缓存) ===================
                coords_candidates, candidate_labels = self.subspace_sampler.sample_batch_for_infonce_gpu_fast(
                    anchor_labels=anchor_labels,
                    n_subspaces_to_sample=self.n_subspaces_to_sample,
                    n_points_per_subspace=self.n_points_per_subspace,
                    include_anchor_subspace=True,
                )  # [2B, N_candidates, 4], [2B, N_candidates]

                N_candidates = coords_candidates.shape[1]

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

                # MetricNet 坐标编码
                coords_encoded_metric = self.pos_encoder_metric(coords_6d_flat[:, :5])

                # 重塑
                feats_grid = feats_grid_flat.view(BatchSize, N_candidates, -1)
                coords_metric_encoded = coords_encoded_metric.view(BatchSize, N_candidates, -1)

                # =================== 5. MetricNet 前向 ===================
                log_energy = self.metric_net(
                    feat_query=feats_vis,
                    feat_ref=feats_grid,
                    coord_ref_encoded=coords_metric_encoded
                )  # [2B, N_candidates]

                # 可选：应用 Softplus 保证能量非负
                if self.use_softplus:
                    log_energy = self.softplus(log_energy)

                # =================== 6. 计算 InfoNCE Loss ===================
                loss = self._compute_infonce_loss(
                    log_energy=log_energy,
                    candidate_labels=candidate_labels,
                    anchor_labels=anchor_labels,
                    temperature=self.infonce_temperature
                )

                # =================== 7. 反向传播 ===================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 日志记录
                if it % 10 == 0:
                    self.logger.info(f'Iter {it}: Loss={loss.item():.6f}')
                    if self.writer is not None:
                        self.writer.add_scalar('loss_infonce', loss.item(), step)

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

            # 每个epoch结束后运行评估（可视化 + 测试）
            self._run_epoch_evaluation(
                epoch=epoch,
                run_visualization=True,  # 是否运行可视化
                n_test_samples=256  # 测试样本数量
            )
            # 每个epoch结束后保存checkpoint
            self._save_checkpoint(
                epoch,
                {**self.param2optimize, **self.param2freeze},
                self.optimizer
            )
            #debug:
            # use_train_uav = False
            # name = 'train' if use_train_uav else 'test'
            # viz_save_path = f"/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results/udf_combined_epoch{epoch}_uav_{name}.html"
            # # 保持训练模式进行可视化（以保持dropout等训练行为一致性）
            # self.visualize_udf_combined(
            #     save_path=viz_save_path,
            #     show_plot=False,
            #     n_samples_per_dim=32,
            #     delta=0.15,
            #     rot_span=torch.pi,
            #     adaptive_z_scale=True,
            #     use_train_uav=use_train_uav,
            #     use_train_mode=True,  # 使用训练模式进行可视化
            # )
            # self.logger.info(f"✅ 可视化已保存: {viz_save_path}")

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
