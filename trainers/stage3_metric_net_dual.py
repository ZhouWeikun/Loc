#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3: Projector Trainer

训练目标：
- projector (低维流形投影器)

前置条件：
- Stage 1: vis_encoder + vis_aggregator (冻结)
- Stage 2: grid + grid_mlp (可选冻结/微调)

训练策略：
- 将 feat_q (视觉特征) 和 feat_ref (Grid特征) 投影到共享低维流形
- 使用对比学习 loss 训练
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


class ProjectorTrainer(GridHashFitTrainer):
    """
    Stage 3: Projector Trainer

    继承自GridHashFitTrainer，在其基础上添加Projector
    """

    def __init__(self, opt=None):
        """初始化Stage 3 Trainer"""
        # 调用父类初始化（会初始化vis_encoder, grid等）
        super().__init__(opt)

        # 加载Stage 2的Grid权重（如果指定）
        if self.opt.load_stage2_ckpt:
            self._load_stage2_checkpoint()

        # 初始化Projector
        self._init_projector()

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


    def _init_projector(self):
        """初始化Projector（低维流形投影器）"""
        print("\n" + "="*80)
        print("初始化 Projector (低维流形投影器)")
        print("="*80)

        # 从配置读取Projector参数
        projector_bottleneck_dim = getattr(self.opt, 'projector_bottleneck_dim', 64)
        projector_output_dim = getattr(self.opt, 'projector_output_dim', 128)

        # 实例化 Projector
        from models.projector_mlp_sample import Projector
        self.projector = Projector(
            input_dim=self.feat_q_dim,
            bottleneck_dim=projector_bottleneck_dim,
            output_dim=projector_output_dim
        ).to(self.device)

        print(f"Projector配置:")
        print(f"  - input_dim: {self.feat_q_dim}")
        print(f"  - bottleneck_dim: {projector_bottleneck_dim}")
        print(f"  - output_dim: {projector_output_dim}")
        print(f"  - 参数量: {self.projector.num_parameters:,}")

        print("✅ Projector 初始化完成")
        print("="*80 + "\n")


    def _setup_trainable_params_stage3(self):
        """重新设置可训练参数（Stage 3专用）"""
        # 选项1：仅训练projector（Grid冻结）
        if getattr(self.opt, 'freeze_grid', True):
            for param in self.grid.parameters():
                param.requires_grad = False
            for param in self.grid_mlp.parameters():
                param.requires_grad = False

            self.param2optimize = {
                'projector': self.projector
            }

            print("参数配置 (freeze_grid=True):")
            print("可训练: projector")
            print("冻结:   vis_encoder, vis_aggregator, grid, grid_mlp\n")

        # 选项2：同时微调grid_mlp
        else:
            for param in self.grid.parameters():
                param.requires_grad = False

            self.param2optimize = {
                'grid_mlp': self.grid_mlp,
                'projector': self.projector
            }

            print("参数配置 (freeze_grid=False):")
            print("  可训练: grid_mlp, projector")
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
        print("开始 Stage 3 训练: Projector")
        print("🚀"*40 + "\n")

        # 0. 初始化GradScaler（如果使用autocast）
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam', opt=self.opt)

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

        # 4.6 初始化分层坐标采样器
        from trainer_depends.utils.util_hierarchical_coord_sampler import create_hierarchical_sampler_from_dataset
        self.coord_sampler = create_hierarchical_sampler_from_dataset(
            sat_dataset=self.sat_dataset,
            bottom_abs_rc_std=self.sat_dataset.halfimg_radius_nrc,
            num_uniform_samples=getattr(opt, 'sampler_num_uniform', 512),
            device=self.device
        )

        # 5. 配置Loss（TODO: 用户后续补充具体 loss 实现）

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

                # ========== 提取特征 ==========
                # 提取视觉特征（冻结的 vis_encoder）
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([uavimgs, satimgs], dim=0)
                )  # [2B, feat_dim]

                # Ground truth坐标
                coords_gt = torch.cat([coords_uav, coords_sat], dim=0)  # [2B, 4]

                # 使用分层采样器生成负样本坐标
                coords_rand_hierarchical = self.coord_sampler.sample(coords_gt)  # [2B, N_samples, 4]
                coords_rand = coords_rand_hierarchical.view(-1, 4)  # [2B*N_samples, 4]

                # 构建所有坐标
                coords_all = torch.cat([coords_gt, coords_rand], dim=0)  # [2B + 2B*N_samples, 4]
                coords_all_6d = self.coord_normer.raw_to_norm(coords_all, append_linear_rot=True)

                # 从Grid提取特征
                feats_all_grid = self._get_feats_fm_grid(
                    torch.cat([coords_all_6d[:, :2], coords_all_6d[:, -1:]], dim=-1)
                )
                # 位置编码
                coords_all_encoded = self.pos_encoder(coords_all_6d[:, :5])
                # Grid MLP调制
                feats_all_grid = self.grid_mlp(
                    inputs=feats_all_grid,
                    condition_features=coords_all_encoded
                )
                feats_all_grid = TF.normalize(feats_all_grid, dim=-1)

                # ========== Projector 前向传播 ==========
                # feat_q: 视觉特征 [2B, feat_dim]
                # feat_ref: Grid特征 [N_total, feat_dim]
                feat_q_proj = self.projector(feats_vis)            # [2B, proj_dim]
                feat_ref_proj = self.projector(feats_all_grid)     # [N_total, proj_dim]

                # ========== 计算 Loss ==========
                # TODO: 用户后续补充具体的对比学习 loss 实现
                # 示例：可以使用 InfoNCE loss
                # loss = compute_infonce_loss(feat_q_proj, feat_ref_proj, ...)
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # 占位

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
                    self.logger.info(f'Iter {it}: loss={loss.item():.6f}')
                step += 1

            # 每个epoch结束后保存checkpoint
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


    def test(self):
        """
        Stage 3测试函数

        TODO: 用户后续补充具体的测试逻辑
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

        # TODO: 用户后续补充测试逻辑
        print("✅ Stage 3 测试完成！")


    def _load_checkpoints_for_test(self):
        """
        测试时加载checkpoint的统一方法（自适应版本）

        自适应加载逻辑：
        1. Stage 3: 使用 self.param2optimize 自动加载当前stage的模型
        2. Stage 2: 加载 grid, grid_mlp
        3. Stage 1: 加载 vis_encoder, vis_aggregator
        """
        print("\n" + "="*80)
        print("加载测试用的checkpoint（自适应）")
        print("="*80)

        # --- 1. 加载Stage 3的checkpoint (当前stage) - 自适应 ---
        stage3_ckpt_path = self._get_stage3_checkpoint_path()

        if stage3_ckpt_path:
            print(f"\n📦 Stage 3 checkpoint: {stage3_ckpt_path}")
            # 自适应：使用 param2optimize 中的所有模型
            self._load_checkpoint(
                stage3_ckpt_path,
                self.param2optimize,
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 3的checkpoint，无法进行测试。")

        # --- 2. 加载Stage 2的checkpoint ---
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

        # --- 3. 加载Stage 1的checkpoint ---
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


if __name__ == "__main__":
    import argparse
    import sys

    # 添加 --test_only 参数
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    args, remaining_argv = parser.parse_known_args()

    # 如果没有指定配置文件，使用 stage3 的默认配置
    if '--p_yaml' not in ' '.join(remaining_argv):
        remaining_argv.extend(['--p_yaml', 'trainer_depends/configs/stage3_projector.yaml'])

    sys.argv[1:] = remaining_argv

    trainer = ProjectorTrainer()

    if args.test_only:
        trainer.test()
    else:
        trainer.train()
