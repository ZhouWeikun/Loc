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

        print("参数配置:")
        print("  可训练: vis_aggregator")
        print("  冻结:   vis_encoder\n")


    def _init_multi_scene_dataloader(self):
        """初始化多场景训练数据加载器"""
        from dataset_wingtra_4d_uav_sat_pair import UAVSatPairDataset, collate_uav_sat_pair
        from util_sample_neg_nrcs import BoundedNegativeCoordinateSampler

        opt = self.opt
        scenes = opt.scenes_setting['scenes']

        # 为每个场景创建pair dataloader
        pair_dataloaders = {}

        for scene in scenes:
            scene_name = scene['name']

            # 创建该场景的 UAVSatPairDataset
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_train = self.uav_datasets_train[scene_name]

            satmap_sampler = BoundedNegativeCoordinateSampler(self.device)
            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                satmap_sampler=satmap_sampler,
                device=self.device,
                n_neg_per_sample=1,
            )

            # 创建DataLoader
            pair_dataloader = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=opt.batchsize_sat,
                num_workers=opt.num_worker,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_uav_sat_pair,
                persistent_workers=True
            )

            pair_dataloaders[scene_name] = pair_dataloader
            self.logger.info(f"  {scene_name}: {len(pair_dataset)} pairs, {len(pair_dataloader)} batches")

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

        # 6. 配置Loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False)

        # 7. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                # 获取数据
                uavimgs = batch['uav_imgs'].to(self.device)  # [B, C, H, W]
                satimgs_pos = batch['sat_imgs_pos'].to(self.device)  # [B, C, H, W]
                satimgs_neg = batch['sat_imgs_neg'].to(self.device)  # [B, C, H, W]

                # 视觉编码器提取特征（冻结）
                imgs_input = torch.cat([uavimgs, satimgs_pos, satimgs_neg], dim=0)
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

                # 构建正样本mask
                if not hasattr(self, 'pos_mask_mat'):
                    self.pos_mask_mat = torch.cat([
                        torch.eye(B),
                        torch.zeros([B, B])
                    ], dim=-1).bool()  # [B, 2B]

                # 计算Triplet Loss
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


if __name__ == "__main__":
    trainer = VisualEncoderTrainer()
    trainer.train()
