#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer基类

提供所有Trainer共享的基础功能：
- 设备管理
- 数据集初始化
- Checkpoint管理
- 日志管理
"""

import torch
import os
from torch.utils.tensorboard import SummaryWriter

from trainer_depends.config.parser import get_parse
from trainer_depends.utils.logger_utils import get_logger
from trainer_depends.datasets.dataset_neuloc_4d import UAVDataset, SatDataset


class BaseTrainer:
    """
    所有Trainer的基类

    提供基础功能：
    - 配置管理
    - 设备初始化
    - 数据集加载
    - Checkpoint加载/保存
    - 日志管理
    """

    def __init__(self, opt=None):
        """
        初始化基类

        Args:
            opt: 配置对象，如果为None则调用get_parse()
        """
        # 1. 配置管理
        self.opt = opt if opt is not None else get_parse()

        # 2. 设备配置
        self._init_device()

        # 3. 日志（延迟初始化，在train()中调用）
        self.logger = None
        self.writer = None
        self.exp_dir2save = None
        self.log_dir2save = None
        self.ckpt_dir2save = None
        self._exp_backup_saved = False


    def _init_device(self):
        """初始化设备（GPU/CPU）"""
        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False

        self.device = device
        print(f"✅ 设备初始化: {self.device}")


    def _get_train_log_filename(self, exp_name):
        """返回训练日志文件名。子类可按需覆盖。"""
        return "train.log"


    def _init_logger(self, exp_name=None):
        """
        初始化日志系统

        Args:
            exp_name: 实验名称，如果为None则使用opt.exp_name
        """
        opt = self.opt

        if exp_name is None:
            exp_name = opt.exp_name

        log_root = getattr(opt, 'dir2save_log', None) or getattr(opt, 'exps_dir', 'gen_fm_exps/logs')
        ckpt_root = getattr(opt, 'dir2save_ckpt', None) or log_root
        os.makedirs(log_root, exist_ok=True)

        exp_name_unique = exp_name
        counter = 0
        while (
            os.path.exists(os.path.join(log_root, exp_name_unique))
            or os.path.exists(os.path.join(ckpt_root, exp_name_unique))
        ):
            counter += 1
            exp_name_unique = f"{exp_name}_{counter}"

        opt.exp_name = exp_name_unique
        self.log_dir2save = os.path.join(log_root, exp_name_unique)
        self.ckpt_dir2save = os.path.join(ckpt_root, exp_name_unique)
        self.exp_dir2save = self.ckpt_dir2save
        os.makedirs(self.log_dir2save, exist_ok=True)

        # 初始化logger
        log_filename = self._get_train_log_filename(exp_name_unique)
        log_path = os.path.join(self.log_dir2save, log_filename)
        self.logger = get_logger(log_path, 'trainer_logger')
        self.logger.info(f"日志目录: {self.log_dir2save}")
        self.logger.info(f"日志文件: {log_path}")
        self.logger.info(f"Checkpoint目录: {self.ckpt_dir2save}")

        # 初始化tensorboard
        if opt.tensorboard:
            tb_path = os.path.join(self.log_dir2save, "train_tensorboard.log")
            self.writer = SummaryWriter(tb_path)
            self.logger.info(f"Tensorboard日志: {tb_path}")

        print(f"✅ 日志初始化完成: {exp_name_unique}")


    def _init_datasets(self, create_train_loader=False):
        """
        初始化多场景数据集

        Args:
            create_train_loader: 是否创建训练用的dataloader
        """
        opt = self.opt
        scenes = opt.scenes_setting['scenes']

        # Choose where satellite tensors are cached/cropped.
        sat_device = self.device
        if getattr(opt, "satmaps_on_cpu", False):
            sat_device = torch.device("cpu")
            msg = "satmaps_on_cpu=True; using CPU for SatDataset cache and cropping."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        elif getattr(opt, "num_worker", 0) > 0 and self.device.type == "cuda":
            sat_device = torch.device("cpu")
            msg = ("num_worker>0 with CUDA SatDataset triggers CUDA init in forked workers; "
                   "using CPU for SatDataset. Set num_worker=0 to keep CUDA dataset ops.")
            if self.logger:
                self.logger.warning(msg)
            else:
                print(f"Warning: {msg}")

        # 存储每个场景的数据集
        self.sat_datasets = {}
        self.uav_datasets_train = {}
        self.uav_datasets_test = {}

        if create_train_loader:
            self.pair_dataloaders = {}

        pad_mode = getattr(opt, "pad_mode", None)

        for scene in scenes:
            scene_name = scene['name']

            # 日志输出
            log_msg = f"正在初始化场景: {scene_name}"
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)

            # 创建SatDataset
            sat_dataset = SatDataset(
                p_satinfo_json=scene['p_satinfo_json'],
                p_uav_geocsv=scene['p_uav_geocsv'],
                imgsize2net=opt.imgsize2net,
                name=scene_name,
                device=sat_device,
                pad_mode=pad_mode,
            )
            self.sat_datasets[scene_name] = sat_dataset

            # 创建UAVDataset (train)
            uav_dataset_train = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                p_uav_geocsv=scene['p_uav_geocsv'],
                sat_dataset=sat_dataset,
                stage='train',
                name=scene_name,
                device='cpu',
                pad_mode=pad_mode,
                dataset_name=self.opt.scenes_setting['dataset_name'],
                split_train_ratio=opt.split_train_ratio,
                split_mode=opt.split_mode,
            )
            self.uav_datasets_train[scene_name] = uav_dataset_train

            # 创建UAVDataset (test)
            uav_dataset_test = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                p_uav_geocsv=scene['p_uav_geocsv'],
                sat_dataset=sat_dataset,
                stage='test',
                name=scene_name,
                pad_mode=pad_mode,
                dataset_name=self.opt.scenes_setting['dataset_name'],
                split_train_ratio=opt.split_train_ratio,
                split_mode=opt.split_mode,
            )
            self.uav_datasets_test[scene_name] = uav_dataset_test

            # 创建训练用的DataLoader（如果需要）
            if create_train_loader:
                # 这里可以添加特定的DataLoader创建逻辑
                # 例如使用 UAVSatPairDataset + MultiSceneDataLoader
                pass

        # 保存第一个场景作为主数据集（用于UDF等计算）
        first_scene_name = scenes[0]['name']
        self.sat_dataset = self.sat_datasets[first_scene_name]
        self.uav_dataset_train = self.uav_datasets_train[first_scene_name]
        self.uav_dataset_test = self.uav_datasets_test[first_scene_name]

        if self.logger:
            self.logger.info(f"✅ 数据集初始化完成，共{len(scenes)}个场景")
        else:
            print(f"✅ 数据集初始化完成，共{len(scenes)}个场景")


    def _load_checkpoint(self, ckpt_config, modules_dict, optimizer=None, mode='train'):
        """
        加载checkpoint

        Args:
            ckpt_config: checkpoint配置（dict或str）
            modules_dict: 需要加载的模块字典
                         例如 {'vis_encoder': self.vis_encoder, 'grid': self.grid}
            optimizer: 优化器对象（仅在mode='train'且继续训练时需要）
            mode: 'train' 或 'test'

        Returns:
            begin_epoch: 起始epoch（如果是新训练则为0）
        """
        from tool.util_ckpt_handler import load_param

        begin_epoch = 0

        # 处理dict格式的checkpoint配置
        if isinstance(ckpt_config, dict):
            for module_name, ckpt_path in ckpt_config.items():
                if ckpt_path and ckpt_path != "":
                    if module_name in modules_dict:
                        load_param(ckpt_path, {module_name: modules_dict[module_name]})
                        msg = f"✅ 加载{module_name}模块: {ckpt_path}"
                        if self.logger:
                            self.logger.info(msg)
                        else:
                            print(msg)

        # 处理str格式的checkpoint配置
        elif isinstance(ckpt_config, str) and ckpt_config != "":
            # 加载所有模块
            params_to_load = {**modules_dict}
            if optimizer is not None and mode == 'train':
                params_to_load['optimizer_state'] = optimizer
            loaded_params = load_param(ckpt_config, params_to_load)

            # 获取epoch信息
            if 'epoch' in loaded_params:
                begin_epoch = loaded_params['epoch'] + 1

            msg = f"✅ 加载checkpoint: {ckpt_config}, 从epoch {begin_epoch}继续训练"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        return begin_epoch


    def _save_checkpoint(self, epoch, modules_dict, optimizer=None):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch
            modules_dict: 需要保存的模块字典
            optimizer: 优化器对象（可选）
        """
        from tool.util_ckpt_handler import save_param

        # 准备要保存的参数
        params_to_save = {**modules_dict}

        # 添加优化器和epoch信息
        if optimizer is not None:
            params_to_save['optimizer_state'] = optimizer
        params_to_save['epoch'] = epoch

        # 保存
        save_dir = self.ckpt_dir2save or self.exp_dir2save or self.opt.exp_name
        save_param(save_dir, params_to_save)

        if (not self._exp_backup_saved) and save_dir:
            try:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(save_dir, self.opt)
                self._exp_backup_saved = True
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"首次checkpoint后备份实验信息失败: {exc}")
                else:
                    print(f"Warning: 首次checkpoint后备份实验信息失败: {exc}")

        if self.logger:
            self.logger.info(f"✅ 已保存checkpoint: epoch {epoch}, dir={save_dir}")


    def _get_feats_fm_imgs(self, imgs):
        """
        从图像提取特征

        Args:
            imgs: [B, C, H, W] 图像tensor

        Returns:
            feats: [B, feat_dim] 特征向量
        """
        with torch.no_grad():
            feats_patch = self.vis_encoder(imgs)
            feats = self.vis_aggregator(feats_patch)
        return feats


    def train(self):
        """
        训练主循环

        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现train()方法")


    def test(self):
        """
        测试/评估

        子类可选实现
        """
        raise NotImplementedError("子类必须实现test()方法")
