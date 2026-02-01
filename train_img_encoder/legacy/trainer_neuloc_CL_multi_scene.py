# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
# from torch.ao.nn.quantized.functional import threshold
import numpy as np
import time
# from triton.language import dtype
# from tool.utils import load_network_wstate, save_network_wstate
import warnings
from torch.utils.tensorboard import SummaryWriter
import glob

warnings.filterwarnings("ignore")
import yaml
import os
import json
# custom:
# from datasets.make_dataloader import make_dataloader
# from datasets_custom.make_dataloder_classify import make_dataloader_train
# from datasets_custom.make_dataloader_xmu import make_dataloader_xmu
# from datasets_custom.make_dataloader_gta import make_dataloader_gta
# from datasets_custom.make_dataloader_wingtra import make_dataloader_wingtra
# from exps.exp24.datasets_custom.make_dataloader_dsalad import  make_dataloader
# from train_img_encoder.nets_taskflow import mk_vis_encoder
from tool.utils_fm_duav import get_logger, get_unique_exp_dir
from dataset_wingtra_4d import UAVDataset, SatDataset
from util_udf_computer import UDFComputer


def get_parse():
    parser = argparse.ArgumentParser(description='Training')

    # ==================== 核心配置 ====================
    # YAML配置文件路径
    parser.add_argument('--p_yaml',
                        default='/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/opts_cl_wingtra_multi-scene.yaml',
                        type=str, help='YAML配置文件路径')

    # 实验配置
    parser.add_argument('--exps_dir', default='.exps/', type=str, help='实验保存目录')
    parser.add_argument('--exp_name', default='debug', type=str, help='实验名称')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='是否使用tensorboard')

    # Checkpoint配置 (支持字典或字符串，YAML中会覆盖)
    parser.add_argument('--load2test', default="", type=str, help='测试时加载的checkpoint')
    parser.add_argument('--load2train', default="", type=str, help='继续训练时加载的checkpoint')

    # 硬件配置
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs, 例如: 0 或 0,1,2')
    parser.add_argument('--num_worker', default=8, type=int, help='DataLoader worker数量')
    parser.add_argument('--autocast', action='store_true', default=False, help='是否使用混合精度训练')

    # 网络配置
    parser.add_argument('--backbone', default="dinov3", type=str,
                        help='Backbone类型: ViTB-224, dinov2, dinov3等')

    # 训练配置
    parser.add_argument('--num_epochs', default=100, type=int, help='训练轮数')

    # ==================== 向后兼容参数（用于单场景模式） ====================
    # 这些参数仅在YAML中没有scenes_setting时使用
    parser.add_argument('--p_satinfo_json', default='', type=str, help='卫星图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uavinfo_json', default='', type=str, help='UAV图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uav_geocsv', default='', type=str, help='UAV地理信息CSV路径（向后兼容）')
    parser.add_argument('--dataset_name', default='default', type=str, help='数据集名称（向后兼容）')

    # 优先级：命令行参数 > YAML 文件参数 > Python 脚本中的默认参数
    # --- 获取命令行参数的原始默认值 ---
    # parse_args([]) 或 parse_args(args=[]) 会解析空列表作为命令行参数，从而获取所有参数的默认值, 这避免了命令行参数被 YAML 覆盖
    default_args = parser.parse_args(args=[])

    # --- 解析命令行参数 (真实传入的参数) ---
    # 这一步将解析用户在命令行中实际传入的参数，如果某个参数在命令行中被指定，它将覆盖 default_args 中的值。
    opt = parser.parse_args()

    # --- 读取 YAML 文件并更新默认值 ---
    yaml_file_path = opt.p_yaml  # 获取 YAML 文件路径
    if os.path.exists(yaml_file_path):
        print(f"从 YAML 文件 '{yaml_file_path}' 加载配置...")
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            # 遍历 YAML 配置并更新 opt 对象
        for section, params in yaml_config.items():
            # 特殊处理 scenes_setting
            if section == 'scenes_setting':
                opt.scenes_setting = params
                continue
            if isinstance(params, dict):
                # 处理嵌套的配置 (例如 hardware_setting, data_setting 等)
                for key, value in params.items():
                    # 优先级：命令行参数 > YAML文件参数 > argparse默认参数
                    if hasattr(opt, key):
                        # 参数在argparse中定义过，检查是否使用默认值
                        if getattr(opt, key) == getattr(default_args, key):
                            # 是默认值，可以用YAML覆盖
                            setattr(opt, key, value)
                        # else: 命令行参数已修改，保持命令行参数
                    else:
                        # YAML-only参数（如batchsize_sat, batchsize_uav等），直接添加
                        setattr(opt, key, value)
            else:
                # 处理非嵌套的顶级配置
                if hasattr(opt, section) and getattr(opt, section) == getattr(default_args, section):
                    setattr(opt, section, params)

        # 检查并设置场景配置
        if not hasattr(opt, 'scenes_setting'):
            # 如果 YAML 中没有 scenes_setting，使用旧的单场景配置方式（向后兼容）
            print("警告：未找到 scenes_setting，将使用命令行参数作为单场景配置")
            opt.scenes_setting = {
                'sampling_strategy': 'round_robin',
                'scenes': [{
                    'name': getattr(opt, 'dataset_name', 'default'),
                    'p_satinfo_json': opt.p_satinfo_json,
                    'p_uavinfo_json': opt.p_uavinfo_json,
                    'p_uav_geocsv': opt.p_uav_geocsv,
                    'weight': 1.0
                }]
            }

        # 打印场景配置信息
        num_scenes = len(opt.scenes_setting['scenes'])
        print(f"\n{'='*60}")
        print(f"场景配置: {'多场景模式' if num_scenes > 1 else '单场景模式'} ({num_scenes}个场景)")
        for i, scene in enumerate(opt.scenes_setting['scenes']):
            print(f"  场景{i+1}: {scene['name']}")
        if num_scenes > 1:
            print(f"  采样策略: {opt.scenes_setting['sampling_strategy']}")
        print(f"{'='*60}\n")

    # --- 组织参数到 group_dict,为了后续保存为yaml时按分层组织 ---
    # 包含 argparse 参数和 YAML-only 参数，与简化后的 YAML 结构保持一致
    group_info = {
        'exp_setting': ['p_yaml', 'exp_name', 'exps_dir', 'load2train', 'load2test', 'tensorboard',
                        'save_freq', 'val', 'val_freq',  # YAML-only 参数
                        'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'],  # 向后兼容参数
        'data_setting': ['imgsize2net', 'satimgsize2crop', 'n_rand2sample_per_pos'],  # YAML-only
        'hardware_setting': ['gpu_ids', 'num_worker', 'autocast',
                            'batchsize_sat', 'batchsize_uav'],  # 后两个为 YAML-only
        'network_setting': ['backbone'],
        'learning_setting': ['num_epochs'],
        # scenes_setting 在 YAML 中单独维护，不包含在 group_dict 中
    }
    opt.group_dict = group_info
    print(opt)  # 打印最终的参数

    return opt


class MultiSceneDataLoader:
    """
    多场景数据加载器
    根据采样策略从多个场景的 dataloader 中采样数据

    Epoch 结束策略：
    - 每个 epoch 的迭代次数等于所有场景中最长的 dataloader 的长度
    - 较短的场景会在epoch内循环多次
    """
    def __init__(self, scene_dataloaders_dict, sampling_strategy='round_robin'):
        """
        Args:
            scene_dataloaders_dict: dict, {scene_name: dataloader}
            sampling_strategy: str, 'round_robin'/'random'/'weighted'
        """
        self.dataloaders = scene_dataloaders_dict
        self.scene_names = list(scene_dataloaders_dict.keys())
        self.sampling_strategy = sampling_strategy
        self.num_scenes = len(self.scene_names)

        # 计算总迭代次数（取所有场景的最大长度）
        self.total_iters = max(len(dl) for dl in self.dataloaders.values())

        # 初始化场景迭代器
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}

        # round_robin 模式下的场景索引
        self.current_scene_idx = 0

        # 当前 epoch 的迭代计数器
        self.current_iter = 0

    def __len__(self):
        """返回每个 epoch 的迭代次数（最长场景的长度）"""
        return self.total_iters

    def __iter__(self):
        """开始新的 epoch，重置计数器"""
        self.current_iter = 0
        self.current_scene_idx = 0
        return self

    def __next__(self):
        """根据采样策略返回下一个 batch"""
        # 检查是否完成一个 epoch
        if self.current_iter >= self.total_iters:
            raise StopIteration

        # 选择场景
        if self.sampling_strategy == 'round_robin':
            scene_name = self.scene_names[self.current_scene_idx]
            self.current_scene_idx = (self.current_scene_idx + 1) % self.num_scenes
        elif self.sampling_strategy == 'random':
            scene_name = np.random.choice(self.scene_names)
        elif self.sampling_strategy == 'weighted':
            # TODO: 实现加权采样
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

    def reset(self):
        """重置所有迭代器（通常不需要手动调用，__iter__ 会自动重置）"""
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        self.current_scene_idx = 0
        self.current_iter = 0


class Trainer(object):
    def __init__(self):
        self.opt = get_parse()

        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

        # config the img_encoder
        from models.Backbone.util_mk_backbone import make_backbone
        self.vis_encoder = make_backbone(self.opt.backbone).to(self.device)
        feat_q_dim = self.vis_encoder.output_channel
        self.feat_q_dim = feat_q_dim
        # freeze the para
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        # config the aggregator
        # self.decoder = LocalDecoderFiLM(dim=feat_q_dim,c_dim=feat_q_dim,hidden_size=1024,n_blocks=3,output_dim=1,norm_type='none',leaky=True).to(self.device)
        # from models.Head.G2M import G2M
        # self.vis_aggregator = G2M(in_channels=feat_q_dim,out_channels=feat_q_dim,rank=1024).to(self.device)
        # from models.Head.salad import SALAD
        # self.vis_aggregator = SALAD(input_feat_dim=feat_q_dim, global_token_dim=128, pathchsize=16, num_clusters=14, cluster_dim=64).to(self.device)
        from models.Head.salad_residual import SALAD_Residual
        self.vis_aggregator = SALAD_Residual(input_feat_dim=feat_q_dim, base_dim=feat_q_dim, patchsize=16,
                                             num_clusters=16, cluster_dim=64).to(self.device)
        # from models.Head.salad_film import SALAD_FiLM
        # self.vis_aggregator = SALAD_FiLM(input_feat_dim=feat_q_dim,base_dim=feat_q_dim,patchsize=16, num_clusters=16, cluster_dim=64).to(self.device)
        self.agg_name = 'salad'


        # define the param to save/laod
        self.param2optimize = {
            'vis_aggregator': self.vis_aggregator,
        }
        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
        }
        for name, module in self.param2freeze.items():
            for param in module.parameters():
                param.requires_grad = False

    def _init_datasets(self, create_train_loader=False):
        """
        初始化所有场景的数据集

        Args:
            create_train_loader: bool, 是否创建训练用的dataloader（用于train模式）
        """
        from dataset_wingtra_4d_uav_sat_pair import UAVSatPairDataset, collate_uav_sat_pair
        from util_sample_neg_nrcs import BoundedNegativeCoordinateSampler

        opt = self.opt
        scenes = opt.scenes_setting['scenes']
        num_scenes = len(scenes)

        # 存储每个场景的数据集
        self.sat_datasets = {}
        self.uav_datasets_train = {}
        self.uav_datasets_test = {}
        if create_train_loader:
            self.pair_dataloaders = {}

        for scene in scenes:
            scene_name = scene['name']
            log_msg = f"正在初始化场景: {scene_name}"
            if hasattr(self, 'logger'):
                self.logger.info(log_msg)
            else:
                print(log_msg)

            # 创建该场景的 SatDataset
            sat_dataset = SatDataset(
                p_satinfo_json=scene['p_satinfo_json'],
                p_uav_geocsv=scene['p_uav_geocsv'],
                imgsize2net=opt.imgsize2net,
            )
            self.sat_datasets[scene_name] = sat_dataset

            # 创建该场景的 UAVDataset (train)
            uav_dataset_train = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                geo_res_m=sat_dataset.geo_res_m,
                trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
                stage='train'
            )
            self.uav_datasets_train[scene_name] = uav_dataset_train

            # 创建该场景的 UAVDataset (test)
            uav_dataset_test = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                geo_res_m=sat_dataset.geo_res_m,
                trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
                stage='test'
            )
            self.uav_datasets_test[scene_name] = uav_dataset_test

            # 如果需要，创建训练用的 DataLoader
            if create_train_loader:
                # 创建该场景的 UAVSatPairDataset
                satmap_sampler = BoundedNegativeCoordinateSampler(self.device)
                pair_dataset = UAVSatPairDataset(
                    uav_dataset=uav_dataset_train,
                    sat_dataset=sat_dataset,
                    satmap_sampler=satmap_sampler,
                    device=self.device,
                    n_neg_per_sample=1,
                )

                # 创建该场景的 DataLoader
                pair_dataloader = torch.utils.data.DataLoader(
                    pair_dataset,
                    batch_size=opt.batchsize_sat,
                    num_workers=opt.num_worker,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=True,
                    persistent_workers=True,
                    collate_fn=collate_uav_sat_pair
                )
                self.pair_dataloaders[scene_name] = pair_dataloader

        # 根据场景数量选择训练模式
        first_scene = scenes[0]['name']
        if num_scenes == 1:
            # 单场景模式：直接使用第一个场景
            self.sat_dataset = self.sat_datasets[first_scene]
            self.uav_datset_trian = self.uav_datasets_train[first_scene]
            self.uav_datset_test = self.uav_datasets_test[first_scene]

            if create_train_loader:
                self.dataloader_train = self.pair_dataloaders[first_scene]
                log_msg = f"单场景模式: {first_scene}"
                if hasattr(self, 'logger'):
                    self.logger.info(log_msg)
                else:
                    print(log_msg)
        else:
            # 多场景模式
            self.sat_dataset = self.sat_datasets[first_scene]
            self.uav_datset_trian = self.uav_datasets_train[first_scene]
            self.uav_datset_test = self.uav_datasets_test[first_scene]

            if create_train_loader:
                # 使用 MultiSceneDataLoader
                self.dataloader_train = MultiSceneDataLoader(
                    self.pair_dataloaders,
                    sampling_strategy=opt.scenes_setting['sampling_strategy']
                )
                log_msg = f"多场景模式: {num_scenes}个场景, 采样策略={opt.scenes_setting['sampling_strategy']}"
                if hasattr(self, 'logger'):
                    self.logger.info(log_msg)
                else:
                    print(log_msg)

        # 为测试创建统一的 dataloader (使用第一个场景)
        if create_train_loader:
            self.uav_dataloader_test = torch.utils.data.DataLoader(
                self.uav_datset_test,
                batch_size=128,
                num_workers=self.opt.num_worker,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                persistent_workers=True
            )

        completion_msg = f"数据集初始化完成，共 {num_scenes} 个场景"
        if hasattr(self, 'logger'):
            self.logger.info(completion_msg)
        else:
            print(completion_msg)

    def _load_checkpoint(self, ckpt_config, mode='train', optimizer=None):
        """
        统一的checkpoint加载方法

        Args:
            ckpt_config: 可以是字典或字符串
                - 字典形式: {'vis_encoder': path, 'vis_aggregator': path, 'optimizer': path}
                - 字符串形式: 单个checkpoint路径（向后兼容）
            mode: 'train' 或 'test'
            optimizer: 优化器对象（仅在mode='train'且需要加载时使用）

        Returns:
            begin_epoch: 如果加载了epoch信息则返回，否则返回0
        """
        from tool.util_ckpt_handler import load_param
        begin_epoch = 0

        if ckpt_config is None:
            return begin_epoch

        if isinstance(ckpt_config, dict):
            # 字典形式：分别加载各个部件
            for key, ckpt_path in ckpt_config.items():
                if ckpt_path and len(ckpt_path) > 0:
                    print(f"[{mode}模式] 加载 {key} 从: {ckpt_path}")

                    if key == 'vis_encoder':
                        load_param(ckpt_path, {'vis_encoder': self.vis_encoder})
                    elif key == 'vis_aggregator':
                        load_param(ckpt_path, {'vis_aggregator': self.vis_aggregator})
                    elif key == 'optimizer' and optimizer is not None:
                        load_param(ckpt_path, {'optimizer_state': optimizer})
                    elif key == 'epoch':
                        # epoch信息通常包含在checkpoint中，这里可以扩展
                        pass
                    else:
                        if key != 'epoch':  # epoch是保留关键字，不警告
                            print(f"警告：未知的checkpoint键: {key}")

        elif isinstance(ckpt_config, str) and len(ckpt_config) > 0:
            # 字符串形式（向后兼容）：加载全部
            print(f"[{mode}模式] 加载checkpoint从: {ckpt_config}")

            if mode == 'test':
                # 测试模式：加载配置文件更新opt
                suffix = '.yaml'
                pattern = os.path.join(os.path.dirname(ckpt_config), f'*{suffix}')
                config_paths = glob.glob(pattern)
                if config_paths:
                    config_path = config_paths[0]
                    with open(config_path, 'r') as stream:
                        config = yaml.load(stream, Loader=yaml.FullLoader)
                    for group_dict_key, group_dict in config.items():
                        if group_dict_key == 'network_setting':
                            for cfg, value in group_dict.items():
                                setattr(self.opt, cfg, value)
                        else:
                            for cfg, value in group_dict.items():
                                if not hasattr(self.opt, cfg):
                                    setattr(self.opt, cfg, value)

                # 加载所有参数
                params_dict = {}
                params_dict.update(self.param2optimize)
                params_dict.update(self.param2freeze)
                load_param(ckpt_config, params_dict)
            else:
                # 训练模式：加载所有可训练参数
                params_to_load = dict(self.param2optimize)
                if optimizer is not None:
                    params_to_load['optimizer_state'] = optimizer
                load_param(ckpt_config, params_to_load)

        return begin_epoch


    def _test_ready(self):
        opt = self.opt

        # 使用统一的checkpoint加载方法
        self._load_checkpoint(opt.load2test, mode='test')

        # 设置为评估模式
        for k, v in self.param2optimize.items():
            v.eval()
        for k, v in self.param2freeze.items():
            v.eval()

        # config the datalaoder
        from dataset_wingtra_4d import SatDataset
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=self.opt.imgsize2net,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=1,
                                                          num_workers=self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=False,
                                                          persistent_workers=True)

    def train(self):
        opt = self.opt

        # config the optimizer :
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize,'adam')

        # load the ckpt for continuing train if necessray
        begin_epoch = self._load_checkpoint(opt.load2train, mode='train', optimizer=self.optimizer)

        # config the logger&writer (移到数据集初始化之前)
            # make the dir to save the exp
        exp_name = get_unique_exp_dir(opt.exps_dir, opt.exp_name)
        opt.exp_name = exp_name
        exp_dir2save = os.path.join(opt.exps_dir, opt.exp_name)
        os.makedirs(exp_dir2save, exist_ok=True)
            # config the logger
        logger = get_logger("{}/{}/train.log".format(opt.exps_dir, opt.exp_name), 'trainer_logger')
        self.logger = logger
        self.logger.info(f"exp ready!, exp_name={exp_name}")
            # config tensorborad if necessary
        self.writer = SummaryWriter("exps/{}/train_tensorboard.log".format(opt.exp_name)) if opt.tensorboard else None

        # 初始化多场景数据集（包括训练dataloader）
        self._init_datasets(create_train_loader=True)

        # config loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat, MSLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False, )
        loss_ms = MSLoss_fm_mat()
        loss_mse = torch.nn.MSELoss(reduction='mean')
        temperature = 1.
        targets = torch.arange(opt.batchsize_sat).to(self.device)  # [0, 1, 2, ..., B-1]

        # ready to trian
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0
        for epoch in range(begin_epoch, num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                # uavimgs, coords_q = data
                uavimgs = batch['uav_imgs'].to(
                    self.device)  # [B, C, H, W]                                                                                                                                              │ │
                satimgs_pos = batch['sat_imgs_pos'].to(
                    self.device)  # [B, C, H, W]                                                                                                                                              │ │
                satimgs_neg = batch['sat_imgs_neg'].to(
                    self.device)  # [B, C, H, W]                                                                                                                                              │ │
                # coords_q = batch['coords_uav'].to(self.device)  # [B, 4]
                # debug for vis
                # q_id = 2
                # uav2vis = self.uav_datset_trian.denormalize_img(uavimgs[q_id])
                # satimg2vis = self.sat_dataset.denormalize_img(satimgs_pos[q_id])
                # from matplotlib import pyplot as plt
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
                # axes[0].imshow(uav2vis)
                # axes[1].imshow(satimg2vis)
                # plt.tight_layout()
                # plt.show()

                # get feat from vis_encoder
                imgs_input = torch.concatenate([uavimgs, satimgs_pos, satimgs_neg], dim=0)
                feats_input = self.vis_encoder(imgs_input.to(self.device))
                if self.agg_name == 'g2m':
                    feats_input = self.vis_aggregator(
                        feats_input[:, 1:, :].permute(0, 2, 1).reshape(feats_input.shape[0], self.feat_q_dim, 14,14))  # for patchsize=16,imgszie=224
                elif self.agg_name == 'salad':
                    feats_input = self.vis_aggregator(feats_input)  # aggregator = salad
                feats_q, feats_ref = feats_input[:uavimgs.shape[0]], feats_input[uavimgs.shape[0]:]
                feat_dist_mat = torch.norm(feats_q.unsqueeze(1) - feats_ref.unsqueeze(0), p=2, dim=-1)
                # feat_dist_mat_np = feat_dist_mat.detach().cpu().numpy()

                if not hasattr(self, 'pos_mask_mat'):
                    self.pos_mask_mat = torch.concatenate(
                        [torch.eye(uavimgs.shape[0]), torch.zeros([feats_q.shape[0], satimgs_neg.shape[0]])],
                        dim=-1).bool()
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

                # 输出loss指标
                if it % 10 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss, step)
                    recall1 = (torch.argmin(feat_dist_mat, dim=-1) == torch.arange(0, feat_dist_mat.shape[0],
                                                                                   device=feat_dist_mat.device)).sum() / \
                              feat_dist_mat.shape[0]
                    # 在多场景模式下显示场景名称
                    scene_info = f" [{batch.get('scene_name', 'unknown')}]" if len(opt.scenes_setting['scenes']) > 1 else ""
                    self.logger.info(f'training set{scene_info} recall1={recall1.item():.4f}')
                step = step + 1

            # modify the learning rate
            # scheduler.step()

            # save the network's para
            if (epoch % 5 == 0) and (epoch > 0):  # for running
                # if epoch % 2 == 0: #for debugging
                # if (epoch == 10) or (epoch % 10 == 9 and epoch >= 110):
                # if ((epoch > 0) and (epoch % opt.save_freq == 0)) or ( epoch % 10 == 9 and epoch >= 110 ):
                from tool.util_ckpt_handler import save_param
                params_to_add = {
                    "optimizer_state": self.optimizer,
                    "epoch": epoch,
                }
                self.param2optimize.update(params_to_add)
                save_param(opt.exp_name, self.param2optimize)
                # eval
                self.test_xy_scale_fm_vis_encoder()
                # log info:
                feat_mean_per_dim = torch.mean(feats_q, dim=0)
                self.logger.info(f"feat的均值: {feat_mean_per_dim.mean().item():.6f}")
                feat_var_per_dim = torch.var(feats_q, dim=0)
                self.logger.info(f"feat方差的均值: {feat_var_per_dim.mean().item():.6f}")
                feat_var_of_var = torch.var(feat_var_per_dim) # 方差的方差（衡量特征维度的差异性）
                self.logger.info(f"feat方差的方差: {feat_var_of_var.item():.6f}")

                if self.writer is not None:
                    self.writer.add_scalar(' feat_mean', feat_mean_per_dim.mean().item(), epoch)
                    self.writer.add_scalar(' feat_var_mean', feat_var_per_dim.mean().item(), epoch)
                    self.writer.add_scalar(' feat_var_var', feat_var_of_var.item(), epoch)

            # log info
            str2log = ''
            str2log += f'epcoh={epoch}'
            logger.info(f'loss={loss.item()}')
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('epoch{:.0f} finished in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 6))
            logger.info('-' * 50)
            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss, epoch)

            # backup the py info, at least have trained a epoch
            if epoch == 0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(exp_dir2save, self.opt)

    """"##############################
    functions awaiting internal invocation
    ##############################"""

    def _get_feats_fm_imgs(self, imgs_flatten, feat_fm_agg=True):
        batchsize = 512
        feat_galllery = []
        for batch in torch.split(imgs_flatten, batchsize, dim=0):
            feat = self.vis_encoder(batch.to(self.device)
                                    )
            if feat_fm_agg:
                if self.agg_name == 'salad':
                    feat = self.vis_aggregator.forward(feat)
                elif self.agg_name == 'g2m':
                    feat = self.vis_aggregator.forward(
                        feat[:, 1:, :].permute(0, 2, 1).reshape(feat.shape[0], self.feat_q_dim, 14, 14), normalize=True)
            else:
                feat = feat[:, 0, :]
            feat_galllery.append(feat.detach().cpu())
        feat_galllery = torch.cat(feat_galllery)
        return feat_galllery


    """"##############################
    functions about testing
    ##############################"""

    def test_xy_scale_fm_vis_encoder(self):
        """
        测试所有场景的 xy_scale 性能
        支持两种模式：
        1. 训练中调用：数据集已初始化，直接测试
        2. 独立测试：需要初始化数据集和加载checkpoint
        """
        # 检查是否为独立测试模式（数据集未初始化）
        is_standalone_test = not hasattr(self, 'sat_datasets') or not self.sat_datasets

        if is_standalone_test:
            print("检测到独立测试模式，正在初始化...")
            # 初始化数据集
            self._init_datasets()
            # 加载checkpoint
            if hasattr(self.opt, 'load2test') and self.opt.load2test:
                print("加载测试checkpoint...")
                self._load_checkpoint(self.opt.load2test, mode='test')
            # 设置为评估模式
            for k, v in self.param2optimize.items():
                v.eval()
            for k, v in self.param2freeze.items():
                v.eval()
        else:
            print("检测到训练中调用模式，使用已有数据集")

        # 测试所有场景
        scenes = self.opt.scenes_setting['scenes']
        for scene in scenes:
            scene_name = scene['name']
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_test = self.uav_datasets_test[scene_name]
            self._test_single_scene_xy_scale(scene_name, sat_dataset, uav_dataset_test)

    def _test_single_scene_xy_scale(self, scene_name, sat_dataset, uav_dataset_test):
        """测试单个场景的 xy_scale 性能"""
        print(f"\n{'='*60}")
        print(f"测试场景: {scene_name}")
        print(f"{'='*60}")

        overlap = 0.25

        # 为该场景创建测试dataloader
        uav_dataloader_test = torch.utils.data.DataLoader(
            uav_dataset_test,
            batch_size=128,
            num_workers=self.opt.num_worker,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=False  # 测试时不需要持久化
        )

        # ==================== 生成尺度列表 ====================
        n_scales = 3
        scale_list, satimgsize_list = sat_dataset.mk_sacle_levels(n_scales)
        print(f"\n尺度列表:")
        for i, (scale, imgsize) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"  Level {i}: scale={scale:.3f}, imgsize={imgsize:.1f}px")

        # ==================== 构造特征库 ====================
        gallery_features = []  # 存储所有尺度的特征
        gallery_coords = []  # 存储对应的4D坐标
        gallery_shape = []
        for scale_idx, (scale, satimgsize2crop) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"\n{'=' * 60}")
            print(f"处理尺度 {scale_idx + 1}/{n_scales}: scale={scale:.3f}")
            print(f"{'=' * 60}")

            # ========== 1. 均匀裁剪卫星地图 ==========
            sat_tiles, nrcs_gallery = sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap
            )
            n_rows, n_cols = sat_tiles.shape[:2]
            print(f"  裁剪网格大小: {n_rows} x {n_cols} = {n_rows * n_cols} 个位置")

            # resize到网络输入尺寸
            sat_tiles_resized = sat_dataset.scale_transform(
                sat_tiles.flatten(start_dim=0, end_dim=1))  # [n_pos, C, H, W]

            # ========== 3. 批量提取特征 ==========
            print(f"步骤3: 提取特征...")
            feats_all = self._get_feats_fm_imgs(sat_tiles_resized)

            # ========== 4. 构造4D坐标 ==========
            print(f"步骤4: 构造4D坐标...")
            nrcs_gallery_torch = torch.from_numpy(nrcs_gallery)  # [n_rows, n_cols, 2]
            rots_zero = torch.zeros((n_rows, n_cols, 1))  # [n_rows, n_cols, 1]
            scales_expanded = torch.full((n_rows, n_cols, 1), scale)  # [n_rows, n_cols, 1]
            coords_4d = torch.cat([
                nrcs_gallery_torch,  # [n_rows, n_cols, 2]
                rots_zero,  # [n_rows, n_cols, 1]
                scales_expanded  # [n_rows, n_cols, 1]
            ], dim=-1)
            # Flatten坐标: [n_rows * n_cols * n_rots, 4]
            coords_4d_flat = coords_4d.flatten(start_dim=0, end_dim=1)

            gallery_features.append(feats_all)
            gallery_coords.append(coords_4d_flat)
            gallery_shape.append(torch.Size([n_rows, n_cols]))

        feat_gallery_flatten_all = torch.concatenate(gallery_features, dim=0)
        coords_gallery_flatten_all = torch.concatenate(gallery_coords, dim=0)

        # sample query from dataloader
        uavimgs, coords_uav = next(iter(uav_dataloader_test))
        rot_to_align_deg = torch.rad2deg(-coords_uav[:, 2]).cpu().numpy()  # 逆向旋转角度（度数）
        # 旋转UAV图像
        from util_batch_rotation import batch_rotate_images_per_sample
        uavimgs_wo_rot = batch_rotate_images_per_sample(
            uavimgs,  # [B, C, H, W]
            rot_to_align_deg  # [B] - 每个图像对应一个角度
        )  # 输出: [B, C, H, W]
        coords_uav_wo_rot = coords_uav.clone()
        coords_uav_wo_rot[:, 2] = 0  # rot = 0

        # ========== 采样对应的正样本卫星图（也是rot=0） ==========
        satimgs_pos = sat_dataset.crop_satimg_by_4d_coords(coords_uav_wo_rot)
        with torch.no_grad():
            feats_q = self._get_feats_fm_imgs(uavimgs_wo_rot)
            feats_pos = self._get_feats_fm_imgs(satimgs_pos)

        # eval
        import faiss
        topN = 50
        feat_gallery_index_l2 = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index_l2.add(feat_gallery_flatten_all.detach().cpu().numpy())
        feat_dist_l2, indices_l2 = feat_gallery_index_l2.search(feats_q.detach().cpu().numpy(), k=topN)

        coords_gallery_topN = coords_gallery_flatten_all[indices_l2[:, :topN]]
        dist_nrc_topN = torch.norm(
            coords_uav[:, None, :2].to(coords_gallery_topN.device) - coords_gallery_topN[:, :, :2], p=2, dim=-1)
        nrc_loc_success = dist_nrc_topN < sat_dataset.halfimg_radius_nrc
        k_values = [1, 5, 10, 20, 50]
        recalls = [(nrc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log = f"[{scene_name}] Recall@K: " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)])
        print(info2log)
        if hasattr(self, 'logger'):
            self.logger.info(info2log)

        # 估检索到的 top1 特征与真实positive 特征的质量差异
        # feat_dist_pos2q = torch.norm(feats_pos-feats_q,dim=-1,p=2)
        # margin = feat_dist_pos2q - feat_dist_l2[:,0]
        # ratio = feat_dist_pos2q / feat_dist_l2[:,0]
        # 可视化响应分布
        # res_map = gallery_features[0] @ feats_q[0].T
        # res_map = res_map.reshape(gallery_shape[0])
        # from util_vis_retrieval_in_2d import visualize_response_map_3d
        # visualize_response_map_3d(torch.exp(-res_map))


    def test_xy_rot_scale_fm_vis_encoder(self):#todo:待完善
        # self._test_ready()
        overlap = 0.25

        # ==================== 生成尺度列表 ====================
        n_scales = 3
        scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
        print(f"\n尺度列表:")
        for i, (scale, imgsize) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"  Level {i}: scale={scale:.3f}, imgsize={imgsize:.1f}px")

        # ==================== 准备旋转器 ====================
        delta_rot_rangle = 20
        rots_deg = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
        rots_rad = torch.tensor(np.deg2rad(np.stack(rots_deg)), dtype=torch.float32)
        n_rots = len(rots_rad)

        # ==================== 构造特征库 ====================
        gallery_features = []  # 存储所有尺度的特征
        gallery_coords = []  # 存储对应的4D坐标
        gallery_shape = []
        for scale_idx, (scale, satimgsize2crop) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"\n{'=' * 60}")
            print(f"处理尺度 {scale_idx + 1}/{n_scales}: scale={scale:.3f}")
            print(f"{'=' * 60}")

            # ========== 1. 均匀裁剪卫星地图 ==========
            sat_tiles, nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap
            )
            n_rows, n_cols = sat_tiles.shape[:2]
            print(f"  裁剪网格大小: {n_rows} x {n_cols} = {n_rows * n_cols} 个位置")

            # Flatten位置维度
            sat_tiles_flatten = sat_tiles.flatten(start_dim=0, end_dim=1)  # [n_pos, C, H, W]
            # nrcs_gallery_flatten = torch.from_numpy(nrcs_gallery).flatten(start_dim=0, end_dim=1)  # [n_pos, 2]
            # n_positions = sat_tiles_flatten.shape[0]

            # ========== 2. 对每个位置旋转多个角度 ==========
            # 预处理：resize到网络输入尺寸
            sat_tiles_resized = self.sat_dataset.scale_transform(sat_tiles_flatten)  # [n_pos, C, H, W]

            # 旋转所有角度
            from util_batch_rotation import batch_rotate_images
            sat_tiles_rotated = batch_rotate_images(sat_tiles_resized, rots_deg)

            # ========== 3. 批量提取特征 ==========
            print(f"步骤3: 提取特征...")
            feats_all = self._get_feats_fm_imgs(sat_tiles_rotated.flatten(start_dim=0, end_dim=1))

            # ========== 4. 构造4D坐标 ==========
            print(f"步骤4: 构造4D坐标...")
            nrcs_gallery_torch = torch.from_numpy(nrcs_gallery)
            # 扩展nrcs: [n_rows, n_cols, 2] -> [n_rows, n_cols, n_rots, 2]
            nrcs_expanded = nrcs_gallery_torch[:, :, None, :].repeat(1, 1, n_rots, 1)
            # 扩展rots: [n_rots] -> [n_rows, n_cols, n_rots, 1]
            rots_expanded = rots_rad[None, None, :, None].repeat(n_rows, n_cols, 1, 1)
            # 扩展scales: scalar -> [n_rows, n_cols, n_rots, 1]
            scales_expanded = torch.full((n_rows, n_cols, n_rots, 1), scale)
            # 拼接: [n_rows, n_cols, n_rots, 4]
            coords_4d = torch.cat([
                nrcs_expanded,  # 已经是tensor
                rots_expanded,
                scales_expanded
            ], dim=-1)
            # Flatten坐标: [n_rows * n_cols * n_rots, 4]
            coords_4d_flat = coords_4d.flatten(start_dim=0, end_dim=2)

            gallery_features.append(feats_all)
            gallery_coords.append(coords_4d_flat)
            gallery_shape.append(torch.Size([n_rows, n_cols, n_rots]))

        feat_gallery_flatten_all = torch.concatenate(gallery_features, dim=0)
        coords_gallery_flatten_all = torch.concatenate(gallery_coords, dim=0)

        # sample query from dataloader
        uavimgs, coords_uav = next(iter(self.uav_dataloader_test))
        satimgs_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_uav)
        with torch.no_grad():
            feats_q = self._get_feats_fm_imgs(uavimgs)
            feats_pos = self._get_feats_fm_imgs(satimgs_pos)

        # eval
        import faiss
        topN = 50
        feat_gallery_index_l2 = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index_l2.add(feat_gallery_flatten_all.detach().cpu().numpy())
        feat_dist_l2, indices_l2 = feat_gallery_index_l2.search(feats_q.detach().cpu().numpy(), k=topN)

        coords_gallery_topN = coords_gallery_flatten_all[indices_l2[:, :topN]]
        dist_nrc_topN = torch.norm(
            coords_uav[:, None, :2].to(coords_gallery_topN.device) - coords_gallery_topN[:, :, :2], p=2, dim=-1)
        nrc_loc_success = dist_nrc_topN < self.sat_dataset.halfimg_radius_nrc * 2
        k_values = [1, 5, 10, 20, 50]
        recalls = [(nrc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        print(f"Recall@K: " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)]))

        # 估检索到的 top1 特征与真实positive 特征的质量差异
        # feat_dist_pos2q = torch.norm(feats_pos - feats_q, dim=-1, p=2)
        # margin = feat_dist_pos2q - feat_dist_l2[:, 0]
        # ratio = feat_dist_pos2q / feat_dist_l2[:, 0]

        # debug for vis
        # satimg2vis = self.sat_dataset.denormalize_img(satimgs_pos[1])
        # uav2vis = self.uav_datset_test.denormalize_img(uavimgs[1])
        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
        # axes[0].imshow(uav2vis)
        # axes[1].imshow(satimg2vis)
        # plt.tight_layout()
        # plt.show()


if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(2025)
    trainer = Trainer()
    # trainer.train()
    trainer.test_xy_scale_fm_vis_encoder()
    # trainer.test_xy_rot()
    # trainer.mk_map_feats()
    # trainer.test_xy()
    # trainer.output_test_res()
    # trainer.test_rot()
    # trainer.test_radon_wo_translate()
    # trainer.test_radon_wo_translate_crossdomain()