#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置解析模块

提供统一的配置解析功能，支持：
- 命令行参数
- YAML配置文件
- 多场景配置
"""

import argparse
import yaml
import os


def get_parse():
    """
    解析命令行参数和YAML配置

    优先级：命令行参数 > YAML文件参数 > 默认参数

    Returns:
        opt: 配置对象
    """
    parser = argparse.ArgumentParser(description='Training')

    # ==================== 核心配置 ====================
    # YAML配置文件路径
    parser.add_argument('--p_yaml',
                        default='/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/opts_cl_wingtra_metricnet.yaml',
                        type=str, help='YAML配置文件路径')

    # 实验配置
    parser.add_argument('--exps_dir', default='.exps/', type=str, help='实验保存目录')
    parser.add_argument('--exp_name', default='debug', type=str, help='实验名称')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='是否使用tensorboard')

    # Checkpoint配置 (支持字典或字符串，YAML中会覆盖)
    parser.add_argument('--load2test', default="", type=str, help='测试时加载的checkpoint')
    parser.add_argument('--load2train', default="", type=str, help='继续训练时加载的checkpoint')

    # Stage checkpoint配置（用于多阶段训练）
    parser.add_argument('--load_stage1_ckpt', default="", type=str, help='Stage 1的checkpoint路径')
    parser.add_argument('--load_stage2_ckpt', default="", type=str, help='Stage 2的checkpoint路径')

    # 硬件配置
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs, 例如: 0 或 0,1,2')
    parser.add_argument('--num_worker', default=8, type=int, help='DataLoader worker数量')
    parser.add_argument('--autocast', action='store_true', default=False, help='是否使用混合精度训练')

    # 网络配置
    parser.add_argument('--backbone', default="dinov3", type=str,
                        help='Backbone类型: ViTB-224, dinov2, dinov3等')
    parser.add_argument('--aggregator_type', default='salad', type=str,
                        help='聚合器类型: salad, g2m')

    # 训练配置
    parser.add_argument('--num_epochs', default=100, type=int, help='训练轮数')

    # Grid配置
    parser.add_argument('--freeze_grid', action='store_true', default=False,
                        help='是否冻结Grid（Stage 3中使用）')

    # ==================== 向后兼容参数（用于单场景模式） ====================
    # 这些参数仅在YAML中没有scenes_setting时使用
    parser.add_argument('--p_satinfo_json', default='', type=str, help='卫星图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uavinfo_json', default='', type=str, help='UAV图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uav_geocsv', default='', type=str, help='UAV地理信息CSV路径（向后兼容）')
    parser.add_argument('--dataset_name', default='default', type=str, help='数据集名称（向后兼容）')

    # 优先级：命令行参数 > YAML 文件参数 > Python 脚本中的默认参数
    # --- 获取命令行参数的原始默认值 ---
    default_args = parser.parse_args(args=[])

    # --- 解析命令行参数 (真实传入的参数) ---
    opt = parser.parse_args()

    # --- 读取 YAML 文件并更新默认值 ---
    yaml_file_path = opt.p_yaml
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
                # 处理嵌套的配置
                for key, value in params.items():
                    # 优先级：命令行参数 > YAML文件参数 > argparse默认参数
                    if hasattr(opt, key):
                        # 参数在argparse中定义过，检查是否使用默认值
                        if getattr(opt, key) == getattr(default_args, key):
                            # 是默认值，可以用YAML覆盖
                            setattr(opt, key, value)
                    else:
                        # YAML-only参数，直接添加
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

    # --- 组织参数到 group_dict ---
    group_info = {
        'exp_setting': ['p_yaml', 'exp_name', 'exps_dir', 'load2train', 'load2test',
                        'load_stage1_ckpt', 'load_stage2_ckpt', 'tensorboard',
                        'save_freq', 'val', 'val_freq',
                        'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'],
        'data_setting': ['imgsize2net', 'satimgsize2crop', 'n_rand2sample_per_pos'],
        'hardware_setting': ['gpu_ids', 'num_worker', 'autocast',
                            'batchsize_sat', 'batchsize_uav'],
        'network_setting': ['backbone', 'aggregator_type', 'freeze_grid'],
        'learning_setting': ['num_epochs'],
    }
    opt.group_dict = group_info

    print(opt)

    return opt
