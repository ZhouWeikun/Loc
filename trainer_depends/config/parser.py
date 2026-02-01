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
    优先级：命令行参数 > 实验日志YAML > 基础配置YAML > 默认参数
    """
    parser = argparse.ArgumentParser(description='Training')

    # ==================== 核心配置 ====================
    parser.add_argument('--p_yaml',
                        default='trainer_depends/configs/stage1_visual_encoder.yaml',
                        type=str, help='YAML配置文件路径 (通常指向opts.yaml或基础配置)')

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
    parser.add_argument('--p_satinfo_json', default='', type=str, help='卫星图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uavinfo_json', default='', type=str, help='UAV图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uav_geocsv', default='', type=str, help='UAV地理信息CSV路径（向后兼容）')
    parser.add_argument('--dataset_name', default='default', type=str, help='数据集名称（向后兼容）')

    # --- 1. 初始解析，获取命令行参数 ---
    opt = parser.parse_args()
    default_args = parser.parse_args(args=[]) # 保存默认值用于比较
    
    # 获取当前文件 (parser.py) 的绝对路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # 从当前文件目录向上两级到达项目根目录
    project_root = os.path.dirname(os.path.dirname(current_file_dir))

    # --- 2. 合并配置字典的函数 ---
    def merge_configs(base_dict, override_dict):
        """
        递归合并字典：override_dict 中的值会覆盖 base_dict 中的值。
        """
        for k, v in override_dict.items():
            if isinstance(v, dict) and isinstance(base_dict.get(k), dict):
                base_dict[k] = merge_configs(base_dict[k], v)
            else:
                base_dict[k] = v
        return base_dict

    # --- 3. 加载基础配置和实验日志配置 ---
    config_from_base_yaml = {}
    config_from_exp_log = {}
    
    # 尝试加载命令行参数指定的YAML文件 (可能是基础配置，也可能是实验日志)
    if opt.p_yaml:
        first_yaml_path_abs = os.path.join(project_root, opt.p_yaml)
        if os.path.exists(first_yaml_path_abs):
            print(f"从命令行指定YAML文件 '{first_yaml_path_abs}' 加载配置...")
            with open(first_yaml_path_abs, 'r', encoding='utf-8') as f:
                temp_config = yaml.safe_load(f)
            
            # 检查这个YAML文件是否是实验日志 (通过是否存在p_yaml键来判断)
            if 'p_yaml' in temp_config or ('exp_setting' in temp_config and 'p_yaml' in temp_config['exp_setting']):
                # 这是一个实验日志
                config_from_exp_log = temp_config
                base_yaml_path_rel = config_from_exp_log.get('p_yaml') or (config_from_exp_log.get('exp_setting', {})).get('p_yaml')

                # 尝试加载基础配置文件（作为默认值的补充）
                # 如果实验日志已经包含完整配置，基础配置文件可以不存在
                if base_yaml_path_rel:
                    base_yaml_path_abs = os.path.join(project_root, base_yaml_path_rel)
                    if os.path.exists(base_yaml_path_abs):
                        print(f"从实验日志中解析到基础配置文件 '{base_yaml_path_abs}' 加载配置（作为默认值补充）...")
                        with open(base_yaml_path_abs, 'r', encoding='utf-8') as f:
                            config_from_base_yaml = yaml.safe_load(f)
                    else:
                        print(f"提示: 基础配置文件 '{base_yaml_path_abs}' 不存在，将使用实验日志中的完整配置。")
                else:
                    print(f"提示: 实验日志 '{first_yaml_path_abs}' 中未指定基础配置文件，将使用实验日志中的完整配置。")
            else:
                # 这是一个基础配置文件
                config_from_base_yaml = temp_config
                print(f"检测到 '{first_yaml_path_abs}' 是基础配置文件。")
        else:
            print(f"警告: 命令行指定的YAML文件 '{first_yaml_path_abs}' 不存在。")

    # 如果没有通过命令行指定，或者命令行指定的是exp_log但exp_log中没有p_yaml，则加载默认的基础配置文件
    if not config_from_base_yaml:
        default_base_yaml_path_rel = default_args.p_yaml
        default_base_yaml_path_abs = os.path.join(project_root, default_base_yaml_path_rel)
        if os.path.exists(default_base_yaml_path_abs):
            print(f"加载默认基础配置文件 '{default_base_yaml_path_abs}' 加载配置...")
            with open(default_base_yaml_path_abs, 'r', encoding='utf-8') as f:
                config_from_base_yaml = yaml.safe_load(f)
        else:
            print(f"警告: 默认基础配置文件 '{default_base_yaml_path_abs}' 不存在。")

    # --- 4. 合并配置 (基础配置 -> 实验日志) ---
    final_merged_config = merge_configs(config_from_base_yaml, config_from_exp_log)

    # --- 5. 将合并后的配置应用到opt对象 (尊重命令行参数) ---
    # 先将合并后的YAML配置应用到opt对象
    for section, params in final_merged_config.items():
        if section == 'scenes_setting':
            setattr(opt, 'scenes_setting', params)
            continue

        if isinstance(params, dict):
            for key, value in params.items():
                # 只有当命令行参数是其默认值时，才允许YAML覆盖
                if hasattr(opt, key) and getattr(opt, key) == getattr(default_args, key):
                    setattr(opt, key, value)
                elif not hasattr(opt, key): # YAML-only参数，直接添加
                    setattr(opt, key, value)
        else: # 顶级参数
            if hasattr(opt, section) and getattr(opt, section) == getattr(default_args, section):
                setattr(opt, section, params)
            elif not hasattr(opt, section): # YAML-only参数，直接添加
                setattr(opt, section, params)
    
    # --- 6. 检查并设置场景配置 (向后兼容) ---
    if not hasattr(opt, 'scenes_setting') or not opt.scenes_setting:
        print("警告：未找到 scenes_setting，将使用命令行参数作为单场景配置")
        # 如果 YAML 中没有 scenes_setting，使用旧的单场景配置方式（向后兼容）
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

    # --- 7. 打印场景配置信息 ---
    num_scenes = len(opt.scenes_setting['scenes'])
    print(f"\n{'='*60}")
    print(f"场景配置: {'多场景模式' if num_scenes > 1 else '单场景模式'} ({num_scenes}个场景)")
    for i, scene in enumerate(opt.scenes_setting['scenes']):
        print(f"  场景{i+1}: {scene['name']}")
    if num_scenes > 1:
        print(f"  采样策略: {opt.scenes_setting['sampling_strategy']}")
    print(f"{'='*60}\n")

    # --- 8. 组织参数到 group_dict ---
    group_info = {
        'exp_setting': ['p_yaml', 'exp_name', 'exps_dir', 'load2train', 'load2test',
                        'load_stage1_ckpt', 'load_stage2_ckpt', 'tensorboard',
                        'save_freq', 'val', 'val_freq',
                        'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'],
        'data_setting': ['imgsize2net', 'satimgsize2crop', 'n_rand2sample_per_pos'],
        'hardware_setting': ['gpu_ids', 'num_worker', 'autocast',
                            'batchsize_sat', 'batchsize_uav', 'batchsize_uav_test'],
        'network_setting': ['backbone', 'aggregator_type', 'freeze_grid'],
        'learning_setting': ['num_epochs'],
    }
    opt.group_dict = group_info

    # --- 9. 最终打印和返回 ---
    print("最终配置:")
    for group_name, keys in opt.group_dict.items():
        print(f"  [{group_name}]")
        for key in keys:
            if hasattr(opt, key):
                print(f"    {key}: {getattr(opt, key)}")
    if hasattr(opt, 'scenes_setting'):
        print(f"  [scenes_setting]: {opt.scenes_setting}")

    return opt
