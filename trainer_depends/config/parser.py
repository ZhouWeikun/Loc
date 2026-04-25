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
import copy


def print_config_summary(opt, header="最终配置:"):
    """打印当前opt中的配置摘要。"""
    scenes_setting = getattr(opt, 'scenes_setting', None)
    if isinstance(scenes_setting, dict):
        scenes = scenes_setting.get('scenes', [])
        print(f"\n{'='*60}")
        print(f"场景配置: {'多场景模式' if len(scenes) > 1 else '单场景模式'} ({len(scenes)}个场景)")
        for i, scene in enumerate(scenes):
            print(f"  场景{i+1}: {scene.get('name', f'scene_{i+1}')}")
        if len(scenes) > 1 and 'sampling_strategy' in scenes_setting:
            print(f"  采样策略: {scenes_setting['sampling_strategy']}")
        print(f"{'='*60}\n")

    print(header)
    for group_name, keys in getattr(opt, 'group_dict', {}).items():
        print(f"  [{group_name}]")
        for key in keys:
            if hasattr(opt, key):
                print(f"    {key}: {getattr(opt, key)}")
    if hasattr(opt, 'scenes_setting'):
        print(f"  [scenes_setting]: {opt.scenes_setting}")

    opt._config_summary_printed = True


def _expand_selected_scene_config(scenes_setting):
    if not isinstance(scenes_setting, dict):
        return scenes_setting

    scene_registry = scenes_setting.get('scene_registry', None)
    if not isinstance(scene_registry, dict) or not scene_registry:
        return scenes_setting

    selected_scene_name = scenes_setting.get('selected_scene_name', None)
    if not selected_scene_name:
        raise ValueError("scenes_setting.scene_registry is provided, but selected_scene_name is missing.")
    if selected_scene_name not in scene_registry:
        available = ", ".join(sorted(scene_registry.keys()))
        raise KeyError(
            f"selected_scene_name '{selected_scene_name}' not found in scene_registry. "
            f"Available scenes: {available}"
        )

    selected_scene = copy.deepcopy(scene_registry[selected_scene_name])
    if not isinstance(selected_scene, dict):
        raise TypeError(
            f"scene_registry['{selected_scene_name}'] must be a dict, "
            f"got {type(selected_scene).__name__}"
        )
    selected_scene.setdefault('name', selected_scene_name)

    expanded = copy.deepcopy(scenes_setting)
    expanded['selected_scene_name'] = selected_scene_name
    if selected_scene.get('dataset_name'):
        expanded['dataset_name'] = selected_scene['dataset_name']
    expanded['scenes'] = [selected_scene]
    return expanded


def get_parse(print_summary=True):
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
    parser.add_argument('--dir2save_log', default='gen_fm_exps/logs', type=str, help='日志与TensorBoard保存根目录')
    parser.add_argument('--dir2save_ckpt', default='gen_fm_exps/ckpts', type=str, help='Checkpoint保存根目录')
    parser.add_argument('--exps_dir', default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument('--exp_name', default='debug', type=str, help='实验名称')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='是否使用tensorboard')

    # Checkpoint配置 (支持字典或字符串，YAML中会覆盖)
    parser.add_argument('--load2test', default="", type=str, help='测试时加载的checkpoint')
    parser.add_argument('--load2train', default="", type=str, help='继续训练时加载的checkpoint')

    # Stage checkpoint配置（用于多阶段训练）
    parser.add_argument('--load_stage1_ckpt', default="", type=str, help='Stage 1的checkpoint路径')
    parser.add_argument('--load_stage2_ckpt', default="", type=str, help='Stage 2的checkpoint路径')
    parser.add_argument('--stage3_analysis_export_root', default="", type=str, help='Stage 3测试分析结果导出目录')
    parser.add_argument('--stage3_recall_cfg', default="per_scene", type=str, help='Stage 3 recall阈值配置名；per_scene表示按场景映射')
    parser.add_argument('--stage3_recall_cfg_yaml', default="trainer_depends/configs/stage3_recall_thresholds.yaml", type=str, help='Stage 3 recall阈值配置YAML')
    parser.add_argument('--inherit_stage1_yaml', default="", type=str, help='Stage 2初始化时继承Stage 1配置参数的opts/base yaml路径')
    parser.add_argument('--inherit_stage1_scope', default='network,data', type=str, help='Stage 2从Stage 1 YAML继承的范围: network,data,scenes,hardware，可用逗号分隔')
    parser.add_argument('--inherit_stage2_yaml', default="", type=str, help='Stage 3初始化时继承Stage 2网络结构参数的opts/base yaml路径')
    parser.add_argument('--inherit_stage2_scope', default='network', type=str, help='Stage 3从Stage 2 YAML继承的范围: network,data,scenes,hardware，可用逗号分隔')
    parser.add_argument('--selected_scene_name', default="", type=str, help='覆盖 scenes_setting.selected_scene_name')
    parser.add_argument(
        '--inherit_stage1_fm_stage2',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='Stage 3是否通过Stage 2的配置继续继承Stage 1相关默认参数',
    )
    parser.add_argument(
        '--inherit_stage1_fm_stage2_scope',
        default="",
        type=str,
        help='Stage 3经由Stage 2继续继承Stage 1时使用的scope；为空时回退到Stage 2记录的 inherit_stage1_scope',
    )

    # 硬件配置
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs, 例如: 0 或 0,1,2')
    parser.add_argument('--num_worker', default=8, type=int, help='DataLoader worker数量')
    parser.add_argument(
        '--satmaps_on_cpu',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否将 SatDataset 的整图缓存与裁图留在 CPU 上',
    )
    parser.add_argument('--autocast', action='store_true', default=False, help='是否使用混合精度训练')

    # 网络配置
    parser.add_argument('--backbone', default="dinov3", type=str,
                        help='Backbone类型: ViTB-224, dinov2, dinov3等')
    parser.add_argument(
        '--freeze_backbone',
        default=True,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否冻结视觉backbone；False时允许backbone参与训练',
    )
    parser.add_argument('--aggregator_type', default='salad', type=str,
                        help='聚合器类型: salad, g2m, g2m_scalar_p, g2m_channelwise_p, gem, fsra, lpn, netvlad')

    # 训练配置
    parser.add_argument('--num_epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--split_train_ratio', default=0.9, type=float, help='UAV 数据集训练集比例')
    parser.add_argument(
        '--split_mode',
        default='segment',
        type=str,
        choices=('segment', 'interval', 'random'),
        help='UAV 数据集切分模式: segment=前段训练后段测试, interval=按固定间隔交错切分, random=固定随机种子划分',
    )
    parser.add_argument(
        '--add_random_satimg_negs',
        default=True,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否为每个query额外添加随机采样的satellite negatives',
    )
    parser.add_argument(
        '--reject_sampling', '--reject_sampleing',
        dest='reject_sampling',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否对随机 satellite negatives 启用正样本邻域拒绝采样',
    )
    parser.add_argument(
        '--reject_batch_aware',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否在 batch collate 阶段额外剔除落入任意 query 邻域的 satellite negatives',
    )
    parser.add_argument(
        '--sat_as_query',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否将sat image也作为query构造配对样本',
    )
    parser.add_argument(
        '--pair_alignment_mode',
        default='full_4d',
        type=str,
        choices=('full_4d', 'xy_only'),
        help='query-positive 配对对齐方式: full_4d=xy/rot/scale对齐, xy_only=仅xy对齐且sat positive固定rot=0、scale=train-mean',
    )

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
            setattr(opt, 'scenes_setting', _expand_selected_scene_config(params))
            continue

        if isinstance(params, dict):
            if section == 'data_setting' and 'reject_sampleing' in params and 'reject_sampling' not in params:
                params = dict(params)
                params['reject_sampling'] = params['reject_sampleing']
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
    else:
        if getattr(opt, 'selected_scene_name', ''):
            if not isinstance(opt.scenes_setting, dict):
                raise TypeError("selected_scene_name override requires scenes_setting to be a dict")
            opt.scenes_setting = copy.deepcopy(opt.scenes_setting)
            opt.scenes_setting['selected_scene_name'] = opt.selected_scene_name
        opt.scenes_setting = _expand_selected_scene_config(opt.scenes_setting)

    legacy_exps_dir = getattr(opt, 'exps_dir', None)
    if not getattr(opt, 'dir2save_log', None):
        opt.dir2save_log = default_args.dir2save_log
    if not getattr(opt, 'dir2save_ckpt', None):
        opt.dir2save_ckpt = default_args.dir2save_ckpt
    if legacy_exps_dir:
        if opt.dir2save_log == default_args.dir2save_log:
            opt.dir2save_log = legacy_exps_dir
        if opt.dir2save_ckpt == default_args.dir2save_ckpt:
            opt.dir2save_ckpt = legacy_exps_dir
        print("提示: `exps_dir` 已废弃，请改用 `dir2save_log` 和 `dir2save_ckpt`。")

    # 兼容仍然直接读取 opt.exps_dir 的旧代码；语义上将其映射为日志根目录。
    opt.exps_dir = opt.dir2save_log

    backbone_config = getattr(opt, 'backbone_config', {})
    if backbone_config is None:
        backbone_config = {}
    if not isinstance(backbone_config, dict):
        raise TypeError(f"backbone_config must be a dict, got {type(backbone_config).__name__}")
    opt.backbone_config = dict(backbone_config)

    aggregator_config = getattr(opt, 'aggregator_config', {})
    if aggregator_config is None:
        aggregator_config = {}
    if not isinstance(aggregator_config, dict):
        raise TypeError(f"aggregator_config must be a dict, got {type(aggregator_config).__name__}")
    opt.aggregator_config = dict(aggregator_config)
    if hasattr(opt, 'reject_sampleing'):
        opt.reject_sampling = bool(getattr(opt, 'reject_sampleing'))

    adapter_config = getattr(opt, 'adapter_config', {})
    if adapter_config is None:
        adapter_config = {}
    if not isinstance(adapter_config, dict):
        raise TypeError(f"adapter_config must be a dict, got {type(adapter_config).__name__}")
    opt.adapter_config = dict(adapter_config)

    hash_lod_aggregator = getattr(opt, 'hash_lod_aggregator', {})
    if hash_lod_aggregator is None:
        hash_lod_aggregator = {}
    if not isinstance(hash_lod_aggregator, dict):
        raise TypeError(f"hash_lod_aggregator must be a dict, got {type(hash_lod_aggregator).__name__}")
    opt.hash_lod_aggregator = dict(hash_lod_aggregator)

    # --- 7. 组织参数到 group_dict ---
    group_info = {
        'exp_setting': ['p_yaml', 'exp_name', 'dir2save_log', 'dir2save_ckpt', 'load2train', 'load2test',
                        'load_stage1_ckpt', 'load_stage2_ckpt',
                        'stage3_analysis_export_root',
                        'stage3_recall_cfg', 'stage3_recall_cfg_yaml',
                        'inherit_stage1_yaml', 'inherit_stage1_scope',
                        'inherit_stage2_yaml', 'inherit_stage2_scope',
                        'selected_scene_name',
                        'inherit_stage1_fm_stage2', 'inherit_stage1_fm_stage2_scope',
                        'tensorboard',
                        'save_freq', 'val', 'val_freq',
                        'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'],
        'data_setting': [
            'pad_mode',
            'imgsize2net',
            'satimgsize2crop',
            'n_rand2sample_per_pos',
            'split_train_ratio',
            'split_mode',
            'add_random_satimg_negs',
            'reject_sampling',
            'reject_batch_aware',
            'sat_as_query',
            'pair_alignment_mode',
        ],
        'hardware_setting': ['gpu_ids', 'num_worker', 'satmaps_on_cpu', 'autocast',
                            'batchsize_sat', 'batchsize_uav', 'batchsize_uav_test'],
        'network_setting': [
            'backbone',
            'freeze_backbone',
            'backbone_config',
            'aggregator_type',
            'aggregator_config',
            'adapter_config',
            'freeze_grid',
            'p_grid_config_yaml',
            'hash_lod_aggregator',
            'posenc_multires_rc',
            'posenc_multires_rot',
            'posenc_multires_scale',
            'grid_mlp_hidden_dim',
            'grid_mlp_num_blocks',
            'apr_target_mode',
            'apr_mlp_hidden_dim',
            'apr_mlp_num_layers',
            'apr_mlp_norm',
            'apr_mlp_dropout',
            'apr_mlp_activation',
            'apr_output_tanh',
            'apr_normalize_input_feat',
            'apr_eval_clamp',
            'apr_require_stage1_ckpt',
        ],
        'learning_setting': [
            'num_epochs',
            'loss_type',
            'infonce_temperature',
            'infonce_negative_mode',
            'apr_optimizer',
            'apr_lr',
            'apr_weight_decay',
            'apr_loss_weight_nrc',
            'apr_loss_weight_rot',
            'apr_loss_weight_scale',
            'apr_max_eval_batches',
        ],
    }
    opt.group_dict = group_info
    opt._config_summary_printed = False

    # --- 8. 最终打印和返回 ---
    if print_summary:
        print_config_summary(opt)

    return opt
