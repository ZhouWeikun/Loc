#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实数据集验证分层坐标采样器

用法:
    # 方式1: 直接运行（从项目根目录）
    python trainer_depends/utils/util_visualize_hierarchical_sampler_with_real_data.py

    # 方式2: 单个样本可视化
    python trainer_depends/utils/util_visualize_hierarchical_sampler_with_real_data.py --mode single

    # 方式3: 多样本批量测试
    python trainer_depends/utils/util_visualize_hierarchical_sampler_with_real_data.py --mode multiple --n_samples 10
"""

import sys
import os
import torch

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # trainer_depends/utils -> trainer_depends -> project_root
sys.path.insert(0, project_root)

from trainer_depends.utils.util_hierarchical_coord_sampler import (
    create_hierarchical_sampler_from_dataset,
    visualize_hierarchical_sampling
)
from math import pi


def load_dataset_from_config(config_path='trainer_depends/configs/stage3_metric_net.yaml'):
    """
    从配置文件加载数据集（与 BaseTrainer._init_datasets 保持一致）

    Args:
        config_path: 配置文件路径（相对于项目根目录）

    Returns:
        sat_dataset: 卫星数据集
        uav_dataset_train: UAV训练数据集
        uav_dataset_test: UAV测试数据集
        opt: 配置对象
    """
    # 确保使用绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    print(f"📄 读取配置文件: {config_path}")

    # 导入配置解析器
    from trainer_depends.config.parser import get_parse

    # 备份并修改sys.argv
    original_argv = sys.argv.copy()
    sys.argv = ['visualize_script', '--p_yaml', config_path]
    opt = get_parse()
    sys.argv = original_argv  # 恢复原始argv

    # 获取场景配置
    scenes = opt.scenes_setting['scenes']
    scene = scenes[0]  # 使用第一个场景
    scene_name = scene['name']

    print(f"📍 使用场景: {scene_name}")

    # 初始化卫星数据集（与 BaseTrainer._init_datasets 一致）
    from trainer_depends.datasets.dataset_wingtra_4d import SatDataset
    sat_dataset = SatDataset(
        p_satinfo_json=scene['p_satinfo_json'],
        p_uav_geocsv=scene['p_uav_geocsv'],
        imgsize2net=opt.imgsize2net,
    )

    # 初始化UAV数据集（训练集）
    from trainer_depends.datasets.dataset_wingtra_4d import UAVDataset
    uav_dataset_train = UAVDataset(
        p_uavinfo_json=scene['p_uavinfo_json'],
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        stage='train',
    )

    # 初始化UAV数据集（测试集）
    uav_dataset_test = UAVDataset(
        p_uavinfo_json=scene['p_uavinfo_json'],
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        stage='test',
    )

    return sat_dataset, uav_dataset_train, uav_dataset_test, opt


def visualize_with_real_data():
    """使用真实数据进行可视化验证"""
    print("\n" + "="*80)
    print("使用真实数据集验证分层坐标采样器")
    print("="*80 + "\n")

    # 1. 加载数据集
    print("📦 加载数据集...")
    sat_dataset, uav_dataset_train, uav_dataset_test, opt = load_dataset_from_config()

    print(f"✅ 卫星数据集: {len(sat_dataset)} 个样本")
    print(f"✅ UAV训练集: {len(uav_dataset_train)} 个样本")
    print(f"✅ UAV测试集: {len(uav_dataset_test)} 个样本")

    # 使用测试集进行可视化
    uav_dataset = uav_dataset_test

    # 2. 打印数据集的坐标范围
    print(f"\n数据集坐标范围:")
    print(f"  - RC边界: row [{sat_dataset.nr2sample_min:.4f}, {sat_dataset.nr2sample_max:.4f}], "
          f"col [{sat_dataset.nc2sample_min:.4f}, {sat_dataset.nc2sample_max:.4f}]")
    print(f"  - 旋转边界: [-π, π] = [{-pi:.4f}, {pi:.4f}] rad")
    print(f"  - 尺度边界: {sat_dataset.satimgsize_scale_to_refm_boundary}")

    # 3. 配置采样器参数（使用新的基于Span-Ratio的策略）
    # 使用 sat_dataset.halfimg_radius_nrc 作为 bottom 层的绝对RC标准差
    bottom_abs_rc_std = sat_dataset.halfimg_radius_nrc
    num_uniform_samples = getattr(opt, 'sampler_num_uniform', 128)

    print(f"\n采样器配置参数:")
    print(f"  - bottom_abs_rc_std (from sat_dataset.halfimg_radius_nrc): {bottom_abs_rc_std:.6f}")
    print(f"  - num_uniform_samples: {num_uniform_samples}")

    # 4. 创建采样器
    print(f"\n🔧 创建分层坐标采样器...")
    sampler = create_hierarchical_sampler_from_dataset(
        sat_dataset=sat_dataset,
        bottom_abs_rc_std=bottom_abs_rc_std,
        num_uniform_samples=num_uniform_samples,
        device='cpu'
    )

    # 5. 从sat_dataset随机采样一个GT坐标（更有代表性）
    print(f"\n📍 从sat_dataset随机采样一个GT坐标...")
    uav_coord_4d = sat_dataset.mk_rand_coords_4d(
        n_rand=1,
        return_tensor=True
    )[0]  # 获取第一个样本

    print(f"随机采样的GT坐标:")
    print(f"  row={uav_coord_4d[0]:.4f}, col={uav_coord_4d[1]:.4f}, "
          f"rot={uav_coord_4d[2]:.4f} rad, scale={uav_coord_4d[3]:.4f}")

    # 6. 可视化采样结果
    print(f"\n🎨 开始可视化分层采样...")

    # 设置保存路径（使用项目根目录的绝对路径）
    vis_dir = os.path.join(project_root, 'trainers', 'vis_results')
    os.makedirs(vis_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(vis_dir, 'hierarchical_sampling_real_data.png')

    results = visualize_hierarchical_sampling(
        sampler=sampler,
        gt_coord_4d=torch.tensor(uav_coord_4d),
        n_samples=1,
        save_path=save_path,
        show_interactive=False
    )

    # 7. 打印总结
    print("\n" + "="*80)
    print("验证完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - hierarchical_sampling_real_data.png (2D投影图)")
    print(f"  - hierarchical_sampling_real_data.html (3D交互式图表)")
    print(f"\n采样统计:")
    print(f"  - 层级数量: {len(results['layer_stats'])}")
    print(f"  - 总采样点数: {results['total_samples']}")
    print(f"  - GT坐标: {results['gt_coord']}")


    print("\n" + "="*80 + "\n")


def test_multiple_samples(n_samples=5):
    """测试多个真实样本的采样效果"""
    print("\n" + "="*80)
    print(f"测试 {n_samples} 个随机采样的GT坐标")
    print("="*80 + "\n")

    # 加载数据集
    sat_dataset, uav_dataset_train, uav_dataset_test, opt = load_dataset_from_config()

    # 配置采样器参数
    bottom_abs_rc_std = sat_dataset.halfimg_radius_nrc
    num_uniform_samples = getattr(opt, 'sampler_num_uniform', 128)

    # 创建采样器
    sampler = create_hierarchical_sampler_from_dataset(
        sat_dataset=sat_dataset,
        bottom_abs_rc_std=bottom_abs_rc_std,
        num_uniform_samples=num_uniform_samples,
        device='cpu'
    )

    # 从sat_dataset随机采样GT坐标（更有代表性）
    print(f"从sat_dataset随机采样 {n_samples} 个GT坐标...")
    gt_coords_batch = sat_dataset.mk_rand_coords_4d(
        n_rand=n_samples,
        return_tensor=True
    )  # [n_samples, 4]

    print(f"批量采样 {n_samples} 个GT坐标:")
    for i, coord in enumerate(gt_coords_batch):
        print(f"  样本{i}: row={coord[0]:.4f}, col={coord[1]:.4f}, "
              f"rot={coord[2]:.4f}, scale={coord[3]:.4f}")

    # 批量采样（启用verbose查看标准差限制）
    print(f"\n开始批量采样（verbose模式）...")
    print("-" * 80)
    sampled_coords = sampler.sample(gt_coords_batch, verbose=True)
    print("-" * 80)

    print(f"\n✅ 批量采样完成！")
    print(f"  - 输入: {gt_coords_batch.shape} -> {n_samples} 个GT坐标")
    print(f"  - 输出: {sampled_coords.shape} -> 每个GT生成 {sampler.total_samples_per_gt} 个采样点")

    # 统计每个样本的采样分布
    print(f"\n各样本的采样分布统计:")
    for i in range(n_samples):
        gt_coord = gt_coords_batch[i]
        samples = sampled_coords[i]  # [total_samples_per_gt, 4]

        # 计算距离
        diff = samples - gt_coord
        rc_dist = torch.norm(diff[:, :2], dim=-1)

        print(f"\n样本{i}:")
        print(f"  - GT位置: (r={gt_coord[0]:.4f}, c={gt_coord[1]:.4f})")
        print(f"  - RC距离范围: [{rc_dist.min():.4f}, {rc_dist.max():.4f}]")
        print(f"  - RC距离均值: {rc_dist.mean():.4f} ± {rc_dist.std():.4f}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='使用真实数据验证分层坐标采样器')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multiple'],
                       help='验证模式: single=单个样本可视化, multiple=多样本批量测试')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='多样本模式下的样本数量')

    args = parser.parse_args()

    if args.mode == 'single':
        # 单个样本的详细可视化
        visualize_with_real_data()
    else:
        # 多个样本的批量测试
        test_multiple_samples(n_samples=args.n_samples)