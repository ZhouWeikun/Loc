#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本：显示地图的2D网格划分和无人机位点

功能：
1. 基于SubspaceSampler的网格划分绘制2D网格
2. 显示训练集和测试集的无人机位点（不同颜色）
3. 保存可视化图像

用法：
    python tool/visualize_grid_and_uav_points.py --p_yaml <config_path>
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from matplotlib.collections import LineCollection

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def visualize_grid_and_points(
    sat_dataset,
    uav_dataset_train,
    uav_dataset_test,
    n_coarse=(40, 30, 12, 1),
    save_path='vis_results/grid_and_uav_points.png',
    dpi=300,
    figsize=(12, 10),
    max_points_train=1000,
    max_points_test=500,
    show_grid=True,
    show_labels=True
):
    """
    可视化地图网格划分和无人机位点

    Args:
        sat_dataset: SatDataset 对象
        uav_dataset_train: 训练集 UAV 数据集
        uav_dataset_test: 测试集 UAV 数据集
        n_coarse: 网格划分参数 (NR, NC, Rot, Scale)
        save_path: 保存路径
        dpi: 图像分辨率
        figsize: 图像尺寸
        max_points_train: 显示的最大训练点数
        max_points_test: 显示的最大测试点数
        show_grid: 是否显示网格线
        show_labels: 是否显示轴标签
    """

    # 获取地图坐标范围
    nr_min = sat_dataset.nr2sample_min
    nr_max = sat_dataset.nr2sample_max
    nc_min = sat_dataset.nc2sample_min
    nc_max = sat_dataset.nc2sample_max

    nr_range = nr_max - nr_min
    nc_range = nc_max - nc_min

    # 网格参数
    n_nr, n_nc = n_coarse[0], n_coarse[1]

    # 计算网格线位置
    nr_edges = np.linspace(nr_min, nr_max, n_nr + 1)
    nc_edges = np.linspace(nc_min, nc_max, n_nc + 1)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ========== 1. 绘制网格线 ==========
    if show_grid:
        # 垂直网格线（沿NC方向）
        vlines = []
        for nc_edge in nc_edges:
            vlines.append([(nc_edge, nr_min), (nc_edge, nr_max)])

        # 水平网格线（沿NR方向）
        hlines = []
        for nr_edge in nr_edges:
            hlines.append([(nc_min, nr_edge), (nc_max, nr_edge)])

        # 绘制网格
        lc_v = LineCollection(vlines, colors='gray', linewidths=0.5, alpha=0.5, linestyle='--')
        lc_h = LineCollection(hlines, colors='gray', linewidths=0.5, alpha=0.5, linestyle='--')
        ax.add_collection(lc_v)
        ax.add_collection(lc_h)

        # 边界加粗
        ax.plot([nc_min, nc_max, nc_max, nc_min, nc_min],
                [nr_min, nr_min, nr_max, nr_max, nr_min],
                'k-', linewidth=2, label='Map Boundary')

    # ========== 2. 采样和绘制无人机位点 ==========

    # 训练集位点
    n_train = len(uav_dataset_train)
    n_train_sample = min(max_points_train, n_train)
    train_indices = np.random.choice(n_train, n_train_sample, replace=False)

    train_coords = []
    for idx in train_indices:
        _, coords = uav_dataset_train[int(idx)]
        train_coords.append(coords.numpy() if torch.is_tensor(coords) else coords)

    train_coords = np.array(train_coords)  # [N_train, 4]
    train_nr = train_coords[:, 0]
    train_nc = train_coords[:, 1]

    # 测试集位点
    n_test = len(uav_dataset_test)
    n_test_sample = min(max_points_test, n_test)
    test_indices = np.random.choice(n_test, n_test_sample, replace=False)

    test_coords = []
    for idx in test_indices:
        _, coords = uav_dataset_test[int(idx)]
        test_coords.append(coords.numpy() if torch.is_tensor(coords) else coords)

    test_coords = np.array(test_coords)  # [N_test, 4]
    test_nr = test_coords[:, 0]
    test_nc = test_coords[:, 1]

    # 绘制位点
    ax.scatter(train_nc, train_nr, c='blue', s=10, alpha=0.6,
               label=f'Train Set ({n_train_sample}/{n_train})', marker='o', edgecolors='none')
    ax.scatter(test_nc, test_nr, c='red', s=10, alpha=0.6,
               label=f'Test Set ({n_test_sample}/{n_test})', marker='^', edgecolors='none')

    # ========== 3. 设置坐标轴和标签 ==========
    ax.set_xlim(nc_min - 0.02*nc_range, nc_max + 0.02*nc_range)
    ax.set_ylim(nr_min - 0.02*nr_range, nr_max + 0.02*nr_range)

    # 反转Y轴（因为图像坐标系中行从上到下）
    ax.invert_yaxis()

    if show_labels:
        ax.set_xlabel('Normalized Column (NC)', fontsize=12)
        ax.set_ylabel('Normalized Row (NR)', fontsize=12)
        ax.set_title(f'Map Grid ({n_nr}×{n_nc}) and UAV Points Distribution',
                     fontsize=14, fontweight='bold')

    # 图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # 网格信息文本
    info_text = (
        f'Grid: {n_nr} × {n_nc} = {n_nr * n_nc} cells\n'
        f'NR range: [{nr_min:.3f}, {nr_max:.3f}]\n'
        f'NC range: [{nc_min:.3f}, {nc_max:.3f}]\n'
        f'Cell size: {nr_range/n_nr:.4f} × {nc_range/n_nc:.4f}'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 设置纵横比为相等
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ 可视化图像已保存: {save_path}")
    print(f"   - 图像尺寸: {figsize[0]}×{figsize[1]} inches")
    print(f"   - 分辨率: {dpi} DPI")
    print(f"   - 训练点数: {n_train_sample}/{n_train}")
    print(f"   - 测试点数: {n_test_sample}/{n_test}")
    print(f"   - 网格大小: {n_nr}×{n_nc} = {n_nr*n_nc} cells")

    # 可选：显示图像
    # plt.show()
    plt.close()

    return fig, ax


def visualize_with_heatmap(
    sat_dataset,
    uav_dataset_train,
    uav_dataset_test,
    n_coarse=(40, 30, 12, 1),
    save_path='vis_results/grid_heatmap.png',
    dpi=300,
    figsize=(14, 10)
):
    """
    使用热力图可视化UAV位点密度分布

    Args:
        sat_dataset: SatDataset 对象
        uav_dataset_train: 训练集 UAV 数据集
        uav_dataset_test: 测试集 UAV 数据集
        n_coarse: 网格划分参数 (NR, NC, Rot, Scale)
        save_path: 保存路径
        dpi: 图像分辨率
        figsize: 图像尺寸
    """

    # 获取地图坐标范围
    nr_min = sat_dataset.nr2sample_min
    nr_max = sat_dataset.nr2sample_max
    nc_min = sat_dataset.nc2sample_min
    nc_max = sat_dataset.nc2sample_max

    # 网格参数
    n_nr, n_nc = n_coarse[0], n_coarse[1]

    # 计算网格边界
    nr_edges = np.linspace(nr_min, nr_max, n_nr + 1)
    nc_edges = np.linspace(nc_min, nc_max, n_nc + 1)

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    for idx, (dataset, ax, title, color) in enumerate([
        (uav_dataset_train, axes[0], 'Training Set', 'Blues'),
        (uav_dataset_test, axes[1], 'Test Set', 'Reds')
    ]):
        # 收集所有坐标
        coords = []
        for i in range(len(dataset)):
            _, coord = dataset[i]
            coords.append(coord.numpy() if torch.is_tensor(coord) else coord)

        coords = np.array(coords)  # [N, 4]
        nr_coords = coords[:, 0]
        nc_coords = coords[:, 1]

        # 计算2D直方图（热力图）
        heatmap, _, _ = np.histogram2d(
            nr_coords, nc_coords,
            bins=[nr_edges, nc_edges]
        )

        # 绘制热力图
        im = ax.imshow(
            heatmap,
            extent=[nc_min, nc_max, nr_max, nr_min],
            aspect='auto',
            cmap=color,
            interpolation='nearest',
            alpha=0.8
        )

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Point Density', fontsize=10)

        # 绘制网格线
        for nr_edge in nr_edges:
            ax.axhline(nr_edge, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')
        for nc_edge in nc_edges:
            ax.axvline(nc_edge, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')

        # 边界
        ax.plot([nc_min, nc_max, nc_max, nc_min, nc_min],
                [nr_min, nr_min, nr_max, nr_max, nr_min],
                'k-', linewidth=2)

        # 设置标签
        ax.set_xlabel('Normalized Column (NC)', fontsize=11)
        ax.set_ylabel('Normalized Row (NR)', fontsize=11)
        ax.set_title(f'{title}\n({len(dataset)} points, {n_nr}×{n_nc} grid)',
                     fontsize=12, fontweight='bold')

        # 统计信息
        total_points = len(coords)
        occupied_cells = np.sum(heatmap > 0)
        avg_density = total_points / (n_nr * n_nc)

        info_text = (
            f'Total: {total_points} points\n'
            f'Occupied: {occupied_cells}/{n_nr*n_nc} cells\n'
            f'Avg density: {avg_density:.1f} pts/cell'
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ 热力图已保存: {save_path}")
    plt.close()

    return fig, axes


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化地图网格和UAV位点')
    parser.add_argument('--p_yaml', type=str,
                        default='trainer_depends/configs/stage3_metric_net.yaml',
                        help='配置文件路径')
    parser.add_argument('--save_dir', type=str, default='vis_results',
                        help='保存目录')
    parser.add_argument('--dpi', type=int, default=500,
                        help='图像分辨率')
    parser.add_argument('--max_train', type=int, default=1000,
                        help='显示的最大训练点数')
    parser.add_argument('--max_test', type=int, default=500,
                        help='显示的最大测试点数')
    parser.add_argument('--show_heatmap', action='store_true',
                        help='是否生成热力图')

    args = parser.parse_args()

    # 加载配置
    from tool.util_ckpt_handler import get_parse
    opt = get_parse()

    print("\n" + "="*80)
    print("开始可视化地图网格和UAV位点")
    print("="*80)
    print(f"配置文件: {opt.p_yaml}")
    print(f"保存目录: {args.save_dir}")
    print("="*80 + "\n")

    # 初始化数据集
    print("正在加载数据集...")
    from trainer_depends.datasets.dataset_wingtra_4d import SatDataset, UAVDataset

    # 获取第一个场景的配置
    scenes = opt.scenes_setting['scenes']
    scene = scenes[0]  # 使用第一个场景
    scene_name = scene['name']

    print(f"使用场景: {scene_name}")

    # Sat数据集
    sat_dataset = SatDataset(
        p_satinfo_json=scene['p_satinfo_json'],
        p_uav_geocsv=scene['p_uav_geocsv'],
        imgsize2net=opt.imgsize2net,
    )

    # UAV训练集
    uav_dataset_train = UAVDataset(
        p_uavinfo_json=scene['p_uavinfo_json'],
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        stage='train',
    )

    # UAV测试集
    uav_dataset_test = UAVDataset(
        p_uavinfo_json=scene['p_uavinfo_json'],
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        stage='test',
    )

    print(f"✅ 数据集加载完成")
    print(f"   - 训练集: {len(uav_dataset_train)} 个样本")
    print(f"   - 测试集: {len(uav_dataset_test)} 个样本")
    print()

    # 获取网格划分参数
    n_coarse = getattr(opt, 'n_coarse', (40, 30, 12, 1))
    print(f"网格划分参数: {n_coarse}")
    print(f"   - NR (行): {n_coarse[0]} 格")
    print(f"   - NC (列): {n_coarse[1]} 格")
    print(f"   - Rot (旋转): {n_coarse[2]} 格")
    print(f"   - Scale (尺度): {n_coarse[3]} 格")
    print()

    # 1. 生成散点图
    print("生成散点图...")
    save_path_scatter = os.path.join(args.save_dir, 'grid_and_uav_points.png')
    visualize_grid_and_points(
        sat_dataset=sat_dataset,
        uav_dataset_train=uav_dataset_train,
        uav_dataset_test=uav_dataset_test,
        n_coarse=n_coarse,
        save_path=save_path_scatter,
        dpi=args.dpi,
        max_points_train=args.max_train,
        max_points_test=args.max_test
    )
    print()

    # 2. 生成热力图（可选）
    if args.show_heatmap:
        print("生成热力图...")
        save_path_heatmap = os.path.join(args.save_dir, 'grid_heatmap.png')
        visualize_with_heatmap(
            sat_dataset=sat_dataset,
            uav_dataset_train=uav_dataset_train,
            uav_dataset_test=uav_dataset_test,
            n_coarse=n_coarse,
            save_path=save_path_heatmap,
            dpi=args.dpi
        )
        print()

    print("="*80)
    print("✅ 可视化完成！")
    print("="*80)