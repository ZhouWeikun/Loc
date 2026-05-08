#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 dataset_wingtra_4d.py 中4D坐标每个维度的数值范围
"""

import numpy as np
import torch
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(__file__))

from dataset_wingtra_4d import SatDataset, UAVDataset


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verify_sat_dataset_ranges(sat_dataset):
    """验证SatDataset的4D坐标范围"""
    print_section("1. SatDataset 坐标范围验证")

    # 1.1 归一化坐标范围
    print("\n📍 归一化坐标 (nr, nc) 范围：")
    print(f"  nr2sample_min = {sat_dataset.nr2sample_min:.6f}")
    print(f"  nr2sample_max = {sat_dataset.nr2sample_max:.6f}")
    print(f"  nr2sample_h   = {sat_dataset.nr2sample_h:.6f} (范围宽度)")
    print(f"  nc2sample_min = {sat_dataset.nc2sample_min:.6f}")
    print(f"  nc2sample_max = {sat_dataset.nc2sample_max:.6f}")
    print(f"  nc2sample_w   = {sat_dataset.nc2sample_w:.6f} (范围宽度)")

    # 1.2 尺度范围
    print("\n📏 尺度比例 (scale_ratio_to_refm) 范围：")
    print(f"  scale_ref_m = {sat_dataset.scale_ref_m} 米 (参考尺度)")
    print(f"  geo_res_m   = {sat_dataset.geo_res_m:.3f} 米/像素 (卫星图分辨率)")
    print(f"  satimgsize2crop_boundary (像素):")
    print(f"    min = {sat_dataset.satimgsize2crop_boundary[0]:.2f} 像素")
    print(f"    max = {sat_dataset.satimgsize2crop_boundary[1]:.2f} 像素")
    print(f"  satimgsize_scale_to_refm_boundary (比例):")
    print(f"    min = {sat_dataset.satimgsize_scale_to_refm_boundary[0]:.4f}")
    print(f"    max = {sat_dataset.satimgsize_scale_to_refm_boundary[1]:.4f}")
    print(f"    范围宽度 = {sat_dataset.satimgsize_scale_to_refm_boundary[1] - sat_dataset.satimgsize_scale_to_refm_boundary[0]:.4f}")

    # 1.3 半径阈值
    print("\n🎯 正样本阈值 (halfimg_radius_nrc)：")
    print(f"  halfimg_radius_nrc   = {sat_dataset.halfimg_radius_nrc:.6f} (归一化空间)")
    print(f"  halfimg_radius_meter = {sat_dataset.halfimg_radius_meter:.2f} 米 (真实空间)")

    # 1.4 卫星图信息
    print("\n🗺️  卫星图尺寸信息：")
    print(f"  satmap_h      = {sat_dataset.satmap_h} 像素")
    print(f"  satmap_w      = {sat_dataset.satmap_w} 像素")
    print(f"  satmap_hw_max = {sat_dataset.satmap_hw_max} 像素")
    print(f"  satmap_edge_pixs = {sat_dataset.satmap_edge_pixs:.2f} 像素 (边界留白)")


def verify_uav_dataset_ranges(uav_dataset):
    """验证UAVDataset的4D坐标范围"""
    print_section("2. UAVDataset 坐标范围验证")

    # 2.1 统计训练集坐标范围
    coords_4d_train = uav_dataset.uav_coords_4d_torch_train  # [N, 4]

    print("\n📊 训练集4D坐标统计 (N={})：".format(coords_4d_train.shape[0]))
    print("\n  维度 0 - nr (归一化行):")
    print(f"    min = {coords_4d_train[:, 0].min().item():.6f}")
    print(f"    max = {coords_4d_train[:, 0].max().item():.6f}")
    print(f"    mean = {coords_4d_train[:, 0].mean().item():.6f}")
    print(f"    std = {coords_4d_train[:, 0].std().item():.6f}")

    print("\n  维度 1 - nc (归一化列):")
    print(f"    min = {coords_4d_train[:, 1].min().item():.6f}")
    print(f"    max = {coords_4d_train[:, 1].max().item():.6f}")
    print(f"    mean = {coords_4d_train[:, 1].mean().item():.6f}")
    print(f"    std = {coords_4d_train[:, 1].std().item():.6f}")

    print("\n  维度 2 - rotation_rad (旋转角度):")
    print(f"    min = {coords_4d_train[:, 2].min().item():.6f} rad ({np.rad2deg(coords_4d_train[:, 2].min().item()):.2f}°)")
    print(f"    max = {coords_4d_train[:, 2].max().item():.6f} rad ({np.rad2deg(coords_4d_train[:, 2].max().item()):.2f}°)")
    print(f"    mean = {coords_4d_train[:, 2].mean().item():.6f} rad ({np.rad2deg(coords_4d_train[:, 2].mean().item()):.2f}°)")
    print(f"    std = {coords_4d_train[:, 2].std().item():.6f} rad ({np.rad2deg(coords_4d_train[:, 2].std().item()):.2f}°)")

    print("\n  维度 3 - scale_ratio_to_refm (尺度比例):")
    print(f"    min = {coords_4d_train[:, 3].min().item():.6f}")
    print(f"    max = {coords_4d_train[:, 3].max().item():.6f}")
    print(f"    mean = {coords_4d_train[:, 3].mean().item():.6f}")
    print(f"    std = {coords_4d_train[:, 3].std().item():.6f}")

    # 2.2 测试集统计
    coords_4d_test = uav_dataset.uav_coords_4d_torch_test
    print("\n📊 测试集4D坐标统计 (N={})：".format(coords_4d_test.shape[0]))
    print(f"  nr:    [{coords_4d_test[:, 0].min().item():.6f}, {coords_4d_test[:, 0].max().item():.6f}]")
    print(f"  nc:    [{coords_4d_test[:, 1].min().item():.6f}, {coords_4d_test[:, 1].max().item():.6f}]")
    print(f"  rot:   [{coords_4d_test[:, 2].min().item():.6f}, {coords_4d_test[:, 2].max().item():.6f}] rad")
    print(f"  scale: [{coords_4d_test[:, 3].min().item():.6f}, {coords_4d_test[:, 3].max().item():.6f}]")


def verify_random_sampling(sat_dataset, n_samples=10000):
    """验证随机采样的4D坐标范围"""
    print_section("3. 随机采样4D坐标验证")

    print(f"\n🎲 生成 {n_samples} 个随机4D坐标进行统计...")
    rand_coords_4d = sat_dataset.mk_rand_coords_4d(n_samples, return_tensor=True)  # [N, 4]

    print("\n📊 随机采样统计：")
    print("\n  维度 0 - nr (归一化行):")
    print(f"    min = {rand_coords_4d[:, 0].min().item():.6f}")
    print(f"    max = {rand_coords_4d[:, 0].max().item():.6f}")
    print(f"    理论范围: [{sat_dataset.nr2sample_min:.6f}, {sat_dataset.nr2sample_max:.6f}]")

    print("\n  维度 1 - nc (归一化列):")
    print(f"    min = {rand_coords_4d[:, 1].min().item():.6f}")
    print(f"    max = {rand_coords_4d[:, 1].max().item():.6f}")
    print(f"    理论范围: [{sat_dataset.nc2sample_min:.6f}, {sat_dataset.nc2sample_max:.6f}]")

    print("\n  维度 2 - rotation_rad (旋转角度):")
    print(f"    min = {rand_coords_4d[:, 2].min().item():.6f} rad ({np.rad2deg(rand_coords_4d[:, 2].min().item()):.2f}°)")
    print(f"    max = {rand_coords_4d[:, 2].max().item():.6f} rad ({np.rad2deg(rand_coords_4d[:, 2].max().item()):.2f}°)")
    print(f"    理论范围: [-π, π] = [{-np.pi:.6f}, {np.pi:.6f}] rad")

    print("\n  维度 3 - scale_ratio_to_refm (尺度比例):")
    print(f"    min = {rand_coords_4d[:, 3].min().item():.6f}")
    print(f"    max = {rand_coords_4d[:, 3].max().item():.6f}")
    print(f"    理论范围: [{sat_dataset.satimgsize_scale_to_refm_boundary[0]:.6f}, {sat_dataset.satimgsize_scale_to_refm_boundary[1]:.6f}]")


def verify_udf_normalization_factors(sat_dataset):
    """验证UDF归一化因子"""
    print_section("4. UDF归一化因子验证")

    # 计算归一化因子
    import math
    norm_factor_rc = math.sqrt(sat_dataset.nr2sample_h ** 2 + sat_dataset.nc2sample_w ** 2)
    norm_factor_rot = np.pi
    scale_min = sat_dataset.satimgsize_scale_to_refm_boundary[0]
    scale_max = sat_dataset.satimgsize_scale_to_refm_boundary[1]
    norm_factor_scale = math.log(scale_max / scale_min)

    print("\n📐 归一化因子计算：")
    print(f"  norm_factor_rc    = sqrt({sat_dataset.nr2sample_h:.6f}² + {sat_dataset.nc2sample_w:.6f}²)")
    print(f"                    = {norm_factor_rc:.6f}")
    print(f"  norm_factor_rot   = π = {norm_factor_rot:.6f}")
    print(f"  norm_factor_scale = log({scale_max:.6f} / {scale_min:.6f})")
    print(f"                    = {norm_factor_scale:.6f}")

    print("\n📊 各维度的最大可能距离（归一化前）：")
    max_rc_dist = norm_factor_rc
    max_rot_dist = np.pi
    max_scale_dist = norm_factor_scale

    print(f"  max_rc_dist    = {max_rc_dist:.6f} (对角线长度)")
    print(f"  max_rot_dist   = {max_rot_dist:.6f} (180° = π rad)")
    print(f"  max_scale_dist = {max_scale_dist:.6f} (log比值)")

    print("\n✅ 归一化后的最大距离（应都为1.0）：")
    print(f"  max_rc_dist_normed    = {max_rc_dist / norm_factor_rc:.6f}")
    print(f"  max_rot_dist_normed   = {max_rot_dist / norm_factor_rot:.6f}")
    print(f"  max_scale_dist_normed = {max_scale_dist / norm_factor_scale:.6f}")

    print("\n🎯 UDF阈值与归一化距离的关系：")
    print(f"  halfimg_radius_nrc (归一化空间) = {sat_dataset.halfimg_radius_nrc:.6f}")
    print(f"  这个值用于判断正样本: udf_dist < {sat_dataset.halfimg_radius_nrc:.6f}")
    print(f"  但UDF计算后的值域约为 [0, 1]")
    print(f"  ⚠️  需要检查两者是否匹配！")

    # 计算权重后的最大UDF距离
    w_rc = 0.6
    w_r = 0.3
    w_s = 0.1
    max_udf_dist = math.sqrt(w_rc * 1.0**2 + w_r * 1.0**2 + w_s * 1.0**2)

    print(f"\n📏 UDF最大距离（加权后）：")
    print(f"  weights: w_rc={w_rc}, w_r={w_r}, w_s={w_s}")
    print(f"  max_udf = sqrt({w_rc}*1² + {w_r}*1² + {w_s}*1²)")
    print(f"          = {max_udf_dist:.6f}")
    print(f"\n  ⚠️  如果 halfimg_radius_nrc ({sat_dataset.halfimg_radius_nrc:.6f}) << max_udf ({max_udf_dist:.6f})")
    print(f"      则几乎没有样本会被判定为正样本！")


def verify_dataset_samples(sat_dataset, uav_dataset, n_samples=5):
    """验证实际采样的数据"""
    print_section("5. 实际数据采样验证")

    print(f"\n🔍 采样 {n_samples} 个SatDataset样本：")
    for i in range(min(n_samples, len(sat_dataset))):
        satimg, coords_4d = sat_dataset[i]
        print(f"\n  样本 {i}:")
        print(f"    satimg shape: {satimg.shape}")
        print(f"    coords_4d: {coords_4d.numpy()}")
        print(f"      nr={coords_4d[0].item():.6f}, nc={coords_4d[1].item():.6f}, "
              f"rot={coords_4d[2].item():.4f} rad ({np.rad2deg(coords_4d[2].item()):.2f}°), "
              f"scale={coords_4d[3].item():.4f}")

    print(f"\n🔍 采样 {n_samples} 个UAVDataset样本（训练集）：")
    uav_dataset.switch_stage('train')
    for i in range(min(n_samples, len(uav_dataset))):
        uavimg, coords_4d = uav_dataset[i]
        print(f"\n  样本 {i}:")
        print(f"    uavimg shape: {uavimg.shape}")
        print(f"    coords_4d: {coords_4d.numpy()}")
        print(f"      nr={coords_4d[0].item():.6f}, nc={coords_4d[1].item():.6f}, "
              f"rot={coords_4d[2].item():.4f} rad ({np.rad2deg(coords_4d[2].item()):.2f}°), "
              f"scale={coords_4d[3].item():.4f}")


def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("  验证 dataset_wingtra_4d.py 中4D坐标的数值范围")
    print("🚀" * 40)

    # 配置数据集路径（来自dataset_wingtra_4d.py的示例）
    p_satinfo_json = '/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json'
    p_uavinfo_json = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json'
    p_uav_geocsv = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv'

    # 检查文件是否存在
    if not os.path.exists(p_satinfo_json):
        print(f"\n❌ 错误：找不到文件 {p_satinfo_json}")
        print("请修改脚本中的路径配置！")
        return

    if not os.path.exists(p_uavinfo_json):
        print(f"\n❌ 错误：找不到文件 {p_uavinfo_json}")
        print("请修改脚本中的路径配置！")
        return

    if not os.path.exists(p_uav_geocsv):
        print(f"\n❌ 错误：找不到文件 {p_uav_geocsv}")
        print("请修改脚本中的路径配置！")
        return

    print(f"\n✅ 数据集配置：")
    print(f"  p_satinfo_json: {p_satinfo_json}")
    print(f"  p_uavinfo_json: {p_uavinfo_json}")
    print(f"  p_uav_geocsv:   {p_uav_geocsv}")

    # 初始化数据集
    print("\n⏳ 正在加载数据集...")
    sat_dataset = SatDataset(
        p_satinfo_json=p_satinfo_json,
        p_uav_geocsv=p_uav_geocsv,
        imgsize2net=224,
    )

    uav_dataset = UAVDataset(
        p_uavinfo_json=p_uavinfo_json,
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        stage='train',
    )

    print("✅ 数据集加载完成！")

    # 执行验证
    verify_sat_dataset_ranges(sat_dataset)
    verify_uav_dataset_ranges(uav_dataset)
    verify_random_sampling(sat_dataset, n_samples=10000)
    verify_udf_normalization_factors(sat_dataset)
    verify_dataset_samples(sat_dataset, uav_dataset, n_samples=3)

    # 总结
    print_section("✅ 验证完成")
    print("\n主要发现：")
    print("  1. nr, nc ∈ [nr2sample_min, nr2sample_max] (归一化空间，非 [0,1])")
    print("  2. rotation_rad ∈ [-π, π] (弧度)")
    print("  3. scale_ratio_to_refm ∈ [scale_min, scale_max] (相对于200米的比例)")
    print("  4. UDF归一化因子确保各维度归一化后都在 [0, 1] 范围")
    print("  5. ⚠️ 需要检查 halfimg_radius_nrc 与 UDF距离的量级是否匹配！")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
