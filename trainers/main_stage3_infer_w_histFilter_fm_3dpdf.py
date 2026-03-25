#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于保存的3D网格概率进行历史滤波推理的脚本

该脚本用于：
1. 读取 _test_3d_classification_accuracy 保存的3D网格概率分布
2. 还原3D概率数据结构
3. 对序列数据进行历史滤波（temporal filtering）
4. 评估滤波后的定位性能
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/data/zwk/pyproj_neuloc_v0')
from trainer_depends.datasets.dataset_wingtra_4d import UAVDataset, SatDataset

def load_3d_pdf_predictions(npz_path):
    """
    读取并还原3D网格概率预测结果

    Args:
        npz_path: .npz文件路径

    Returns:
        dict: 包含以下键值的字典
            - pred_pdf_3d_all: [N, NR*NC*Rot] 预测的3D概率分布
            - q_label_3d_all: [N] GT的3D标签
            - coords_gt_all: [N, 4] GT坐标 (nr, nc, rot, scale)
            - n_coarse_3d: [3] 网格维度 (NR, NC, Rot)
            - cell_centers_3d: [NR, NC, Rot, 3] 每个3D cell的中心坐标
            - temperature: float, softmax温度参数
            - n_samples: int, 样本数量
            - data_type: str, 数据类型 ('train' or 'test')
    """
    print(f"\n{'='*60}")
    print(f"读取3D网格概率预测数据")
    print(f"{'='*60}")
    print(f"文件路径: {npz_path}")

    # 检查文件是否存在
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)

    # 提取数据
    result = {
        'pred_pdf_3d_all': data['pred_pdf_3d_all'],  # [N, NR*NC*Rot]
        'q_label_3d_all': data['q_label_3d_all'],    # [N]
        'coords_gt_all': data['coords_gt_all'],      # [N, 4]
        'n_coarse_3d': data['n_coarse_3d'],          # [NR, NC, Rot]
        'cell_centers_3d': data['cell_centers_3d'],  # [NR, NC, Rot, 3]
        'n_samples': int(data['n_samples']),
        'data_type': str(data['data_type']),
    }

    # 打印数据信息
    print(f"\n数据加载成功:")
    print(f"  数据类型: {result['data_type']}")
    print(f"  样本数量: {result['n_samples']}")
    print(f"  网格维度: {result['n_coarse_3d']} (NR={result['n_coarse_3d'][0]}, NC={result['n_coarse_3d'][1]}, Rot={result['n_coarse_3d'][2]})")
    print(f"  总网格数: {result['n_coarse_3d'][0] * result['n_coarse_3d'][1] * result['n_coarse_3d'][2]}")
    print(f"\n数据形状:")
    print(f"  pred_pdf_3d_all: {result['pred_pdf_3d_all'].shape}")
    print(f"  q_label_3d_all: {result['q_label_3d_all'].shape}")
    print(f"  coords_gt_all: {result['coords_gt_all'].shape}")
    print(f"  cell_centers_3d: {result['cell_centers_3d'].shape}")
    print(f"{'='*60}\n")

    return result


def reshape_pdf_to_3d(pred_pdf_flat, n_coarse_3d):
    """
    将展平的3D概率分布还原为3D形状

    Args:
        pred_pdf_flat: [N, NR*NC*Rot] 展平的概率分布
        n_coarse_3d: [3] 网格维度 (NR, NC, Rot)

    Returns:
        pred_pdf_3d: [N, NR, NC, Rot] 3D形状的概率分布
    """
    n_samples = pred_pdf_flat.shape[0]
    nr, nc, n_rot = n_coarse_3d
    pred_pdf_3d = pred_pdf_flat.reshape(n_samples, nr, nc, n_rot)
    return pred_pdf_3d


def init_uav_dataset(p_satinfo_json, p_uavinfo_json, p_uavgeo_csv, stage='test', use_augmentation=False):
    """
    初始化UAV数据集以获取序列相对位移信息

    Args:
        p_satinfo_json: str, 卫星地图信息JSON文件路径
        p_uavinfo_json: str, UAV信息JSON文件路径
        stage: str, 'train' 或 'test' (默认: 'test')
        use_augmentation: bool, 是否使用数据增强 (默认: False，用于测试)

    Returns:
        dict: 包含以下键值的字典
            - sat_dataset: SatDataset对象
            - uav_dataset: UAVDataset对象
            - uav_coords_4d: np.ndarray, [N, 4] UAV的4D坐标 (nr, nc, rot, scale)
            - uav_georcs: np.ndarray, [N, 2] UAV的地理坐标
    """
    print(f"\n{'='*60}")
    print(f"初始化UAV数据集")
    print(f"{'='*60}")
    print(f"卫星地图JSON: {p_satinfo_json}")
    print(f"UAV信息JSON: {p_uavinfo_json}")
    print(f"数据集阶段: {stage}")
    print(f"数据增强: {use_augmentation}")

    # 1. 初始化SatDataset（需要获取geo_res_m和坐标转换函数）
    sat_dataset = SatDataset(
        p_satinfo_json=p_satinfo_json,
        p_uav_geocsv=p_uavgeo_csv,  # 这个参数可以不需要，因为我们主要用SAT的坐标转换功能
        imgsize2net=224,
        scale_ref_m=200,
    )
    print(f"\n✓ SatDataset初始化成功")
    print(f"  地理分辨率: {sat_dataset.geo_res_m:.4f} m/pixel")
    print(f"  EPSG代码: {sat_dataset.epsg_code}")

    # 2. 初始化UAVDataset
    uav_dataset = UAVDataset(
        p_uavinfo_json=p_uavinfo_json,
        imgsize2net=224,
        scale_ref_m=200,
        geo_res_m=sat_dataset.geo_res_m,
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        stage=stage,
        use_augmentation=use_augmentation
    )
    print(f"\n✓ UAVDataset初始化成功")
    print(f"  样本总数: {len(uav_dataset.uavimg_paths)}")

    # 3. 获取UAV坐标信息
    if stage == 'train':
        uav_coords_4d = uav_dataset.uav_coords_4d_torch_train.numpy()
        n_samples = uav_dataset.n_train
    else:  # test
        uav_coords_4d = uav_dataset.uav_coords_4d_torch_test.numpy()
        n_samples = len(uav_dataset.uav_coords_4d_torch_test)

    uav_georcs = uav_dataset.uav_georcs  # [N, 2] (geo_row, geo_col)

    print(f"  {stage}集样本数: {n_samples}")
    print(f"  4D坐标形状: {uav_coords_4d.shape}")
    print(f"  坐标范围:")
    print(f"    nr: [{uav_coords_4d[:, 0].min():.4f}, {uav_coords_4d[:, 0].max():.4f}]")
    print(f"    nc: [{uav_coords_4d[:, 1].min():.4f}, {uav_coords_4d[:, 1].max():.4f}]")
    print(f"    rot: [{uav_coords_4d[:, 2].min():.4f}, {uav_coords_4d[:, 2].max():.4f}] rad")
    print(f"    scale: [{uav_coords_4d[:, 3].min():.4f}, {uav_coords_4d[:, 3].max():.4f}]")
    print(f"{'='*60}\n")

    return {
        'sat_dataset': sat_dataset,
        'uav_dataset': uav_dataset,
        'uav_coords_4d': uav_coords_4d,
        'uav_georcs': uav_georcs,
        'n_samples': n_samples,
    }

from scripts.analysis.util_stage3_analyze_pred3d import (
    compute_top_k_accuracy,
    print_accuracy_results,
    compute_2d_plane_accuracy,
    compute_rotation_accuracy_at_gt_position,
)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='读取并分析3D网格概率预测结果，并初始化UAV数据集用于历史滤波')
    parser.add_argument('--npz_path', type=str,
                        default='/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_neural_proxy_nr40_nc30_r36_LossLogsum_9/loc_results/pred_3d_ep001_nr40_nc30_nrot36.npz',
                        help='保存的.npz文件路径')
    parser.add_argument('--p_satinfo_json', type=str,
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json',
                        help='卫星地图信息JSON文件路径')
    parser.add_argument('--p_uavinfo_json', type=str,
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json',
                        help='UAV信息JSON文件路径')
    parser.add_argument('--p_uavgeo_csv', type=str,
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv',
                        help='UAV信息CSV文件路径')
    parser.add_argument('--stage', type=str, default='test', choices=['train', 'test'],
                        help='数据集阶段 (train/test)')
    args = parser.parse_args()

    # 1. 读取3D预测数据
    data = load_3d_pdf_predictions(args.npz_path)

    # 2. 还原3D形状
    pred_pdf_3d = reshape_pdf_to_3d(
        data['pred_pdf_3d_all'],
        data['n_coarse_3d']
    )
    print(f"还原3D形状: {pred_pdf_3d.shape} (N, NR, NC, Rot)")

    # 3. 计算单帧准确率（验证数据正确性）
    print("\n计算单帧准确率（验证数据）:")
    single_frame_results = compute_top_k_accuracy(
        data['pred_pdf_3d_all'],
        data['q_label_3d_all'],
        k_values=[1, 8, 27, 64, 128, 256, 512]
    )
    print_accuracy_results(single_frame_results, title="3D定位准确率（单帧）")

    results_2d_before = compute_2d_plane_accuracy(
        data['pred_pdf_3d_all'],
        data['q_label_3d_all'],
        shape_3d=data['n_coarse_3d'],
        k_values=[1, 8, 27, 64,128,256,512]
    )
    print_accuracy_results(results_2d_before, title="2D平面定位准确率 (滤波前)")

    results_rot_before = compute_rotation_accuracy_at_gt_position(
        data['pred_pdf_3d_all'],
        gt_labels_3d=data['q_label_3d_all'],
        shape_3d=data['n_coarse_3d'],
        dim_order='HWO',
        k_values=[1, 2, 3, 4]
    )
    print_accuracy_results(results_rot_before, title="rot定位准确率 (滤波前)")

    # # 4. 数据统计
    # print(f"\n数据统计信息:")
    # print(f"  预测概率范围: [{data['pred_pdf_3d_all'].min():.6f}, {data['pred_pdf_3d_all'].max():.6f}]")
    # print(f"  预测概率和（验证归一化）: 平均={data['pred_pdf_3d_all'].sum(axis=1).mean():.6f}")
    # print(f"  GT标签范围: [{data['q_label_3d_all'].min()}, {data['q_label_3d_all'].max()}]")
    # print(f"  GT坐标范围:")
    # print(f"    nr: [{data['coords_gt_all'][:, 0].min():.4f}, {data['coords_gt_all'][:, 0].max():.4f}]")
    # print(f"    nc: [{data['coords_gt_all'][:, 1].min():.4f}, {data['coords_gt_all'][:, 1].max():.4f}]")
    # print(f"    rot: [{data['coords_gt_all'][:, 2].min():.4f}, {data['coords_gt_all'][:, 2].max():.4f}]")

    # 5. 初始化UAV数据集
    p2uav_coords_4d = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uav_4dcoords_test.npz'
    # dataset_info = init_uav_dataset(
    #     p_satinfo_json=args.p_satinfo_json,
    #     p_uavinfo_json=args.p_uavinfo_json,
    #     p_uavgeo_csv=args.p_uavgeo_csv,
    #     stage=args.stage,
    #     use_augmentation=False
    # )
    # uav_coords_4d = dataset_info['uav_coords_4d']
    # np.savez(p2uav_coords_4d,uav_coords_4d)
    uav_coords_4d = torch.from_numpy(np.load(p2uav_coords_4d)['arr_0'])
    raw_diff = torch.diff(uav_coords_4d[:,2])
    diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi

    from util_core_histogram_filter_3d import HistogramFilter3D
    H ,W , O = data['n_coarse_3d'][0],data['n_coarse_3d'][1],data['n_coarse_3d'][2]
    histfilter = HistogramFilter3D(H=H,W=W,O=O)
    pred_pdf_3d_hist = torch.from_numpy(pred_pdf_3d).to(histfilter.device).permute(0,3,1,2)
    preds_filtered = []
    histfilter.belief = histfilter.belief * pred_pdf_3d_hist[0:1]
    preds_filtered.append(histfilter.belief.clone())

    for i in range(diff_rot_rad.shape[0]):
        if i == diff_rot_rad.shape[0]:
            break
        # version0:the best for Integral Inference Scheme
        # histfilter.predict(move_rot=diff_rot_rad[i],noise_std_rot=30/180*torch.pi,direction_aware=False,noise_std_xy=0.65,xy_k_size=5)
        # histfilter.update(pred_pdf_3d_hist[i+1:i+2],alpha=0.25)
        # version1:
        histfilter.predict(move_rot=diff_rot_rad[i],noise_std_rot=30/180*torch.pi,direction_aware=False,noise_std_xy=0.65,xy_k_size=2)
        histfilter.update(pred_pdf_3d_hist[i+1:i+2],alpha=0.5)

        preds_filtered.append(histfilter.belief.clone())
    preds_filtered = torch.cat(preds_filtered)
    preds_filtered = preds_filtered.permute(0,2,3,1)

    preds_filtered_results = compute_top_k_accuracy(
        preds_filtered.reshape(preds_filtered.shape[0],-1).cpu().numpy(),
        data['q_label_3d_all'],
        k_values=[1, 8, 27, 64, 128 ,256,512]
    )
    print_accuracy_results(preds_filtered_results, title="3D定位准确率（滤波后）")

    results_2d_filtered = compute_2d_plane_accuracy(
        preds_filtered,
        data['q_label_3d_all'],
        shape_3d=data['n_coarse_3d'],
        dim_order='HWO',
        k_values=[1, 2, 3, 4, 9, 16]
    )
    print_accuracy_results(results_2d_filtered, title="2D平面定位准确率 (滤波后)")

    results_rot_filtered = compute_rotation_accuracy_at_gt_position(
        preds_filtered,
        gt_labels_3d=data['q_label_3d_all'],
        shape_3d=data['n_coarse_3d'],
        dim_order='HWO',
        k_values=[1, 2, 3, 4, 5, 6, 12,18]
    )
    print_accuracy_results(results_rot_filtered, title="rot定位准确率 (滤波后)")

    from scripts.analysis.util_stage3_analyze_pred3d import compute_2d_neighbors_recall
    compute_2d_neighbors_recall(preds_filtered,gt_labels=data['q_label_3d_all'], dim_order='HWO',len_neighbors=3)

    acc_3D_top512 = preds_filtered_results['top512_acc']
    torch.save(preds_filtered.permute(0,2,3,1),os.path.join(os.path.dirname(args.npz_path),os.path.basename(args.npz_path).split('.npz')[0]+f'_filtered_3dTop512Acc{acc_3D_top512:.1f}.pt'))







