#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D网格序列定位测试函数

用于添加到 stage3_project_integrateRot_classify.py 的 MetricNetTrainer 类中
"""

import torch
import torch.nn.functional as TF
import numpy as np


def _test_2d_sequence_localization_accuracy(
    self,
    n_samples=None,
    use_train_uav=False,
    temperature=0.5,
    seq_window_len=5,
    len_neighbors=2,
    shuffle=False
):
    """
    测试2D平面序列定位精度（支持序列聚合）

    这个函数是 _test_2d_classification_accuracy 的升级版，支持：
    1. 单帧定位测试
    2. 序列聚合定位测试
    3. n×n邻域聚合定位测试

    Args:
        n_samples: 测试样本数量（None表示使用全部数据）
        use_train_uav: 是否使用训练集UAV数据
        temperature: softmax温度参数
        seq_window_len: 序列聚合窗口长度
        len_neighbors: 邻域大小（2表示2×2=4个邻域）
        shuffle: 是否打乱数据顺序（序列测试时应该为False）

    Returns:
        dict: 包含各种准确率指标的字典
    """
    from trainer_depends.datasets.util_loc_in_girds import (
        agg_seq_pdf,
        compute_agged_pred_nneighbors_id
    )

    print(f"\n{'='*80}")
    print(f"2D序列定位测试")
    print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
    print(f"序列聚合窗口: {seq_window_len}")
    print(f"邻域大小: {len_neighbors}×{len_neighbors} = {len_neighbors**2}")
    print(f"{'='*80}\n")

    # 获取数据集
    dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

    # 确定测试样本数
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))

    # 创建DataLoader（序列测试通常不打乱）
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    # ==================== 预计算候选点特征 ====================
    print("预计算所有候选点的特征...")

    coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
        n_points_per_subspace=1, use_fine=False
    )
    n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
    n_coarse_2d = n_coarse[:2]  # [NR, NC]
    n_grid_h, n_grid_w = n_coarse_2d[0], n_coarse_2d[1]

    # 预计算所有候选点的grid特征
    coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
    coords_6d_flat = self.coord_normer.raw_to_norm(coords_flat, append_linear_rot=True)
    grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

    with torch.no_grad():
        feats_grid_raw = self._get_feats_fm_grid(grid_input)
        coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
        feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
        feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

    # 获取每个2D grid cell的中心坐标
    coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
    cell_centers_2d = coords_reshaped[:, :, 0, 0, :2]  # [NR, NC, 2]

    print(f"✅ 候选点特征预计算完成，网格大小: {n_grid_h}×{n_grid_w}\n")

    # ==================== 收集所有预测和GT ====================
    print("开始推理并收集预测概率...")

    pred_pdf_list = []  # 收集所有预测的概率分布
    q_label_list = []   # 收集所有GT标签
    coords_gt_list = [] # 收集所有GT坐标
    processed = 0

    with torch.no_grad():
        for batch in test_loader:
            if processed >= n_samples:
                break

            imgs = batch[0].to(self.device)
            coords_gt = batch[1].to(self.device)  # [B, 4]
            batch_size = imgs.shape[0]

            # 提取视觉特征
            feats_vis = self._get_feats_fm_imgs(imgs)  # [B, C]

            # 计算能量
            energys = self.projector.compute_energy(
                feats_vis, feats_grid_all, metric='euclidean'
            )  # [B, N_total]

            # Reshape成 [B, NR, NC, Rot, Scale]
            energys_reshaped = energys.reshape(batch_size, *n_coarse)

            # 边缘化Rot和Scale维度 (使用logsumexp)
            neg_dists = -energys_reshaped
            scaled_logits = neg_dists / temperature
            logits_2d = torch.logsumexp(scaled_logits, dim=[-2, -1])  # [B, NR, NC]

            # 转换为概率分布
            pred_pdf = torch.softmax(logits_2d.view(batch_size, -1), dim=-1)  # [B, H*W]
            pred_pdf_list.append(pred_pdf.cpu())

            # 计算GT标签
            gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
            gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
            gt_nr = gt_indices_multi[:, 0]
            gt_nc = gt_indices_multi[:, 1]
            gt_flat_idx = gt_nr * n_coarse_2d[1] + gt_nc
            q_label_list.append(gt_flat_idx.cpu())
            coords_gt_list.append(coords_gt.cpu())

            processed += batch_size

    # 拼接所有结果
    pred_pdf_all = torch.cat(pred_pdf_list, dim=0)  # [N, H*W]
    q_label_all = torch.cat(q_label_list, dim=0)    # [N]
    coords_gt_all = torch.cat(coords_gt_list, dim=0)  # [N, 4]
    n_total_samples = pred_pdf_all.shape[0]

    print(f"✅ 收集完成，共 {n_total_samples} 个样本\n")

    # ==================== 单帧定位测试 ====================
    print("="*80)
    print("1. 单帧定位测试")
    print("="*80)

    results_single = _compute_2d_localization_metrics(
        pred_pdf_all,
        q_label_all,
        coords_gt_all,
        cell_centers_2d,
        n_grid_h,
        n_grid_w,
        title="单帧定位"
    )

    # ==================== 序列聚合定位测试 ====================
    print("\n" + "="*80)
    print(f"2. 序列聚合定位测试 (窗口长度={seq_window_len})")
    print("="*80)

    # 进行序列聚合
    pred_pdf_agged = agg_seq_pdf(
        pred_pdf_all.to(self.device),
        window_len=seq_window_len,
        padding=False
    ).cpu()  # [N-window_len+1, H*W]

    # GT也需要截取对应的部分
    q_label_agged = q_label_all[seq_window_len-1:]
    coords_gt_agged = coords_gt_all[seq_window_len-1:]

    results_seq = _compute_2d_localization_metrics(
        pred_pdf_agged,
        q_label_agged,
        coords_gt_agged,
        cell_centers_2d,
        n_grid_h,
        n_grid_w,
        title=f"序列聚合(窗口={seq_window_len})"
    )

    # ==================== 邻域聚合测试 ====================
    print("\n" + "="*80)
    print(f"3. {len_neighbors}×{len_neighbors}邻域聚合测试")
    print("="*80)

    # 计算单帧的邻域聚合
    id_neighbors_1d, id_neighbors_2d = compute_agged_pred_nneighbors_id(
        pred_pdf_all.reshape(-1, n_grid_h, n_grid_w).to(self.device),
        len_neighbors,
        ret_2d=True
    )  # [N, n*n], [N, n*n, 2]

    k_values = list(range(1, len_neighbors**2 + 1))
    results_neighbors = _compute_neighbors_recall(
        q_label_all.numpy(),
        id_neighbors_1d.cpu().numpy(),
        k_values,
        title=f"单帧+{len_neighbors**2}邻域"
    )

    # 计算序列聚合+邻域聚合
    id_neighbors_seq_1d, id_neighbors_seq_2d = compute_agged_pred_nneighbors_id(
        pred_pdf_agged.reshape(-1, n_grid_h, n_grid_w).to(self.device),
        len_neighbors,
        ret_2d=True
    )

    results_seq_neighbors = _compute_neighbors_recall(
        q_label_agged.numpy(),
        id_neighbors_seq_1d.cpu().numpy(),
        k_values,
        title=f"序列聚合+{len_neighbors**2}邻域"
    )

    # ==================== 汇总结果 ====================
    results = {
        # 基本信息
        'n_samples': n_total_samples,
        'n_grid_cells': n_grid_h * n_grid_w,
        'n_grid_h': n_grid_h,
        'n_grid_w': n_grid_w,
        'seq_window_len': seq_window_len,
        'len_neighbors': len_neighbors,

        # 单帧结果
        'single_frame': results_single,

        # 序列聚合结果
        'sequence_agg': results_seq,

        # 邻域结果
        'neighbors': results_neighbors,
        'seq_neighbors': results_seq_neighbors,
    }

    # ==================== 打印总结 ====================
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总样本数: {n_total_samples}")
    print(f"网格大小: {n_grid_h}×{n_grid_w} = {n_grid_h*n_grid_w} cells")
    print(f"\n【单帧定位】")
    print(f"  Top-1: {results_single['top1_acc']:.2f}%")
    print(f"  Top-5: {results_single['top5_acc']:.2f}%")
    print(f"  平均距离误差: {results_single['mean_dist_error']:.4f}")
    print(f"\n【序列聚合 (窗口={seq_window_len})】")
    print(f"  Top-1: {results_seq['top1_acc']:.2f}%")
    print(f"  Top-5: {results_seq['top5_acc']:.2f}%")
    print(f"  平均距离误差: {results_seq['mean_dist_error']:.4f}")
    print(f"\n【{len_neighbors**2}邻域聚合】")
    print(f"  单帧@1: {results_neighbors['recall@1']:.2f}%")
    print(f"  单帧@{len_neighbors**2}: {results_neighbors[f'recall@{len_neighbors**2}']:.2f}%")
    print(f"  序列@1: {results_seq_neighbors['recall@1']:.2f}%")
    print(f"  序列@{len_neighbors**2}: {results_seq_neighbors[f'recall@{len_neighbors**2}']:.2f}%")
    print("="*80 + "\n")

    return results


def _compute_2d_localization_metrics(
    pred_pdf,
    q_labels,
    coords_gt,
    cell_centers_2d,
    n_grid_h,
    n_grid_w,
    title="定位测试"
):
    """
    计算2D定位指标

    Args:
        pred_pdf: [N, H*W] 预测概率
        q_labels: [N] GT标签
        coords_gt: [N, 4] GT坐标
        cell_centers_2d: [H, W, 2] 每个cell的中心坐标
        n_grid_h, n_grid_w: 网格大小
        title: 标题
    """
    # 预测的cell索引
    pred_indices = pred_pdf.argmax(dim=-1)  # [N]
    pred_nr = pred_indices // n_grid_w
    pred_nc = pred_indices % n_grid_w

    # GT的cell索引
    gt_nr = q_labels // n_grid_w
    gt_nc = q_labels % n_grid_w

    # 计算排名
    sorted_indices = pred_pdf.argsort(dim=-1, descending=True)
    ranks = []
    for i in range(len(q_labels)):
        rank = (sorted_indices[i] == q_labels[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    # 计算距离误差
    pred_centers = cell_centers_2d[pred_nr, pred_nc]  # [N, 2]
    gt_coords_2d = coords_gt[:, :2]  # [N, 2]
    dist_errors = torch.norm(pred_centers - gt_coords_2d, dim=-1, p=2).numpy()

    # 汇总指标
    results = {
        'n_samples': len(ranks),
        'top1_acc': (ranks == 1).mean() * 100,
        'top2_acc': (ranks <= 2).mean() * 100,
        'top3_acc': (ranks <= 3).mean() * 100,
        'top4_acc': (ranks <= 4).mean() * 100,
        'top5_acc': (ranks <= 5).mean() * 100,
        'top10_acc': (ranks <= 10).mean() * 100,
        'mean_rank': ranks.mean(),
        'median_rank': np.median(ranks),
        'mean_dist_error': dist_errors.mean(),
        'median_dist_error': np.median(dist_errors),
        'dist_error_std': dist_errors.std(),
    }

    # 打印结果
    print(f"{title}结果:")
    print(f"  样本数: {results['n_samples']}")
    print(f"  Top-1: {results['top1_acc']:.2f}%")
    print(f"  Top-2: {results['top2_acc']:.2f}%")
    print(f"  Top-3: {results['top3_acc']:.2f}%")
    print(f"  Top-5: {results['top5_acc']:.2f}%")
    print(f"  Top-10: {results['top10_acc']:.2f}%")
    print(f"  平均排名: {results['mean_rank']:.2f}")
    print(f"  中位数排名: {results['median_rank']:.2f}")
    print(f"  距离误差 - 平均: {results['mean_dist_error']:.4f}")
    print(f"  距离误差 - 中位数: {results['median_dist_error']:.4f}")
    print(f"  距离误差 - 标准差: {results['dist_error_std']:.4f}")

    return results


def _compute_neighbors_recall(q_labels, id_neighbors, k_values, title="邻域Recall"):
    """
    计算邻域recall

    Args:
        q_labels: [N] GT标签
        id_neighbors: [N, K] 预测的K个邻域cell索引
        k_values: list of int，要计算的k值
        title: 标题
    """
    # 使用现成的函数
    try:
        # 尝试从pyproj_pylib_zwk导入
        import sys
        sys.path.insert(0, '/home/data/zwk/pyproj_pylib_zwk')
        from uavloc_utils.eval_recall_fm_salad import compute_recall_by_label
        recall_dict = compute_recall_by_label(q_labels, id_neighbors, k_values, title=title)
    except:
        # 如果导入失败，手动计算
        print(f"⚠️  无法导入compute_recall_by_label，使用简化版计算")
        recall_dict = {}
        for k in k_values:
            # 检查GT是否在top-k中
            correct = np.any(id_neighbors[:, :k] == q_labels[:, None], axis=1)
            recall = correct.mean() * 100
            recall_dict[f'recall@{k}'] = recall
            print(f"  Recall@{k}: {recall:.2f}%")

    return recall_dict
