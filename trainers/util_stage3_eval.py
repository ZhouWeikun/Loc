import os
import numpy as np
import torch
import torch.nn.functional as TF

def _run_epoch_evaluation(self, epoch, run_visualization=False, n_test_samples=256):
        """
        在每个epoch结束时运行评估

        Args:
            epoch: 当前epoch编号
            run_visualization: 是否运行可视化（默认False）
            n_test_samples: 测试样本数量
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Epoch {epoch} 评估开始")
        self.logger.info(f"{'=' * 60}")

        # 切换为eval模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 运行3D分类测试
        results_3d = self._test_3d_classification_accuracy(
            n_samples=n_test_samples,
            use_train_uav=False,
            temperature=self.energy_temperature,
        )

        # 记录测试结果到日志
        self.logger.info(f"\n[Epoch {epoch}] 3D分类测试结果:")
        self.logger.info(f"  样本数: {results_3d['n_samples']}")
        self.logger.info(f"  总cell数: {results_3d['n_total_cells']}")
        self.logger.info(f"  Top-1准确率: {results_3d['top1_acc']:.2f}%")
        self.logger.info(f"  Top-8准确率: {results_3d['top8_acc']:.2f}%")
        self.logger.info(f"  Top-27准确率: {results_3d['top27_acc']:.2f}%")
        self.logger.info(f"  Top-64准确率: {results_3d['top64_acc']:.2f}%")
        self.logger.info(f"  Top-256准确率: {results_3d['top256_acc']:.2f}%")
        self.logger.info(f"  Top-512准确率: {results_3d['top512_acc']:.2f}%")
        self.logger.info(f"  平均排名: {results_3d['mean_rank']:.2f}")
        self.logger.info(f"  中位数排名: {results_3d['median_rank']:.2f}")
        self.logger.info(f"  2D位置误差 - 平均: {results_3d['mean_dist_error_2d']:.4f}")
        self.logger.info(f"  2D位置误差 - 中位数: {results_3d['median_dist_error_2d']:.4f}")
        self.logger.info(f"  旋转误差 - 平均: {results_3d['mean_rot_error_deg']:.2f}°")
        self.logger.info(f"  旋转误差 - 中位数: {results_3d['median_rot_error_deg']:.2f}°")
        self.logger.info(f"  固定位置旋转Top-1准确率: {results_3d['rot_only_top1_acc']:.2f}%")
        self.logger.info(f"  固定位置旋转Top-2准确率: {results_3d['rot_only_top2_acc']:.2f}%")
        self.logger.info(f"  固定位置旋转Top-3准确率: {results_3d['rot_only_top3_acc']:.2f}%")
        self.logger.info(f"{'=' * 60}\n")

        # 记录到TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('test/top1_acc', results_3d['top1_acc'], epoch)
            self.writer.add_scalar('test/top8_acc', results_3d['top8_acc'], epoch)
            self.writer.add_scalar('test/top27_acc', results_3d['top27_acc'], epoch)
            self.writer.add_scalar('test/top64_acc', results_3d['top64_acc'], epoch)
            self.writer.add_scalar('test/top256_acc', results_3d['top256_acc'], epoch)
            self.writer.add_scalar('test/top512_acc', results_3d['top512_acc'], epoch)
            self.writer.add_scalar('test/mean_rank', results_3d['mean_rank'], epoch)
            self.writer.add_scalar('test/mean_dist_error_2d', results_3d['mean_dist_error_2d'], epoch)
            self.writer.add_scalar('test/mean_rot_error_deg', results_3d['mean_rot_error_deg'], epoch)
            self.writer.add_scalar('test/rot_only_top1_acc', results_3d['rot_only_top1_acc'], epoch)

        # 恢复为train模式
        for model in self.param2optimize.values():
            model.train()

        self.logger.info(f"Epoch {epoch} 评估完成\n")
        return results_3d

def _test_2d_classification_accuracy(self, n_samples=256, use_train_uav=False, temperature=0.5):
        """
        测试2D平面分类正确性

        Args:
            n_samples: 测试样本数量
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数

        Returns:
            dict: 包含各种准确率指标的字典
        """
        print(f"\n{'=' * 60}")
        print(f"2D平面分类测试")
        print(f"测试样本数: {n_samples}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, n_samples),
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False
        )
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_2d = n_coarse[:2]  # [NR, NC]

        # 预计算所有候选点的grid特征
        coords_flat = coords_candidates.view(-1, 4)  # [N_total, 4]
        coords_6d_flat = self.coord_normer.raw_to_net(coords_flat, append_linear_rot=True)
        grid_input = torch.cat([coords_6d_flat[:, :2], coords_6d_flat[:, -1:]], dim=-1)

        with torch.no_grad():
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_6d_flat[:, :5])
            feats_grid_all = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
            feats_grid_all = TF.normalize(feats_grid_all, dim=-1)  # [N_total, C]

        # 获取每个2D grid cell的中心坐标（用于计算距离误差）
        # coords_candidates shape: [1, N_total, 4], 需要reshape成 [NR, NC, Rot, Scale, 4]
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        # 取每个2D cell的中心（对Rot和Scale取均值）
        cell_centers_2d = coords_reshaped[:, :, 0, 0, :2]  # [NR, NC, 2] 取第一个rot/scale的坐标

        # 统计容器
        all_ranks = []
        all_dist_errors = []
        all_rot_ranks_weighted = []  # 加权聚合后的旋转排名
        all_rot_ranks_marginalized = []  # 直接边缘化后的旋转排名
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

                # 获取预测的2D索引
                logits_flat = logits_2d.view(batch_size, -1)  # [B, NR*NC]
                pred_indices = logits_flat.argmax(dim=-1)  # [B]
                pred_nr = pred_indices // n_coarse_2d[1]
                pred_nc = pred_indices % n_coarse_2d[1]

                # 计算GT的2D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_flat_idx = gt_nr * n_coarse_2d[1] + gt_nc

                # 计算排名
                sorted_indices = logits_flat.argsort(dim=-1, descending=True)
                for i in range(batch_size):
                    rank = (sorted_indices[i] == gt_flat_idx[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_ranks.append(rank)

                # 计算距离误差（预测cell中心到GT坐标的距离）
                pred_centers = cell_centers_2d[pred_nr, pred_nc]  # [B, 2]
                gt_coords_2d = coords_gt[:, :2]  # [B, 2]
                dist_errors = torch.norm(pred_centers - gt_coords_2d, dim=-1, p=2)
                all_dist_errors.extend(dist_errors.cpu().numpy().tolist())

                # ========== 计算加权旋转分类精度（类似train的逻辑） ==========
                # 1. 获取4D logits: [B, NR, NC, Rot] (边缘化Scale维度)
                logits_4d = torch.logsumexp(scaled_logits, dim=-1)  # [B, NR, NC, Rot]

                # 2. 计算2D空间概率分布（作为权重）
                probs_2d = torch.softmax(logits_2d.view(batch_size, -1) / temperature, dim=-1)  # [B, NR*NC]
                probs_2d = probs_2d.reshape(batch_size, n_coarse[0], n_coarse[1])  # [B, NR, NC]

                # 3. 计算每个2D位置的旋转概率分布
                probs_rot_local = torch.softmax(logits_4d / temperature, dim=-1)  # [B, NR, NC, Rot]

                # 4. 用2D空间权重对旋转分布进行加权聚合
                spatial_weights = probs_2d.unsqueeze(-1)  # [B, NR, NC, 1]
                probs_rot_aggregated = (probs_rot_local * spatial_weights).sum(dim=[1, 2])  # [B, Rot]

                # 5. 计算GT旋转的排名
                gt_rot = gt_indices_multi[:, 2]  # [B]
                for i in range(batch_size):
                    # 对聚合后的旋转分布排序
                    sorted_rot_indices = probs_rot_aggregated[i].argsort(descending=True)
                    # 找到GT旋转的排名
                    rot_rank_weighted = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks_weighted.append(rot_rank_weighted)

                # ========== 计算边缘化旋转分类精度（直接对除Rot外的维度求和） ==========
                # 1. 计算概率体积: [B, NR, NC, Rot, Scale]
                probs_5d = torch.softmax(scaled_logits.view(batch_size, -1) / temperature, dim=-1)
                probs_5d = probs_5d.reshape(batch_size, *n_coarse)  # [B, NR, NC, Rot, Scale]

                # 2. 边缘化除Rot外的所有维度（求和）
                probs_rot_marginalized = probs_5d.sum(dim=[1, 2, 4])  # [B, Rot]

                # 3. 计算GT旋转的排名
                for i in range(batch_size):
                    # 对边缘化后的旋转分布排序
                    sorted_rot_indices = probs_rot_marginalized[i].argsort(descending=True)
                    # 找到GT旋转的排名
                    rot_rank_marg = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks_marginalized.append(rot_rank_marg)

                processed += batch_size

        # 计算指标
        all_ranks = np.array(all_ranks)
        all_dist_errors = np.array(all_dist_errors)
        all_rot_ranks_weighted = np.array(all_rot_ranks_weighted)
        all_rot_ranks_marginalized = np.array(all_rot_ranks_marginalized)
        n_total_cells = n_coarse_2d[0] * n_coarse_2d[1]

        results = {
            'n_samples': len(all_ranks),
            'n_total_cells': n_total_cells,
            'top1_acc': (all_ranks == 1).mean() * 100,
            'top4_acc': (all_ranks <= 4).mean() * 100,
            'top9_acc': (all_ranks <= 9).mean() * 100,
            'top16_acc': (all_ranks <= 16).mean() * 100,
            'mean_rank': all_ranks.mean(),
            'median_rank': np.median(all_ranks),
            'mean_dist_error': all_dist_errors.mean(),
            'median_dist_error': np.median(all_dist_errors),
            'dist_error_std': all_dist_errors.std(),
            # 加权旋转分类准确率
            'rot_weighted_top1_acc': (all_rot_ranks_weighted == 1).mean() * 100,
            'rot_weighted_top2_acc': (all_rot_ranks_weighted <= 2).mean() * 100,
            'rot_weighted_top3_acc': (all_rot_ranks_weighted <= 3).mean() * 100,
            'rot_weighted_mean_rank': all_rot_ranks_weighted.mean(),
            'rot_weighted_median_rank': np.median(all_rot_ranks_weighted),
            # 边缘化旋转分类准确率
            'rot_marginalized_top1_acc': (all_rot_ranks_marginalized == 1).mean() * 100,
            'rot_marginalized_top2_acc': (all_rot_ranks_marginalized <= 2).mean() * 100,
            'rot_marginalized_top3_acc': (all_rot_ranks_marginalized <= 3).mean() * 100,
            'rot_marginalized_mean_rank': all_rot_ranks_marginalized.mean(),
            'rot_marginalized_median_rank': np.median(all_rot_ranks_marginalized),
        }

        # 打印结果
        print(f"\n{'=' * 60}")
        print(f"2D分类测试结果 (共{n_total_cells}个cell)")
        print(f"{'=' * 60}")
        print(f"Top-1  准确率: {results['top1_acc']:.2f}%")
        print(f"Top-4  准确率: {results['top4_acc']:.2f}%")
        print(f"Top-9  准确率: {results['top9_acc']:.2f}%")
        print(f"Top-16 准确率: {results['top16_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"距离误差统计:")
        print(f"  平均误差: {results['mean_dist_error']:.4f}")
        print(f"  中位数误差: {results['median_dist_error']:.4f}")
        print(f"  标准差: {results['dist_error_std']:.4f}")
        print(f"{'=' * 60}")
        print(f"加权旋转分类准确率 (共{n_coarse[2]}个rot, 使用2D预测概率加权):")
        print(f"  Top-1 准确率: {results['rot_weighted_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_weighted_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_weighted_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_weighted_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_weighted_median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"边缘化旋转分类准确率 (共{n_coarse[2]}个rot, 直接对除Rot外维度求和):")
        print(f"  Top-1 准确率: {results['rot_marginalized_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_marginalized_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_marginalized_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_marginalized_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_marginalized_median_rank']:.2f}")
        print(f"{'=' * 60}\n")

        return results

def _test_2d_sequence_localization_accuracy(
            self,
            n_samples=None,
            use_train_uav=False,
            temperature=0.5,
            seq_window_len=5,
            len_neighbors=2,
            shuffle=False,
            save_pred_pdf=True
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
            save_pred_pdf: 是否保存预测概率分布到checkpoint文件夹（默认True）

        Returns:
            dict: 包含各种准确率指标的字典
        """
        from trainer_depends.datasets.util_core_loc_in_girds import (
            agg_seq_pdf,
            compute_agged_pred_nneighbors_id
        )

        print(f"\n{'=' * 80}")
        print(f"2D序列定位测试")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"序列聚合窗口: {seq_window_len}")
        print(f"邻域大小: {len_neighbors}×{len_neighbors} = {len_neighbors ** 2}")
        print(f"{'=' * 80}\n")

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
        coords_6d_flat = self.coord_normer.raw_to_net(coords_flat, append_linear_rot=True)
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
        q_label_list = []  # 收集所有GT标签
        coords_gt_list = []  # 收集所有GT坐标
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
        q_label_all = torch.cat(q_label_list, dim=0)  # [N]
        coords_gt_all = torch.cat(coords_gt_list, dim=0)  # [N, 4]
        n_total_samples = pred_pdf_all.shape[0]

        print(f"✅ 收集完成，共 {n_total_samples} 个样本\n")

        # ==================== 保存预测概率分布 ====================
        if save_pred_pdf:
            import os

            # 获取checkpoint路径
            stage3_ckpt_path = self._get_stage3_checkpoint_path()
            if stage3_ckpt_path:
                # 保存到checkpoint同目录
                ckpt_dir = os.path.dirname(stage3_ckpt_path)
                save_dir = os.path.join(ckpt_dir, 'loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 准备保存数据
                save_data = {
                    'pred_pdf_all': pred_pdf_all.numpy(),  # [N, H*W]
                    'q_label_all': q_label_all.numpy(),  # [N]
                    'coords_gt_all': coords_gt_all.numpy(),  # [N, 4]
                    'n_grid_h': n_grid_h,
                    'n_grid_w': n_grid_w,
                    'n_samples': n_total_samples,
                    'cell_centers_2d': cell_centers_2d.cpu().numpy(),  # [H, W, 2]
                    'test_config': {
                        'use_train_uav': use_train_uav,
                        'temperature': temperature,
                    }
                }

                # 保存为npz文件
                dataset_name = 'train' if use_train_uav else 'test'
                save_path = os.path.join(save_dir, f'pred_pdf_{dataset_name}_n{n_total_samples}.npz')
                np.savez_compressed(save_path, **save_data)

                print(f"💾 已保存预测概率分布: {save_path}")
                print(f"   - pred_pdf_all shape: {pred_pdf_all.shape}")
                print(f"   - q_label_all shape: {q_label_all.shape}")
                print(f"   - coords_gt_all shape: {coords_gt_all.shape}")
                print(f"   - cell_centers_2d shape: {cell_centers_2d.shape}\n")
            else:
                print("⚠️  未找到checkpoint路径，跳过保存预测概率分布\n")

        # ==================== 单帧定位测试 ====================
        print("=" * 80)
        print("1. 单帧定位测试")
        print("=" * 80)

        results_single = self._compute_2d_loc_metrics(
            pred_pdf_all,
            q_label_all,
            coords_gt_all,
            cell_centers_2d,
            n_grid_w,
            title="单帧定位"
        )

        # ==================== 序列聚合定位测试 ====================
        print("\n" + "=" * 80)
        print(f"2. 序列聚合定位测试 (窗口长度={seq_window_len})")
        print("=" * 80)

        #debug2vis:
        if True:  # 设置为False可关闭可视化
            import matplotlib.pyplot as plt
            import os

            k_samples = 16  # 可视化前k个样本
            n_cols = 4  # 每行显示4个
            n_rows = (k_samples + n_cols - 1) // n_cols

            # Reshape概率分布 [k, H*W] -> [k, H, W]
            pred_pdf_vis = pred_pdf_all[:k_samples].reshape(k_samples, n_grid_h, n_grid_w).numpy()
            q_labels_vis = q_label_all[:k_samples].numpy()

            # 创建子图
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), dpi=100)
            if n_rows == 1:
                axes = axes[None, :]

            for idx in range(k_samples):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                # 绘制热力图
                im = ax.imshow(pred_pdf_vis[idx], cmap='hot', interpolation='bilinear')

                # 标记GT位置
                gt_label = q_labels_vis[idx]
                gt_nr = gt_label // n_grid_w
                gt_nc = gt_label % n_grid_w
                ax.plot(gt_nc, gt_nr, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)

                # 标记预测位置
                pred_label = pred_pdf_vis[idx].argmax()
                pred_nr = pred_label // n_grid_w
                pred_nc = pred_label % n_grid_w
                ax.plot(pred_nc, pred_nr, 'bx', markersize=12, markeredgewidth=2)

                # 标题
                is_correct = (pred_label == gt_label)
                title_color = 'green' if is_correct else 'red'
                ax.set_title(f'Sample {idx}\n{"✓" if is_correct else "✗"}',
                             fontsize=10, color=title_color)

                # 添加颜色条
                plt.colorbar(im, ax=ax, fraction=0.046)

                ax.axis('off')

            # 隐藏多余的子图
            for idx in range(k_samples, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')

            plt.tight_layout()

            # 保存可视化
            vis_save_dir = os.path.dirname(self.opt.load2test)
            epoch_id = os.path.basename(self.opt.load2test)[-7:-4]
            vis_save_path = os.path.join(vis_save_dir, f'pred_pdf_samples_' + epoch_id + '.png')
            plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"🎨 已保存概率分布可视化: {vis_save_path}")
            print(f"   - 绿色星号(*) = GT位置")
            print(f"   - 蓝色叉号(×) = 预测位置")
            print(f"   - 绿色标题 = 预测正确，红色标题 = 预测错误\n")

        # 进行序列聚合
        pred_pdf_agged = agg_seq_pdf(
            pred_pdf_all.to(self.device),
            window_len=seq_window_len,
            padding=False
        ).cpu()  # [N-window_len+1, H*W]

        # GT也需要截取对应的部分
        q_label_agged = q_label_all[seq_window_len - 1:]
        coords_gt_agged = coords_gt_all[seq_window_len - 1:]

        results_seq = self._compute_2d_loc_metrics(
            pred_pdf_agged,
            q_label_agged,
            coords_gt_agged,
            cell_centers_2d,
            n_grid_w,
            title=f"序列聚合(窗口={seq_window_len})"
        )

        # ==================== 邻域聚合测试 ====================
        print("\n" + "=" * 80)
        print(f"3. {len_neighbors}×{len_neighbors}邻域聚合测试")
        print("=" * 80)

        # 计算单帧的邻域聚合
        id_neighbors_1d, id_neighbors_2d = compute_agged_pred_nneighbors_id(
            pred_pdf_all.reshape(-1, n_grid_h, n_grid_w).to(self.device),
            len_neighbors,
            ret_2d=True
        )  # [N, n*n], [N, n*n, 2]

        k_values = list(range(1, len_neighbors ** 2 + 1))
        results_neighbors = self._compute_neighbors_recall(
            q_label_all.numpy(),
            id_neighbors_1d.cpu().numpy(),
            k_values,
            title=f"单帧+{len_neighbors ** 2}邻域"
        )

        # 计算序列聚合+邻域聚合
        id_neighbors_seq_1d, id_neighbors_seq_2d = compute_agged_pred_nneighbors_id(
            pred_pdf_agged.reshape(-1, n_grid_h, n_grid_w).to(self.device),
            len_neighbors,
            ret_2d=True
        )

        results_seq_neighbors = self._compute_neighbors_recall(
            q_label_agged.numpy(),
            id_neighbors_seq_1d.cpu().numpy(),
            k_values,
            title=f"序列聚合+{len_neighbors ** 2}邻域"
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
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        print(f"总样本数: {n_total_samples}")
        print(f"网格大小: {n_grid_h}×{n_grid_w} = {n_grid_h * n_grid_w} cells")
        print(f"\n【单帧定位】")
        print(f"  Top-1: {results_single['top1_acc']:.2f}%")
        print(f"  Top-4: {results_single['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_single['mean_dist_error']:.4f}")
        print(f"\n【序列聚合 (窗口={seq_window_len})】")
        print(f"  Top-1: {results_seq['top1_acc']:.2f}%")
        print(f"  Top-4: {results_seq['top4_acc']:.2f}%")
        print(f"  平均距离误差: {results_seq['mean_dist_error']:.4f}")
        print(f"\n【{len_neighbors ** 2}邻域聚合】")
        print(f"  单帧@1: {results_neighbors['recall@1']:.2f}%")
        print(f"  单帧@{len_neighbors ** 2}: {results_neighbors[f'recall@{len_neighbors ** 2}']:.2f}%")
        print(f"  序列@1: {results_seq_neighbors['recall@1']:.2f}%")
        print(f"  序列@{len_neighbors ** 2}: {results_seq_neighbors[f'recall@{len_neighbors ** 2}']:.2f}%")
        print("=" * 80 + "\n")

        return results

def _compute_2d_loc_metrics(
            self,
            pred_pdf,
            q_labels,
            coords_gt,
            cell_centers_2d,
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
        dist_errors = torch.norm(pred_centers - gt_coords_2d.to(pred_centers.device), dim=-1, p=2)

        # 汇总指标
        results = {
            'n_samples': len(ranks),
            'top1_acc': (ranks == 1).mean() * 100,
            'top2_acc': (ranks <= 2).mean() * 100,
            'top3_acc': (ranks <= 3).mean() * 100,
            'top4_acc': (ranks <= 4).mean() * 100,
            'top9_acc': (ranks <= 9).mean() * 100,
            'top16_acc': (ranks <= 16).mean() * 100,
            'mean_rank': ranks.mean(),
            'median_rank': np.median(ranks),
            'mean_dist_error': dist_errors.mean().item(),
            'median_dist_error': torch.median(dist_errors).item(),
            'dist_error_std': dist_errors.std().item(),
        }

        # 打印结果
        print(f"{title}结果:")
        print(f"  样本数: {results['n_samples']}")
        print(f"  Top-1: {results['top1_acc']:.2f}%")
        print(f"  Top-2: {results['top2_acc']:.2f}%")
        print(f"  Top-3: {results['top3_acc']:.2f}%")
        print(f"  Top-4: {results['top4_acc']:.2f}%")
        print(f"  Top-16: {results['top16_acc']:.2f}%")
        print(f"  平均排名: {results['mean_rank']:.2f}")
        print(f"  中位数排名: {results['median_rank']:.2f}")
        print(f"  距离误差 - 平均: {results['mean_dist_error']:.4f}")
        print(f"  距离误差 - 中位数: {results['median_dist_error']:.4f}")
        print(f"  距离误差 - 标准差: {results['dist_error_std']:.4f}")

        return results

def _compute_neighbors_recall(self, q_labels, id_neighbors, k_values, title="邻域Recall"):
        """
        计算邻域recall

        Args:
            q_labels: [N] GT标签
            id_neighbors: [N, K] 预测的K个邻域cell索引
            k_values: list of int，要计算的k值
            title: 标题
        """
        recall_dict = {}

        # 尝试使用现成的函数
        try:
            import sys
            sys.path.insert(0, '/home/data/zwk/pyproj_pylib_zwk')
            from uavloc_utils.eval_recall_fm_salad import compute_recall_by_label
            recall_dict_raw = compute_recall_by_label(q_labels, id_neighbors, k_values, title=title)

            # 确保键格式为 'recall@k'
            for k in k_values:
                # 尝试不同的键格式
                if f'recall@{k}' in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[f'recall@{k}']
                elif k in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[k]
                elif f'Recall@{k}' in recall_dict_raw:
                    recall_dict[f'recall@{k}'] = recall_dict_raw[f'Recall@{k}']

        except Exception as e:
            # 如果导入失败，手动计算
            print(f"⚠️  无法导入compute_recall_by_label ({str(e)})，使用简化版计算")
            print(f"{title}:")
            for k in k_values:
                # 检查GT是否在top-k中
                correct = np.any(id_neighbors[:, :k] == q_labels[:, None], axis=1)
                recall = correct.mean() * 100
                recall_dict[f'recall@{k}'] = recall
                print(f"  Recall@{k}: {recall:.2f}%")

        return recall_dict

def _get_dir2save(self, ret_epoch=False):
        if hasattr(self, 'exp_dir2save') and self.exp_dir2save and os.path.exists(self.exp_dir2save):
            # 训练模式：保存到当前实验目录
            checkpoint_dir = self.exp_dir2save[1:]
            save_dir = os.path.join(checkpoint_dir, 'loc_results')
            os.makedirs(save_dir, exist_ok=True)

            # 从实验目录找到最新的epoch号
            ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch') and f.endswith('.pth')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace('epoch', '').replace('.pth', '')))
                epoch_num = ckpts[-1].replace('epoch', '').replace('.pth', '')
            else:
                epoch_num = 'current'
        else:
            # 测试模式：使用 load2test 指定的路径
            stage3_ckpt_path = self._get_stage3_checkpoint_path()
            if stage3_ckpt_path:
                checkpoint_dir = os.path.dirname(stage3_ckpt_path)
                save_dir = os.path.join(checkpoint_dir, 'loc_results')
                os.makedirs(save_dir, exist_ok=True)

                # 从checkpoint文件名提取epoch号
                epoch_num = os.path.basename(stage3_ckpt_path).replace('epoch', '').replace('.pth', '')

        if ret_epoch:
            return save_dir, epoch_num
        else:
            return save_dir

def _save_pred_pdf_3d(
        self,
        pred_pdf_3d_all,
        q_label_3d_all,
        coords_gt_all,
        n_coarse_3d,
        cell_centers_3d,
        temperature,
        data_type,
        tag=""
    ):
        save_dir, epoch_num = self._get_dir2save(ret_epoch=True)
        tag_part = f"_{tag}" if tag else ""
        save_filename = (
            f'pred_3d{tag_part}_ep{epoch_num}_nr{n_coarse_3d[0]}_nc{n_coarse_3d[1]}_nrot{n_coarse_3d[2]}.npz'
        )
        save_path = os.path.join(save_dir, save_filename)

        if isinstance(pred_pdf_3d_all, torch.Tensor):
            pred_pdf_3d_all = pred_pdf_3d_all.detach().cpu().numpy()
        if isinstance(q_label_3d_all, torch.Tensor):
            q_label_3d_all = q_label_3d_all.detach().cpu().numpy()
        if isinstance(coords_gt_all, torch.Tensor):
            coords_gt_all = coords_gt_all.detach().cpu().numpy()
        if isinstance(cell_centers_3d, torch.Tensor):
            cell_centers_3d = cell_centers_3d.detach().cpu().numpy()

        save_data = {
            'pred_pdf_3d_all': pred_pdf_3d_all,
            'q_label_3d_all': q_label_3d_all,
            'coords_gt_all': coords_gt_all,
            'n_coarse_3d': n_coarse_3d,
            'cell_centers_3d': cell_centers_3d,
            'temperature': temperature,
            'n_samples': pred_pdf_3d_all.shape[0],
            'data_type': data_type,
        }

        np.savez(save_path, **save_data)
        print(f"\n✓ 已保存3D预测概率分布到: {save_path}")
        print(f"  - pred_pdf_3d_all: {pred_pdf_3d_all.shape}")
        print(f"  - q_label_3d_all: {q_label_3d_all.shape}")
        print(f"  - coords_gt_all: {coords_gt_all.shape}")
        print(f"  - n_coarse_3d: {n_coarse_3d}")
        print(f"  - cell_centers_3d: {cell_centers_3d.shape}")

def _test_3d_classification_accuracy(
        self,
        n_samples=256,
        use_train_uav=False,
        shuffle=False,
        temperature=10,
        energy_fn=None,
        save_pred_pdf=True,
        chunk_size=2048,
    ):
        """
        测试3D分类正确性 (NR, NC, Rot)，只边缘化Scale维度

        Args:
            n_samples: 测试样本数量（None表示使用全部数据）
            use_train_uav: 是否使用训练集UAV数据
            temperature: softmax温度参数
            shuffle: 是否打乱数据顺序（序列测试时应该为False）
            save_pred_pdf: 是否保存预测概率分布到checkpoint文件夹（默认True）

        Returns:
            dict: 包含各种准确率指标的字典
        """
        temperature = temperature if temperature is not None else self.energy_temperature
        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"{'=' * 60}\n")

        # 获取数据集
        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test

        # 确定测试样本数
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        # 创建DataLoader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        # 预采样所有子空间坐标（只需要采一次）
        coords_candidates, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=1, use_fine=False, rand_offset=False,
        )
        coords_candidates_flat = coords_candidates.view(-1, 4)
        n_coarse = self.subspace_sampler.n_coarse  # [NR, NC, Rot, Scale]
        n_coarse_3d = n_coarse[:3]  # [NR, NC, Rot]

        if energy_fn is None:
            energy_fn = self._compute_metric_from_query_and_points

        # 获取每个3D grid cell的中心坐标
        # coords_candidates shape: [1, N_total, 4], reshape成 [NR, NC, Rot, Scale, 4]
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        # 取每个3D cell的中心（对Scale取第一个值）
        cell_centers_3d = coords_reshaped[:, :, :, 0, :3]  # [NR, NC, Rot, 3] -> (nr, nc, rot)

        # 统计容器
        all_ranks = []
        all_dist_errors_2d = []  # 2D位置误差
        all_rot_errors = []  # 旋转误差
        all_rot_ranks = []  # 固定位置时的旋转排名
        processed = 0

        # 用于保存的容器
        pred_pdf_3d_list = []  # 收集3D概率分布
        q_label_3d_list = []  # 收集GT的3D标签
        coords_gt_list = []  # 收集GT坐标

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
                possibilities = energy_fn(
                    feats_vis,
                    coords_candidates_flat,
                    metric='possibility',
                    coord_space='raw',
                    temperature=temperature,
                    chunk_size=chunk_size,
                    feat_type='projector',
                )  # [B, N_total]

                # Reshape成 [B, NR, NC, Rot, Scale]
                possibilities_reshaped = possibilities.reshape(batch_size, *n_coarse)

                # 只边缘化Scale维度
                logits_3d = possibilities_reshaped.sum(dim=-1)  # [B, NR, NC, Rot]

                # 获取预测的3D索引
                logits_flat = logits_3d.view(batch_size, -1)  # [B, NR*NC*Rot]
                pred_indices = logits_flat.argmax(dim=-1)  # [B]

                # 解析3D索引
                pred_rot = pred_indices % n_coarse_3d[2]
                pred_nc = (pred_indices // n_coarse_3d[2]) % n_coarse_3d[1]
                pred_nr = pred_indices // (n_coarse_3d[1] * n_coarse_3d[2])

                # 计算GT的3D索引
                gt_indices_flat = self.subspace_sampler.coords_to_coarse_indices(coords_gt)  # [B]
                gt_indices_multi = self.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)  # [B, 4]
                gt_nr = gt_indices_multi[:, 0]
                gt_nc = gt_indices_multi[:, 1]
                gt_rot = gt_indices_multi[:, 2]
                gt_flat_idx = gt_nr * (n_coarse_3d[1] * n_coarse_3d[2]) + gt_nc * n_coarse_3d[2] + gt_rot

                # 计算排名
                sorted_indices = logits_flat.argsort(dim=-1, descending=True)
                for i in range(batch_size):
                    rank = (sorted_indices[i] == gt_flat_idx[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_ranks.append(rank)

                # 计算2D距离误差
                pred_centers = cell_centers_3d[pred_nr, pred_nc, pred_rot]  # [B, 3]
                gt_coords_2d = coords_gt[:, :2]  # [B, 2]
                dist_errors_2d = torch.norm(pred_centers[:, :2] - gt_coords_2d, dim=-1, p=2)
                all_dist_errors_2d.extend(dist_errors_2d.cpu().numpy().tolist())

                # 计算旋转误差（弧度）
                pred_rot_val = pred_centers[:, 2]  # [B]
                gt_rot_val = coords_gt[:, 2]  # [B]
                # 处理角度环绕 [-pi, pi]
                rot_diff = pred_rot_val - gt_rot_val
                rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))  # 归一化到[-pi, pi]
                rot_errors = torch.abs(rot_diff)
                all_rot_errors.extend(rot_errors.cpu().numpy().tolist())

                # 计算固定位置时的旋转排名（只在正确的nr, nc位置上比较rot）
                for i in range(batch_size):
                    # 提取GT位置(nr, nc)处所有旋转的logits
                    logits_at_gt_pos = logits_3d[i, gt_nr[i], gt_nc[i], :]  # [n_rot]
                    # 排序找到GT旋转的排名
                    sorted_rot_indices = logits_at_gt_pos.argsort(descending=True)
                    rot_rank = (sorted_rot_indices == gt_rot[i]).nonzero(as_tuple=True)[0].item() + 1
                    all_rot_ranks.append(rot_rank)

                # 收集用于保存的数据
                if save_pred_pdf:
                    # 将logits转换为概率分布
                    pred_pdf_3d = logits_flat  # [B, NR*NC*Rot]
                    pred_pdf_3d_list.append(pred_pdf_3d.cpu())
                    q_label_3d_list.append(gt_flat_idx.cpu())
                    coords_gt_list.append(coords_gt.cpu())

                processed += batch_size

            # 保存预测结果到checkpoint文件夹
            if save_pred_pdf and len(pred_pdf_3d_list) > 0:
                # 拼接所有批次的数据
                pred_pdf_3d_all = torch.cat(pred_pdf_3d_list, dim=0)  # [N, NR*NC*Rot]
                q_label_3d_all = torch.cat(q_label_3d_list, dim=0)  # [N]
                coords_gt_all = torch.cat(coords_gt_list, dim=0)  # [N, 4]
                data_type = 'train' if use_train_uav else 'test'
                self._save_pred_pdf_3d(
                    pred_pdf_3d_all=pred_pdf_3d_all,
                    q_label_3d_all=q_label_3d_all,
                    coords_gt_all=coords_gt_all,
                    n_coarse_3d=n_coarse_3d,
                    cell_centers_3d=cell_centers_3d,
                    temperature=temperature,
                    data_type=data_type,
                )

        # 计算指标
        all_ranks = np.array(all_ranks)
        all_dist_errors_2d = np.array(all_dist_errors_2d)
        all_rot_errors = np.array(all_rot_errors)
        all_rot_errors_deg = np.rad2deg(all_rot_errors)  # 转为度
        all_rot_ranks = np.array(all_rot_ranks)
        n_total_cells = n_coarse_3d[0] * n_coarse_3d[1] * n_coarse_3d[2]

        results = {
            'n_samples': len(all_ranks),
            'n_total_cells': n_total_cells,
            'n_coarse_3d': n_coarse_3d,
            'top1_acc': (all_ranks == 1).mean() * 100,
            'top8_acc': (all_ranks <= 8).mean() * 100,
            'top27_acc': (all_ranks <= 27).mean() * 100,
            'top64_acc': (all_ranks <= 64).mean() * 100,
            'top256_acc': (all_ranks <= 256).mean() * 100,
            'top512_acc': (all_ranks <= 512).mean() * 100,
            'mean_rank': all_ranks.mean(),
            'median_rank': np.median(all_ranks),
            # 2D位置误差
            'mean_dist_error_2d': all_dist_errors_2d.mean(),
            'median_dist_error_2d': np.median(all_dist_errors_2d),
            # 旋转误差（度）
            'mean_rot_error_deg': all_rot_errors_deg.mean(),
            'median_rot_error_deg': np.median(all_rot_errors_deg),
            # 固定位置时的旋转准确率
            'rot_only_top1_acc': (all_rot_ranks == 1).mean() * 100,
            'rot_only_top2_acc': (all_rot_ranks <= 2).mean() * 100,
            'rot_only_top3_acc': (all_rot_ranks <= 3).mean() * 100,
            'rot_only_mean_rank': all_rot_ranks.mean(),
            'rot_only_median_rank': np.median(all_rot_ranks),
        }

        # 打印结果
        print(f"\n{'=' * 60}")
        print(f"3D分类测试结果 (共{n_total_cells}个cell: {n_coarse_3d[0]}x{n_coarse_3d[1]}x{n_coarse_3d[2]})")
        print(f"{'=' * 60}")
        print(f"Top-1   准确率: {results['top1_acc']:.2f}%")
        print(f"Top-8   准确率: {results['top8_acc']:.2f}%")
        print(f"Top-27  准确率: {results['top27_acc']:.2f}%")
        print(f"Top-64  准确率: {results['top64_acc']:.2f}%")
        print(f"Top-256 准确率: {results['top256_acc']:.2f}%")
        print(f"Top-512 准确率: {results['top512_acc']:.2f}%")
        print(f"平均排名: {results['mean_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
        print(f"{'=' * 60}")
        print(f"2D位置误差统计:")
        print(f"  平均误差: {results['mean_dist_error_2d']:.4f}")
        print(f"  中位数误差: {results['median_dist_error_2d']:.4f}")
        print(f"{'=' * 60}")
        print(f"旋转误差统计 (度):")
        print(f"  平均误差: {results['mean_rot_error_deg']:.2f}°")
        print(f"  中位数误差: {results['median_rot_error_deg']:.2f}°")
        print(f"{'=' * 60}")
        print(f"固定位置(NR,NC)时的旋转准确率 (共{n_coarse_3d[2]}个rot):")
        print(f"  Top-1 准确率: {results['rot_only_top1_acc']:.2f}%")
        print(f"  Top-2 准确率: {results['rot_only_top2_acc']:.2f}%")
        print(f"  Top-3 准确率: {results['rot_only_top3_acc']:.2f}%")
        print(f"  平均排名: {results['rot_only_mean_rank']:.2f}")
        print(f"  中位数排名: {results['rot_only_median_rank']:.2f}")
        print(f"{'=' * 60}\n")

        return results
