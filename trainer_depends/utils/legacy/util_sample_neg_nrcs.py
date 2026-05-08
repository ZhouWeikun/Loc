import torch
import numpy as np
from typing import Tuple, Optional, Union


class BoundedNegativeCoordinateSampler:
    """
    在指定区间内采样负样本坐标
    """

    def __init__(self, device: Union[str, torch.device] = 'cuda'):
        """
        Args:
            device: 计算设备，可以是字符串 'cuda'/'cpu' 或 torch.device 对象
        """
        # 规范化设备类型
        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                device = 'cpu'
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"BoundedNegativeCoordinateSampler initialized on device: {self.device}")

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        确保tensor在正确的设备上

        Args:
            tensor: 输入tensor

        Returns:
            在self.device上的tensor
        """
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def sample_negatives(self,
                         nrcs: torch.Tensor,
                         threshold: float,
                         row_range: Tuple[float, float],
                         col_range: Tuple[float, float],
                         num_negatives: int = 10,
                         max_attempts: int = 1000,
                         batch_size: int = 100,
                         return_on_cpu: bool = False) -> torch.Tensor:
        """
        为每个query在指定区间内采样远离的负样本

        Args:
            nrcs: [N, 2] - batch的归一化坐标 [row, col]，范围[0, 1]
            threshold: 距离阈值（归一化单位）
            row_range: (min_row, max_row) - 行的采样区间，归一化值
            col_range: (min_col, max_col) - 列的采样区间，归一化值
            num_negatives: 每个query需要的负样本数量
            max_attempts: 最大采样尝试次数
            batch_size: 批量采样的大小
            return_on_cpu: 是否将结果返回到CPU（默认False，保持在采样器的设备上）

        Returns:
            neg_coords: [N, num_negatives, 2] - 采样的负样本归一化坐标 [row, col]
        """
        # 确保输入在正确的设备上
        nrcs = self._ensure_device(nrcs)
        N = nrcs.size(0)

        # 验证区间合法性
        assert 0 <= row_range[0] < row_range[1] <= 1, f"Invalid row_range: {row_range}"
        assert 0 <= col_range[0] < col_range[1] <= 1, f"Invalid col_range: {col_range}"

        # 存储结果
        neg_coords_list = []

        for i in range(N):
            query_coord = nrcs[i]  # [2]

            # 采样这个query的负样本
            neg_coords = self._sample_for_single_query(
                query_coord,
                nrcs,  # 需要避开所有batch坐标
                threshold,
                row_range,
                col_range,
                num_negatives,
                max_attempts,
                batch_size
            )

            neg_coords_list.append(neg_coords)

        # 堆叠成 [N, num_negatives, 2]
        result = torch.stack(neg_coords_list, dim=0)

        # 根据需要移动到CPU
        if return_on_cpu and result.device.type != 'cpu':
            result = result.cpu()

        return result

    def _sample_for_single_query(self,
                                 query_coord: torch.Tensor,
                                 all_coords: torch.Tensor,
                                 threshold: float,
                                 row_range: Tuple[float, float],
                                 col_range: Tuple[float, float],
                                 num_negatives: int,
                                 max_attempts: int,
                                 batch_size: int) -> torch.Tensor:
        """
        为单个query采样负样本（内部方法，假设所有tensor已在正确设备上）
        """
        collected = []
        attempts = 0

        row_min, row_max = row_range
        col_min, col_max = col_range

        while len(collected) < num_negatives and attempts < max_attempts:
            # 在指定区间内批量采样候选点
            random_rows = torch.rand(batch_size, device=self.device)
            random_cols = torch.rand(batch_size, device=self.device)

            # 缩放到指定区间
            rows = row_min + random_rows * (row_max - row_min)
            cols = col_min + random_cols * (col_max - col_min)

            candidates = torch.stack([rows, cols], dim=1)  # [batch_size, 2]

            # 计算candidates到所有batch坐标的距离
            # [batch_size, N]
            distances_to_all = torch.cdist(
                candidates,  # [batch_size, 2]
                all_coords,  # [N, 2]
                p=2
            )

            # 找到距离所有点都 > threshold 的候选
            # [batch_size]
            min_distances = distances_to_all.min(dim=1)[0]
            valid_mask = min_distances > threshold

            # 收集有效的候选
            valid_candidates = candidates[valid_mask]

            if len(valid_candidates) > 0:
                # 添加到结果中
                num_to_add = min(
                    len(valid_candidates),
                    num_negatives - len(collected)
                )
                collected.append(valid_candidates[:num_to_add])

            attempts += batch_size

        # 拼接所有收集的坐标
        if len(collected) == 0:
            # 如果实在采样不到，返回区间内的随机点（并警告）
            print(f"Warning: Could not sample {num_negatives} negatives "
                  f"with threshold {threshold} in range "
                  f"row=[{row_min:.3f}, {row_max:.3f}], "
                  f"col=[{col_min:.3f}, {col_max:.3f}]. "
                  f"Using random points in range.")

            random_rows = torch.rand(num_negatives, device=self.device)
            random_cols = torch.rand(num_negatives, device=self.device)
            return torch.stack([
                row_min + random_rows * (row_max - row_min),
                col_min + random_cols * (col_max - col_min)
            ], dim=1)

        neg_coords = torch.cat(collected, dim=0)

        # 如果还不够，补充区间内的随机点
        if len(neg_coords) < num_negatives:
            shortage = num_negatives - len(neg_coords)
            random_rows = torch.rand(shortage, device=self.device)
            random_cols = torch.rand(shortage, device=self.device)
            extra = torch.stack([
                row_min + random_rows * (row_max - row_min),
                col_min + random_cols * (col_max - col_min)
            ], dim=1)
            neg_coords = torch.cat([neg_coords, extra], dim=0)

        return neg_coords[:num_negatives]

    def sample_negatives_fast(self,
                              nrcs: torch.Tensor,
                              threshold: float,
                              row_range: Tuple[float, float],
                              col_range: Tuple[float, float],
                              num_negatives: int = 10,
                              oversample_ratio: float = 3.0,
                              return_on_cpu: bool = False) -> torch.Tensor:
        """
        快速版本：先大量采样，再过滤

        Args:
            nrcs: [N, 2] - batch的归一化坐标
            threshold: 距离阈值
            row_range: (min_row, max_row) - 行的采样区间
            col_range: (min_col, max_col) - 列的采样区间
            num_negatives: 每个query需要的负样本数
            oversample_ratio: 过采样倍率
            return_on_cpu: 是否将结果返回到CPU

        Returns:
            neg_coords: [N, num_negatives, 2]
        """
        # 确保输入在正确的设备上
        nrcs = self._ensure_device(nrcs)
        N = nrcs.size(0)

        row_min, row_max = row_range
        col_min, col_max = col_range

        # 验证区间
        assert 0 <= row_min < row_max <= 1, f"Invalid row_range: {row_range}"
        assert 0 <= col_min < col_max <= 1, f"Invalid col_range: {col_range}"

        # 一次性采样大量候选点（在指定区间内）
        total_candidates = int(N * num_negatives * oversample_ratio)

        random_rows = torch.rand(total_candidates, device=self.device)
        random_cols = torch.rand(total_candidates, device=self.device)

        # 缩放到指定区间
        rows = row_min + random_rows * (row_max - row_min)
        cols = col_min + random_cols * (col_max - col_min)

        candidates = torch.stack([rows, cols], dim=1)  # [total_candidates, 2]

        # 计算candidates到所有batch坐标的距离
        # [total_candidates, N]
        distances = torch.cdist(candidates, nrcs, p=2)

        # 对每个batch坐标，找到足够远的候选
        neg_coords_list = []

        for i in range(N):
            # 找到距离当前query > threshold的候选
            valid_mask = distances[:, i] > threshold
            valid_candidates = candidates[valid_mask]

            if len(valid_candidates) >= num_negatives:
                # 随机选择num_negatives个
                indices = torch.randperm(len(valid_candidates), device=self.device)[:num_negatives]
                neg_coords = valid_candidates[indices]
            else:
                # 不够的话，先用所有valid的，再补充区间内的随机点
                shortage = num_negatives - len(valid_candidates)

                random_rows = torch.rand(shortage, device=self.device)
                random_cols = torch.rand(shortage, device=self.device)
                extra = torch.stack([
                    row_min + random_rows * (row_max - row_min),
                    col_min + random_cols * (col_max - col_min)
                ], dim=1)

                if len(valid_candidates) > 0:
                    neg_coords = torch.cat([valid_candidates, extra], dim=0)
                else:
                    neg_coords = extra

            neg_coords_list.append(neg_coords)

        result = torch.stack(neg_coords_list, dim=0)

        # 根据需要移动到CPU
        if return_on_cpu and result.device.type != 'cpu':
            result = result.cpu()

        return result

    def sample_negatives_shared_fast(self,
                                     nrcs: torch.Tensor,
                                     threshold: float,
                                     row_range: Tuple[float, float],
                                     col_range: Tuple[float, float],
                                     total_num_negatives: int,
                                     oversample_ratio: float = 3.0,
                                     return_on_cpu: bool = False) -> torch.Tensor:
        """
        快速版本：为整个batch共享采样负样本
        先大量采样，再过滤

        Args:
            nrcs: [N, 2] - batch的归一化坐标
            threshold: 距离阈值
            row_range: (min_row, max_row) - 行的采样区间
            col_range: (min_col, max_col) - 列的采样区间
            total_num_negatives: 总共需要的负样本数量
            oversample_ratio: 过采样倍率（建议3-5倍）
            return_on_cpu: 是否将结果返回到CPU

        Returns:
            neg_coords: [total_num_negatives, 2]
        """
        # 确保输入在正确的设备上
        nrcs = self._ensure_device(nrcs)

        row_min, row_max = row_range
        col_min, col_max = col_range

        # 验证区间
        assert 0 <= row_min < row_max <= 1, f"Invalid row_range: {row_range}"
        assert 0 <= col_min < col_max <= 1, f"Invalid col_range: {col_range}"

        # 一次性采样大量候选点
        total_candidates = int(total_num_negatives * oversample_ratio)

        random_rows = torch.rand(total_candidates, device=self.device)
        random_cols = torch.rand(total_candidates, device=self.device)

        # 缩放到指定区间
        rows = row_min + random_rows * (row_max - row_min)
        cols = col_min + random_cols * (col_max - col_min)

        candidates = torch.stack([rows, cols], dim=1)  # [total_candidates, 2]

        # 计算candidates到所有batch坐标的距离
        # [total_candidates, N]
        distances = torch.cdist(candidates, nrcs, p=2)

        # 找到距离所有query都 > threshold 的候选
        min_distances = distances.min(dim=1)[0]  # [total_candidates]
        valid_mask = min_distances > threshold

        valid_candidates = candidates[valid_mask]

        if len(valid_candidates) >= total_num_negatives:
            # 随机选择total_num_negatives个
            indices = torch.randperm(
                len(valid_candidates),
                device=self.device
            )[:total_num_negatives]
            result = valid_candidates[indices]
        else:
            # 不够的话，先用所有valid的，再补充随机点
            shortage = total_num_negatives - len(valid_candidates)

            print(f"Warning: Only found {len(valid_candidates)}/{total_num_negatives} "
                  f"valid negatives. Filling {shortage} with random points.")

            random_rows = torch.rand(shortage, device=self.device)
            random_cols = torch.rand(shortage, device=self.device)
            extra = torch.stack([
                row_min + random_rows * (row_max - row_min),
                col_min + random_cols * (col_max - col_min)
            ], dim=1)

            if len(valid_candidates) > 0:
                result = torch.cat([valid_candidates, extra], dim=0)
            else:
                result = extra

        # 根据需要移动到CPU
        if return_on_cpu and result.device.type != 'cpu':
            result = result.cpu()

        return result

    def visualize_sampling(self,
                           nrcs: torch.Tensor,
                           neg_coords: torch.Tensor,
                           threshold: float,
                           row_range: Tuple[float, float],
                           col_range: Tuple[float, float],
                           save_path: Optional[str] = None):
        """
        可视化采样结果

        Args:
            nrcs: [N, 2] - batch坐标
            neg_coords: [N, num_negatives, 2] - 负样本坐标
            threshold: 距离阈值
            row_range: 行的采样区间
            col_range: 列的采样区间
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # 移动到CPU进行可视化
        nrcs_np = nrcs.cpu().numpy()
        neg_coords_np = neg_coords.cpu().numpy()

        row_min, row_max = row_range
        col_min, col_max = col_range

        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制整个地图边界
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 绘制采样区间（绿色矩形）
        sampling_rect = patches.Rectangle(
            (col_min, row_min),
            col_max - col_min,
            row_max - row_min,
            linewidth=2,
            edgecolor='green',
            facecolor='green',
            alpha=0.1,
            label='Sampling region'
        )
        ax.add_patch(sampling_rect)

        # 绘制batch坐标（红色）
        ax.scatter(nrcs_np[:, 1], nrcs_np[:, 0],
                   c='red', s=200, marker='x',
                   label='Batch coords', linewidths=3, zorder=5)

        # 绘制排除区域（半透明圆圈）
        for coord in nrcs_np:
            circle = patches.Circle(
                (coord[1], coord[0]),
                threshold,
                color='red',
                alpha=0.15,
                zorder=2
            )
            ax.add_patch(circle)

        # 绘制负样本（蓝色）
        for i, negs in enumerate(neg_coords_np):
            ax.scatter(negs[:, 1], negs[:, 0],
                       c='blue', s=50, alpha=0.6,
                       label='Negatives' if i == 0 else '', zorder=3)

        ax.set_xlabel('Col (normalized)', fontsize=12)
        ax.set_ylabel('Row (normalized)', fontsize=12)
        ax.set_title(
            f'Negative Sampling\n'
            f'threshold={threshold:.3f}, '
            f'row=[{row_min:.2f}, {row_max:.2f}], '
            f'col=[{col_min:.2f}, {col_max:.2f}]',
            fontsize=14
        )
        ax.legend(fontsize=12, loc='upper right')

        # 反转Y轴（因为图像坐标系）
        ax.invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

    def get_sampling_statistics(self,
                                nrcs: torch.Tensor,
                                threshold: float,
                                row_range: Tuple[float, float],
                                col_range: Tuple[float, float],
                                grid_size: int = 100) -> dict:
        """
        分析在给定区间内的采样统计

        Args:
            nrcs: [N, 2] - batch坐标
            threshold: 距离阈值
            row_range: 行的采样区间
            col_range: 列的采样区间
            grid_size: 网格分辨率

        Returns:
            stats: 统计信息字典
        """
        # 确保输入在正确的设备上
        nrcs = self._ensure_device(nrcs)
        N = nrcs.size(0)

        row_min, row_max = row_range
        col_min, col_max = col_range

        # 在采样区间内创建网格
        grid_row = torch.linspace(row_min, row_max, grid_size, device=self.device)
        grid_col = torch.linspace(col_min, col_max, grid_size, device=self.device)

        grid_coords = torch.stack(
            torch.meshgrid(grid_row, grid_col, indexing='ij'),
            dim=-1
        ).reshape(-1, 2)  # [grid_size^2, 2]

        # 计算到所有batch坐标的最小距离
        distances = torch.cdist(grid_coords, nrcs, p=2)  # [grid_size^2, N]
        min_distances = distances.min(dim=1)[0]  # [grid_size^2]

        valid_mask = min_distances > threshold
        valid_ratio = valid_mask.float().mean().item()

        # 计算采样区间的面积占比
        region_area = (row_max - row_min) * (col_max - col_min)

        return {
            'valid_area_ratio_in_region': valid_ratio,
            'sampling_region_area': region_area,
            'total_map_area': 1.0,
            'region_coverage': region_area,
            'batch_size': N,
            'threshold': threshold,
            'row_range': row_range,
            'col_range': col_range,
            'estimated_valid_samples': int(valid_ratio * grid_size ** 2),
            'recommendation': self._get_recommendation(valid_ratio, region_area, threshold),
            'device': str(self.device)
        }

    def _get_recommendation(self, valid_ratio: float, region_area: float, threshold: float) -> str:
        """根据统计给出建议"""
        if valid_ratio < 0.2:
            return (f"Warning: Only {valid_ratio:.1%} of sampling region is valid. "
                    f"Consider: 1) Reducing threshold, 2) Enlarging sampling region, "
                    f"3) Using smaller batch size")
        elif region_area < 0.1:
            return (f"Sampling region is small ({region_area:.1%} of map). "
                    f"Consider enlarging if possible.")
        elif valid_ratio > 0.8 and region_area > 0.5:
            return "Sampling configuration looks excellent!"
        else:
            return "Sampling configuration is acceptable."


# ============= 使用示例 =============

def example_usage():
    """使用示例（修复了设备问题）"""

    # 自动选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 初始化采样器
    sampler = BoundedNegativeCoordinateSampler(device=device)

    # 创建batch坐标（归一化）- 可以在CPU或GPU上
    batch_size = 8
    nrcs = torch.rand(batch_size, 2)  # 默认在CPU上
    print(f"Input nrcs device: {nrcs.device}")

    # 设置采样区间
    row_range = (0., 1.)
    col_range = (0, 1.)
    threshold = 0.014

    print("\n=== Batch Coordinates ===")
    print(f"Shape: {nrcs.shape}")
    print(f"Device: {nrcs.device}")
    print(f"Row range: [{nrcs[:, 0].min():.3f}, {nrcs[:, 0].max():.3f}]")
    print(f"Col range: [{nrcs[:, 1].min():.3f}, {nrcs[:, 1].max():.3f}]")

    print("\n=== Sampling Region ===")
    print(f"Row range: {row_range}")
    print(f"Col range: {col_range}")
    print(f"Threshold: {threshold}")

    # === 方法1：标准采样 ===
    print("\n=== Method 1: Standard Sampling ===")
    neg_coords_1 = sampler.sample_negatives(
        nrcs=nrcs,  # 自动移动到正确设备
        threshold=threshold,
        row_range=row_range,
        col_range=col_range,
        num_negatives=10,
        return_on_cpu=False  # 保持在采样器设备上
    )
    print(f"Sampled shape: {neg_coords_1.shape}")
    print(f"Sampled device: {neg_coords_1.device}")

    # === 方法2：快速采样，返回到CPU ===
    print("\n=== Method 2: Fast Sampling (return to CPU) ===")
    neg_coords_2 = sampler.sample_negatives_fast(
        nrcs=nrcs,
        threshold=threshold,
        row_range=row_range,
        col_range=col_range,
        num_negatives=10,
        return_on_cpu=True  # 返回到CPU
    )
    print(f"Sampled shape: {neg_coords_2.shape}")
    print(f"Sampled device: {neg_coords_2.device}")

    # === 验证约束 ===
    print("\n=== Verification ===")

    # 移动到同一设备进行验证
    nrcs_for_check = nrcs.to(neg_coords_1.device)

    # 1. 验证距离约束
    print("Distance constraints:")
    for i in range(min(3, batch_size)):  # 只打印前3个
        query_coord = nrcs_for_check[i]
        neg_coords = neg_coords_1[i]

        distances = torch.norm(neg_coords - query_coord.unsqueeze(0), dim=1)
        min_dist = distances.min().item()

        print(f"  Query {i}: min_distance = {min_dist:.4f}, "
              f"threshold = {threshold:.4f}, "
              f"satisfied = {min_dist > threshold}")

    # 2. 验证区间约束
    print("\nRegion constraints:")
    all_negs = neg_coords_1.reshape(-1, 2).cpu()  # 移到CPU验证

    row_in_range = ((all_negs[:, 0] >= row_range[0]) &
                    (all_negs[:, 0] <= row_range[1])).all()
    col_in_range = ((all_negs[:, 1] >= col_range[0]) &
                    (all_negs[:, 1] <= col_range[1])).all()

    print(f"  All rows in [{row_range[0]:.2f}, {row_range[1]:.2f}]: {row_in_range}")
    print(f"  All cols in [{col_range[0]:.2f}, {col_range[1]:.2f}]: {col_in_range}")
    print(f"  Actual row range: [{all_negs[:, 0].min():.3f}, {all_negs[:, 0].max():.3f}]")
    print(f"  Actual col range: [{all_negs[:, 1].min():.3f}, {all_negs[:, 1].max():.3f}]")

    # === 采样统计 ===
    print("\n=== Sampling Statistics ===")
    stats = sampler.get_sampling_statistics(
        nrcs=nrcs,
        threshold=threshold,
        row_range=row_range,
        col_range=col_range,
        grid_size=100
    )

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # === 可视化 ===
    print("\n=== Visualization ===")
    sampler.visualize_sampling(
        nrcs=nrcs,
        neg_coords=neg_coords_2,
        threshold=threshold,
        row_range=row_range,
        col_range=col_range,
        save_path='bounded_negative_sampling.png'
    )


if __name__ == "__main__":
    # 运行基础示例
    example_usage()