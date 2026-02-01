# trainer_depends/datasets/util_SubspaceSampler.py

import numpy as np
import torch
from typing import Tuple, Optional, Union, List


class SubspaceSampler:
    """
    4D子空间划分与采样器

    层级结构：
    - 粗划分（coarse）：定义子空间/类别数量
    - 细划分（fine）：每个粗子空间内的采样网格，保证采样均匀性

    坐标格式：[nr, nc, rot, scale]
    - nr, nc: 归一化空间坐标
    - rot: 旋转角度 (rad)，范围 [-π, π]，具有周期性
    - scale: 尺度比例
    """

    def __init__(
            self,
            sat_dataset,
            n_coarse: Tuple[int, int, int, int] = (8, 8, 12, 3),
            n_fine_per_coarse: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ):
        """
        Args:
            sat_dataset: SatDataset对象，用于获取坐标范围
            n_coarse: 粗划分数量 (n_nr, n_nc, n_rot, n_scale)
            n_fine_per_coarse: 每个粗子空间内的细划分数量
        """
        self.sat_dataset = sat_dataset
        self.n_coarse = np.array(n_coarse, dtype=np.int32)
        self.n_fine_per_coarse = np.array(n_fine_per_coarse, dtype=np.int32)

        # 总细分数量
        self.n_fine_total = self.n_coarse * self.n_fine_per_coarse

        # 子空间总数（分类类别数）
        self.n_subspaces = int(np.prod(self.n_coarse))

        # 每个粗子空间内的细分格子数
        self.n_fine_cells_per_subspace = int(np.prod(self.n_fine_per_coarse))

        # 获取各维度的范围
        self._init_coord_ranges()

        # 计算各维度的步长
        self._compute_bin_sizes()

        # GPU 缓存（延迟初始化）
        self._gpu_cache = {}

    def _get_gpu_cache(self, device: torch.device) -> dict:
        """
        获取或创建 GPU 缓存（懒加载）

        预计算内容：
        1. all_coarse_starts: [n_subspaces, 4] 所有粗子空间起始坐标
        2. fine_offsets: [n_fine_cells_per_subspace, 4] 细分格子相对偏移
        3. fine_bin_sizes: [4] 细分格子大小
        """
        device_key = str(device)
        if device_key not in self._gpu_cache:
            # 基础常量
            coord_mins = torch.from_numpy(self.coord_ranges[:, 0]).to(device=device, dtype=torch.float32)
            coarse_bin_sizes = torch.from_numpy(self.coarse_bin_sizes).to(device=device, dtype=torch.float32)
            fine_bin_sizes = torch.from_numpy(self.fine_bin_sizes_in_coarse).to(device=device, dtype=torch.float32)

            # 预计算粗子空间起始坐标
            all_coarse_starts = self._precompute_all_coarse_starts(coord_mins, coarse_bin_sizes, device)

            # 预计算细分格子偏移（相对于粗子空间起点）
            fine_offsets = self._precompute_fine_offsets(fine_bin_sizes, device)

            self._gpu_cache[device_key] = {
                'coord_ranges': torch.from_numpy(self.coord_ranges).to(device=device, dtype=torch.float32),
                'coarse_bin_sizes': coarse_bin_sizes,
                'coord_mins': coord_mins,
                'all_coarse_starts': all_coarse_starts,  # [n_subspaces, 4]
                'fine_offsets': fine_offsets,            # [n_fine_cells_per_subspace, 4]
                'fine_bin_sizes': fine_bin_sizes,        # [4]
            }
        return self._gpu_cache[device_key]

    def _precompute_all_coarse_starts(self, coord_mins: torch.Tensor,
                                       coarse_bin_sizes: torch.Tensor,
                                       device: torch.device) -> torch.Tensor:
        """
        预计算所有粗子空间的起始坐标

        Returns:
            all_starts: [n_subspaces, 4] 每个粗子空间的起始坐标
        """
        indices = [torch.arange(n, device=device) for n in self.n_coarse]
        grid = torch.meshgrid(*indices, indexing='ij')
        multi_indices = torch.stack([g.flatten() for g in grid], dim=1).float()
        all_starts = coord_mins + multi_indices * coarse_bin_sizes
        return all_starts

    def _precompute_fine_offsets(self, fine_bin_sizes: torch.Tensor,
                                  device: torch.device) -> torch.Tensor:
        """
        预计算粗子空间内所有细分格子的相对偏移

        Returns:
            fine_offsets: [n_fine_cells_per_subspace, 4] 细分格子相对于粗子空间起点的偏移
        """
        indices = [torch.arange(n, device=device) for n in self.n_fine_per_coarse]
        grid = torch.meshgrid(*indices, indexing='ij')
        multi_indices = torch.stack([g.flatten() for g in grid], dim=1).float()
        fine_offsets = multi_indices * fine_bin_sizes
        return fine_offsets

    def _init_coord_ranges(self):
        """初始化各维度的坐标范围"""
        sd = self.sat_dataset

        # 空间维度范围
        self.nr_range = np.array([sd.nr2sample_min, sd.nr2sample_max], dtype=np.float32)
        self.nc_range = np.array([sd.nc2sample_min, sd.nc2sample_max], dtype=np.float32)

        # 旋转维度范围 [-π, π]（周期性）
        self.rot_range = np.array([-np.pi, np.pi], dtype=np.float32)

        # 尺度维度范围
        self.scale_range = sd.satimgsize_scale_to_refm_boundary.astype(np.float32)

        # 合并为统一格式 [4, 2]
        self.coord_ranges = np.stack([
            self.nr_range,
            self.nc_range,
            self.rot_range,
            self.scale_range
        ], axis=0)  # [4, 2]

    def _compute_bin_sizes(self):
        """计算粗划分和细划分的步长"""
        ranges = self.coord_ranges[:, 1] - self.coord_ranges[:, 0]  # [4]

        # 粗划分步长（显式转换为float32）
        self.coarse_bin_sizes = (ranges / self.n_coarse).astype(np.float32)  # [4]

        # 细划分步长（相对于整个范围）
        self.fine_bin_sizes = (ranges / self.n_fine_total).astype(np.float32)  # [4]

        # 细划分步长（相对于单个粗格子）
        self.fine_bin_sizes_in_coarse = (self.coarse_bin_sizes / self.n_fine_per_coarse).astype(np.float32)  # [4]

    # ==================== 周期性处理 ====================

    def _wrap_rotation(self, rot: np.ndarray) -> np.ndarray:
        """将旋转角度wrap到 [-π, π]"""
        return ((rot + np.pi) % (2 * np.pi)) - np.pi

    def _wrap_rotation_torch(self, rot: torch.Tensor) -> torch.Tensor:
        """将旋转角度wrap到 [-π, π] (torch版本)"""
        return ((rot + np.pi) % (2 * np.pi)) - np.pi

    # ==================== 坐标 → 索引 ====================

    def coords_to_coarse_indices(
            self,
            coords_4d: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将4D坐标转换为粗子空间索引（单个整数）

        Args:
            coords_4d: [..., 4] 任意batch shape的4D坐标

        Returns:
            indices: [...] 对应的粗子空间索引 (0 ~ n_subspaces-1)
        """
        is_tensor = torch.is_tensor(coords_4d)
        if is_tensor:
            device = coords_4d.device
            coords = coords_4d.cpu().numpy()
        else:
            coords = np.asarray(coords_4d)

        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 4)  # [N, 4]

        # 计算各维度的bin索引
        bin_indices = self._coords_to_bin_indices(coords_flat, self.n_coarse)  # [N, 4]

        # 转换为单个整数索引（row-major order）
        indices = self._multi_index_to_flat(bin_indices, self.n_coarse)  # [N]

        indices = indices.reshape(original_shape)

        if is_tensor:
            return torch.from_numpy(indices).to(device).long()
        return indices

    def coords_to_fine_indices(
            self,
            coords_4d: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        将4D坐标转换为 (粗索引, 细索引) 对

        Returns:
            coarse_indices: [...] 粗子空间索引
            fine_indices: [...] 在粗子空间内的细分索引
        """
        is_tensor = torch.is_tensor(coords_4d)
        if is_tensor:
            device = coords_4d.device
            coords = coords_4d.cpu().numpy()
        else:
            coords = np.asarray(coords_4d)

        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 4)

        # 计算全局细分bin索引
        fine_global_indices = self._coords_to_bin_indices(coords_flat, self.n_fine_total)  # [N, 4]

        # 分解为粗索引和细索引
        coarse_bin_indices = fine_global_indices // self.n_fine_per_coarse  # [N, 4]
        fine_bin_indices = fine_global_indices % self.n_fine_per_coarse  # [N, 4]

        # 转换为单个整数索引
        coarse_indices = self._multi_index_to_flat(coarse_bin_indices, self.n_coarse)
        fine_indices = self._multi_index_to_flat(fine_bin_indices, self.n_fine_per_coarse)

        coarse_indices = coarse_indices.reshape(original_shape)
        fine_indices = fine_indices.reshape(original_shape)

        if is_tensor:
            return (torch.from_numpy(coarse_indices).to(device).long(),
                    torch.from_numpy(fine_indices).to(device).long())
        return coarse_indices, fine_indices

    def _coords_to_bin_indices(
            self,
            coords: np.ndarray,
            n_bins: np.ndarray
    ) -> np.ndarray:
        """
        将坐标转换为各维度的bin索引

        Args:
            coords: [N, 4] 坐标
            n_bins: [4] 各维度的bin数量

        Returns:
            bin_indices: [N, 4] 各维度的bin索引
        """
        # 处理旋转维度的周期性
        coords = coords.copy()
        coords[:, 2] = self._wrap_rotation(coords[:, 2])

        # 归一化到 [0, 1]
        normalized = (coords - self.coord_ranges[:, 0]) / (self.coord_ranges[:, 1] - self.coord_ranges[:, 0])

        # 转换为bin索引并clip
        bin_indices = (normalized * n_bins).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        return bin_indices

    def _multi_index_to_flat(
            self,
            multi_indices: np.ndarray,
            shape: np.ndarray
    ) -> np.ndarray:
        """
        将多维索引转换为扁平索引 (row-major order)

        Args:
            multi_indices: [N, 4] 各维度的索引
            shape: [4] 各维度的大小

        Returns:
            flat_indices: [N] 扁平索引
        """
        strides = np.array([
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1
        ], dtype=np.int64)

        return (multi_indices * strides).sum(axis=-1)

    def _flat_to_multi_index(
            self,
            flat_indices: np.ndarray,
            shape: np.ndarray
    ) -> np.ndarray:
        """
        将扁平索引转换为多维索引

        Args:
            flat_indices: [N] 扁平索引
            shape: [4] 各维度的大小

        Returns:
            multi_indices: [N, 4] 各维度的索引
        """
        flat_indices = np.asarray(flat_indices).flatten()
        multi_indices = np.zeros((len(flat_indices), 4), dtype=np.int32)
        remainder = flat_indices.copy()

        for i in range(4):
            divisor = int(np.prod(shape[i + 1:])) if i < 3 else 1
            multi_indices[:, i] = remainder // divisor
            remainder = remainder % divisor

        return multi_indices

    def flat_to_multi_index_flexible(
            self,
            flat_indices: Union[np.ndarray, torch.Tensor],
            shape: Optional[Union[np.ndarray, torch.Tensor, Tuple[int, int, int, int]]] = None,
            use_fine: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        灵活的扁平索引到多维索引转换（支持tensor和自定义shape）

        Args:
            flat_indices: [...] 扁平索引
            shape: [4] 各维度的大小，可选
                - None: 自动选择（根据use_fine决定使用n_coarse或n_fine_total）
                - tuple/list/array: 自定义形状 (nr, nc, rot, scale)
            use_fine: 当shape=None时，是否使用fine划分（默认False使用coarse）

        Returns:
            multi_indices: [..., 4] 多维索引 [nr_idx, nc_idx, rot_idx, scale_idx]

        Examples:
            >>> # 使用默认coarse划分
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat)

            >>> # 使用fine划分
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat, use_fine=True)

            >>> # 使用自定义shape
            >>> labels_multi = sampler.flat_to_multi_index_flexible(labels_flat, shape=(10, 10, 6, 2))
        """
        is_tensor = torch.is_tensor(flat_indices)

        # 确定shape
        if shape is None:
            # 自动选择：根据use_fine和是否有fine划分
            if use_fine and self.n_fine_cells_per_subspace > 1:
                shape_to_use = self.n_fine_total
            else:
                shape_to_use = self.n_coarse
        else:
            # 使用自定义shape
            if torch.is_tensor(shape):
                shape_to_use = shape.cpu().numpy()
            elif isinstance(shape, (list, tuple)):
                shape_to_use = np.array(shape, dtype=np.int32)
            else:
                shape_to_use = np.asarray(shape, dtype=np.int32)

        # 转换为numpy进行计算
        if is_tensor:
            device = flat_indices.device
            indices = flat_indices.cpu().numpy()
        else:
            indices = np.asarray(flat_indices)

        original_shape = indices.shape

        # 调用底层numpy实现
        multi_indices = self._flat_to_multi_index(indices, shape_to_use)
        multi_indices = multi_indices.reshape(*original_shape, 4)

        # 转回tensor（如果输入是tensor）
        if is_tensor:
            return torch.from_numpy(multi_indices).to(device).long()
        return multi_indices

    def coarse_indices_to_multi(
            self,
            flat_indices: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将粗子空间的扁平索引转换为多维索引

        Args:
            flat_indices: [...] 扁平的粗子空间索引

        Returns:
            multi_indices: [..., 4] 多维索引 [nr_idx, nc_idx, rot_idx, scale_idx]
        """
        is_tensor = torch.is_tensor(flat_indices)
        if is_tensor:
            device = flat_indices.device
            indices = flat_indices.cpu().numpy()
        else:
            indices = np.asarray(flat_indices)

        original_shape = indices.shape
        multi_indices = self._flat_to_multi_index(indices, self.n_coarse)
        multi_indices = multi_indices.reshape(*original_shape, 4)

        if is_tensor:
            return torch.from_numpy(multi_indices).to(device).long()
        return multi_indices

    # ==================== 索引 → 坐标 ====================

    def coarse_index_to_center(
            self,
            coarse_indices: Union[int, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        获取粗子空间的中心坐标

        Args:
            coarse_indices: 粗子空间索引，可以是标量或数组

        Returns:
            centers: [N, 4] 或 [4] 子空间中心坐标
        """
        is_scalar = np.isscalar(coarse_indices)
        if is_scalar:
            indices = np.array([coarse_indices])
        elif torch.is_tensor(coarse_indices):
            indices = coarse_indices.cpu().numpy().flatten()
        else:
            indices = np.asarray(coarse_indices).flatten()

        # 转换为多维索引
        multi_indices = self._flat_to_multi_index(indices, self.n_coarse)  # [N, 4]

        # 计算中心坐标
        centers = (self.coord_ranges[:, 0] +
                   (multi_indices + 0.5) * self.coarse_bin_sizes)  # [N, 4]

        if is_scalar:
            return centers[0]
        return centers.astype(np.float32)

    # ==================== 相邻子空间查询 ====================

    def get_adjacent_subspaces(
            self,
            coarse_idx: int,
            include_diagonal: bool = False
    ) -> np.ndarray:
        """
        获取相邻的子空间索引（考虑旋转周期性）

        Args:
            coarse_idx: 粗子空间索引
            include_diagonal: 是否包含对角相邻

        Returns:
            adjacent_indices: 相邻子空间索引数组
        """
        # 转换为多维索引
        multi_idx = self._flat_to_multi_index(np.array([coarse_idx]), self.n_coarse)[0]

        adjacent = []

        # 定义偏移量
        if include_diagonal:
            offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * 4)).T.reshape(-1, 4)
            offsets = offsets[~np.all(offsets == 0, axis=1)]  # 移除零偏移
        else:
            offsets = np.array([
                [-1, 0, 0, 0], [1, 0, 0, 0],
                [0, -1, 0, 0], [0, 1, 0, 0],
                [0, 0, -1, 0], [0, 0, 1, 0],
                [0, 0, 0, -1], [0, 0, 0, 1]
            ])

        for offset in offsets:
            neighbor = multi_idx + offset

            # 处理边界（旋转维度周期性，其他维度clip）
            neighbor[0] = np.clip(neighbor[0], 0, self.n_coarse[0] - 1)
            neighbor[1] = np.clip(neighbor[1], 0, self.n_coarse[1] - 1)
            neighbor[2] = neighbor[2] % self.n_coarse[2]  # 周期性
            neighbor[3] = np.clip(neighbor[3], 0, self.n_coarse[3] - 1)

            # 转换为扁平索引
            flat_idx = self._multi_index_to_flat(neighbor.reshape(1, 4), self.n_coarse)[0]

            if flat_idx != coarse_idx:
                adjacent.append(flat_idx)

        return np.unique(adjacent)

    # ==================== 采样方法 ====================

    def sample_all_subspaces_gpu(
            self,
            n_points_per_subspace: int = 4,
            use_fine: bool = None,
            device: torch.device = None,
            rand_offset=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一次性采样所有子空间（遍历，无重复）

        自动适配 coarse=fine 和 coarse≠fine 两种情况：
        - coarse=fine: 直接在子空间内随机采样
        - coarse≠fine: 使用 fine 细分进行更均匀的采样

        适用场景：
        - 需要覆盖所有子空间的完整采样
        - 构建全局特征库/索引

        Args:
            n_points_per_subspace: 每个子空间的采样点数 (P)
            device: GPU 设备
            use_fine: 是否使用 fine 细分采样。None=自动判断（fine>1时使用）
            rand_offset: 是否在子空间内随机偏移。False 时使用中心点采样。

        Returns:
            coords: [n_subspaces, P, 4] 采样坐标
            labels: [n_subspaces, P] 子空间标签
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]

        P = n_points_per_subspace
        N = self.n_subspaces

        # 自动判断是否使用 fine 细分
        if use_fine is None:
            use_fine = self.n_fine_cells_per_subspace > 1

        if use_fine and self.n_fine_cells_per_subspace > 1:
            # coarse ≠ fine: 使用细分采样
            fine_offsets = cache['fine_offsets']      # [n_fine_cells, 4]
            fine_bin_sizes = cache['fine_bin_sizes']  # [4]
            n_fine_cells = fine_offsets.shape[0]

            # 为每个子空间的每个点随机选择一个细分格子
            selected_fine = torch.randint(0, n_fine_cells, (N, P), device=device)
            fine_off = fine_offsets[selected_fine]  # [N, P, 4]

            if rand_offset:
                # 细分格子内随机偏移
                rand_off = torch.rand(N, P, 4, device=device) * fine_bin_sizes
            else:
                # 细分格子中心
                rand_off = fine_bin_sizes.view(1, 1, 4) * 0.5

            # coords = coarse_start + fine_offset + rand_offset
            coords = all_coarse_starts.unsqueeze(1) + fine_off + rand_off  # [N, P, 4]
        else:
            # coarse = fine: 直接在子空间内随机采样
            coarse_bin_sizes = cache['coarse_bin_sizes']  # [4]
            if rand_offset:
                rand_offsets = torch.rand(N, P, 4, device=device) * coarse_bin_sizes
            else:
                rand_offsets = coarse_bin_sizes.view(1, 1, 4) * 0.5
            coords = all_coarse_starts.unsqueeze(1) + rand_offsets  # [N, P, 4]

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # 标签: 每个子空间的索引
        labels = torch.arange(N, device=device).unsqueeze(1).expand(-1, P)

        return coords, labels


    def sample_batch_gpu_coarse(
            self,
            anchor_labels: torch.Tensor,
            n_subspaces_to_sample: int = 128,
            n_points_per_subspace: int = 4,
            include_anchor_subspace: bool = True,
            rand_offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        极速 GPU 版本：预计算查表 + 随机偏移

        核心优化：
        1. 预计算所有子空间起始坐标 (只在首次调用时计算一次)
        2. 采样时直接索引查表，无需 flat_idx → multi_idx 转换
        3. 只需一次索引 + 一次随机偏移

        Args:
            anchor_labels: [B] anchor 的子空间标签 (GPU tensor)
            n_subspaces_to_sample: 采样的子空间数量
            n_points_per_subspace: 每个子空间的采样点数
            include_anchor_subspace: 是否强制包含 anchor 所属子空间
            rand_offset: 是否在子空间内随机偏移。False 时使用中心点采样。

        Returns:
            coords: [B, N_total, 4] 采样坐标
            candidate_labels: [B, N_total] 每个候选点的子空间标签
        """
        device = anchor_labels.device
        B = anchor_labels.shape[0]

        # 获取 GPU 缓存 (包含预计算的所有子空间起始坐标)
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4] 预计算的查找表
        coarse_bin_sizes = cache['coarse_bin_sizes']    # [4]

        n_total = n_subspaces_to_sample * n_points_per_subspace

        # ========== 1. 随机选择子空间 ==========
        selected_subspaces = torch.randint(
            0, self.n_subspaces, (B, n_subspaces_to_sample), device=device
        )

        if include_anchor_subspace:
            selected_subspaces[:, 0] = anchor_labels

        # ========== 2. 直接索引查表获取起始坐标 (核心优化) ==========
        # all_coarse_starts: [n_subspaces, 4]
        # selected_subspaces: [B, K]
        # 结果: [B, K, 4]
        coarse_starts = all_coarse_starts[selected_subspaces]

        # ========== 3. 子空间内随机采样 ==========
        # rand_offsets: [B, K, P, 4] 在 [0, bin_size) 内均匀采样
        if rand_offset:
            rand_offsets = torch.rand(
                B, n_subspaces_to_sample, n_points_per_subspace, 4, device=device
            ) * coarse_bin_sizes
        else:
            rand_offsets = coarse_bin_sizes.view(1, 1, 1, 4) * 0.5

        # 坐标 = 起点 + 偏移
        coords = coarse_starts.unsqueeze(2) + rand_offsets  # [B, K, P, 4]
        coords = coords.view(B, n_total, 4)

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # ========== 4. 生成标签 ==========
        candidate_labels = selected_subspaces.unsqueeze(2).expand(
            -1, -1, n_points_per_subspace
        ).reshape(B, n_total)

        return coords, candidate_labels


    def sample_batch_gpu_fine(
            self,
            anchor_labels: torch.Tensor,
            n_subspaces_to_sample: int = 128,
            n_points_per_subspace: int = 4,
            include_anchor_subspace: bool = True,
            rand_offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        极速 GPU 版本（通用）：支持 coarse ≠ fine 的细分采样

        预计算策略：
        1. all_coarse_starts[n_subspaces, 4]: 粗子空间起始坐标
        2. fine_offsets[n_fine_cells, 4]: 细分格子相对偏移
        3. fine_bin_sizes[4]: 细分格子大小

        采样流程：
        coords = coarse_starts[selected_subspaces]      # 查表
               + fine_offsets[selected_fine_cells]      # 查表
               + rand() * fine_bin_sizes                # 随机偏移

        Args:
            anchor_labels: [B] anchor 的子空间标签 (GPU tensor)
            n_subspaces_to_sample: 采样的子空间数量 (K)
            n_points_per_subspace: 每个子空间的采样点数 (P)
            include_anchor_subspace: 是否强制包含 anchor 所属子空间
            rand_offset: 是否在细分格子内随机偏移。False 时使用中心点采样。

        Returns:
            coords: [B, K*P, 4] 采样坐标
            candidate_labels: [B, K*P] 每个候选点的子空间标签
        """
        device = anchor_labels.device
        B = anchor_labels.shape[0]
        K = n_subspaces_to_sample
        P = n_points_per_subspace

        # 获取 GPU 缓存
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]
        fine_offsets = cache['fine_offsets']            # [n_fine_cells, 4]
        fine_bin_sizes = cache['fine_bin_sizes']        # [4]

        n_fine_cells = fine_offsets.shape[0]
        n_total = K * P

        # ========== 1. 随机选择粗子空间 ==========
        selected_subspaces = torch.randint(0, self.n_subspaces, (B, K), device=device)

        if include_anchor_subspace:
            selected_subspaces[:, 0] = anchor_labels

        # 查表获取粗子空间起始坐标: [B, K, 4]
        coarse_starts = all_coarse_starts[selected_subspaces]

        # ========== 2. 为每个点随机选择细分格子 ==========
        # selected_fine: [B, K, P] 每个点选择哪个细分格子
        selected_fine = torch.randint(0, n_fine_cells, (B, K, P), device=device)

        # 查表获取细分偏移: [B, K, P, 4]
        fine_off = fine_offsets[selected_fine]

        # ========== 3. 细分格子内随机偏移 ==========
        if rand_offset:
            rand_off = torch.rand(B, K, P, 4, device=device) * fine_bin_sizes
        else:
            rand_off = fine_bin_sizes.view(1, 1, 1, 4) * 0.5

        # ========== 4. 组合最终坐标 ==========
        # coords = coarse_start + fine_offset + rand_offset
        coords = coarse_starts.unsqueeze(2) + fine_off + rand_off  # [B, K, P, 4]
        coords = coords.view(B, n_total, 4)

        # Wrap rotation to [-π, π]
        coords[:, :, 2] = ((coords[:, :, 2] + torch.pi) % (2 * torch.pi)) - torch.pi

        # ========== 5. 生成标签 ==========
        candidate_labels = selected_subspaces.unsqueeze(2).expand(-1, -1, P).reshape(B, n_total)

        return coords, candidate_labels

    def sample_grid_in_subspaces_gpu(
            self,
            subspace_indices: torch.Tensor,
            grid_dims: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
         在指定的子空间内生成规则的网格采样点(Top-K Refinement)

        Args:
            subspace_indices: [B, K] 需要采样的子空间索引 (Flattened)
            grid_dims: (nr, nc, rot, scale) 每个子空间内部的网格密度
                       例如 (4, 4, 3, 1) 表示在每个粗格子内生成 4x4x3x1 个点

        Returns:
            coords: [B, K, Total_Points, 4] 采样点的物理坐标
                    其中 Total_Points = nr * nc * rot * scale
        """
        device = subspace_indices.device
        B, K = subspace_indices.shape

        # 1. 获取基础信息
        cache = self._get_gpu_cache(device)
        all_coarse_starts = cache['all_coarse_starts']  # [n_subspaces, 4]
        coarse_bin_sizes = cache['coarse_bin_sizes']  # [4]

        # 2. 获取选中子空间的起始坐标
        # [B, K, 4]
        start_coords = all_coarse_starts[subspace_indices]

        # 3. 生成局部归一化网格 [0, 1]
        # grid_dims: [D_r, D_c, D_rot, D_s]
        # linspace 生成每个维度的中心点，例如 dim=4 -> [0.125, 0.375, 0.625, 0.875]
        grids = []
        for dim_size in grid_dims:
            # 生成 0.5/N, 1.5/N, ..., (N-0.5)/N
            # 这样保证点位于每个细分格子的中心
            if dim_size > 0:
                grid = (torch.arange(dim_size, device=device, dtype=torch.float32) + 0.5) / dim_size
            else:
                grid = torch.tensor([0.5], device=device)  # 防御性编程
            grids.append(grid)

        # 生成 Meshgrid
        # mesh: [D_r, D_c, D_rot, D_s]
        mesh = torch.meshgrid(*grids, indexing='ij')

        # Flatten -> [Total_Points, 4]
        # Total_Points = D_r * D_c * D_rot * D_s
        local_grid_normalized = torch.stack([m.flatten() for m in mesh], dim=1)

        # 4. 映射到物理坐标
        # Coords = Start + Normalized_Grid * Bin_Size
        # Start: [B, K, 1, 4]
        # Grid:  [1, 1, Total_Points, 4]
        # Bin:   [1, 1, 1, 4]

        # [B, K, Total_Points, 4]
        coords = start_coords.unsqueeze(2) + local_grid_normalized.view(1, 1, -1, 4) * coarse_bin_sizes.view(1, 1, 1, 4)

        # 5. 处理旋转周期性
        coords[..., 2] = self._wrap_rotation_torch(coords[..., 2])

        return coords

    def sample_grid_at_2d_indices_gpu(
            self,
            indices_2d: Union[List, np.ndarray, torch.Tensor],
            grid_dims: Tuple[int, int, int, int],
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在指定的2D位置索引处，沿所有旋转和尺度维度进行均匀采样

        功能说明：
        - 给定2D空间位置索引 [(nr_idx, nc_idx), ...], 在这些位置处
        - 遍历所有可能的旋转和尺度粗格子
        - 在每个4D粗格子内进行均匀细分采样

        Args:
            indices_2d: 2D位置索引坐标
                - List of tuples: [(nr_idx0, nc_idx0), (nr_idx1, nc_idx1), ...]
                - np.ndarray: [N, 2] 格式
                - torch.Tensor: [N, 2] 格式
            grid_dims: (nr, nc, rot, scale) 每个子空间内部的网格密度
                       例如 (4, 4, 3, 1) 表示在每个粗格子内生成 4x4x3x1 个点
            device: 目标设备，如果为None则自动选择

        Returns:
            coords: [N, K_rot * K_scale, Total_Points, 4] 采样点的物理坐标
                    其中 K_rot = n_coarse[2], K_scale = n_coarse[3]
                    Total_Points = nr * nc * rot * scale
            labels: [N, K_rot * K_scale] 对应的子空间标签（扁平索引）

        示例：
            >>> # 在位置 (0, 0) 和 (1, 1) 处采样所有旋转和尺度
            >>> indices_2d = [(0, 0), (1, 1)]
            >>> coords, labels = sampler.sample_grid_at_2d_indices_gpu(
            ...     indices_2d, grid_dims=(4, 4, 3, 1)
            ... )
            >>> # coords.shape: [2, n_rot*n_scale, 48, 4]
            >>> # labels.shape: [2, n_rot*n_scale]
        """
        # 1. 处理输入格式，转换为torch.Tensor
        if isinstance(indices_2d, list):
            indices_2d = np.array(indices_2d, dtype=np.int32)
        if isinstance(indices_2d, np.ndarray):
            indices_2d = torch.from_numpy(indices_2d).long()

        if device is None:
            device = indices_2d.device if isinstance(indices_2d, torch.Tensor) else torch.device('cpu')
        indices_2d = indices_2d.to(device)

        # 确保是 [N, 2] 格式
        if indices_2d.ndim == 1:
            indices_2d = indices_2d.unsqueeze(0)
        assert indices_2d.shape[-1] == 2, f"Expected shape [N, 2], got {indices_2d.shape}"

        N = indices_2d.shape[0]
        K_rot = int(self.n_coarse[2])   # 旋转维度粗划分数量
        K_scale = int(self.n_coarse[3])  # 尺度维度粗划分数量
        K_total = K_rot * K_scale

        # 2. 生成所有旋转和尺度的索引组合
        # rot_indices: [K_rot], scale_indices: [K_scale]
        rot_indices = torch.arange(K_rot, device=device)
        scale_indices = torch.arange(K_scale, device=device)

        # 生成笛卡尔积: [K_rot, K_scale] -> [K_total, 2]
        rot_mesh, scale_mesh = torch.meshgrid(rot_indices, scale_indices, indexing='ij')
        rot_scale_pairs = torch.stack([rot_mesh.flatten(), scale_mesh.flatten()], dim=1)  # [K_total, 2]

        # 3. 为每个2D位置构建完整的4D子空间索引
        # indices_2d: [N, 2] -> [N, 1, 2]
        # rot_scale_pairs: [K_total, 2] -> [1, K_total, 2]
        # 拼接: [N, K_total, 4]
        spatial_2d_expanded = indices_2d.unsqueeze(1).expand(N, K_total, 2)  # [N, K_total, 2]
        rot_scale_expanded = rot_scale_pairs.unsqueeze(0).expand(N, K_total, 2)  # [N, K_total, 2]

        multi_indices_4d = torch.cat([spatial_2d_expanded, rot_scale_expanded], dim=-1)  # [N, K_total, 4]

        # 4. 转换为扁平子空间索引
        # multi_indices_4d: [N, K_total, 4] -> [N*K_total, 4]
        multi_indices_flat = multi_indices_4d.reshape(-1, 4).cpu().numpy()
        subspace_indices_flat = self._multi_index_to_flat(multi_indices_flat, self.n_coarse)  # [N*K_total]
        subspace_indices = torch.from_numpy(subspace_indices_flat).to(device).long().reshape(N, K_total)

        # 5. 调用原有的采样函数
        coords = self.sample_grid_in_subspaces_gpu(subspace_indices, grid_dims)  # [N, K_total, Total_Points, 4]

        return coords, subspace_indices

    def sample_grid_around_coords_gpu(
            self,
            center_coords: Union[np.ndarray, torch.Tensor],
            grid_dims: Tuple[int, int, int, int],
            sample_space_size: Optional[Union[np.ndarray, torch.Tensor, Tuple]] = None,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        以给定坐标为中心进行均匀网格采样

        功能说明：
        - 给定中心坐标和每个维度的采样数量
        - 以中心为基准，在指定的采样空间内生成均匀分布的网格点
        - 默认采样空间大小为 coarse_bin_sizes（可自定义）
        - 套用 sample_grid_in_subspaces_gpu 的网格生成逻辑

        Args:
            center_coords: 中心坐标
                - np.ndarray: [N, 4] 或 [4] 格式
                - torch.Tensor: [N, 4] 或 [4] 格式
            grid_dims: (nr, nc, rot, scale) 每个维度的采样点数
                       例如 (4, 4, 3, 2) 表示生成 4x4x3x2 = 96 个点
            sample_space_size: 采样空间大小 [4]，表示在每个维度上的采样范围
                - None: 使用 coarse_bin_sizes（默认）
                - tuple/list: (size_nr, size_nc, size_rot, size_scale)
                - np.ndarray/torch.Tensor: [4] 格式
            device: 目标设备，如果为None则自动选择

        Returns:
            coords: [N, Total_Points, 4] 采样点的物理坐标
                    其中 Total_Points = nr * nc * rot * scale
                    如果输入是单个坐标 [4]，返回 [Total_Points, 4]

        示例：
            >>> # 在单个坐标周围采样
            >>> center = np.array([0.5, 0.5, 0.0, 1.0])
            >>> coords = sampler.sample_grid_around_coords_gpu(
            ...     center, grid_dims=(4, 4, 3, 2)
            ... )
            >>> # coords.shape: [96, 4]

            >>> # 在多个坐标周围批量采样
            >>> centers = np.array([[0.5, 0.5, 0.0, 1.0],
            ...                     [0.3, 0.7, 1.5, 1.2]])
            >>> coords = sampler.sample_grid_around_coords_gpu(
            ...     centers, grid_dims=(4, 4, 3, 2)
            ... )
            >>> # coords.shape: [2, 96, 4]

            >>> # 使用自定义采样空间大小
            >>> coords = sampler.sample_grid_around_coords_gpu(
            ...     center, grid_dims=(4, 4, 3, 2),
            ...     sample_space_size=(0.1, 0.1, 0.5, 0.2)
            ... )
        """
        # 1. 处理输入坐标，转换为 torch.Tensor
        if isinstance(center_coords, np.ndarray):
            center_coords = torch.from_numpy(center_coords).float()
        elif not torch.is_tensor(center_coords):
            center_coords = torch.tensor(center_coords, dtype=torch.float32)

        # 确定设备
        if device is None:
            device = center_coords.device if center_coords.is_cuda else torch.device('cpu')
        center_coords = center_coords.to(device)

        # 处理单个坐标的情况: [4] -> [1, 4]
        is_single_coord = False
        if center_coords.ndim == 1:
            center_coords = center_coords.unsqueeze(0)
            is_single_coord = True

        assert center_coords.shape[-1] == 4, f"Expected last dim=4, got {center_coords.shape}"
        N = center_coords.shape[0]

        # 2. 确定采样空间大小
        if sample_space_size is None:
            # 使用 coarse_bin_sizes 作为默认采样空间
            cache = self._get_gpu_cache(device)
            space_size = cache['coarse_bin_sizes']  # 应该是 [4]
        else:
            # 使用自定义采样空间大小
            if isinstance(sample_space_size, (tuple, list)):
                space_size = torch.tensor(sample_space_size, dtype=torch.float32, device=device)
            elif isinstance(sample_space_size, np.ndarray):
                space_size = torch.from_numpy(sample_space_size).float().to(device)
            else:
                space_size = sample_space_size.to(device)

        # 确保 space_size 是 1D 张量，形状为 [4]
        if space_size.ndim > 1:
            # 如果是多维的，只取第一行或flatten（取决于实际需求）
            space_size = space_size.flatten()[:4]
        assert space_size.shape[0] == 4, f"Expected space_size shape [4], got {space_size.shape}"

        # 3. 生成局部归一化网格 [0, 1]
        # 类似 sample_grid_in_subspaces_gpu 的逻辑
        grids = []
        for dim_size in grid_dims:
            if dim_size > 0:
                # 生成 0.5/N, 1.5/N, ..., (N-0.5)/N
                # 保证点位于每个细分格子的中心
                grid = (torch.arange(dim_size, device=device, dtype=torch.float32) + 0.5) / dim_size
            else:
                grid = torch.tensor([0.5], device=device)
            grids.append(grid)

        # 生成 Meshgrid
        mesh = torch.meshgrid(*grids, indexing='ij')

        # Flatten -> [Total_Points, 4]
        local_grid_normalized = torch.stack([m.flatten() for m in mesh], dim=1)
        # Total_Points = local_grid_normalized.shape[0]

        # 4. 将归一化网格从 [0, 1] 映射到 [-0.5, 0.5]（以中心为原点）
        # 然后乘以采样空间大小，得到相对于中心的偏移
        # offset = (normalized - 0.5) * space_size
        local_offsets = (local_grid_normalized - 0.5) * space_size.view(1, 4)  # [Total_Points, 4]

        # 5. 计算最终坐标: center + offset
        # center_coords: [N, 4] -> [N, 1, 4]
        # local_offsets: [Total_Points, 4] -> [1, Total_Points, 4]
        coords = center_coords.unsqueeze(1) + local_offsets.unsqueeze(0)  # [N, Total_Points, 4]

        # 6. 处理旋转维度的周期性
        coords[..., 2] = self._wrap_rotation_torch(coords[..., 2])

        # 7. 如果输入是单个坐标，返回 [Total_Points, 4]
        if is_single_coord:
            coords = coords.squeeze(0)

        return coords


    # ==================== 辅助方法 ====================
    def get_subspace_info(self) -> dict:
        """获取子空间划分信息"""
        return {
            'n_coarse': self.n_coarse.tolist(),
            'n_fine_per_coarse': self.n_fine_per_coarse.tolist(),
            'n_subspaces': self.n_subspaces,
            'n_fine_cells_per_subspace': self.n_fine_cells_per_subspace,
            'coord_ranges': self.coord_ranges.tolist(),
            'coarse_bin_sizes': self.coarse_bin_sizes.tolist(),
            'fine_bin_sizes': self.fine_bin_sizes.tolist(),
        }

    def __repr__(self) -> str:
        return (f"SubspaceSampler(\n"
                f"  n_coarse={self.n_coarse.tolist()},\n"
                f"  n_fine_per_coarse={self.n_fine_per_coarse.tolist()},\n"
                f"  n_subspaces={self.n_subspaces},\n"
                f"  n_fine_cells_per_subspace={self.n_fine_cells_per_subspace}\n"
                f")")
