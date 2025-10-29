"""
UAV-Satellite Pair Dataset
组合UAV和Satellite数据集，在__getitem__中返回三元组样本
利用DataLoader的多进程机制自动并行采样
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiSceneDataLoader:
    """
    多场景数据加载器
    根据采样策略从多个场景的 dataloader 中采样数据

    Epoch 结束策略：
    - 每个 epoch 的迭代次数等于所有场景中最长的 dataloader 的长度
    - 较短的场景会在epoch内循环多次
    """
    def __init__(self, scene_dataloaders_dict, sampling_strategy='round_robin'):
        """
        Args:
            scene_dataloaders_dict: dict, {scene_name: dataloader}
            sampling_strategy: str, 'round_robin'/'random'/'weighted'
        """
        self.dataloaders = scene_dataloaders_dict
        self.scene_names = list(scene_dataloaders_dict.keys())
        self.sampling_strategy = sampling_strategy
        self.num_scenes = len(self.scene_names)

        # 计算总迭代次数（取所有场景的最大长度）
        self.total_iters = max(len(dl) for dl in self.dataloaders.values())

        # 初始化场景迭代器
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}

        # round_robin 模式下的场景索引
        self.current_scene_idx = 0

        # 当前 epoch 的迭代计数器
        self.current_iter = 0

    def __len__(self):
        """返回每个 epoch 的迭代次数（最长场景的长度）"""
        return self.total_iters

    def __iter__(self):
        """开始新的 epoch，重置计数器"""
        self.current_iter = 0
        self.current_scene_idx = 0
        return self

    def __next__(self):
        """根据采样策略返回下一个 batch"""
        # 检查是否完成一个 epoch
        if self.current_iter >= self.total_iters:
            raise StopIteration

        # 选择场景
        if self.sampling_strategy == 'round_robin':
            scene_name = self.scene_names[self.current_scene_idx]
            self.current_scene_idx = (self.current_scene_idx + 1) % self.num_scenes
        elif self.sampling_strategy == 'random':
            scene_name = np.random.choice(self.scene_names)
        elif self.sampling_strategy == 'weighted':
            # TODO: 实现加权采样
            weights = [self.dataloaders[name].dataset.weight for name in self.scene_names]
            scene_name = np.random.choice(self.scene_names, p=np.array(weights)/sum(weights))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # 获取该场景的 batch
        try:
            batch = next(self.scene_iters[scene_name])
        except StopIteration:
            # 该场景遍历完，重新开始（在epoch内循环）
            self.scene_iters[scene_name] = iter(self.dataloaders[scene_name])
            batch = next(self.scene_iters[scene_name])

        # 在 batch 中添加场景名称标记
        batch['scene_name'] = scene_name

        # 更新迭代计数器
        self.current_iter += 1

        return batch

    def reset(self):
        """重置所有迭代器（通常不需要手动调用，__iter__ 会自动重置）"""
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        self.current_scene_idx = 0
        self.current_iter = 0


class MultiSceneDataLoader:
    def __init__(self, scene_dataloaders, strategy='round_robin'):
        self.dataloaders = scene_dataloaders  # {scene_name: dataloader}
        self.strategy = strategy
        self.scene_iters = {k: iter(v) for k, v in self.dataloaders.items()}

    def __iter__(self):
        while True:
            for scene_name in self._get_scene_order():
                try:
                    batch = next(self.scene_iters[scene_name])
                    batch['scene_name'] = scene_name  # 标记场景
                    yield batch
                except StopIteration:
                    # 该场景遍历完，重新开始
                    self.scene_iters[scene_name] = iter(self.dataloaders[scene_name])


class UAVSatPairDataset(Dataset):
    """
    UAV-Satellite配对数据集

    在__getitem__中返回：
    - uav_img: UAV图像
    - sat_img_pos: 正样本卫星图（与UAV对应位置）
    - sat_imgs_neg: 负样本卫星图（随机采样，如果n_neg=0则为None）
    - coords_uav: UAV的4D坐标 [nr, nc, rot, scale]

    这样可以利用DataLoader的num_workers并行采样，无需额外的异步逻辑
    """

    def __init__(
        self,
        uav_dataset,
        sat_dataset,
        satmap_sampler=None,
        device='cuda',
        n_neg_per_sample=1,
    ):
        """
        Args:
            uav_dataset: UAVDataset实例
            sat_dataset: SatDataset实例
            satmap_sampler: BoundedNegativeCoordinateSampler实例（n_neg=0时可为None）
            device: 设备（用于负样本采样）
            n_neg_per_sample: 每个样本的负样本数量（0表示不采样负样本）
        """
        self.uav_dataset = uav_dataset
        self.sat_dataset = sat_dataset
        self.satmap_sampler = satmap_sampler
        self.device = device
        self.n_neg_per_sample = n_neg_per_sample

        if n_neg_per_sample > 0 and satmap_sampler is None:
            raise ValueError("satmap_sampler cannot be None when n_neg_per_sample > 0")

    def __len__(self):
        return len(self.uav_dataset)

    def __getitem__(self, index):
        """
        获取一个样本

        Returns:
            dict: {
                'uav_img': [C, H, W] UAV图像
                'sat_img_pos': [C, H, W] 正样本卫星图
                'sat_imgs_neg': [n_neg, C, H, W] 或 [C, H, W] 或 None（如果n_neg=0）
                'coords_uav': [4] UAV的4D坐标
            }
        """
        # 1. 从UAV数据集获取UAV图像和增强后的4D坐标
        # UAVDataset现在直接返回 (uav_img, coords_4d)
        uav_img, coords_uav = self.uav_dataset[index]

        # 2. 采样正样本卫星图（与UAV位置对应）
        sat_img_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_uav)

        # 3. 采样负样本（如果需要）
        sat_imgs_neg = None
        if self.n_neg_per_sample > 0:
            # 提取nrcs（前两维）
            nrcs_uav = coords_uav[:2]
            nrcs_uav_np = nrcs_uav.numpy() if isinstance(nrcs_uav, torch.Tensor) else nrcs_uav

            # 采样负样本坐标
            nrcs_neg = self._sample_negatives_cpu(
                nrcs_uav_np,
                threshold=self.sat_dataset.halfimg_radius_nrc,
                row_range=self.sat_dataset.nr2sample_range,
                col_range=self.sat_dataset.nc2sample_range,
                total_num_negatives=self.n_neg_per_sample
            )

            # 随机采样旋转和尺度
            rots_neg = -np.pi + 2 * np.pi * np.random.rand(self.n_neg_per_sample)
            scales_neg = np.random.rand(self.n_neg_per_sample) * \
                         (self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
                          self.sat_dataset.satimgsize_scale_to_200m_boundary[0]) + \
                         self.sat_dataset.satimgsize_scale_to_200m_boundary[0]

            coords_neg = np.concatenate([
                nrcs_neg,
                rots_neg[:, np.newaxis],
                scales_neg[:, np.newaxis]
            ], axis=-1)

            # 采样负样本卫星图
            sat_imgs_neg_list = []
            for i in range(self.n_neg_per_sample):
                sat_img_neg = self.sat_dataset.crop_satimg_by_4d_coords(coords_neg[i])
                sat_imgs_neg_list.append(sat_img_neg)

            if self.n_neg_per_sample == 1:
                sat_imgs_neg = sat_imgs_neg_list[0]
            else:
                sat_imgs_neg = torch.stack(sat_imgs_neg_list, dim=0)

        return {
            'uav_img': uav_img,
            'sat_img_pos': sat_img_pos,
            'sat_imgs_neg': sat_imgs_neg,
            'coords_uav': coords_uav,
        }

    def _sample_negatives_cpu(self, nrcs, threshold, row_range, col_range, total_num_negatives):
        """
        CPU版本的负样本采样（用于DataLoader的worker进程）

        Args:
            nrcs: [2] 或 [1, 2] 的numpy数组
            threshold: 排除半径
            row_range: [min, max] 行采样范围
            col_range: [min, max] 列采样范围
            total_num_negatives: 采样数量

        Returns:
            neg_samples: [total_num_negatives, 2] 负样本坐标
        """
        if nrcs.ndim == 1:
            nrcs = nrcs[np.newaxis, :]

        nrc = nrcs[0]

        # 定义排除区域
        nr_min = max(row_range[0], nrc[0] - threshold)
        nr_max = min(row_range[1], nrc[0] + threshold)
        nc_min = max(col_range[0], nrc[1] - threshold)
        nc_max = min(col_range[1], nrc[1] + threshold)

        # 采样负样本
        neg_samples = []
        attempts = 0
        max_attempts = total_num_negatives * 100

        while len(neg_samples) < total_num_negatives and attempts < max_attempts:
            # 在整个范围内随机采样
            nr_cand = np.random.uniform(row_range[0], row_range[1])
            nc_cand = np.random.uniform(col_range[0], col_range[1])

            # 检查是否在排除区域外
            if not (nr_min <= nr_cand <= nr_max and nc_min <= nc_cand <= nc_max):
                neg_samples.append([nr_cand, nc_cand])

            attempts += 1

        # 如果采样失败，使用边界点
        while len(neg_samples) < total_num_negatives:
            neg_samples.append([row_range[0], col_range[0]])

        return np.array(neg_samples[:total_num_negatives], dtype=np.float32)


def collate_uav_sat_pair(batch):
    """
    自定义collate函数，将多个样本组合成batch

    Args:
        batch: list of dict，每个dict包含 uav_img, sat_img_pos, sat_imgs_neg, coords_uav

    Returns:
        dict: {
            'uav_imgs': [B, C, H, W]
            'sat_imgs_pos': [B, C, H, W]
            'sat_imgs_neg': [B, C, H, W] 或 [B, n_neg, C, H, W] 或 None（如果n_neg=0）
            'coords_uav': [B, 4]
        }
    """
    uav_imgs = torch.stack([item['uav_img'] for item in batch])
    sat_imgs_pos = torch.stack([item['sat_img_pos'] for item in batch])
    coords_uav = torch.stack([item['coords_uav'] for item in batch])

    # 处理负样本（可能为None）
    if batch[0]['sat_imgs_neg'] is not None:
        sat_imgs_neg = torch.stack([item['sat_imgs_neg'] for item in batch])
    else:
        sat_imgs_neg = None

    return {
        'uav_imgs': uav_imgs,
        'sat_imgs_pos': sat_imgs_pos,
        'sat_imgs_neg': sat_imgs_neg,
        'coords_uav': coords_uav,
    }


# 使用示例
if __name__ == '__main__':
    from dataset_wingtra_4d import UAVDataset, SatDataset
    from util_sample_neg_nrcs import BoundedNegativeCoordinateSampler
    from torch.utils.data import DataLoader
    import time

    # 初始化数据集
    sat_dataset = SatDataset(
        p_satinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/blocks12_res03m.json',
        p_uav_geocsv='/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv',
        imgsize2net=224,
    )

    uav_dataset = UAVDataset(
        p_uavinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info.json',
        geo_res_m=sat_dataset.geo_res_m,
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        stage='train'
    )

    satmap_sampler = BoundedNegativeCoordinateSampler(device='cuda')

    # ========== 测试1：n_neg=1 ==========
    print("=" * 50)
    print("测试1：n_neg=1（标准对比学习）")
    print("=" * 50)

    pair_dataset = UAVSatPairDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        satmap_sampler=satmap_sampler,
        device='cuda',
        n_neg_per_sample=1,
    )

    print(f"Dataset size: {len(pair_dataset)}")

    # 测试单个样本
    print("\n测试单个样本...")
    sample = pair_dataset[0]
    print(f"UAV image shape: {sample['uav_img'].shape}")
    print(f"Sat pos image shape: {sample['sat_img_pos'].shape}")
    print(f"Sat neg image shape: {sample['sat_imgs_neg'].shape}")
    print(f"Coords shape: {sample['coords_uav'].shape}")

    # 测试DataLoader
    print("\n测试DataLoader (num_workers=0)...")
    dataloader = DataLoader(
        pair_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_uav_sat_pair,
    )

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  UAV images: {batch['uav_imgs'].shape}")
        print(f"  Sat pos images: {batch['sat_imgs_pos'].shape}")
        print(f"  Sat neg images: {batch['sat_imgs_neg'].shape}")
        print(f"  Coords: {batch['coords_uav'].shape}")
        if i >= 1:
            break

    # ========== 测试2：n_neg=0 ==========
    print("\n" + "=" * 50)
    print("测试2：n_neg=0（只采样正样本）")
    print("=" * 50)

    pair_dataset_no_neg = UAVSatPairDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        satmap_sampler=None,  # n_neg=0时可以为None
        device='cuda',
        n_neg_per_sample=0,
    )

    print(f"\n测试单个样本...")
    sample = pair_dataset_no_neg[0]
    print(f"UAV image shape: {sample['uav_img'].shape}")
    print(f"Sat pos image shape: {sample['sat_img_pos'].shape}")
    print(f"Sat neg image: {sample['sat_imgs_neg']}")  # 应该是None
    print(f"Coords shape: {sample['coords_uav'].shape}")

    # 测试DataLoader
    print("\n测试DataLoader...")
    dataloader_no_neg = DataLoader(
        pair_dataset_no_neg,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_uav_sat_pair,
    )

    for i, batch in enumerate(dataloader_no_neg):
        print(f"Batch {i}:")
        print(f"  UAV images: {batch['uav_imgs'].shape}")
        print(f"  Sat pos images: {batch['sat_imgs_pos'].shape}")
        print(f"  Sat neg images: {batch['sat_imgs_neg']}")  # 应该是None
        print(f"  Coords: {batch['coords_uav'].shape}")
        if i >= 1:
            break

    # ========== 测试3：多进程加速 ==========
    print("\n" + "=" * 50)
    print("测试3：多进程加速对比")
    print("=" * 50)

    # 单进程
    print("\n单进程 (num_workers=0)...")
    dataloader_single = DataLoader(
        pair_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_uav_sat_pair,
    )

    start_time = time.time()
    for i, batch in enumerate(dataloader_single):
        if i >= 5:
            break
    time_single = time.time() - start_time
    print(f"Time: {time_single:.2f}s for 5 batches")

    # 多进程
    print("\n多进程 (num_workers=4)...")
    dataloader_multi = DataLoader(
        pair_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_uav_sat_pair,
        persistent_workers=True,
    )

    start_time = time.time()
    for i, batch in enumerate(dataloader_multi):
        if i >= 5:
            break
    time_multi = time.time() - start_time
    print(f"Time: {time_multi:.2f}s for 5 batches")
    print(f"Speedup: {time_single / time_multi:.2f}x")
