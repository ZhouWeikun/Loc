"""
UAV-Satellite Pair Dataset
组合UAV和Satellite数据集，在__getitem__中返回三元组样本
利用DataLoader的多进程机制自动并行采样
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os

from .dataset_neuloc_4d  import UAVDataset, SatDataset
from torch.utils.data import DataLoader
import time

class UAVSatPairDataset(Dataset):
    """
    UAV-Satellite配对数据集

    在__getitem__中返回：
    - uavimg: UAV图像
    - satimg_query: 查询卫星图（多遥感图时随机采样）
    - satimgs_pos: 正样本卫星图（与UAV对应位置）
    - satimgs_neg: 负样本卫星图（随机采样，如果n_neg=0则为None）
    - coords_uav: UAV的4D坐标 [nr, nc, rot, scale]

    这样可以利用DataLoader的num_workers并行采样，无需额外的异步逻辑
    """

    def __init__(
        self,
        uav_dataset,
        sat_dataset,
        device='cuda',
        n_neg_per_query=1,
        sat_as_query=False,
        nrc_reject_sampling=False,
    ):
        """
        Args:
            uav_dataset: UAVDataset实例
            sat_dataset: SatDataset实例
            device: 设备（用于负样本采示样）
            n_neg_per_query: 每个样本的负样本数量（0表不采样负样本,此时负样本来自其他正样本）
            nrc_reject_sampling: bool, 是否使用正样本领域拒绝采样（True）还是完全随机采样（False）
        """
        self.uav_dataset = uav_dataset
        self.sat_dataset = sat_dataset
        self.device = device
        self.n_neg_per_query = n_neg_per_query
        self.nrc_reject_sampling = nrc_reject_sampling
        # 如果有多时相遥感图，则遥感图之间可相互检索
        self.n_satmaps = len(self.sat_dataset.satmaps)
        self.sat_as_query = sat_as_query
        if self.sat_as_query and self.n_satmaps <= 1:
            raise ValueError(
                f"sat_as_query=True requires at least 2 satellite images, got n_satmaps={self.n_satmaps}"
            )

    def __len__(self):
        return len(self.uav_dataset)

    def __getitem__(self, index):
        """
        获取一个样本

        Returns:
            dict: {
                'uavimg': [C, H, W] UAV图像
                'satimg_query': [C, H, W] 查询卫星图
                'satimgs_pos': [C, H, W] 正样本卫星图
                'satimgs_neg': [n_neg, C, H, W] 或 [C, H, W] 或 None（如果n_neg=0）
                'coords_uav': [4] UAV的4D坐标
            }
        """
        # 1. 从UAV数据集获取UAV图像和增强后的4D坐标
        # UAVDataset现在直接返回 (uavimg, coords_4d)
        uavimg, coords_uav = self.uav_dataset[index]

        # 2. 采样正样本卫星图（与UAV位置对应）
        satimgs_pos, satmap_id_pos = self.sat_dataset.crop_satimg_by_4d_coords_fast(
            coords_uav, return_satmap_id=True
        )

        # 多遥感图时，为 query/pos 各自随机采样一张图
        satmap_id_query = None
        satmap_id_pos2satimg_query = None
        if self.sat_as_query:
            perm = torch.randperm(self.n_satmaps)
            query_id, pos_id = perm[0], perm[1]
            satmap_id_query = int(query_id)
            satmap_id_pos2satimg_query = int(pos_id)
            coords_sat_query = self.sat_dataset.mk_rand_coords_4d(n_rand=1, return_tensor=True).squeeze()
            satimg_query = self.sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_sat_query, id_satmap2sample=query_id,
            ).squeeze()
            satimg_pos2satimg_query = self.sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_sat_query, id_satmap2sample=pos_id,
            ).squeeze()
        
        # debug
        # img2vis_uav = self.uav_dataset.denormalize_img(uavimg)
        # img2vis_sat = self.sat_dataset.denormalize_img(satimgs_pos)
        # from matplotlib import pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        # ax1.imshow(img2vis_uav)
        # ax2.imshow(img2vis_sat)
        # plt.show()

        # 3. 采样负样本（如果需要）
        satimgs_neg = None
        coords_uav_neg = None
        satmap_id_neg = None
        if self.n_neg_per_query > 0:
            # 根据开关选择采样策略
            if self.nrc_reject_sampling:
                # 策略1: 正样本领域拒绝采样（排除正样本附近区域）
                nrcs_uav = coords_uav[:2]
                nrcs_uav_np = nrcs_uav.detach().cpu().numpy() if isinstance(nrcs_uav, torch.Tensor) else nrcs_uav

                # 采样负样本坐标（排除正样本附近区域）
                nrcs_neg = self._sample_negatives_cpu(
                    nrcs_uav_np,
                    threshold=self.sat_dataset.halfimg_radius_nrc*1.1,
                    row_range=self.sat_dataset.nr2sample_range,
                    col_range=self.sat_dataset.nc2sample_range,
                    total_num_negatives=self.n_neg_per_query,
                    ret_tensor=True,
                )
            else:
                # 策略2: 完全随机采样（不考虑与正样本的距离）
                nrcs_neg = self.sat_dataset.mk_rand_nrcs(self.n_neg_per_query,return_tensor=True)

            # 随机采样旋转和尺度
            device = nrcs_neg.device
            dtype_t = nrcs_neg.dtype
            rots_neg = (torch.rand(self.n_neg_per_query, device=device, dtype=dtype_t) * 2 * torch.pi) - torch.pi
            scale_min = float(self.sat_dataset.satimgsize_scale_to_ref_m_boundary[0])
            scale_max = float(self.sat_dataset.satimgsize_scale_to_ref_m_boundary[1])
            scales_neg = torch.rand(self.n_neg_per_query, device=device, dtype=dtype_t) * (scale_max - scale_min) + scale_min

            coords_uav_neg = torch.cat([
                nrcs_neg,
                rots_neg[:, None],
                scales_neg[:, None]
            ], dim=-1)

            # 采样负样本卫星图
            satimgs_neg, satmap_id_neg = self.sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_uav_neg, random_satmap=True, return_satmap_id=True
            )

        dict2return = {
            'uavimg': uavimg,
            'satimgs_pos': satimgs_pos,
            'satimgs_neg': satimgs_neg,
            'coords_uav': coords_uav,
            'coords_uav_neg': coords_uav_neg,
            'satmap_id_pos': satmap_id_pos,
            'satmap_id_neg': satmap_id_neg,
        }
        if self.sat_as_query:
            dict2return.update({
            'satimg_query': satimg_query,
            'satimg_pos2satimg_query': satimg_pos2satimg_query,
            'coords_sat_query': coords_sat_query,
            'satmap_id_query': satmap_id_query,
            'satmap_id_pos2satimg_query': satmap_id_pos2satimg_query,
            })
        # self._maybe_debug_visualize_sample(index, dict2return,save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results')
        return dict2return

    def _maybe_debug_visualize_sample(self, index, sample_dict, max_samples=1, save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results'):
        func = type(self)._maybe_debug_visualize_sample
        debug_vis_count = int(getattr(func, "_debug_vis_count", 0))
        if max_samples > 0 and debug_vis_count >= max_samples:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[UAVSatPairDataset] debug visualize skipped: matplotlib unavailable ({exc})")
            return

        def _coords_to_tensor(coords):
            if coords is None:
                return None
            if isinstance(coords, torch.Tensor):
                return coords.detach().cpu().to(dtype=torch.float32)
            try:
                return torch.as_tensor(coords, dtype=torch.float32)
            except Exception:
                return None

        def _wrap_rad(rad_value):
            if rad_value is None:
                return None
            return float(np.arctan2(np.sin(rad_value), np.cos(rad_value)))

        def _format_deg(rad_value):
            if rad_value is None:
                return "None"
            return f"{(float(rad_value) * 180.0 / np.pi):.1f}deg"

        def _short_name(path_str, keep=32):
            if not path_str:
                return "unknown"
            base = os.path.basename(path_str)
            if len(base) <= keep:
                return base
            head = max(keep // 2 - 2, 8)
            tail = max(keep - head - 3, 8)
            return f"{base[:head]}...{base[-tail:]}"

        def _get_uav_raw_coords(index_value):
            stage = getattr(self.uav_dataset, "stage", "train")
            if stage == 'train' and hasattr(self.uav_dataset, 'uav_coords_4d_torch_train'):
                return self.uav_dataset.uav_coords_4d_torch_train[index_value]
            if stage != 'train' and hasattr(self.uav_dataset, 'uav_coords_4d_torch_test'):
                return self.uav_dataset.uav_coords_4d_torch_test[index_value]
            return None

        def _format_uav_title(title_prefix, coords_tensor, index_value):
            raw_coords = _coords_to_tensor(_get_uav_raw_coords(index_value))
            final_coords = _coords_to_tensor(coords_tensor)
            raw_rot = float(raw_coords[2].item()) if raw_coords is not None and raw_coords.numel() >= 3 else None
            final_rot = float(final_coords[2].item()) if final_coords is not None and final_coords.numel() >= 3 else None
            aug_rot = None
            if raw_rot is not None and final_rot is not None:
                aug_rot = _wrap_rad(final_rot - raw_rot)
            return (
                f"{title_prefix}\n"
                f"rot_raw={_format_deg(raw_rot)}\n"
                f"rot_aug={_format_deg(aug_rot)}\n"
                f"rot_final={_format_deg(final_rot)}"
            )

        def _format_sat_title(title_prefix, coords_tensor):
            satmap_id = None
            satmap_label = None
            if "|" in title_prefix:
                title_prefix, satmap_label = title_prefix.split("|", 1)
                if satmap_label.startswith("satmap_id="):
                    try:
                        satmap_id = int(satmap_label.split("=", 1)[1])
                    except Exception:
                        satmap_id = None
            rot_deg = None
            if coords_tensor is not None and coords_tensor.numel() >= 3:
                rot_deg = float(coords_tensor[2].item()) * 180.0 / np.pi
            satmap_desc = ""
            if satmap_id is not None:
                satmap_paths = self.sat_dataset.satinfo_dict.get('filepaths', [])
                if 0 <= satmap_id < len(satmap_paths):
                    satmap_desc = (
                        f"\nsatmap_id={satmap_id}"
                        f"\nsrc={_short_name(satmap_paths[satmap_id])}"
                    )
                else:
                    satmap_desc = f"\nsatmap_id={satmap_id}"
            rot_line = ""
            if rot_deg is not None:
                rot_line = f"\nrot={rot_deg:.1f}deg"
            return (
                f"{title_prefix}\n"
                f"{rot_line}"
                f"{satmap_desc}"
            )

        def _append_images(images, title_prefix, tensor, denorm_fn, coords=None, satmap_id=None, max_items=4):
            if tensor is None:
                return
            if not isinstance(tensor, torch.Tensor):
                return
            tensor_cpu = tensor.detach().cpu()
            coords_cpu = _coords_to_tensor(coords)
            if tensor_cpu.ndim == 3:
                if title_prefix == "uavimg":
                    title_prefix = _format_uav_title(title_prefix, coords_cpu, int(index))
                if title_prefix.startswith("sat"):
                    title_prefix = _format_sat_title(f"{title_prefix}|satmap_id={satmap_id}", coords_cpu)
                images.append((title_prefix, denorm_fn(tensor_cpu)))
                return
            if tensor_cpu.ndim == 4:
                n_show = min(int(tensor_cpu.shape[0]), int(max_items))
                for i in range(n_show):
                    item_title = f"{title_prefix}[{i}]"
                    item_coords = None
                    if coords_cpu is not None:
                        if coords_cpu.ndim == 1:
                            item_coords = coords_cpu
                        elif coords_cpu.ndim >= 2 and i < int(coords_cpu.shape[0]):
                            item_coords = coords_cpu[i]
                    if title_prefix.startswith("sat"):
                        item_title = _format_sat_title(f"{item_title}|satmap_id={satmap_id}", item_coords)
                    images.append((item_title, denorm_fn(tensor_cpu[i])))

        images = []
        _append_images(
            images, "uavimg", sample_dict.get('uavimg'), self.uav_dataset.denormalize_img,
            coords=sample_dict.get('coords_uav'), max_items=1
        )
        _append_images(
            images, "satimgs_pos", sample_dict.get('satimgs_pos'), self.sat_dataset.denormalize_img,
            coords=sample_dict.get('coords_uav'), satmap_id=sample_dict.get('satmap_id_pos'),
        )
        _append_images(
            images, "satimgs_neg", sample_dict.get('satimgs_neg'), self.sat_dataset.denormalize_img,
            coords=sample_dict.get('coords_uav_neg'), satmap_id=sample_dict.get('satmap_id_neg'),
        )
        _append_images(
            images, "satimg_query", sample_dict.get('satimg_query'), self.sat_dataset.denormalize_img,
            coords=sample_dict.get('coords_sat_query'), satmap_id=sample_dict.get('satmap_id_query'), max_items=1,
        )
        _append_images(
            images, "satimg_pos2satimg_query", sample_dict.get('satimg_pos2satimg_query'), self.sat_dataset.denormalize_img,
            coords=sample_dict.get('coords_sat_query'), satmap_id=sample_dict.get('satmap_id_pos2satimg_query'), max_items=1,
        )

        if not images:
            return

        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        if len(images) == 1:
            axes = [axes]
        for ax, (title, image_np) in zip(axes, images):
            ax.imshow(image_np)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f"UAVSatPairDataset debug sample idx={index}, pid={os.getpid()}")
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"pair_debug_pid{os.getpid()}_idx{int(index)}_{debug_vis_count:03d}.png",
            )
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[UAVSatPairDataset] debug visualization saved to: {save_path}")
        else:
            plt.show()
        plt.close(fig)
        setattr(func, "_debug_vis_count", debug_vis_count + 1)

    def _sample_negatives_cpu(self, nrcs, threshold, row_range, col_range, total_num_negatives, ret_tensor=False):
        """
        CPU版本的负样本采样（用于DataLoader的worker进程）

        Args:
            nrcs: [2] 或 [1, 2] 的numpy数组
            threshold: 排除半径
            row_range: [min, max] 行采样范围
            col_range: [min, max] 列采样范围
            total_num_negatives: 采样数量
            ret_tensor: 是否返回torch.Tensor

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

        neg_samples = np.array(neg_samples[:total_num_negatives], dtype=np.float32)
        if ret_tensor:
            return torch.from_numpy(neg_samples)
        return neg_samples


def collate_uav_sat_pair(batch):
    """
    自定义collate函数，将多个样本组合成batch

    Args:
        batch: list of dict，每个dict包含 uavimg, satimgs_pos, satimgs_neg, coords_uav

    Returns:
        dict: {
            'uavimgs': [B, C, H, W]
            'satimgs_pos': [B, C, H, W]
            'satimgs_neg': [B, C, H, W] 或 [B, n_neg, C, H, W] 或 None（如果n_neg=0）
            'coords_uav': [B, 4]
        }
    """
    #basic:
    uavimgs = torch.stack([item['uavimg'] for item in batch])
    satimgs_pos = torch.stack([item['satimgs_pos'] for item in batch])
    coords_uav = torch.stack([item['coords_uav'] for item in batch])
    dict2return = {
        'uavimgs': uavimgs,
        'satimgs_pos': satimgs_pos,
        'coords_uav': coords_uav,
    }

    #negative samples for per query:
    has_neg = ('coords_uav_neg' in  batch[0]) and ( batch[0]['coords_uav_neg'] is not None)
    if has_neg:
        coords_uav_neg = torch.stack([item['coords_uav_neg'] for item in batch])
        satimgs_neg = torch.stack([item['satimgs_neg'] for item in batch])
        dict2return.update({
            'coords_uav_neg': coords_uav_neg,
            'satimgs_neg': satimgs_neg,
        })

    #samples from satimg as query:
    has_satimg_query = ('satimg_query' in  batch[0]) and ( batch[0]['coords_sat_query'] is not None)
    if has_satimg_query:
        satimgs_query = torch.stack([item['satimg_query'] for item in batch])
        satimgs_pos2satimg_query = torch.stack( [item['satimg_pos2satimg_query'] for item in batch])
        coords_sat_query = torch.stack( [item['coords_sat_query'] for item in batch])
        dict2return.update({
            'satimgs_query': satimgs_query,
            'satimgs_pos2satimg_query':satimgs_pos2satimg_query,
            'coords_sat_query':coords_sat_query,
        })

    return dict2return


# 使用示例
if __name__ == '__main__':

    # 初始化数据集
    sat_dataset = SatDataset(
        p_satinfo_json='/home/data/zwk/dataset_UAV-VisLoc/04/satellite04_epsg32650_res03m_multi_tifs.json',
        p_uav_geocsv='/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_geo_corrected.csv',
        imgsize2net=224,
    )

    uav_dataset = UAVDataset(
        p_uavinfo_json='/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_metainfo.json',
        p_uav_geocsv='/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_geo_corrected.csv',
        sat_dataset=sat_dataset,
        stage='train'
    )

    # ========== 测试1：n_neg=1, 使用bounded sampling ==========
    print("=" * 50)
    print("测试1：n_neg=1 + bounded sampling（正样本领域拒绝采样）")
    print("=" * 50)

    pair_dataset = UAVSatPairDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        device='cuda',
        n_neg_per_query=1,
        nrc_reject_sampling=True,  # 使用正样本领域拒绝采样
    )

    print(f"Dataset size: {len(pair_dataset)}")

    # 测试单个样本
    print("\n测试单个样本...")
    sample = pair_dataset[0]
    print(f"UAV image shape: {sample['uavimg'].shape}")
    print(f"Sat pos image shape: {sample['satimgs_pos'].shape}")
    print(f"Sat neg image shape: {sample['satimgs_neg'].shape}")
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
        print(f"  UAV images: {batch['uavimgs'].shape}")
        print(f"  Sat pos images: {batch['satimgs_pos'].shape}")
        print(f"  Sat neg images: {batch['satimgs_neg'].shape}")
        print(f"  Coords: {batch['coords_uav'].shape}")
        if i >= 1:
            break

    # ========== 测试2：n_neg=1, 使用random sampling ==========
    print("\n" + "=" * 50)
    print("测试2：n_neg=1 + random sampling（完全随机采样）")
    print("=" * 50)

    pair_dataset_random = UAVSatPairDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        device='cuda',
        n_neg_per_query=1,
        nrc_reject_sampling=False,  # 使用完全随机采样
    )

    print(f"\n测试单个样本...")
    sample = pair_dataset_random[0]
    print(f"UAV image shape: {sample['uavimg'].shape}")
    print(f"Sat pos image shape: {sample['satimgs_pos'].shape}")
    print(f"Sat neg image shape: {sample['satimgs_neg'].shape}")
    print(f"Coords shape: {sample['coords_uav'].shape}")

    # ========== 测试3：n_neg=0 ==========
    print("\n" + "=" * 50)
    print("测试3：n_neg=0（只采样正样本）")
    print("=" * 50)

    pair_dataset_no_neg = UAVSatPairDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        device='cuda',
        n_neg_per_query=0,
    )

    print(f"\n测试单个样本...")
    sample = pair_dataset_no_neg[0]
    print(f"UAV image shape: {sample['uavimg'].shape}")
    print(f"Sat pos image shape: {sample['satimgs_pos'].shape}")
    print(f"Sat neg image: {sample['satimgs_neg']}")  # 应该是None
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
        print(f"  UAV images: {batch['uavimgs'].shape}")
        print(f"  Sat pos images: {batch['satimgs_pos'].shape}")
        print(f"  Sat neg images: {batch['satimgs_neg']}")  # 应该是None
        print(f"  Coords: {batch['coords_uav'].shape}")
        if i >= 1:
            break

    # ========== 测试4：多进程加速 ==========
    print("\n" + "=" * 50)
    print("测试4：多进程加速对比")
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
