"""
异步卫星图像采样器
在后台线程中预先采样下一个batch的卫星图像，避免阻塞训练循环
"""
import torch
import threading
import queue
from typing import Optional, Tuple


class AsyncSatelliteSampler:
    """
    异步采样卫星图像的包装器

    使用方式：
        sampler = AsyncSatelliteSampler(sat_dataset, satmap_sampler, device)

        # 在训练循环开始前，先启动第一个采样
        sampler.start_sample(coords_q_first)

        for coords_q in dataloader:
            # 获取上一次采样的结果（会等待直到完成）
            satimgs_pos, satimgs_neg = sampler.get_result()

            # 立即启动下一个batch的采样（异步，不阻塞）
            sampler.start_sample(coords_q_next)

            # 当前batch可以继续训练，不会被采样阻塞
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        sat_dataset,
        satmap_sampler,
        device='cuda',
        queue_size=2,  # 预加载队列大小（至少为2才能实现真正的预取）
    ):
        self.sat_dataset = sat_dataset
        self.satmap_sampler = satmap_sampler
        self.device = device

        # 使用队列进行线程间通信
        # task_queue: 存放待处理的任务
        # result_queue: 存放已完成的结果
        # 需要queue_size >= 2才能实现真正的流水线：一个在处理，一个在等待
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)

        # 控制标志
        self.running = False
        self.worker_thread = None

    def _worker(self):
        """
        后台工作线程，持续从任务队列获取采样任务并执行
        """
        while self.running:
            try:
                # 从任务队列获取坐标（超时1秒，避免阻塞shutdown）
                task = self.task_queue.get(timeout=1.0)

                if task is None:  # 结束信号
                    break

                coords_q = task

                # 执行采样（CPU上进行，避免GPU争用）
                with torch.no_grad():
                    # 1. 采样正样本
                    satimgs_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_q)

                    # 2. 采样负样本坐标
                    n_neg = coords_q.shape[0]
                    nrcs_neg = self.satmap_sampler.sample_negatives_shared_fast(
                        nrcs=coords_q[:, :2],
                        threshold=self.sat_dataset.halfimg_radius_nrc,
                        row_range=self.sat_dataset.nr2sample_range,
                        col_range=self.sat_dataset.nc2sample_range,
                        total_num_negatives=n_neg
                    )

                    rots_neg = -torch.pi + 2 * torch.pi * torch.rand(n_neg, device=self.device)
                    scales_neg = torch.rand(n_neg, device=self.device) * \
                                 (self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
                                  self.sat_dataset.satimgsize_scale_to_200m_boundary[0]) + \
                                 self.sat_dataset.satimgsize_scale_to_200m_boundary[0]

                    coords_neg = torch.cat([
                        nrcs_neg,
                        rots_neg.unsqueeze(1),
                        scales_neg.unsqueeze(1)
                    ], dim=-1)

                    # 3. 采样负样本图像
                    satimgs_neg = self.sat_dataset.crop_satimg_by_4d_coords(coords_neg)

                # 将结果放入结果队列
                self.result_queue.put((satimgs_pos, satimgs_neg))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AsyncSatelliteSampler worker error: {e}")
                self.result_queue.put(None)  # 放入错误标志

    def start(self):
        """启动后台工作线程"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

    def stop(self):
        """停止后台工作线程"""
        if self.running:
            self.running = False
            self.task_queue.put(None)  # 发送结束信号
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=5.0)

    def submit(self, coords_q: torch.Tensor):
        """
        提交一个采样任务（非阻塞）

        Args:
            coords_q: [B, 4] 查询坐标
        """
        if not self.running:
            self.start()

        # 将坐标移到CPU，避免GPU内存争用
        if coords_q.is_cuda:
            coords_q = coords_q.cpu()

        self.task_queue.put(coords_q)

    def get(self, timeout=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取采样结果（阻塞，直到结果可用）

        Returns:
            satimgs_pos: [B, C, H, W] 正样本卫星图像
            satimgs_neg: [B, C, H, W] 负样本卫星图像
        """
        result = self.result_queue.get(timeout=timeout)

        if result is None:
            raise RuntimeError("AsyncSatelliteSampler worker encountered an error")

        return result

    def __del__(self):
        """析构时自动停止工作线程"""
        self.stop()


class SimpleSatelliteSampler:
    """
    同步采样器（无预取），用作对比baseline

    使用方式：
        sampler = SimpleSatelliteSampler(sat_dataset, satmap_sampler, device)
        for coords_q in dataloader:
            satimgs_pos, satimgs_neg = sampler(coords_q)
    """

    def __init__(self, sat_dataset, satmap_sampler, device='cuda'):
        self.sat_dataset = sat_dataset
        self.satmap_sampler = satmap_sampler
        self.device = device

    def __call__(self, coords_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """同步采样（阻塞）"""
        with torch.no_grad():
            # 1. 采样正样本
            satimgs_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_q)

            # 2. 采样负样本
            n_neg = coords_q.shape[0]
            nrcs_neg = self.satmap_sampler.sample_negatives_shared_fast(
                nrcs=coords_q[:, :2],
                threshold=self.sat_dataset.halfimg_radius_nrc,
                row_range=self.sat_dataset.nr2sample_range,
                col_range=self.sat_dataset.nc2sample_range,
                total_num_negatives=n_neg
            )

            rots_neg = -torch.pi + 2 * torch.pi * torch.rand(n_neg, device=self.device)
            scales_neg = torch.rand(n_neg, device=self.device) * \
                         (self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
                          self.sat_dataset.satimgsize_scale_to_200m_boundary[0]) + \
                         self.sat_dataset.satimgsize_scale_to_200m_boundary[0]

            coords_neg = torch.cat([nrcs_neg, rots_neg.unsqueeze(1), scales_neg.unsqueeze(1)], dim=-1)
            satimgs_neg = self.sat_dataset.crop_satimg_by_4d_coords(coords_neg)

        return satimgs_pos, satimgs_neg


class PrefetchSatelliteSampler:
    """
    预取采样器 - 使用简化的同步方式（暂时）

    TODO: 异步预取存在队列同步问题，暂时使用同步方式
    后续可以考虑使用torch.multiprocessing的方式实现

    使用方式：
        sampler = PrefetchSatelliteSampler(sat_dataset, satmap_sampler, device)
        for coords_q in dataloader:
            satimgs_pos, satimgs_neg = sampler(coords_q)
    """

    def __init__(self, sat_dataset, satmap_sampler, device='cuda'):
        # 暂时使用同步采样
        self.sampler = SimpleSatelliteSampler(sat_dataset, satmap_sampler, device)

    def __call__(self, coords_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sampler(coords_q)

    def stop(self):
        """兼容接口"""
        pass

    def __del__(self):
        pass


# 测试代码
if __name__ == '__main__':
    import time
    from dataset_wingtra_4d import SatDataset
    from util_sample_neg_nrcs import BoundedNegativeCoordinateSampler

    # 初始化数据集
    sat_dataset = SatDataset(
        p_satinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/blocks12_res03m.json',
        p_uav_geocsv='/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv',
        imgsize2net=224,
    )

    satmap_sampler = BoundedNegativeCoordinateSampler(device='cuda')

    # 测试异步采样
    print("Testing AsyncSatelliteSampler...")
    async_sampler = AsyncSatelliteSampler(sat_dataset, satmap_sampler, device='cuda')
    async_sampler.start()

    # 模拟训练循环
    batch_size = 32
    num_batches = 10

    # 生成随机坐标
    coords_list = []
    for _ in range(num_batches):
        nrcs = torch.from_numpy(sat_dataset.mk_rand_nrcs(batch_size)).float()
        rots = torch.rand(batch_size, 1) * 2 * torch.pi - torch.pi
        scales = torch.ones(batch_size, 1) * sat_dataset.satimgsize_scale_to_200m_mean
        coords = torch.cat([nrcs, rots, scales], dim=1)
        coords_list.append(coords)

    # 测试同步方式（baseline）
    print("\nBaseline (synchronous):")
    start_time = time.time()
    for coords in coords_list:
        satimgs_pos = sat_dataset.crop_satimg_by_4d_coords(coords)
        nrcs_neg = satmap_sampler.sample_negatives_shared_fast(
            nrcs=coords[:, :2],
            threshold=sat_dataset.halfimg_radius_nrc,
            row_range=sat_dataset.nr2sample_range,
            col_range=sat_dataset.nc2sample_range,
            total_num_negatives=batch_size
        )
    baseline_time = time.time() - start_time
    print(f"Time: {baseline_time:.3f}s")

    # 测试异步方式
    print("\nWith AsyncSatelliteSampler:")
    start_time = time.time()

    # 预先提交第一个batch
    async_sampler.submit(coords_list[0])

    for i in range(len(coords_list)):
        # 获取当前结果
        satimgs_pos, satimgs_neg = async_sampler.get()

        # 提交下一个batch（如果有的话）
        if i + 1 < len(coords_list):
            async_sampler.submit(coords_list[i + 1])

        # 模拟训练时间
        time.sleep(0.01)

    async_time = time.time() - start_time
    print(f"Time: {async_time:.3f}s")
    print(f"Speedup: {baseline_time / async_time:.2f}x")

    async_sampler.stop()

    # 测试简化版
    print("\nTesting PrefetchSatelliteSampler...")
    prefetch_sampler = PrefetchSatelliteSampler(sat_dataset, satmap_sampler, device='cuda')

    start_time = time.time()
    for coords in coords_list:
        satimgs_pos, satimgs_neg = prefetch_sampler(coords)
        time.sleep(0.01)  # 模拟训练时间

    prefetch_time = time.time() - start_time
    print(f"Time: {prefetch_time:.3f}s")
    print(f"Speedup: {baseline_time / prefetch_time:.2f}x")

    prefetch_sampler.stop()
