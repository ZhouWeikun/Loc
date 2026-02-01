import torch
import torch.nn as nn
import math


class NormalizedGaussianSampler(nn.Module):
    """
    纯粹的 Linear-4D 采样器
    Input:  [B, 4] (Linear-4D)
    Output: [B, N, 4] (Linear-4D)
    """

    def __init__(self, norm_std_devs, device='cuda'):
        super().__init__()
        self.device = device
        # [r, c, theta, s]
        self.std_devs = torch.tensor(norm_std_devs, device=device).view(1, 1, -1)

        # 统一边界 [-1, 1]
        self.lower = torch.tensor([-1.0] * 4, device=device).view(1, 1, -1)
        self.upper = torch.tensor([1.0] * 4, device=device).view(1, 1, -1)
        self.periods = torch.tensor([2.0] * 4, device=device).view(1, 1, -1)

        # 只有 Theta (idx 2) 是周期性的
        self.circular_mask = torch.tensor([False, False, True, False], device=device).view(1, 1, -1)
        self.sqrt_2 = math.sqrt(2)
        self.epsilon = 1e-6

    def _phi(self, x):
        return 0.5 * (1 + torch.erf(x / self.sqrt_2))

    def _phi_inv(self, p):
        return self.sqrt_2 * torch.erfinv(2 * p - 1)

    def sample_importance(self, centers_linear, num_samples=16, include_center=False):
        """
        Args:
            centers_linear: [B, 4]
            num_samples: 随机采样的数量 N
            include_center: bool, 是否包含中心点自身。
                            如果为 True，返回形状为 [B, N+1, 4]，中心点位于索引 0。
                            如果为 False，返回形状为 [B, N, 4]。
        """
        B = centers_linear.shape[0]
        centers_expanded = centers_linear.unsqueeze(1)  # [B, 1, 4]

        # ==========================================
        # 1. 正常的随机采样逻辑 (保持不变)
        # ==========================================

        # 1. Z-score
        z_min = (self.lower - centers_expanded) / (self.std_devs + 1e-8)
        z_max = (self.upper - centers_expanded) / (self.std_devs + 1e-8)

        # 2. Probability Interval
        p_min = self._phi(z_min)
        p_max = self._phi(z_max)

        # Handle Circular
        zeros = torch.zeros_like(p_min)
        ones = torch.ones_like(p_max)
        eff_p_min = torch.where(self.circular_mask, zeros, p_min)
        eff_p_max = torch.where(self.circular_mask, ones, p_max)
        eff_p_max = torch.maximum(eff_p_max, eff_p_min + self.epsilon)

        # 3. Sample
        rand_u = torch.rand(B, num_samples, 4, device=self.device)
        p_sample = eff_p_min + rand_u * (eff_p_max - eff_p_min)
        p_sample = torch.clamp(p_sample, self.epsilon, 1.0 - self.epsilon)

        # 4. Inverse Transform
        z_sample = self._phi_inv(p_sample)
        perturbation = z_sample * self.std_devs

        sampled_coords = centers_expanded + perturbation

        # 5. Wrap-around (Only for Theta)
        final_coords = torch.where(
            self.circular_mask,
            torch.remainder(sampled_coords - self.lower, self.periods) + self.lower,
            sampled_coords
        )  # [B, N, 4]

        # ==========================================
        # 2. 拼接中心点逻辑 (新增)
        # ==========================================
        if include_center:
            # 拼接: [B, 1, 4] cat [B, N, 4] -> [B, N+1, 4]
            # 中心点永远在第 0 个位置，方便后续索引
            return torch.cat([centers_expanded, final_coords], dim=1)
        else:
            return final_coords


class GaussianSampler(nn.Module):
    def __init__(self, std_devs, limits, circular_dims=None, device='cuda'):
        """
        动态边界采样器 (Dynamic Boundary Sampler)

        功能：
        1. 对于有界维度 (Bounded Dims)：使用动态截断正态分布，避免边界堆积。
        2. 对于周期维度 (Circular Dims)：使用 Wrap-around (取模) 和最短弧长距离。

        Args:
            std_devs (list or tuple): 各维度的标准差 [sigma_x, sigma_y, sigma_d, sigma_s]
            limits (list of tuples): 各维度的物理边界 [(min_x, max_x), ..., (min_s, max_s)]
            circular_dims (list of int, optional): 指定哪些维度是周期性的 (例如 [3])
            device (str): 计算设备
        """
        super().__init__()
        self.device = device

        # 1. 基础参数转 Tensor (1, 1, D) 用于广播
        self.std_devs = torch.tensor(std_devs, device=device, dtype=torch.float32).view(1, 1, -1)

        # 2. 边界处理
        self.lower_bounds = torch.tensor([lim[0] for lim in limits], device=device, dtype=torch.float32).view(1, 1, -1)
        self.upper_bounds = torch.tensor([lim[1] for lim in limits], device=device, dtype=torch.float32).view(1, 1, -1)
        self.periods = self.upper_bounds - self.lower_bounds

        # 3. 周期性掩码
        # circular_mask: [False, False, False, True]
        self.circular_mask = torch.zeros(len(std_devs), device=device, dtype=torch.bool).view(1, 1, -1)
        if circular_dims:
            self.circular_mask[:, :, circular_dims] = True

        # 常数缓存
        self.sqrt_2 = math.sqrt(2)
        self.epsilon = 1e-6  # 数值稳定性极小值

    def _phi(self, x):
        """标准正态分布 CDF: Phi(x)"""
        return 0.5 * (1 + torch.erf(x / self.sqrt_2))

    def _phi_inv(self, p):
        """标准正态分布 Inverse CDF (Probit): Phi^-1(p)"""
        return self.sqrt_2 * torch.erfinv(2 * p - 1)

    def sample_importance(self, centers, num_samples=16):
        """
        执行重要性采样

        Args:
            centers: (B, D) 锚点坐标
            num_samples: 每个锚点采样的数量 N

        Returns:
            final_coords: (B, N, D) 采样后的坐标，保证在界内且符合物理约束
        """
        B, D = centers.shape
        centers_expanded = centers.unsqueeze(1)  # (B, 1, D)

        # --- 步骤 1: 计算当前中心点相对于物理边界的 Z-score ---
        # Z = (Limit - Center) / Sigma
        # 这告诉我们：边界距离中心有多少个标准差
        z_min = (self.lower_bounds - centers_expanded) / (self.std_devs + 1e-8)
        z_max = (self.upper_bounds - centers_expanded) / (self.std_devs + 1e-8)

        # --- 步骤 2: 将 Z-score 映射到概率空间 (CDF) ---
        # 得到截断正态分布的有效概率区间 [p_min, p_max]
        p_min = self._phi(z_min)
        p_max = self._phi(z_max)

        # --- 步骤 3: 处理周期性维度 ---
        # 对于周期性维度 (如方向)，我们不需要截断，采样全区间 [0, 1]
        # 使用 where 覆盖掉周期维度的 p 值
        zeros = torch.zeros_like(p_min)
        ones = torch.ones_like(p_max)

        # 如果是 circular，范围强制为 [0, 1]；否则保持 [p_min, p_max]
        effective_p_min = torch.where(self.circular_mask, zeros, p_min)
        effective_p_max = torch.where(self.circular_mask, ones, p_max)

        # 确保 p_max > p_min (防止 float 精度导致的错误)
        effective_p_max = torch.maximum(effective_p_max, effective_p_min + self.epsilon)

        # --- 步骤 4: 在有效概率区间内均匀采样 (Inverse Transform Sampling) ---
        # u ~ U(p_min, p_max)
        rand_u = torch.rand(B, num_samples, D, device=self.device)
        p_sample = effective_p_min + rand_u * (effective_p_max - effective_p_min)

        # 数值截断：防止 p_sample 极其接近 0 或 1 导致 erfinv 输出 infinity
        p_sample = torch.clamp(p_sample, self.epsilon, 1.0 - self.epsilon)

        # --- 5. 逆变换还原 Z-score，并转回物理坐标 ---
        z_sample = self._phi_inv(p_sample)
        perturbation = z_sample * self.std_devs

        sampled_coords = centers_expanded + perturbation

        # --- 6. 最终处理周期性维度的 Wrap-around ---
        # 对于有界维度：步骤 4 保证了它一定在界内，不需要 clamp
        # 对于周期维度：使用 remainder 取模
        # 公式: lower + (x - lower) % period
        final_coords = torch.where(
            self.circular_mask,
            torch.remainder(sampled_coords - self.lower_bounds, self.periods) + self.lower_bounds,
            sampled_coords
        )

        return final_coords

    def compute_weights(self, centers, query_points):
        """
        计算各向异性高斯权重 (PDF Kernel)
        自动处理周期性维度的最短距离。
        """
        centers_expanded = centers.unsqueeze(1)  # (B, 1, D)

        # 1. 计算原始差值
        delta = query_points - centers_expanded  # (B, N, D)

        # 2. 周期性距离修正
        # d_cyclic = min(|d|, Period - |d|)
        delta_abs = torch.abs(delta)
        delta_cyclic = torch.min(delta_abs, self.periods - delta_abs)

        # 选择有效差值：周期维度用 cyclic 差值，普通维度用原始差值
        effective_delta = torch.where(self.circular_mask, delta_cyclic, delta)

        # 3. 计算马氏距离 (Mahalanobis Distance)
        normalized_delta = effective_delta / (self.std_devs + 1e-8)
        dist_sq = torch.sum(normalized_delta ** 2, dim=-1)  # (B, N)

        # 4. 计算高斯权重
        # 注意：这里我们只计算 exp 部分作为相对权重，通常这就够了
        weights = torch.exp(-0.5 * dist_sq)

        return weights


# ============================================================================
# 可视化验证函数
# ============================================================================
def visualize_dynamic_boundary_sampling(
        sampler,
        centers: torch.Tensor = None,
        num_samples: int = 1000,
        save_path: str = None,
        show_interactive: bool = False,
        spatial_dims=(0, 1),
        scale_dim: int = 3,
        rot_dim: int = 2,
        dim_names=None,
        coord_normer=None,
        centers_are_normalized: bool = True
):
    """
    可视化高斯采样器的采样分布。

    布局：
    - 左图: (X, Y) 2D散点图
    - 中图: Rotation 1D分布直方图
    - 右图: Scale 1D分布直方图
    """
    import os
    import numpy as np

    print(f"\n{'=' * 80}")
    print(f"Gaussian Sampler 可视化验证")
    print(f"{'=' * 80}\n")

    if save_path is None:
        base_name = "vis_dynamic_boundary_sampling"
    else:
        base_name = os.path.splitext(save_path)[0]

    html_path = f"{base_name}.html"
    png_path = f"{base_name}.png"

    if hasattr(sampler, 'lower_bounds'):
        lower_bounds = sampler.lower_bounds
        upper_bounds = sampler.upper_bounds
    else:
        lower_bounds = sampler.lower
        upper_bounds = sampler.upper

    # 1. 生成或使用中心点
    if centers is None:
        centers_np = (lower_bounds + upper_bounds).squeeze(0).squeeze(0) / 2
        centers = centers_np.unsqueeze(0)

    if coord_normer is not None and not centers_are_normalized:
        norm6 = coord_normer.raw_to_norm(centers, append_linear_rot=True)
        centers = torch.stack(
            [norm6[..., 0], norm6[..., 1], norm6[..., 5], norm6[..., 4]],
            dim=-1
        )

    if centers.dim() != 2:
        raise ValueError(f"centers 应为 (B, D)，但得到 {centers.shape}")

    centers = centers.to(device=sampler.device, dtype=torch.float32)
    centers_np = centers.cpu().numpy()
    B, D = centers.shape

    # 2. 采样
    samples = sampler.sample_importance(centers, num_samples=num_samples)
    samples_np = samples.reshape(-1, D).cpu().numpy()

    # 3. 统计边界一致性
    lower = lower_bounds.squeeze(0).squeeze(0).cpu().numpy()
    upper = upper_bounds.squeeze(0).squeeze(0).cpu().numpy()
    out_of_bounds = (samples_np < lower) | (samples_np > upper)
    out_count = out_of_bounds.sum(axis=0)

    print(f"Centers: {B} 个")
    print(f"Samples: {samples_np.shape[0]} 个")
    for idx in range(D):
        name = dim_names[idx] if dim_names else f"dim{idx}"
        print(f"  - {name}: min={samples_np[:, idx].min():.4f}, "
              f"max={samples_np[:, idx].max():.4f}, out_of_bounds={out_count[idx]}")

    # 4. Matplotlib 2D 可视化
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x_dim, y_dim = spatial_dims
        x_name = dim_names[x_dim] if dim_names else f"dim{x_dim}"
        y_name = dim_names[y_dim] if dim_names else f"dim{y_dim}"

        # --- Subplot 1: 空间散点图 ---
        ax = axes[0]
        ax.scatter(samples_np[:, x_dim], samples_np[:, y_dim], s=8, c='tab:blue',
                   alpha=0.35, edgecolors='none', label='samples')
        ax.scatter(centers_np[:, x_dim], centers_np[:, y_dim], s=150, c='red',
                   marker='*', edgecolors='black', linewidths=1.0, label='centers', zorder=10)

        # 画出边界框
        x_min, x_max = lower[x_dim], upper[x_dim]
        y_min, y_max = lower[y_dim], upper[y_dim]
        ax.plot([x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                color='black', linewidth=1.0, linestyle='--', label='bounds')

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title('Spatial Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # --- Subplot 2: Rotation 直方图 ---
        ax = axes[1]
        rot_range = (lower[rot_dim], upper[rot_dim])
        ax.hist(samples_np[:, rot_dim], bins=50, range=rot_range, density=True,
                color='tab:orange', alpha=0.6, histtype='stepfilled', label='rot')
        ax.axvline(centers_np[0, rot_dim], color='red', linestyle='--', linewidth=2, label='center')
        ax.set_xlabel(dim_names[rot_dim] if dim_names else f"dim{rot_dim}")
        ax.set_ylabel('Density')
        ax.set_title('Rotation Distribution')
        ax.grid(True, alpha=0.3)

        # --- Subplot 3: Scale 直方图 ---
        ax = axes[2]
        scale_range = (lower[scale_dim], upper[scale_dim])
        ax.hist(samples_np[:, scale_dim], bins=50, range=scale_range, density=True,
                color='tab:green', alpha=0.6, histtype='stepfilled', label='scale')
        ax.axvline(centers_np[0, scale_dim], color='red', linestyle='--', linewidth=2, label='center')
        ax.set_xlabel(dim_names[scale_dim] if dim_names else f"dim{scale_dim}")
        ax.set_ylabel('Density')
        ax.set_title('Scale Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"✅ 统计分布图已保存到: {png_path}")

        if show_interactive:
            plt.show()
        else:
            plt.close(fig)

    except ImportError:
        print("⚠️  需要安装matplotlib用于2D可视化")

    # 5. 3D Plotly 可视化 (X, Y, Rot)
    try:
        import plotly.graph_objects as go

        z_dim = rot_dim if rot_dim < D else 2
        z_name = dim_names[z_dim] if dim_names else f"dim{z_dim}"

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=centers_np[:, x_dim], y=centers_np[:, y_dim], z=centers_np[:, z_dim],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='centers'
        ))

        max_points_display = 2000
        vis_samples = samples_np
        if len(vis_samples) > max_points_display:
            indices = np.random.choice(len(vis_samples), max_points_display, replace=False)
            vis_samples = vis_samples[indices]

        fig.add_trace(go.Scatter3d(
            x=vis_samples[:, x_dim], y=vis_samples[:, y_dim], z=vis_samples[:, z_dim],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.5),
            name='samples'
        ))

        fig.update_layout(
            title='3D Sampling Visualization',
            scene=dict(xaxis_title=x_name, yaxis_title=y_name, zaxis_title=z_name),
            height=800
        )
        fig.write_html(html_path)
        print(f"✅ 交互式3D图表已保存到: {html_path}")

        if show_interactive:
            fig.show()

    except Exception as e:
        print(f"⚠️  3D可视化跳过: {e}")

    print(f"\n{'=' * 80}\n")
    return {
        'centers': centers_np,
        'total_samples': samples_np.shape[0],
        'png_path': png_path,
        'html_path': html_path
    }


# ==========================================
#              测试与验证代码
# ==========================================
if __name__ == "__main__":
    import os
    import numpy as np

    # 归一化空间: 4维 [nr_norm, nc_norm, rot_norm, scale_norm]
    # nr/nc/scale/rot 范围均为 [-1, 1]
    circular_dims = [2]  # 第3维是方向 (theta_linear)

    sampler = NormalizedGaussianSampler(
        norm_std_devs=[0.1, 0.1, 0.2, 0.1],
        circular_dims=circular_dims,
        device='cpu'
    )

    print("-" * 30)
    print("测试 1: 边界采样不堆积")
    print("-" * 30)
    # 将中心放在左边界 x=0
    centers_edge = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    samples = sampler.sample_importance(centers_edge, num_samples=1000)

    min_x = samples[:, :, 0].min().item()
    max_x = samples[:, :, 0].max().item()
    count_neg = (samples[:, :, 0] < -1.0).sum().item()
    count_exact_0 = (samples[:, :, 0] == -1.0).sum().item()

    print(f"Center X=-1")
    print(f"采样点 Min X: {min_x:.4f} (应 >= -1)")
    print(f"采样点 Max X: {max_x:.4f}")
    print(f"小于 -1 的点数: {count_neg} (应为 0)")
    print(f"等于 -1 的点数: {count_exact_0} (应很小或为0，不应堆积)")
    # 如果是 Clamp 方法，这里会有大量点等于 -1

    print("\n" + "-" * 30)
    print("测试 2: 周期性 Wrap-around")
    print("-" * 30)
    # 将中心放在角度边界 pi (3.14159)
    centers_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    samples_rot = sampler.sample_importance(centers_rot, num_samples=5)

    print(f"Center Rot (norm) = 1.0")
    print(f"采样点 Rot (norm) (应在 1 附近，可能跳变到 -1):")
    print(samples_rot[0, :, 2])

    print("\n" + "-" * 30)
    print("测试 3: 周期性距离权重计算")
    print("-" * 30)
    # 构造一个跨越边界的情况
    # 中心: pi,  查询点: -pi + 0.1 (物理距离极近)
    center_pt = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    query_pt = torch.tensor([[0.0, 0.0, -1.0 + (0.1 / np.pi), 0.0]])  # 对应 0.1 rad

    weight = sampler.compute_weights(center_pt, query_pt)

    delta_norm = (0.1 / np.pi)
    sigma_norm = 0.2
    print(f"Center (norm): {center_pt[0, 2].item():.2f}, Query (norm): {query_pt[0, 2].item():.2f}")
    print(f"计算权重: {weight.item():.6f}")
    expected = np.exp(-0.5 * ((delta_norm / sigma_norm) ** 2))
    print(f"理论权重: {expected:.6f}")

    if abs(weight.item() - expected) < 1e-4:
        print(">> 验证成功：周期性最短路径生效。")
    else:
        print(">> 验证失败：权重计算错误。")

    print("\n" + "-" * 30)
    print("测试 4: 基于真实数据边界的可视化")
    print("-" * 30)
    # 参考 trainer_depends/datasets/util_verify_dataset_4d_coords_range.py 的配置
    p_satinfo_json = '/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json'
    p_uav_geocsv = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv'

    if not os.path.exists(p_satinfo_json) or not os.path.exists(p_uav_geocsv):
        print("⚠️  未找到数据集路径，请先修改 p_satinfo_json / p_uav_geocsv 后再运行。")
    else:
        from trainer_depends.datasets.dataset_wingtra_4d import SatDataset
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor

        sat_dataset = SatDataset(
            p_satinfo_json=p_satinfo_json,
            p_uav_geocsv=p_uav_geocsv,
            imgsize2net=224,
        )

        coord_normer = CoordsNormProcessor(sat_dataset)
        scale_min, scale_max = sat_dataset.satimgsize_scale_to_refm_boundary
        sigma_scale = 0.3 * (scale_max - scale_min)
        # 与 compute_weight_matrix_from_norm 内部一致的 Sigma 归一化方式
        sigmas = coord_normer.get_normalized_sigmas(
            gs_sigma_nrc = sat_dataset.halfimg_radius_nrc,
            gs_sigma_radrot = np.pi / 180 * 10,
            gs_sigma_scale= sigma_scale
        )
        norm_std_devs = [
            sigmas['nrc'].item(),
            sigmas['nrc'].item(),
            sigmas['rot_linear'],
            sigmas['scale'].item(),
        ]

        sampler_real = NormalizedGaussianSampler(
            norm_std_devs=norm_std_devs,
            circular_dims=[2],
            device='cpu'
        )

        nr_min, nr_max = sat_dataset.nr2sample_min, sat_dataset.nr2sample_max
        nc_min, nc_max = sat_dataset.nc2sample_min, sat_dataset.nc2sample_max
        center_vis = torch.tensor([[
            nr_min,
            nc_min,
            np.pi,
            (scale_min + scale_max) * 0.5,
        ]], dtype=torch.float32)

        vis_dir = '/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results'
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, 'dynamic_boundary_sampling_real_data.png')

        visualize_dynamic_boundary_sampling(
            sampler=sampler_real,
            centers=center_vis,
            num_samples=2000,
            save_path=save_path,
            show_interactive=False,
            dim_names=['nr_norm', 'nc_norm', 'rot_norm', 'scale_norm'],
            coord_normer=coord_normer,
            centers_are_normalized=False
        )
