#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分层坐标采样器 (Span-based Ratio Strategy)

用于在训练MetricNet时，基于GT坐标进行分层随机采样，生成有层次的负样本。
改进策略：
1. 基于边界跨度(Span)和比例(Ratio)计算标准差，适应不同尺度的场景。
2. Bottom层支持绝对阈值控制，确保高精度的收敛能力。

修改记录：
- 移除了 base_std + multiplier 模式
- 引入 span * ratio 模式
- 引入 bottom_abs_rc_std 绝对控制
"""

import torch
import os
from typing import List, Dict, Any, Tuple, Optional
from math import pi
import numpy as np


class HierarchicalCoordSampler:
    """
    分层坐标采样器类 (Ratio Based)

    用于生成围绕GT坐标的分层采样点。
    标准差计算公式： sigma = Span * Ratio
    """

    def __init__(
            self,
            rc_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
            rot_bounds: Tuple[float, float],
            scale_bounds: Tuple[float, float],
            # --- 新增/修改的参数 ---
            bottom_abs_rc_std: Optional[float] = None,  # Bottom层的RC绝对标准差。如果>0则覆盖比例策略
            # --------------------
            num_uniform_samples: int = 0,
            safety_factor: float = 3.0,
            device: str = 'cuda'
    ):
        """
        初始化分层坐标采样器

        Args:
            rc_bounds: (row, col) 的边界范围 ((r_min, r_max), (c_min, c_max))
            rot_bounds: rotation 的边界范围 (rot_min, rot_max)，单位：弧度
            scale_bounds: scale 的边界范围 (s_min, s_max)
            bottom_abs_rc_std: [关键] 碗底层的绝对RC标准差。
                               - 如果设置(>0)，Bottom层将忽略跨度比例，强制使用此精细度。
                               - 适用于在大地图中仍需保持像素级精度的场景。
            num_uniform_samples: 全局均匀采样的数量
            safety_factor: 安全系数，用于自适应标准差计算
            device: 设备 ('cuda' 或 'cpu')
        """
        self.rc_bounds = rc_bounds
        self.rot_bounds = rot_bounds
        self.scale_bounds = scale_bounds
        self.bottom_abs_rc_std = bottom_abs_rc_std
        self.num_uniform_samples = num_uniform_samples
        self.safety_factor = safety_factor
        self.device = device

        # 提取边界值
        (self.r_min, self.r_max), (self.c_min, self.c_max) = rc_bounds
        self.rot_min, self.rot_max = rot_bounds
        self.s_min, self.s_max = scale_bounds

        # 1. 计算各维度的跨度 (Span)
        # RC跨度：取较短边的长度作为基准，比较保守
        self.span_rc = min(self.r_max - self.r_min, self.c_max - self.c_min)

        # 旋转跨度：固定 2pi
        self.span_rot = 2 * pi

        # 尺度跨度：在对数空间计算
        epsilon = 1e-8
        self.span_log_scale = torch.log(torch.tensor(self.s_max + epsilon)) - \
                              torch.log(torch.tensor(self.s_min + epsilon))
        self.span_log_scale = self.span_log_scale.item()

        # 2. 生成基于比例的分层配置
        self.sampling_configs = self._get_ratio_based_configs()

        # 计算总采样数
        self.num_gaussian_samples = sum(cfg['num_samples'] for cfg in self.sampling_configs)
        self.total_samples_per_gt = self.num_gaussian_samples + num_uniform_samples

        self._print_init_info()

    def _get_ratio_based_configs(self) -> List[Dict[str, Any]]:
        """
        生成基于边界跨度比例的采样策略
        """
        # 定义每一层的比例 (Standard Deviation Ratios)
        # 数值代表：标准差占总跨度的百分比

        # 1. 配置 Bottom 层 (碗底)
        # 默认极小比例 (1%)，但如果提供了 absolute std，则会在循环中覆盖
        bottom_cfg = {
            'name': 'bottom',
            'num_samples': 64,
            'rc_ratio': 0.01,  # 1%
            'rot_ratio': 0.01,  # 1%,3.6°
            'scale_ratio': 0.01  # 1%
        }

        # 2. 配置 Slope 层 (碗壁)
        # 捕捉梯度 (5%)
        slope_cfg = {
            'name': 'slope',
            'num_samples': 256,
            'rc_ratio': 0.05,  # 5%
            'rot_ratio': 0.05,  # 5%
            'scale_ratio': 0.05
        }

        # 3. 配置 Rim 层 (边缘)
        # 全局收敛 (25%)
        rim_cfg = {
            'name': 'rim',
            'num_samples': 384,
            'rc_ratio': 0.25,  # 20%
            'rot_ratio': 0.25,  # 20%
            'scale_ratio': 0.25
        }

        raw_configs = [bottom_cfg, slope_cfg, rim_cfg]
        final_configs = []

        for layer in raw_configs:
            # 基础计算：Span * Ratio
            rc_std = self.span_rc * layer['rc_ratio']
            rot_std = self.span_rot * layer['rot_ratio']
            log_s_std = self.span_log_scale * layer['scale_ratio']

            # === 特殊处理 Bottom 层 ===
            if layer['name'] == 'bottom':
                # 如果启用了绝对阈值，覆盖 RC std
                if self.bottom_abs_rc_std is not None and self.bottom_abs_rc_std > 0:
                    rc_std = self.bottom_abs_rc_std

            config = {
                'name': layer['name'],
                'num_samples': layer['num_samples'],
                'rc_std_dev': rc_std,
                'rot_std_dev_rad': rot_std,
                'log_scale_std_dev': log_s_std,
                # 保存比例信息用于debug显示
                'meta_ratio': layer['rc_ratio'] if layer['name'] != 'bottom' or (
                            self.bottom_abs_rc_std is None or self.bottom_abs_rc_std <= 0) else 'ABS'
            }
            final_configs.append(config)

        return final_configs

    def _print_init_info(self):
        print(f"\n初始化分层坐标采样器 (Span-Based):")
        print(f"  - 跨度信息:")
        print(
            f"    RC Span: {self.span_rc:.2f} (Rows [{self.r_min:.2f}, {self.r_max:.2f}], Cols [{self.c_min:.2f}, {self.c_max:.2f}])")
        print(f"    Rot Span: {self.span_rot:.2f} rad")
        print(f"    Log Scale Span: {self.span_log_scale:.2f}")

        print(f"  - 分层采样配置:")
        for cfg in self.sampling_configs:
            ratio_info = f"Ratio={cfg['meta_ratio']:.1%}" if isinstance(cfg['meta_ratio'],
                                                                        float) else f"Mode={cfg['meta_ratio']}"
            print(f"    * {cfg['name']:<7}: {cfg['num_samples']} samples | "
                  f"rc_std={cfg['rc_std_dev']:.4f} ({ratio_info}), "
                  f"rot_std={cfg['rot_std_dev_rad']:.4f}, "
                  f"log_s_std={cfg['log_scale_std_dev']:.4f}")

        print(f"  - 全局均匀采样: {self.num_uniform_samples} samples")
        print(f"  - 每个GT点总采样数: {self.total_samples_per_gt}\n")

    def sample(self, gt_coords_4d: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        基于GT坐标进行分层采样 (代码逻辑保持不变，依赖 config 中的 std 值)
        """
        assert gt_coords_4d.dim() == 2 and gt_coords_4d.shape[1] == 4, \
            f"gt_coords_4d应为 [B, 4]，但得到 {gt_coords_4d.shape}"

        device = gt_coords_4d.device
        dtype = gt_coords_4d.dtype
        batch_size = gt_coords_4d.shape[0]
        two_pi = 2 * pi

        # 分解GT坐标 [B, 1]
        r_t_batch = gt_coords_4d[:, 0:1]
        c_t_batch = gt_coords_4d[:, 1:2]
        rot_t_batch = gt_coords_4d[:, 2:3]
        s_t_batch = gt_coords_4d[:, 3:4]

        # === 分层高斯采样 ===
        all_gaussian_samples = []

        for config in self.sampling_configs:
            num_samples = config['num_samples']
            rc_std = config['rc_std_dev']
            rot_std = config['rot_std_dev_rad']
            log_s_std = config['log_scale_std_dev']

            if num_samples == 0:
                continue

            # --- 1. RC采样（带自适应标准差） ---
            # 计算到边界的距离
            dist_to_r_min = r_t_batch - self.r_min
            dist_to_r_max = self.r_max - r_t_batch
            dist_to_c_min = c_t_batch - self.c_min
            dist_to_c_max = self.c_max - c_t_batch

            # 计算最大允许的标准差（防止采样越界）
            max_r_std = torch.minimum(dist_to_r_min, dist_to_r_max) / self.safety_factor
            max_c_std = torch.minimum(dist_to_c_min, dist_to_c_max) / self.safety_factor
            max_rc_std_batch = torch.cat((max_r_std, max_c_std), dim=1).unsqueeze(1)  # [B, 1, 2]

            # 自适应标准差：取配置值和最大允许值的较小者
            effective_rc_std = torch.minimum(
                torch.tensor(rc_std, device=device, dtype=dtype),
                torch.broadcast_to(max_rc_std_batch, (batch_size, 1, 2))
            )

            if verbose:
                print(f"\n层级 '{config['name']}':")
                print(f"  配置rc_std: {rc_std:.4f}")
                print(f"  最大允许rc_std: {max_rc_std_batch[0, 0].mean().item():.4f}")
                mean_eff = effective_rc_std[0, 0].mean().item()
                print(f"  实际有效rc_std: {mean_eff:.4f}")
                if mean_eff < rc_std * 0.99:
                    print(f"  ⚠️  警告: 实际标准差被限制了 {(1 - mean_eff / rc_std) * 100:.1f}% (靠近边界)")

            # 采样RC坐标
            mean_rc_batch = torch.cat((r_t_batch, c_t_batch), dim=1)  # [B, 2]
            sampled_rc = torch.randn(batch_size, num_samples, 2, device=device,
                                     dtype=dtype) * effective_rc_std + mean_rc_batch.unsqueeze(1)
            sampled_rc[..., 0] = torch.clamp(sampled_rc[..., 0], self.r_min, self.r_max)
            sampled_rc[..., 1] = torch.clamp(sampled_rc[..., 1], self.c_min, self.c_max)

            # --- 2. 旋转采样 ---
            sampled_rot = torch.randn(batch_size, num_samples, device=device, dtype=dtype) * rot_std + rot_t_batch
            sampled_rot = (sampled_rot + pi) % two_pi - pi

            # --- 3. 尺度采样 ---
            epsilon = 1e-8
            log_s_t = torch.log(s_t_batch + epsilon)
            log_s_min = torch.log(torch.tensor(self.s_min, device=device, dtype=dtype) + epsilon)
            log_s_max = torch.log(torch.tensor(self.s_max, device=device, dtype=dtype) + epsilon)

            dist_to_log_min = log_s_t - log_s_min
            dist_to_log_max = log_s_max - log_s_t
            max_log_s_std = torch.minimum(dist_to_log_min, dist_to_log_max) / self.safety_factor

            effective_log_s_std = torch.minimum(
                torch.tensor(log_s_std, device=device, dtype=dtype),
                max_log_s_std
            )

            sampled_log_s = torch.randn(batch_size, num_samples, 1, device=device,
                                        dtype=dtype) * effective_log_s_std.unsqueeze(1) + log_s_t.unsqueeze(1)
            sampled_s = torch.exp(sampled_log_s)
            sampled_s = torch.clamp(sampled_s, self.s_min, self.s_max)

            # 拼接
            gaussian_samples_layer = torch.cat((sampled_rc, sampled_rot.unsqueeze(2), sampled_s), dim=2)
            all_gaussian_samples.append(gaussian_samples_layer)

        # 拼接所有高斯层级
        final_gaussian_samples = torch.cat(all_gaussian_samples, dim=1)

        # === 全局均匀采样 ===
        if self.num_uniform_samples == 0:
            return final_gaussian_samples

        total_uniform_samples = batch_size * self.num_uniform_samples
        sampled_r_uni = (self.r_max - self.r_min) * torch.rand(total_uniform_samples, device=device,
                                                               dtype=dtype) + self.r_min
        sampled_c_uni = (self.c_max - self.c_min) * torch.rand(total_uniform_samples, device=device,
                                                               dtype=dtype) + self.c_min
        sampled_rot_uni = (two_pi * torch.rand(total_uniform_samples, device=device, dtype=dtype)) - pi
        sampled_s_uni = (self.s_max - self.s_min) * torch.rand(total_uniform_samples, device=device,
                                                               dtype=dtype) + self.s_min

        uniform_samples_flat = torch.stack([sampled_r_uni, sampled_c_uni, sampled_rot_uni, sampled_s_uni], dim=1)
        uniform_samples_batch = uniform_samples_flat.view(batch_size, self.num_uniform_samples, 4)

        final_samples = torch.cat([final_gaussian_samples, uniform_samples_batch], dim=1)
        return final_samples

    def get_sampling_info(self) -> Dict[str, Any]:
        return {
            'rc_bounds': self.rc_bounds,
            'span_rc': self.span_rc,
            'sampling_configs': self.sampling_configs,
            'bottom_abs_rc_std': self.bottom_abs_rc_std
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_hierarchical_sampler_from_dataset(
        sat_dataset,
        bottom_abs_rc_std: Optional[float] = None,  # 传递绝对阈值
        num_uniform_samples: int = 0,
        device: str = 'cuda'
) -> HierarchicalCoordSampler:
    """
    从数据集自动创建分层采样器 (Ratio Based)

    Args:
        sat_dataset: 卫星数据集对象
        bottom_abs_rc_std: 碗底层的绝对RC标准差（可选）
        num_uniform_samples: 全局均匀采样数量
        device: 设备

    Returns:
        HierarchicalCoordSampler实例
    """
    r_min = sat_dataset.nr2sample_min
    r_max = sat_dataset.nr2sample_max
    c_min = sat_dataset.nc2sample_min
    c_max = sat_dataset.nc2sample_max

    rot_min, rot_max = -pi, pi
    s_min, s_max = sat_dataset.satimgsize_scale_to_refm_boundary

    sampler = HierarchicalCoordSampler(
        rc_bounds=((r_min, r_max), (c_min, c_max)),
        rot_bounds=(rot_min, rot_max),
        scale_bounds=(s_min, s_max),
        bottom_abs_rc_std=bottom_abs_rc_std,  # 传入新参数
        num_uniform_samples=num_uniform_samples,
        device=device
    )

    return sampler


# ============================================================================
# 可视化验证函数 (保持不变，只是为了完整性再次包含)
# ============================================================================
def visualize_hierarchical_sampling(
        sampler: HierarchicalCoordSampler,
        gt_coord_4d: torch.Tensor = None,
        n_samples: int = 1,
        save_path: str = None,
        show_interactive: bool = False
):
    """
    可视化分层采样的结果。

    布局修改：
    - 左图: (Col, Row) 2D散点图
    - 中图: Rotation 1D分布直方图
    - 右图: Scale 1D分布直方图
    """

    print(f"\n{'=' * 80}")
    print(f"分层采样可视化验证")
    print(f"{'=' * 80}\n")

    if save_path is None:
        base_name = "vis_sampling"
    else:
        base_name = os.path.splitext(save_path)[0]

    html_path = f"{base_name}.html"
    png_path = f"{base_name}.png"

    # 1. 生成或使用GT坐标
    if gt_coord_4d is None:
        (r_min, r_max), (c_min, c_max) = sampler.rc_bounds
        gt_coord_4d = torch.tensor([
            (r_min + r_max) / 2,
            (c_min + c_max) / 2,
            0.0,
            (sampler.s_min + sampler.s_max) / 2
        ], device=sampler.device)

    gt_np = gt_coord_4d.cpu().numpy()
    print(f"GT坐标: Row={gt_np[0]:.2f}, Col={gt_np[1]:.2f}, Rot={gt_np[2]:.2f}, Scale={gt_np[3]:.2f}")

    # 2. 批量采样
    gt_coords_batch = gt_coord_4d.unsqueeze(0).repeat(n_samples, 1)
    # 启用verbose以检查是否有std被限制的情况
    sampled_coords = sampler.sample(gt_coords_batch, verbose=True)

    # 3. 数据准备
    layer_stats = []
    offset = 0
    for config in sampler.sampling_configs:
        num = config['num_samples']
        if num > 0:
            samples_np = sampled_coords[:, offset:offset + num, :].cpu().numpy()
            # 扁平化所有batch的样本以便统计分布
            flat_samples = samples_np.reshape(-1, 4)
            layer_stats.append({
                'name': config['name'],
                'samples': flat_samples
            })
            offset += num

    if sampler.num_uniform_samples > 0:
        samples_np = sampled_coords[:, -sampler.num_uniform_samples:, :].cpu().numpy()
        flat_samples = samples_np.reshape(-1, 4)
        layer_stats.append({
            'name': 'uniform',
            'samples': flat_samples
        })

    # 4. Matplotlib 2D 可视化 (修改为直方图统计)
    try:
        import matplotlib.pyplot as plt

        # 创建 1行3列 的画布
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 定义颜色 (Bottom: Green, Slope: Blue, Rim: Orange, Uniform: Gray)
        colors = {'bottom': 'green', 'slope': 'blue', 'rim': 'orange', 'uniform': 'gray'}

        # --- Subplot 1: (Col, Row) 2D 散点图 ---
        ax = axes[0]
        for stats in layer_stats:
            name = stats['name']
            data = stats['samples']
            # alpha 随层级调整，bottom最不透明
            alpha = 0.6 if name == 'bottom' else 0.4
            size = 20 if name == 'bottom' else 10
            ax.scatter(data[:, 1], data[:, 0], s=size, c=colors.get(name, 'black'),
                       alpha=alpha, label=name, edgecolors='none')

        # 绘制GT
        ax.scatter(gt_np[1], gt_np[0], s=150, c='red', marker='*',
                   edgecolors='black', linewidths=1.5, label='GT', zorder=10)

        ax.set_xlabel('Col (X)')
        ax.set_ylabel('Row (Y)')
        ax.set_title('Spatial Distribution (Row vs Col)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # 保持长宽比，以免地图变形
        # ax.set_aspect('equal')
        ax.set_xlim(sampler.c_min, sampler.c_max)
        ax.set_ylim(sampler.r_min, sampler.r_max)

        # --- Subplot 2: Rotation 1D 直方图 ---
        ax = axes[1]
        for stats in layer_stats:
            name = stats['name']
            rot_data = stats['samples'][:, 2]
            # 使用 density=True 以便对比不同数量级的分布形状
            ax.hist(rot_data, bins=50, range=(-pi, pi), density=True,
                    color=colors.get(name, 'black'), alpha=0.4, label=name, histtype='stepfilled')

        # 绘制GT线
        ax.axvline(gt_np[2], color='red', linestyle='--', linewidth=2, label='GT')

        ax.set_xlabel('Rotation (rad)')
        ax.set_ylabel('Density')
        ax.set_title('Rotation Distribution (Histogram)')
        ax.set_xlim(-pi, pi)
        ax.grid(True, alpha=0.3)

        # --- Subplot 3: Scale 1D 直方图 ---
        ax = axes[2]
        for stats in layer_stats:
            name = stats['name']
            scale_data = stats['samples'][:, 3]
            ax.hist(scale_data, bins=50, range=(sampler.s_min, sampler.s_max), density=True,
                    color=colors.get(name, 'black'), alpha=0.4, label=name, histtype='stepfilled')

        # 绘制GT线
        ax.axvline(gt_np[3], color='red', linestyle='--', linewidth=2, label='GT')

        ax.set_xlabel('Scale')
        ax.set_ylabel('Density')
        ax.set_title('Scale Distribution (Histogram)')
        ax.set_xlim(sampler.s_min, sampler.s_max)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"✅ 统计分布图已保存到: {png_path}")

        # 控制显示
        if show_interactive:
            plt.show()
        else:
            plt.close(fig)

    except ImportError:
        print("⚠️  需要安装matplotlib用于2D可视化")

    # 5. 3D Plotly 可视化 (保持原样，只做少量清理)
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # GT点
        fig.add_trace(go.Scatter3d(
            x=[gt_np[0]], y=[gt_np[1]], z=[gt_np[2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='GT'
        ))

        # 各层采样点 (为了性能，如果点太多进行降采样)
        max_points_display = 2000

        for stats in layer_stats:
            name = stats['name']
            data = stats['samples']

            if len(data) > max_points_display:
                indices = np.random.choice(len(data), max_points_display, replace=False)
                data = data[indices]

            fig.add_trace(go.Scatter3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors.get(name, 'purple'), opacity=0.5),
                name=name
            ))

        fig.update_layout(
            title='3D Sampling Visualization (Row, Col, Rot)',
            scene=dict(xaxis_title='Row', yaxis_title='Col', zaxis_title='Rot'),
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
        'layer_stats': layer_stats,
        'num_uniform_samples': sampler.num_uniform_samples,
        'total_samples': sampler.total_samples_per_gt,
        'gt_coord': gt_coord_4d.cpu().numpy()
    }

if __name__ == "__main__":
    """
    独立验证脚本
    """
    print("\n" + "=" * 80)
    print("Ratio-Based 分层采样器 - 独立验证")
    print("=" * 80 + "\n")


    # CASE B: 使用绝对阈值 (std = 0.5)
    print("\n>>> 测试案例 B: 绝对阈值模式 (Bottom std = 0.5)")
    sampler_abs = HierarchicalCoordSampler(
        rc_bounds=((0, 1), (0, 1)),
        rot_bounds=(-pi, pi),
        scale_bounds=(0.4, 1.0),
        bottom_abs_rc_std=0.015,  # 强制极小值
        num_uniform_samples=200,
        device='cpu'
    )

    # 验证 std 值
    print(f"\n对比 Bottom 层 RC Std:")
    # print(f"  Case A (Ratio): {sampler_ratio.sampling_configs[0]['rc_std_dev']:.4f}")
    print(f"  Case B (Abs):   {sampler_abs.sampling_configs[0]['rc_std_dev']:.4f}")

    # 2. 创建一个测试用的GT坐标
    test_gt_coord = torch.tensor([0.5, 0.5, 1.2, 0.7])  # (row, col, rot, scale)

    # 3. 可视化采样结果 (默认为 False，只保存文件)
    results = visualize_hierarchical_sampling(
        sampler=sampler_abs,
        gt_coord_4d=test_gt_coord,
        n_samples=1,
        save_path='hierarchical_sampling_verification.png',
        show_interactive=False
    )

    print("验证完成！")
    print(f"层级数量: {len(results['layer_stats'])}")
    print(f"总采样点数: {results['total_samples']}")