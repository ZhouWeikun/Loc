import torch
import matplotlib.pyplot as plt
import math  # 用于获取 pi
from typing import Dict, Tuple, Optional


def generate_pose_samples_for_batch(
        p_true_batch: torch.Tensor,
        rc_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        scale_bounds: Tuple[float, float],  # <-- 修正1: 提升为必需参数
        num_gaussian_samples: int,
        rc_std_dev: float,
        direction_std_dev_rad: float,
        log_scale_std_dev: float,
        num_uniform_samples: int = 0,
) -> torch.Tensor:
    """
    【最终健壮版本 v2 - 修正尺度边界】为一批4D位姿锚点生成各自的采样点。
    - RC/Direction/Scale 的局部采样结果都会被严格限制在边界内。

    参数:
    - ... (其他参数不变) ...
    - scale_bounds (Tuple): 尺度的绝对值边界 (s_min, s_max)，对所有采样策略都生效。
    - ...
    """
    # 修正2: 将断言移到函数开头，因为 scale_bounds 现在总是需要
    assert scale_bounds is not None, "必须提供 scale_bounds 以确保所有采样点都在有效范围内。"

    device = p_true_batch.device
    dtype = torch.float32
    batch_size = p_true_batch.shape[0]
    two_pi = 2 * torch.pi
    (r_min, r_max), (c_min, c_max) = rc_bounds
    s_min, s_max = scale_bounds  # <-- 在这里提前解包，方便后续使用

    # --- 策略1: 局部高斯/对数正态采样 ---
    r_t_batch, c_t_batch, d_t_rad_batch, s_t_batch = [t.unsqueeze(1) for t in p_true_batch.T]

    # 1. RC 采样 (加性高斯)
    mean_rc_batch = torch.cat((r_t_batch, c_t_batch), dim=1)
    sampled_rc_gauss = torch.randn(batch_size, num_gaussian_samples, 2, device=device,
                                   dtype=dtype) * rc_std_dev + mean_rc_batch.unsqueeze(1)
    sampled_rc_gauss[..., 0] = torch.clamp(sampled_rc_gauss[..., 0], r_min, r_max)
    sampled_rc_gauss[..., 1] = torch.clamp(sampled_rc_gauss[..., 1], c_min, c_max)

    # 2. 方向采样 (加性高斯 + 周期处理)
    sampled_d_rad_gauss = torch.randn(batch_size, num_gaussian_samples, device=device,
                                      dtype=dtype) * direction_std_dev_rad + d_t_rad_batch
    sampled_d_rad_gauss = (sampled_d_rad_gauss % two_pi + two_pi) % two_pi

    # 3. 尺度采样 (乘性高斯, 即对数正态)
    epsilon = 1e-8
    log_s_t = torch.log(s_t_batch + epsilon)
    sampled_log_s = torch.randn(batch_size, num_gaussian_samples, device=device,
                                dtype=dtype) * log_scale_std_dev + log_s_t
    sampled_s_gauss = torch.exp(sampled_log_s)

    # 核心修正3: 对尺度采样结果进行clamp
    sampled_s_gauss = torch.clamp(sampled_s_gauss, s_min, s_max)

    gaussian_samples = torch.cat((
        sampled_rc_gauss,
        sampled_d_rad_gauss.unsqueeze(2),
        sampled_s_gauss.unsqueeze(2)
    ), dim=2)

    if num_uniform_samples == 0:
        return gaussian_samples

    # --- 策略2: 全局均匀采样 (不变) ---
    sampled_r_uni = (r_max - r_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + r_min
    sampled_c_uni = (c_max - c_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + c_min
    sampled_d_rad_uni = two_pi * torch.rand(num_uniform_samples, device=device, dtype=dtype)
    sampled_s_uni = (s_max - s_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + s_min
    uniform_samples = torch.column_stack((sampled_r_uni, sampled_c_uni, sampled_d_rad_uni, sampled_s_uni))

    # --- 整合结果 (不变) ---
    expanded_uniform_samples = uniform_samples.unsqueeze(0).expand(batch_size, -1, -1)
    final_samples = torch.cat([gaussian_samples, expanded_uniform_samples], dim=1)

    return final_samples


def visualize_sampling_results(
        samples_tensor: torch.Tensor,
        anchors_tensor: torch.Tensor,
        rc_std_dev: float,
        log_scale_std_dev: float,
        direction_std_dev_rad: float
):
    """
    A comprehensive function to visualize and validate the sampling results.
    (All labels and titles are in English).
    """
    # --- Preparation ---
    plt.style.use('seaborn-v0_8-whitegrid')
    samples_np = samples_tensor.cpu().numpy()
    anchors_np = anchors_tensor.cpu().numpy()
    batch_size = samples_np.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, batch_size))

    # Create a 2x2 plot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Comprehensive Visualization of Sampling Results", fontsize=20)

    # --- Loop to plot data for each anchor ---
    for i in range(batch_size):
        anchor = anchors_np[i]
        samples = samples_np[i]
        color = colors[i]

        # 1. RC Distribution (Top-Left)
        ax = axes[0, 0]
        ax.scatter(samples[:, 1], samples[:, 0], s=5, alpha=0.1, color=color, label=f'Anchor {i} Samples')
        ax.scatter(anchor[1], anchor[0], s=200, marker='*', edgecolor='black', color=color,
                   label=f'Anchor {i} True Pose')
        # Draw 2-sigma error ellipse
        ellipse = Ellipse((anchor[1], anchor[0]), width=rc_std_dev * 4, height=rc_std_dev * 4,
                          edgecolor=color, fc='None', lw=2, linestyle='--')
        ax.add_patch(ellipse)

        # 2. Direction Distribution (Top-Right)
        ax = axes[0, 1]
        ax.hist(samples[:, 2], bins=60, range=(0, 2 * math.pi), density=True, alpha=0.7, color=color,
                label=f'Anchor {i} Samples')
        ax.axvline(anchor[2], linestyle='--', color=color, lw=2, label=f'Anchor {i} True Dir')

        # 3. Scale Distribution (Bottom-Left)
        ax = axes[1, 0]
        ax.hist(samples[:, 3], bins=100, density=True, alpha=0.7, color=color, label=f'Anchor {i} Samples')
        ax.axvline(anchor[3], linestyle='--', color=color, lw=2, label=f'Anchor {i} True Scale')

        # 4. Log-Scale Ratio Distribution for Validation (Bottom-Right)
        ax = axes[1, 1]
        log_ratios = np.log(samples[:, 3] / anchor[3])
        # Plot the distribution of the samples
        ax.hist(log_ratios, bins=100, density=True, alpha=0.7, color=color, label=f'Anchor {i} Sample Log-Ratios')
        # Plot the theoretical Gaussian curve
        x = np.linspace(-3 * log_scale_std_dev, 3 * log_scale_std_dev, 100)
        p = norm.pdf(x, 0, log_scale_std_dev)
        ax.plot(x, p, 'k--', linewidth=2, label='Theoretical Gaussian' if i == 0 else "")
        # Display statistics
        mean, std = np.mean(log_ratios), np.std(log_ratios)
        ax.text(0.05, 0.95 - i * 0.1, f'Anchor {i}: Mean={mean:.3f}, Std={std:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color=color)

    # --- Formatting the plots ---
    axes[0, 0].set_title('RC Distribution with 2-Sigma Ellipse')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axis('equal')

    axes[0, 1].set_title('Direction Distribution')
    axes[0, 1].set_xlabel('Direction (radians)')

    axes[1, 0].set_title('Scale Distribution')
    axes[1, 0].set_xlabel('Scale Value')
    axes[1, 0].set_ylabel('Density')

    axes[1, 1].set_title('Validation: Log-Ratio Distribution of Scale')
    axes[1, 1].set_xlabel('Log Ratio')

    for ax in axes.flat:
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


from matplotlib.patches import Ellipse
from scipy.stats import norm
import numpy as np
if __name__ == '__main__':
    # --- 参数定义 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RC_STD_DEV = 150.0
    LOG_SCALE_STD_DEV = 0.2
    DIRECTION_STD_DEV_RAD = math.pi / 4

    p_true_batch_tensor = torch.tensor([
        [768.2, 1024.5, math.pi / 4, 1.5],
        [300.0, 500.0, math.pi * 1.5, 0.2]
    ], device=device, dtype=torch.float32)

    bounds_rc = ((0, 1080), (0, 1920))

    # --- 1. 生成样本 ---
    final_samples_tensor = generate_pose_samples_for_batch(
        p_true_batch=p_true_batch_tensor,
        rc_bounds=bounds_rc,
        num_gaussian_samples=5000,
        rc_std_dev=RC_STD_DEV,
        direction_std_dev_rad=DIRECTION_STD_DEV_RAD,
        log_scale_std_dev=LOG_SCALE_STD_DEV,
    )

    # --- 2. 调用优化后的可视化函数 ---
    visualize_sampling_results(
        samples_tensor=final_samples_tensor,
        anchors_tensor=p_true_batch_tensor,
        rc_std_dev=RC_STD_DEV,
        log_scale_std_dev=LOG_SCALE_STD_DEV,
        direction_std_dev_rad=DIRECTION_STD_DEV_RAD
    )