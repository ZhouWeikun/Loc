import torch
from typing import List, Dict, Any, Tuple
import math
from math import pi

def get_stratified_sampling_configs(
        base_rc_std: float,
        base_dir_std_rad: float = pi/4,
        base_log_s_std: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    动态生成用于分层采样的配置列表。
    这个函数封装了我们推荐的“碗底-碗壁-边缘”三层采样策略。

    参数:
    - base_rc_std (float): 用于RC（行列）维度的基准标准差。
    - base_dir_std_rad (float): 用于方向维度的基准标准差（弧度）。
    - base_log_s_std (float): 用于对数尺度维度的基准标准差。

    返回:
    - List[Dict[str, Any]]: 一个配置列表，可直接用于 generate_pose_samples_stratified 函数。
    """

    # 在这里定义我们的核心策略：层级、样本数、以及相对于基准标准差的乘数。
    # 未来如果想调整策略，只需修改这里即可。
    strategy_definition = [
        # 1. “碗底” (Bottom): 极小标准差，精细刻画最低点
        {'name': 'bottom', 'num_samples': 64, 'rc_multiplier': 1.0, 'dir_multiplier': 1, 'scale_multiplier': 1},

        # 2. “碗壁” (Slope): 中等标准差，学习UDF坡度
        {'name': 'slope', 'num_samples': 128, 'rc_multiplier': 10, 'dir_multiplier': 2, 'scale_multiplier': 1.2},

        # 3. “边缘” (Rim): 较大标准差，学习过渡区域
        {'name': 'rim', 'num_samples': 128, 'rc_multiplier': 20, 'dir_multiplier': 4, 'scale_multiplier': 1.4},
    ]

    # 根据传入的基准值和策略定义，动态生成最终配置
    final_configs = []
    for layer in strategy_definition:
        config = {
            'name': layer['name'],
            'num_samples': layer['num_samples'],
            'rc_std_dev': base_rc_std * layer['rc_multiplier'],
            'direction_std_dev_rad': base_dir_std_rad * layer['dir_multiplier'],
            'log_scale_std_dev': base_log_s_std * layer['scale_multiplier']
        }
        final_configs.append(config)

    return final_configs


def generate_pose_samples_hierarchical(
        p_true_batch: torch.Tensor,
        rc_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        scale_bounds: Tuple[float, float],
        # --- 核心修改1: 使用配置列表代替离散参数 ---
        sampling_configs: List[Dict[str, Any]],
        num_uniform_samples: int = 0,
) -> torch.Tensor:
    """
    【v3 - 分层采样版本】为一批4D位姿锚点生成更有层次的采样点。
    - 使用一个配置列表来定义多个高斯采样层级（如近、中、远）。
    - 这样可以更好地定义UDF场的形状，解决“弥散”问题。

    参数:
    - p_true_batch: 真值位姿锚点, shape [B, 4] -> (r, c, d_rad, s)
    - rc_bounds: (row, col) 的边界
    - scale_bounds: 尺度的边界 (s_min, s_max)
    - sampling_configs: 一个列表，每一项是一个字典，定义一个高斯采样层。
        例如: [
            {'num_samples': 64, 'rc_std_dev': 10.0, 'direction_std_dev_rad': 0.1, 'log_scale_std_dev': 0.05},
            {'num_samples': 128, 'rc_std_dev': 50.0, 'direction_std_dev_rad': 0.5, 'log_scale_std_dev': 0.2},
        ]
    - num_uniform_samples: 全局均匀采样的数量
    """
    assert scale_bounds is not None, "必须提供 scale_bounds。"

    device = p_true_batch.device
    dtype = torch.float32
    batch_size = p_true_batch.shape[0]
    two_pi = 2 * torch.pi
    (r_min, r_max), (c_min, c_max) = rc_bounds
    s_min, s_max = scale_bounds
    safety_factor = 3.0

    # 将真值位姿分解，方便后续广播
    r_t_batch, c_t_batch, d_t_rad_batch, s_t_batch = [t.unsqueeze(1) for t in p_true_batch.T]

    # --- 核心修改2: 循环处理每个高斯采样层级 ---
    all_gaussian_samples = []
    for config in sampling_configs:
        # 从配置中获取该层级的参数
        num_samples = config['num_samples']
        rc_std = config['rc_std_dev']
        dir_std = config['direction_std_dev_rad']
        log_s_std = config['log_scale_std_dev']

        # --- ★★★ 核心修改: 自适应标准差逻辑 (仅针对RC维度) ★★★ ---
        # 1. 计算每个真值点到四个边界的距离
        dist_to_r_min = r_t_batch - r_min
        dist_to_r_max = r_max - r_t_batch
        dist_to_c_min = c_t_batch - c_min
        dist_to_c_max = c_max - c_t_batch

        # 2. 对于R和C，分别取到两个边界中更近的那个距离
        #    我们不希望标准差太大，以至于采样点轻易就越过最近的边界
        #    除以2.0是一个安全系数，确保大约95%的样本（在2-sigma内）落在边界内
        max_r_std = torch.minimum(dist_to_r_min, dist_to_r_max) / safety_factor
        max_c_std = torch.minimum(dist_to_c_min, dist_to_c_max) / safety_factor

        # 3. 构造一个形状为 [B, 1, 2] 的最大标准差张量
        max_rc_std_batch = torch.cat((max_r_std, max_c_std), dim=1).unsqueeze(1)

        # 4. 最终使用的标准差，是配置中给定的值和我们算出的最大值中，较小的那个
        #    使用 torch.broadcast_to 确保形状匹配
        effective_rc_std = torch.minimum(
            torch.tensor(rc_std, device=device, dtype=dtype),
            torch.broadcast_to(max_rc_std_batch, (batch_size, 1, 2))
        )
        # --- 修改结束 ---

        # 1. RC 采样 (使用新的 effective_rc_std)
        mean_rc_batch = torch.cat((r_t_batch, c_t_batch), dim=1)
        sampled_rc = torch.randn(batch_size, num_samples, 2, device=device,
                                 dtype=dtype) * effective_rc_std + mean_rc_batch.unsqueeze(1)
        # 即使有了自适应标准差，clamp 仍然是必须的，以防万一有样本超出
        sampled_rc[..., 0] = torch.clamp(sampled_rc[..., 0], r_min, r_max)
        sampled_rc[..., 1] = torch.clamp(sampled_rc[..., 1], c_min, c_max)

        # 2. 方向采样 (加性高斯 + 周期处理)
        sampled_d_rad = torch.randn(batch_size, num_samples, device=device, dtype=dtype) * dir_std + d_t_rad_batch
        sampled_d_rad = (sampled_d_rad % two_pi + two_pi) % two_pi

        # 3. 尺度采样 (对数正态)
        # --- ★★★ 新增: 尺度的自适应标准差 (在对数空间中) ★★★ ---
        epsilon = 1e-8
        log_s_t = torch.log(s_t_batch + epsilon)
        log_s_min = torch.log(torch.tensor(s_min, device=device, dtype=dtype) + epsilon)
        log_s_max = torch.log(torch.tensor(s_max, device=device, dtype=dtype) + epsilon)

        dist_to_log_min = log_s_t - log_s_min
        dist_to_log_max = log_s_max - log_s_t

        # 安全系数2.0确保大部分样本落在界内
        max_log_s_std = torch.minimum(dist_to_log_min, dist_to_log_max) / safety_factor

        # 最终使用的标准差是配置值和最大允许值中的较小者
        effective_log_s_std = torch.minimum(
            torch.tensor(log_s_std, device=device, dtype=dtype),
            max_log_s_std
        )

        sampled_log_s = torch.randn(batch_size, num_samples, 1, device=device, dtype=dtype) * effective_log_s_std.unsqueeze(1) + log_s_t.unsqueeze(1)
        sampled_s = torch.exp(sampled_log_s)
        sampled_s = torch.clamp(sampled_s, s_min, s_max)

        gaussian_samples_layer = torch.cat((sampled_rc, sampled_d_rad.unsqueeze(2), sampled_s), dim=2)
        all_gaussian_samples.append(gaussian_samples_layer)

    # 将所有高斯层级的样本拼接起来
    final_gaussian_samples = torch.cat(all_gaussian_samples, dim=1)

    # --- 策略2: 全局均匀采样 (保持不变) ---
    if num_uniform_samples == 0:
        return final_gaussian_samples

    sampled_r_uni = (r_max - r_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + r_min
    sampled_c_uni = (c_max - c_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + c_min
    sampled_d_rad_uni = two_pi * torch.rand(num_uniform_samples, device=device, dtype=dtype)
    sampled_s_uni = (s_max - s_min) * torch.rand(num_uniform_samples, device=device, dtype=dtype) + s_min
    uniform_samples = torch.column_stack((sampled_r_uni, sampled_c_uni, sampled_d_rad_uni, sampled_s_uni))
    expanded_uniform_samples = uniform_samples.unsqueeze(0).expand(batch_size, -1, -1)

    final_samples = torch.cat([final_gaussian_samples, expanded_uniform_samples], dim=1)

    return final_samples


import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
# --- 辅助绘图函数 1: 绘制2D散点图 ---
def plot_2d_scatter(ax, p_true, all_samples, sampling_configs, num_uniform_samples,
                    x_dim, y_dim, x_label, y_label, title):
    """在给定的matplotlib axes上绘制2D散点图。"""

    true_x = p_true[x_dim].item()
    true_y = p_true[y_dim].item()

    # 绘制真值
    ax.scatter(true_x, true_y, color='red', marker='X', s=150, label='Ground Truth', zorder=5)

    current_sample_idx = 0
    # 绘制分层高斯采样点
    for i, config in enumerate(sampling_configs):
        num = config['num_samples']
        layer_samples = all_samples[current_sample_idx: current_sample_idx + num]

        plot_x = layer_samples[:, x_dim].cpu().numpy()
        plot_y = layer_samples[:, y_dim].cpu().numpy()

        color = plt.cm.viridis(i / len(sampling_configs))
        ax.scatter(plot_x, plot_y, color=color, s=15, alpha=0.6, label=f'L: {config["name"]}', zorder=2)
        current_sample_idx += num

    # 绘制均匀采样点
    if num_uniform_samples > 0:
        uniform_samples = all_samples[current_sample_idx:]
        plot_x_uni = uniform_samples[:, x_dim].cpu().numpy()
        plot_y_uni = uniform_samples[:, y_dim].cpu().numpy()
        ax.scatter(plot_x_uni, plot_y_uni, color='gray', s=5, alpha=0.3, label=f'Uniform', zorder=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.5)


# --- 辅助绘图函数 2: 绘制1D直方图 ---
def plot_1d_histogram(ax, p_true, all_samples, sampling_configs, dim, label, title):
    """在给定的matplotlib axes上绘制1D直方图。"""
    true_val = p_true[dim].item()

    current_sample_idx = 0
    for i, config in enumerate(sampling_configs):
        num = config['num_samples']
        layer_samples = all_samples[current_sample_idx: current_sample_idx + num]
        data = layer_samples[:, dim].cpu().numpy()
        color = plt.cm.viridis(i / len(sampling_configs))
        ax.hist(data, bins=50, density=True, color=color, alpha=0.7, label=f'L: {config["name"]}')
        current_sample_idx += num

    ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='Ground Truth')
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()


# --- 主可视化函数 ---
def visualize_hierarchical_samples(
        p_true: torch.Tensor,
        rc_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        sampling_configs: List[Dict[str, Any]],
        all_samples: torch.Tensor,
        num_uniform_samples: int
):
    """
    创建一个包含多个子图的仪表盘，全面可视化4D采样点。
    """

    # 创建一个 3x2 的子图网格
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle('Comprehensive Sampling Visualization Dashboard', fontsize=16)

    # --- 绘制 2D 散点图 ---
    # 1. Column vs Row
    plot_2d_scatter(axes[0, 0], p_true, all_samples, sampling_configs, num_uniform_samples,
                    x_dim=1, y_dim=0, x_label="Column (X)", y_label="Row (Y)", title="Spatial Distribution (C vs R)")
    (r_min, r_max), (c_min, c_max) = rc_bounds
    axes[0, 0].set_xlim(c_min, c_max)
    axes[0, 0].set_ylim(r_min, r_max)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_aspect('equal', adjustable='box')

    # 2. Rotation vs Scale
    plot_2d_scatter(axes[0, 1], p_true, all_samples, sampling_configs, num_uniform_samples,
                    x_dim=2, y_dim=3, x_label="Rotation (rad)", y_label="Scale",
                    title="Pose Distribution (Rot vs Scale)")

    # 3. Column vs Rotation
    plot_2d_scatter(axes[1, 0], p_true, all_samples, sampling_configs, num_uniform_samples,
                    x_dim=1, y_dim=2, x_label="Column (X)", y_label="Rotation (rad)", title="C vs Rot")

    # 4. Column vs Scale
    plot_2d_scatter(axes[1, 1], p_true, all_samples, sampling_configs, num_uniform_samples,
                    x_dim=1, y_dim=3, x_label="Column (X)", y_label="Scale", title="C vs Scale")

    # --- 绘制 1D 直方图 ---
    # 5. Rotation Distribution
    plot_1d_histogram(axes[2, 0], p_true, all_samples, sampling_configs,
                      dim=2, label="Rotation (rad)", title="Rotation Distribution by Layer")

    # 6. Scale Distribution
    plot_1d_histogram(axes[2, 1], p_true, all_samples, sampling_configs,
                      dim=3, label="Scale", title="Scale Distribution by Layer")

    # 调整布局并显示图表
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # 调整布局，为总标题和图例留出空间
    plt.show()