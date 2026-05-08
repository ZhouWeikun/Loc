import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DEBUG_XY = False
DEBUG_XY_ROT = False
class HistogramFilter3D(nn.Module):
    def __init__(self, H, W, O, resolution=1.0, device='cuda'):
        super().__init__()
        self.H = H
        self.W = W
        self.O = O
        self.resolution = resolution
        self.device = device

        # 初始化 Belief (1, O, H, W)
        self.belief = torch.ones((1, O, H, W), device=device) / (H * W * O)

        self.counter = 0


    def predict(self, move_rot=0.0, noise_std_rot=0.1, direction_aware=False, move_xy=(0, 0), noise_std_xy=0.5, xy_k_size=5):
        """
        统一的运动更新函数。

        Args:
            move_rot: 相对旋转 (弧度)。正值表示逆时针旋转。
            noise_std_rot: 旋转噪声标准差 (弧度)。
            direction_aware: 是否启用位移预测。False 时仅做位置扩散。
            move_xy: 相对位移 (米)。机体坐标系 (dx_forward, dy_left)。
            noise_std_xy: 位置噪声标准差 (米)。
        """
        # [DEBUG CAPTURE 1]: 初始状态
        if DEBUG_XY_ROT:
            belief_init = self.belief.clone()

        # --- 第一步：平移 (Translation Step) ---
        # 逻辑：必须先平移，再旋转。
        # 因为 move_xy 是基于当前 belief 的朝向 (旧朝向) 定义的。
        if direction_aware:
            # 模式 A: 定向预测 (Prediction)
            # 根据机体位移 move_xy 和当前朝向，生成有偏移的核
            trans_kernel = self._get_trans_kernel(move_xy, noise_std_xy)
        else:
            # 模式 B: 全向扩散 (Diffusion)
            # 强制位移为 0，生成中心对齐的高斯核
            # 此时 belief 会向四周变胖，而不移动
            trans_kernel = self._get_trans_kernel((0, 0), noise_std_xy, k_size=xy_k_size)

        # Grouped Conv2d: 每个方向层 (Channel) 使用自己专属的核进行平移/扩散
        self.belief = F.conv2d(
            self.belief,
            weight=trans_kernel,
            padding='same',
            groups=self.O
        )

        # [DEBUG CAPTURE 2]: 平移后状态
        if DEBUG_XY_ROT:
            belief_trans = self.belief.clone()

        # --- 第二步：旋转 (Rotation Step) ---
        # 逻辑：在 O 维度上做 1D 卷积。
        # 这个卷积核不仅带有高斯模糊 (表示不确定性)，还带有中心偏移 (表示旋转量)。
        # 这样一次操作就完成了 "Rotate" + "Blur"。
        rot_kernel = self._get_rot_kernel(move_rot, noise_std_rot)

        # 准备数据: 将 (1, O, H, W) 转换为 (HxW, 1, O) 以适配 conv1d
        # HxW 变成了 Batch Size，O 变成了 Signal Length
        b_flat = self.belief.permute(2, 3, 0, 1).reshape(self.H * self.W, 1, self.O)

        # Circular Padding: 处理 0度和360度的衔接
        pad_size = rot_kernel.shape[-1] // 2
        b_flat = F.pad(b_flat, (pad_size, pad_size), mode='circular')

        # 1D 卷积
        b_rot_pred = F.conv1d(b_flat, rot_kernel)

        # 恢复形状: (HxW, 1, O) -> (H, W, 1, O) -> (1, O, H, W)
        self.belief = b_rot_pred.view(self.H, self.W, 1, self.O).permute(2, 3, 0, 1)

        # 归一化防止数值消失
        self.normalize_belief()
        self.counter += 1

        # === DEBUG VISUALIZATION (放在最后) ===
        if DEBUG_XY_ROT:
            import time, os
            timestamp = int(time.time())
            # 请修改为你自己的保存路径
            save_dir = '/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_neural_proxy_nr40_nc30_r36_LossLogSumHDDE/seq_loc_results'
            os.makedirs(save_dir, exist_ok=True)

            html_path = f'{save_dir}/predict_evolution_{self.counter}.html'

            print(f"\n[DEBUG] Generating visualization for Predict step...")

            # 调用新的三阶段可视化函数
            # 注意：传入的都是原始 tensor，函数内部会 detach 转 numpy
            visualize_belief_evolution(
                belief_init/(belief_init.sum()),
                belief_trans/(belief_trans.sum()),
                self.belief,
                save_path=html_path,
                resolution=self.resolution
            )
        # =======================================

    def update(self, observation_prob, epsilon=1e-3, alpha=0.5):
        """
        观测更新 (Measurement Step) - 加入鲁棒性机制

        Args:
            observation_prob: 神经场输出的似然 (Likelihood)
            epsilon: 最小概率底座 (Floor)。
                     防止 observation_prob 在真实位置为 0 导致永久丢失目标。
                     相当于认为传感器有一定概率是随机噪声。
            alpha: 更新率 (Update Rate), 范围 (0, 1]。
                   alpha=1.0 为纯贝叶斯更新。
                   alpha < 1.0 时，保留一部分预测的 Belief，防止被单帧噪声瞬间带偏。
        """
        # --- 策略 1: 观测平滑 (解决 "零概率陷阱") ---
        # 混合 原始观测分布 和 一个均匀分布
        # logic: P_robust = (1 - w) * P_obs + w * Uniform
        # 这里用简化的加法实现，效果类似
        obs_smooth = observation_prob + epsilon

        # --- 策略 2: 贝叶斯乘法 ---
        posterior = self.belief * obs_smooth

        # 归一化 (必须在混合之前做，保证尺度一致)
        posterior = posterior / (posterior.sum() + 1e-10)

        # --- 策略 3: 时间动量 (可选，解决 "单帧跳变") ---
        if alpha < 1.0:
            # 这是一个混合模型：新的 Belief 是 "纯贝叶斯后验" 和 "仅靠运动预测的先验" 的加权平均
            # 这能让系统具有一定的"惯性"，不会因为一帧错误的观测就瞬间瞬移
            self.belief = alpha * posterior + (1 - alpha) * self.belief
        else:
            self.belief = posterior

        # 再次归一化确保数值稳定
        self.normalize_belief()

    def normalize_belief(self):
        self.belief = self.belief / (self.belief.sum() + 1e-10)

    def _get_rot_kernel(self, move_rot_rad, sigma_rad):
        """
        生成 1D 旋转核。
        利用高斯核的偏移来模拟旋转，比 torch.roll 更平滑（支持小数个 bin 的旋转）。
        """
        # 1. 确定核大小
        # 这里的单位是 "bin" (网格格数)
        rad_per_bin = 2 * np.pi / self.O

        # 确保输入参数在正确的设备上
        if isinstance(sigma_rad, torch.Tensor):
            sigma_rad = sigma_rad.to(self.device)
        if isinstance(move_rot_rad, torch.Tensor):
            move_rot_rad = move_rot_rad.to(self.device)

        sigma_bin = sigma_rad / rad_per_bin
        shift_bin = move_rot_rad / rad_per_bin

        # 计算核大小时需要转换为标量
        if isinstance(sigma_bin, torch.Tensor):
            sigma_bin_scalar = sigma_bin.item()
        else:
            sigma_bin_scalar = sigma_bin

        k_size = int(np.ceil(3 * sigma_bin_scalar)) * 2 + 1
        k_size = max(3, k_size)
        if k_size % 2 == 0: k_size += 1

        # 2. 生成 1D 网格
        grid = torch.arange(k_size, device=self.device).float() - (k_size - 1) / 2

        # 3. 生成偏移高斯 (1, 1, K)
        # 距离 = grid_index - shift_amount
        kernel = torch.exp(-(grid - shift_bin) ** 2 / (2 * sigma_bin ** 2))
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, -1)

    def _get_trans_kernel(self, move_xy, sigma, k_size=15):
        """
        生成 2D 卷积核。
        注意：你之前的代码非常复杂，混合了Sigmoid距离约束和垂直方向高斯约束。
        为了 4D 神经场定位，最稳健且通用的方式是：
        生成一个以 (dx, dy) 为中心的高斯核。
        """
        # k_size = 15  # 核大小，根据 sigma 和分辨率调整
        # ... 这里可以复用你原来的 get_trans_filters 逻辑 ...
        # 简化的核心逻辑：
        # 对 O 个方向，分别计算其对应的位移 (dx_o, dy_o)。
        # 如果是全局坐标系移动(如GPS差分)，所有通道核一样。
        # 如果是 Ego-motion (前/左)，则每个方向通道的 dx, dy 不同。

        # 假设 move_xy 是全局坐标系下的 (delta_north, delta_east)
        # 那么所有通道共享同一个平移核：
        dx, dy = move_xy[0] / self.resolution, move_xy[1] / self.resolution

        y, x = torch.meshgrid(
            torch.arange(-(k_size // 2), k_size // 2 + 1, device=self.device),
            torch.arange(-(k_size // 2), k_size // 2 + 1, device=self.device)
        )

        dist_sq = (x - dx) ** 2 + (y - dy) ** 2
        kernel_2d = torch.exp(-dist_sq / (2 * (sigma / self.resolution) ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()

        # 扩展为 (O, 1, K, K) 以适应 Group Conv2d
        return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(self.O, 1, 1, 1)

    def _get_gaussian_kernel(self, size, sigma):
        """创建一个标准的 2D 高斯核"""
        coords = torch.arange(size).float() - (size - 1) / 2
        x = coords.reshape(1, -1)
        y = coords.reshape(-1, 1)
        k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        return k.view(1, 1, size, size)



"""#####################################################################"""
def visualize_trans_kernel(trans_kernel, move_xy=(0, 0), sigma=1.0, resolution=1.0,
                           save_path=None, show_directions=None):
    """
    可视化平移滤波核

    Args:
        trans_kernel: torch.Tensor, shape [O, 1, K, K] 平移核
        move_xy: tuple, (dx, dy) 期望的位移 (米)
        sigma: float, 核的标准差 (米)
        resolution: float, 网格分辨率 (米/像素)
        save_path: str or None, 保存路径（如果None则显示图像）
        show_directions: list or None, 显示哪些方向的核（如果None则只显示第一个）

    Returns:
        fig: matplotlib figure对象
    """
    # 转换为numpy并移到CPU
    if torch.is_tensor(trans_kernel):
        kernel_np = trans_kernel.cpu().numpy()
    else:
        kernel_np = trans_kernel

    O, _, K, K_check = kernel_np.shape
    assert K == K_check, "核必须是正方形"

    # 确定显示哪些方向
    if show_directions is None:
        show_directions = [0]  # 默认只显示第一个方向
    else:
        show_directions = [d for d in show_directions if d < O]

    n_show = len(show_directions)

    # 创建子图
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    # 计算期望的中心偏移（像素单位）
    dx_pix = move_xy[0] / resolution
    dy_pix = move_xy[1] / resolution
    sigma_pix = sigma / resolution

    # 核的中心坐标
    center = K // 2

    for idx, direction_idx in enumerate(show_directions):
        ax = axes[idx]

        # 获取当前方向的核
        kernel_2d = kernel_np[direction_idx, 0, :, :]  # [K, K]

        # 绘制核热力图
        im = ax.imshow(kernel_2d, cmap='hot', origin='lower',
                       extent=[-center - 0.5, center + 0.5, -center - 0.5, center + 0.5])
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 标记理论中心（期望的位移位置）
        ax.plot(dx_pix, dy_pix, 'b*', markersize=15,
                label=f'Theoretical Center ({dx_pix:.2f}, {dy_pix:.2f})')

        # 标记实际中心（概率最大值位置）
        max_idx = np.unravel_index(kernel_2d.argmax(), kernel_2d.shape)
        max_y, max_x = max_idx
        max_x_centered = max_x - center
        max_y_centered = max_y - center
        ax.plot(max_x_centered, max_y_centered, 'gx', markersize=12,
                label=f'Actual Peak ({max_x_centered}, {max_y_centered})')

        # 绘制理论1-sigma圆圈
        circle = Circle((dx_pix, dy_pix), sigma_pix,
                       fill=False, color='cyan', linestyle='--',
                       linewidth=2, label=f'1σ circle (σ={sigma_pix:.2f} pix)')
        ax.add_patch(circle)

        # 绘制坐标轴
        ax.axhline(y=0, color='white', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='white', linestyle=':', linewidth=1, alpha=0.5)

        # 设置标题和标签
        if O > 1:
            angle_deg = direction_idx * 360 / O
            ax.set_title(f'Direction={direction_idx}/{O} (Degree={angle_deg:.1f}°)\n'
                        f'Offset=({move_xy[0]:.2f}, {move_xy[1]:.2f})m, σ={sigma:.2f}m',
                        fontsize=10)
        else:
            ax.set_title(f'OffsetKernal\noffset=({move_xy[0]:.2f}, {move_xy[1]:.2f})m, σ={sigma:.2f}m',
                        fontsize=10)

        ax.set_xlabel('X Offset (pixels)')
        ax.set_ylabel('Y Offset (pixels)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 滤波核可视化已保存到: {save_path}")
    else:
        plt.show()

    return fig


def visualize_trans_kernel_comparison(hist_filter, move_xy_list, sigma_list,k_size=5,
                                     resolution=1.0, save_path=None):
    """
    可视化多组不同参数的平移核，用于比较

    Args:
        hist_filter: HistogramFilter3D对象
        move_xy_list: list of tuples, 多组位移参数 [(dx1, dy1), (dx2, dy2), ...]
        sigma_list: list of floats, 对应的标准差列表
        resolution: float, 网格分辨率 (米/像素)
        save_path: str or None, 保存路径

    Returns:
        fig: matplotlib figure对象
    """
    n_cases = len(move_xy_list)
    assert len(sigma_list) == n_cases, "位移和标准差列表长度必须相同"

    # 创建子图
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 5))
    if n_cases == 1:
        axes = [axes]

    for idx, (move_xy, sigma) in enumerate(zip(move_xy_list, sigma_list)):
        ax = axes[idx]

        # 生成核
        with torch.no_grad():
            trans_kernel = hist_filter._get_trans_kernel(move_xy, sigma,k_size=k_size)

        # 获取第一个方向的核
        kernel_2d = trans_kernel[0, 0, :, :].cpu().numpy()  # [K, K]
        K = kernel_2d.shape[0]
        center = K // 2

        # 绘制核热力图
        im = ax.imshow(kernel_2d, cmap='hot', origin='lower',
                      extent=[-center - 0.5, center + 0.5, -center - 0.5, center + 0.5])

        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 计算期望的中心偏移（像素单位）
        dx_pix = move_xy[0] / resolution
        dy_pix = move_xy[1] / resolution
        sigma_pix = sigma / resolution

        # 标记理论中心
        ax.plot(dx_pix, dy_pix, 'b*', markersize=15,
               label=f'Theoretical Center ({dx_pix:.2f}, {dy_pix:.2f})')

        # 标记实际中心
        max_idx = np.unravel_index(kernel_2d.argmax(), kernel_2d.shape)
        max_y, max_x = max_idx
        max_x_centered = max_x - center
        max_y_centered = max_y - center
        ax.plot(max_x_centered, max_y_centered, 'gx', markersize=12,
               label=f'Actual peak ({max_x_centered}, {max_y_centered})')

        # 绘制1-sigma圆圈
        circle = Circle((dx_pix, dy_pix), sigma_pix,
                       fill=False, color='cyan', linestyle='--',
                       linewidth=2, label=f'1σ (σ={sigma_pix:.2f} pix)')
        ax.add_patch(circle)

        # 绘制坐标轴
        ax.axhline(y=0, color='white', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='white', linestyle=':', linewidth=1, alpha=0.5)

        # 设置标题和标签
        ax.set_title(f'kernal {idx+1}\noffset=({move_xy[0]:.2f}, {move_xy[1]:.2f})m\nσ={sigma:.2f}m',
                    fontsize=10)
        ax.set_xlabel('X Offset (pixels)')
        ax.set_ylabel('Y Offset (pixels)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 滤波核对比可视化已保存到: {save_path}")
    else:
        plt.show()

    return fig


def visualize_rot_kernel(rot_kernel, move_rot_rad=0.0, sigma_rad=0.1, O=12, save_path=None):
    """
    可视化旋转滤波核（1D）

    Args:
        rot_kernel: torch.Tensor, shape [1, 1, K] 旋转核
        move_rot_rad: float, 期望的旋转角度 (弧度)
        sigma_rad: float, 核的标准差 (弧度)
        O: int, 总方向数
        save_path: str or None, 保存路径（如果None则显示图像）

    Returns:
        fig: matplotlib figure对象
    """
    # 转换为numpy并移到CPU
    if torch.is_tensor(rot_kernel):
        kernel_np = rot_kernel.cpu().numpy().squeeze()  # [K]
    else:
        kernel_np = rot_kernel.squeeze()

    K = len(kernel_np)

    # 计算bin相关参数
    rad_per_bin = 2 * np.pi / O
    sigma_bin = sigma_rad / rad_per_bin
    shift_bin = move_rot_rad / rad_per_bin

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 5))

    # X轴：以核中心为0的坐标
    x_bins = np.arange(K) - (K - 1) / 2

    # 绘制核权重（条形图）
    bars = ax.bar(x_bins, kernel_np, width=0.8, alpha=0.7, color='steelblue',
                  edgecolor='black', label='Kernel weights')

    # 标记理论中心（期望的旋转偏移）
    ax.axvline(x=shift_bin, color='red', linestyle='--', linewidth=2,
              label=f'Theoretical shift ({shift_bin:.2f} bins = {np.degrees(move_rot_rad):.1f}°)')

    # 标记实际峰值
    peak_bin = x_bins[kernel_np.argmax()]
    ax.axvline(x=peak_bin, color='green', linestyle=':', linewidth=2,
              label=f'Actual peak ({peak_bin:.2f} bins)')

    # 绘制1-sigma范围
    ax.axvspan(shift_bin - sigma_bin, shift_bin + sigma_bin,
              alpha=0.2, color='yellow', label=f'±1σ ({sigma_bin:.2f} bins = ±{np.degrees(sigma_rad):.1f}°)')

    # 设置标题和标签
    ax.set_title(f'Rotation Kernel Visualization\n'
                f'Move: {np.degrees(move_rot_rad):.2f}° ({move_rot_rad:.3f} rad), '
                f'σ: {np.degrees(sigma_rad):.2f}° ({sigma_rad:.3f} rad)\n'
                f'Total directions: {O}, Kernel size: {K}',
                fontsize=11)
    ax.set_xlabel('Bin offset from center', fontsize=10)
    ax.set_ylabel('Probability weight', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 旋转核可视化已保存到: {save_path}")
    else:
        plt.show()

    return fig


def visualize_belief_volume_comparison(belief_before, belief_after,
                                       save_path=None, resolution=1.0):
    """
    并排对比两个概率体的 Volume (用于观察卷积前后的变化)
    """

    # 1. 数据准备
    def prep_data(b):
        if torch.is_tensor(b): b = b.detach().cpu().numpy()
        if b.ndim == 4: b = b[0]
        return b

    b1 = prep_data(belief_before)
    b2 = prep_data(belief_after)

    O, H, W = b1.shape

    # 统一色标范围
    g_max = max(b1.max(), b2.max())
    g_min = g_max * 0.05  # 阈值设为峰值的 5%

    # 2. 构建坐标
    o_idx, h_idx, w_idx = np.meshgrid(np.arange(O), np.arange(H), np.arange(W), indexing='ij')
    x = w_idx.flatten() * resolution
    y = h_idx.flatten() * resolution
    z = o_idx.flatten() * (360.0 / O)

    # 3. 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f'Before (Max: {b1.max():.2e})', f'After (Max: {b2.max():.2e})')
    )

    # 添加 Before Volume
    fig.add_trace(go.Volume(
        x=x, y=y, z=z,
        value=b1.flatten(),
        isomin=g_min, isomax=g_max,
        opacity=0.1, surface_count=15, colorscale='Hot',
        colorbar=dict(x=0.45, title='Prob'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Before'
    ), row=1, col=1)

    # 添加 After Volume
    fig.add_trace(go.Volume(
        x=x, y=y, z=z,
        value=b2.flatten(),
        isomin=g_min, isomax=g_max,
        opacity=0.1, surface_count=15, colorscale='Hot',
        colorbar=dict(x=1.0, title='Prob'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='After'
    ), row=1, col=2)

    fig.update_layout(
        title="Belief Volume Comparison (Translation)",
        width=1600, height=800,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Angle'),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Angle')
    )

    if save_path:
        fig.write_html(save_path)
        print(f"✓ Volume对比可视化已保存: {save_path}")


def visualize_belief_evolution(b_init, b_trans, b_final,
                               save_path=None, resolution=1.0):
    """
    可视化 Belief 的三阶段演变：初始 -> 平移后 -> 旋转后
    """

    # 1. 数据预处理 helper
    def prep_data(b):
        if torch.is_tensor(b): b = b.detach().cpu().numpy()
        if b.ndim == 4: b = b[0]
        return b

    d1 = prep_data(b_init)
    d2 = prep_data(b_trans)
    d3 = prep_data(b_final)

    O, H, W = d1.shape

    # 2. 统一色标 (Global Scale)
    # 这样可以看出随着模糊，峰值高度是否在下降
    g_max = max(d1.max(), d2.max(), d3.max())
    g_min = g_max * 0.05  # 阈值设为最大峰值的 5%

    # 3. 构建坐标
    o_idx, h_idx, w_idx = np.meshgrid(np.arange(O), np.arange(H), np.arange(W), indexing='ij')
    x = w_idx.flatten() * resolution
    y = h_idx.flatten() * resolution
    z = o_idx.flatten() * (360.0 / O)

    # 4. 创建 1x3 子图
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f'1. Initial (Max: {d1.max():.2e})',
            f'2. After Translation (Max: {d2.max():.2e})',
            f'3. After Rotation (Max: {d3.max():.2e})'
        ),
        horizontal_spacing=0.02
    )

    # 通用配置
    vol_kwargs = dict(
        x=x, y=y, z=z,
        isomin=g_min, isomax=g_max,
        opacity=0.1, surface_count=15, colorscale='Hot',
        caps=dict(x_show=False, y_show=False, z_show=False)
    )

    # Trace 1: Initial
    fig.add_trace(go.Volume(
        value=d1.flatten(),
        colorbar=dict(x=0.31, title='Prob', len=0.5),  # 独立的 colorbar
        name='Initial',
        **vol_kwargs
    ), row=1, col=1)

    # Trace 2: Trans
    fig.add_trace(go.Volume(
        value=d2.flatten(),
        colorbar=dict(x=0.64, title='Prob', len=0.5),
        name='Translated',
        **vol_kwargs
    ), row=1, col=2)

    # Trace 3: Rot
    fig.add_trace(go.Volume(
        value=d3.flatten(),
        colorbar=dict(x=0.99, title='Prob', len=0.5),
        name='Rotated',
        **vol_kwargs
    ), row=1, col=3)

    # 5. 布局美化
    scene_layout = dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Angle')
    fig.update_layout(
        title="Belief State Evolution (Predict Step)",
        width=1800, height=700,
        scene=scene_layout, scene2=scene_layout, scene3=scene_layout,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"✓ 演变过程可视化已保存: {save_path}")


if __name__ == "__main__":
    H ,W , O = 40, 30, 12
    filter = HistogramFilter3D(H=40,W=30,O=12)
    moves_xy = np.zeros([O,2])
    sigms_xy = np.ones(O)*0.65
    p2save = '/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_project_integrateRot_classify_12_t0.05/kernals_sigma0.65.png'
    # visualize_trans_kernel_comparison(filter,moves_xy.tolist(),sigms_xy.tolist(),k_size=5,save_path=p2save)

    # kernals_xy = filter._get_trans_kernel(move_xy=(0,0),sigma=0.5,k_size=5)
    # p2save = '/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_project_integrateRot_classify_12_t0.05/kernals_2.png'
    # show_directions = list(np.arange(12))
    # visualize_trans_kernel(kernals_xy,sigma=0.5,save_path=p2save,show_directions=show_directions)

    move_rot_rad = 15/180*np.pi
    sigma_rot_rad = 15/180*np.pi
    kernals_rot = filter._get_rot_kernel(move_rot_rad=move_rot_rad, sigma_rad=sigma_rot_rad)
    p2save = '/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_project_integrateRot_classify_12_t0.05/kernals_rot.png'
    visualize_rot_kernel(kernals_rot,move_rot_rad,sigma_rot_rad,save_path=p2save)



