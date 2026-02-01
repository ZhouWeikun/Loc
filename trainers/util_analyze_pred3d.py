import os.path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_localization_video_with_gt(preds_tensor, gt_coords=None, save_path='localization_video_gt.mp4', fps=10):
    """
    生成带有 Ground Truth 对比的定位视频。

    Args:
        preds_tensor: torch.Tensor or np.array, shape [T, O, H, W]
                      预测的概率体序列。
        gt_coords:    torch.Tensor or np.array, shape [T] (1D) 或 [T, 3] (3D)
                      Ground Truth 坐标。
                      - 如果是 1D: 假设是对 [H, W, O] 形状的 flatten 索引。
                      - 如果是 3D: 默认格式为 [Row(H), Col(W), Rot(O)] (即 nr, nc, rot)。
        save_path:    str, 保存路径 (.mp4)
        fps:          int, 帧率
    """
    # 1. 数据准备
    if torch.is_tensor(preds_tensor):
        data = preds_tensor.detach().cpu().numpy()
    else:
        data = preds_tensor

    T, O, H, W = data.shape

    # 处理 GT 数据
    gt_data = None
    if gt_coords is not None:
        if torch.is_tensor(gt_coords):
            gt_data = gt_coords.detach().cpu().numpy()
        else:
            gt_data = gt_coords

        # 确保 GT 长度匹配
        if len(gt_data) != T:
            print(f"警告: GT长度 ({len(gt_data)}) 与 预测帧数 ({T}) 不一致，将截断或对齐。")
            T = min(T, len(gt_data))

    print(f"开始生成视频，共 {T} 帧 (H={H}, W={W}, O={O})...")

    # 2. 设置画布
    fig, (ax_map, ax_ori) = plt.subplots(1, 2, figsize=(14, 6))
    plt.tight_layout(pad=4.0)

    # --- 初始化左图 (地图) ---
    # 初始帧 (Marginalize O -> [H, W])
    frame0_map = data[0].sum(axis=0)

    # [修改点 1] origin 改为 'upper' (原点在左上)
    im_map = ax_map.imshow(frame0_map, cmap='inferno', origin='upper', animated=True, aspect='equal')

    ax_map.set_title("Position Estimate (Green) vs GT (Red)")
    ax_map.set_xlabel("Width (Col)")
    ax_map.set_ylabel("Height (Row)")
    fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)

    # 标记
    pred_point, = ax_map.plot([], [], 'g+', markersize=18, markeredgewidth=3, label='Pred Peak')
    gt_point, = ax_map.plot([], [], 'r*', markersize=15, markeredgewidth=1.5, label='Ground Truth')

    ax_map.legend(loc='upper right')
    ax_map.set_xlim(-0.5, W - 0.5)

    # [修改点 2] 反转 Y 轴范围：底部是 H，顶部是 0 (向下增长)
    ax_map.set_ylim(H - 0.5, -0.5)

    # --- 初始化右图 (方向) ---
    # 初始帧 (Marginalize H, W: 沿着H,W维度求和)
    frame0_ori = data[0].sum(axis=(1, 2))
    angles = np.arange(O) * (360 / O)
    # 初始柱状图颜色
    bars = ax_ori.bar(angles, frame0_ori, width=(300 / O), color='skyblue', align='edge', alpha=0.7)

    ax_ori.set_title("Orientation Distribution")
    ax_ori.set_xlabel("Angle (Degree)")
    ax_ori.set_ylabel("Probability")
    ax_ori.set_xticks(angles)
    ax_ori.set_ylim(0, 1.0)

    time_text = ax_map.text(0.02, 0.95, '', transform=ax_map.transAxes, color='white', fontweight='bold', fontsize=12)

    # 3. 辅助函数：解析 GT (核心修改点)
    def parse_gt(idx):
        if gt_data is None:
            return None, None, None

        current_gt = gt_data[idx]

        if current_gt.ndim == 0 or gt_data.shape[1] == 1:  # 1D Index
            # === 关键修改 ===
            # 用户指定 flatten 顺序基于 [H, W, O]
            # 意味着数据按 O -> W -> H 的顺序变化
            h_gt, w_gt, o_gt = np.unravel_index(int(current_gt), (H, W, O))

        else:  # 3D Index [nr, nc, rot] -> [h, w, o]
            h_gt = int(current_gt[0])  # Row
            w_gt = int(current_gt[1])  # Col
            o_gt = int(current_gt[2])  # Rot Index

        return h_gt, w_gt, o_gt

    # 4. 动画更新函数
    def update(frame_idx):
        # 获取数据
        current_vol = data[frame_idx]  # shape [O, H, W]
        h_gt, w_gt, o_gt = parse_gt(frame_idx)

        # --- 更新地图 (Position) ---
        prob_map = current_vol.sum(axis=0)  # [H, W]
        im_map.set_data(prob_map)
        im_map.set_clim(vmin=prob_map.min(), vmax=prob_map.max())

        # 预测峰值
        max_idx = np.unravel_index(prob_map.argmax(), prob_map.shape)  # (row, col)
        pred_point.set_data([max_idx[1]], [max_idx[0]])  # plot(x, y) -> (col, row)

        # GT 标记
        if h_gt is not None:
            gt_point.set_data([w_gt], [h_gt])  # plot(x=col, y=row)

        # --- 更新方向 (Orientation) ---
        prob_ori = current_vol.sum(axis=(1, 2))  # [O]
        prob_ori = prob_ori / (prob_ori.sum() + 1e-6)  # Normalize for display

        # 更新柱子高度和颜色
        max_ori_val = 0
        for i, (rect, h) in enumerate(zip(bars, prob_ori)):
            rect.set_height(h)
            max_ori_val = max(max_ori_val, h)

            # GT 颜色逻辑: GT所在的方向显示为红色，其他为蓝色
            if o_gt is not None and i == o_gt:
                rect.set_color('red')
                rect.set_alpha(0.9)
            else:
                rect.set_color('skyblue')
                rect.set_alpha(0.7)

        ax_ori.set_ylim(0, max_ori_val * 1.2 + 0.01)

        # --- 更新文字 ---
        info_str = f'Frame: {frame_idx}/{T}'
        if h_gt is not None:
            # 计算简单的距离误差 (L2 distance in pixels)
            dist_err = np.sqrt((max_idx[0] - h_gt) ** 2 + (max_idx[1] - w_gt) ** 2)
            info_str += f'\nPos Err: {dist_err:.1f} px'

        time_text.set_text(info_str)

        return im_map, pred_point, gt_point, *bars, time_text

    # 5. 生成并保存
    ani = animation.FuncAnimation(fig, update, frames=T, blit=False, interval=100)

    print(f"正在渲染视频到 {save_path} ...")
    try:
        # 尝试使用 ffmpeg
        ani.save(save_path, writer='ffmpeg', dpi=120, fps=fps)
        print(f"✓ 视频生成成功: {save_path}")
    except Exception as e:
        print(f"ffmpeg 保存失败: {e}")
        print("尝试使用 Pillow 保存为 GIF...")
        gif_path = save_path.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"✓ GIF生成成功: {gif_path}")

    plt.close()


def compute_topN_acc_given_threshold(
        coords_pred,
        coords_gt,
        dist_th,
        rot_th_deg=None,  # <--- 改为默认为 None，或者允许传入 None
        scale_th=None,
        k_values=[1, 3, 5, 10]
):
    """
    计算精细定位的Top-K准确率 (Recall@K)，并统计详细误差。

    修改更新：
    - 支持 rot_th_deg 为 None。此时只评价 2D 位置精度 (只要 dist 合格即视为 Hit)。
    - Scale 采用相对误差计算。
    """
    device = coords_pred.device
    coords_gt = coords_gt.to(device)

    B, N_pred, _ = coords_pred.shape
    coords_gt_expanded = coords_gt.unsqueeze(1)  # [B, 1, 4]

    # --- 1. 计算各项误差 ---

    # 1.1 空间距离误差 (x, y)
    dist_errors = torch.norm(coords_pred[..., :2] - coords_gt_expanded[..., :2], p=2, dim=-1)

    # 1.2 旋转误差 (d)
    rot_diff_rad = torch.abs(coords_pred[..., 2] - coords_gt_expanded[..., 2])
    rot_errors_rad = torch.min(rot_diff_rad, 2 * torch.pi - rot_diff_rad)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)

    # 1.3 尺度误差 (s) - 使用相对误差
    pred_s = coords_pred[..., 3]
    gt_s = coords_gt_expanded[..., 3]
    scale_errors_rel = torch.abs(pred_s - gt_s) / (gt_s + 1e-6)

    # --- 2. 判定是否命中 (Hit) ---

    # [修改点] 基础条件仅包含距离
    is_hit = (dist_errors <= dist_th)

    # [修改点] 只有当 rot_th_deg 不为 None 时，才叠加旋转判定
    if rot_th_deg is not None:
        is_hit = is_hit & (rot_errors_deg <= rot_th_deg)

    # 只有当 scale_th 不为 None 时，才叠加尺度判定
    if scale_th is not None:
        is_hit = is_hit & (scale_errors_rel <= scale_th)

    # --- 3. 计算 Metrics (Acc) ---
    metrics = {}
    for k in k_values:
        if k > N_pred:
            metrics[f'top{k}_acc'] = 0.0
            continue
        has_hit_in_k = is_hit[:, :k].any(dim=1)
        metrics[f'top{k}_acc'] = has_hit_in_k.float().mean().item() * 100

    # --- 4. 统计 Top-1 详细误差 ---
    # 即使不卡阈值，我们也依然统计 Top-1 的误差数值，方便观察
    top1_dist = dist_errors[:, 0]
    top1_rot = rot_errors_deg[:, 0]
    top1_scale = scale_errors_rel[:, 0]

    errors = {
        # === 位置误差 ===
        'mean_dist_err_top1': top1_dist.mean().item(),
        'median_dist_err_top1': torch.median(top1_dist).item(),

        # === 旋转误差 ===
        'mean_rot_err_top1': top1_rot.mean().item(),
        'median_rot_err_top1': torch.median(top1_rot).item(),

        # === 尺度误差 ===
        'mean_scale_rel_err_top1': top1_scale.mean().item(),
        'median_scale_rel_err_top1': torch.median(top1_scale).item()
    }

    return metrics, errors


def print_topN_acc_results(metrics, errors, thresholds):
    """
    美化打印精细评估结果。

    Updates:
    - 修复了 f-string 中波浪号导致的 TypeError。
    - 支持 Rot 阈值为 None 的显示 (Ignored)。
    - 支持 Scale 相对误差的百分比显示。
    - 包含 Top-1 的 Mean 和 Median 误差统计。
    """
    # 1. 获取并格式化阈值字符串
    dist_th = thresholds.get('norm_dist', 'N/A')

    # 处理 Rotation 阈值 (支持 None)
    raw_rot_th = thresholds.get('rot')
    if raw_rot_th is None:
        rot_msg = "None (Ignored)"
    else:
        rot_msg = f"{raw_rot_th}°"

    # 处理 Scale 阈值 (支持 None，且显示百分比)
    raw_scale_th = thresholds.get('scale')
    if raw_scale_th is not None:
        # 例如 0.1 -> "0.10 (Rel/~10%)"
        # 修正：将 ~ 放在花括号外部
        scale_msg = f"{raw_scale_th:.2f} (Rel/~{raw_scale_th * 100:.0f}%)"
    else:
        scale_msg = "None (Monitor Only)"

    # 2. 打印头部
    print(f"\n{'=' * 20} Fine-Grained Accuracy Report {'=' * 20}")
    print(f"Thresholds -> Dist: {dist_th}, Rot: {rot_msg}, Scale: {scale_msg}")
    print("-" * 75)
    print(f"{'Metric':<15} | {'Accuracy (%)':<15}")
    print("-" * 35)

    # 3. 打印 Accuracy (按 k 值排序)
    keys = sorted([k for k in metrics.keys() if 'top' in k],
                  key=lambda x: int(x.replace('top', '').replace('_acc', '')))

    for k in keys:
        print(f"{k:<15} | {metrics[k]:<15.2f}")

    # 4. 打印 Top-1 详细误差统计
    print("-" * 75)
    print(f"Top-1 Error Stats (Global Average & Median):")

    # Dist
    d_mean = errors.get('mean_dist_err_top1', 0)
    d_med = errors.get('median_dist_err_top1', 0)
    print(f"  Dist  Error: Mean={d_mean:.4f},   Median={d_med:.4f}")

    # Rot
    r_mean = errors.get('mean_rot_err_top1', 0)
    r_med = errors.get('median_rot_err_top1', 0)
    print(f"  Rot   Error: Mean={r_mean:.2f}°,    Median={r_med:.2f}°")

    # Scale (转换为百分比显示)
    s_mean = errors.get('mean_scale_rel_err_top1', 0) * 100
    s_med = errors.get('median_scale_rel_err_top1', 0) * 100
    print(f"  Scale Error: Mean={s_mean:.2f}%,    Median={s_med:.2f}% (Rel)")

    print(f"{'=' * 75}\n")


def compute_top_k_accuracy(pred_pdf, gt_labels, k_values=[1, 4, 9, 16, 50], dim_order="HWO"):
    """
    计算Top-K准确率，支持自定义维度的解读顺序。
    会自动将输入转置为标准的 [H, W, O] 顺序后计算，以对齐 GT Label 的索引逻辑。

    Args:
        pred_pdf: [N, D1, D2, D3] 多维概率分布 或 [N, C] 扁平化概率
        gt_labels: [N] GT标签 (Flat Indices, 假设基于 H->W->O 顺序生成)
        k_values: list, K值列表
        dim_order: str, 输入 pred_pdf 的维度含义，例如 "OHW" 或 "HWO"
                   H=NR(Row), W=NC(Col), O=Rot(Orientation)

    Returns:
        dict: Top-K准确率字典
    """
    # 1. 确保输入是 Numpy 数组
    if isinstance(pred_pdf, torch.Tensor):
        pred_pdf = pred_pdf.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    n_samples = pred_pdf.shape[0]

    # 2. 处理维度顺序并扁平化
    # 目标：统一转换为 [N, H, W, O] 然后 flatten，以匹配 gt_flat_idx = nr*... + nc*... + rot
    target_order = "HWO"

    if pred_pdf.ndim == 4:  # [N, D1, D2, D3]
        if dim_order != target_order:
            # 创建维度映射: 0是batch维，保持不变
            # 例如 dim_order="OHW", 我们需要转为 "HWO"
            # 原图索引: O=1, H=2, W=3
            # 目标索引: (0, 2, 3, 1) -> (Batch, H, W, O)

            # 找到 H, W, O 在当前 dim_order 中的位置索引 (+1是因为有一个Batch维度)
            idx_map = {char: i + 1 for i, char in enumerate(dim_order)}

            # 构建转置的轴列表
            permute_axes = [0] + [idx_map[c] for c in target_order]

            # 执行转置
            pred_pdf = np.transpose(pred_pdf, permute_axes)
            print(f"DEBUG: 已将预测分布从 [N, {dim_order}] 转置为 [N, {target_order}] 以对齐标签")

        # 扁平化: [N, H, W, O] -> [N, H*W*O]
        pred_pdf_flat = pred_pdf.reshape(n_samples, -1)

    elif pred_pdf.ndim == 2:  # 已经是扁平的 [N, C]
        # 此时无法重排轴，假设输入已经是按照 H->W->O 顺序 flatten 的
        pred_pdf_flat = pred_pdf
    else:
        raise ValueError(f"pred_pdf 形状不正确: {pred_pdf.shape}, 期望是 2D(Flat) 或 4D(Vol)")

    # 3. 对每个样本找到概率最高的K个位置
    # axis=-1 表示在最后一个轴（也就是flatten后的特征轴）上排序
    sorted_indices = np.argsort(pred_pdf_flat, axis=-1)[:, ::-1]  # [N, Total] 降序

    results = {}
    for k in k_values:
        # 检查K值是否超出总类别数
        if k > pred_pdf_flat.shape[1]:
            continue

        # 检查GT标签是否在Top-K中
        top_k_indices = sorted_indices[:, :k]  # [N, K]
        # gt_labels[:, None] 将 [N] 变为 [N, 1]，利用广播机制比较
        correct = np.any(top_k_indices == gt_labels[:, None], axis=1)  # [N]
        accuracy = correct.mean() * 100
        results[f'top{k}_acc'] = accuracy

    # 4. 计算排名 (Mean Rank / Median Rank)
    ranks = []
    for i in range(n_samples):
        # 找到GT label在排序后索引中的位置
        # np.where 返回 tuple, [0] 是行索引数组
        found = np.where(sorted_indices[i] == gt_labels[i])[0]
        if len(found) > 0:
            rank = found[0] + 1
        else:
            # 如果GT label越界（理论上不应发生，除非GT和Pred形状不匹配）
            rank = pred_pdf_flat.shape[1] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    results['mean_rank'] = ranks.mean()
    results['median_rank'] = np.median(ranks)

    return results


def convert_3d_to_2d_predictions(pred_3d, gt_labels_3d, shape_3d=None, dim_order="HWO"):
    """
    将3D预测和GT标签转换为2D（边缘化旋转维度）

    这是一个辅助函数，用于将包含旋转维度的3D预测转换为2D平面预测。

    Args:
        pred_3d: np.array or torch.Tensor
                 - 如果是2D: shape [N, H*W*O]，需要提供shape_3d
                 - 如果是4D: shape [N, H, W, O] 或 [N, O, H, W]，根据dim_order判断
        gt_labels_3d: np.array or torch.Tensor, shape [N]
                      3D GT标签（flatten索引，基于[H, W, O]顺序）
        shape_3d: tuple or None, (H, W, O)
                  仅当pred_3d是2D时需要提供
        dim_order: str, "HWO" 或 "OHW"
                   指定pred_3d的维度顺序（仅对4D输入有效）

    Returns:
        pred_2d: np.array or torch.Tensor, shape [N, H, W]
                 2D平面概率图（旋转维度已边缘化）
        gt_labels_2d: np.array, shape [N]
                      2D GT标签（flatten索引，基于H*W）
        shape_2d: tuple, (H, W)
                  2D形状信息

    Example:
        >>> pred_3d = np.random.rand(100, 40, 30, 12)  # [N, H, W, O]
        >>> gt_labels_3d = np.random.randint(0, 14400, 100)  # [N]
        >>> pred_2d, gt_2d, (H, W) = convert_3d_to_2d_predictions(pred_3d, gt_labels_3d)
        >>> # pred_2d.shape = (100, 40, 30), gt_2d.shape = (100,)
    """
    # 1. 数据准备
    is_torch = torch.is_tensor(pred_3d)
    if is_torch:
        pred_3d_np = pred_3d.detach().cpu().numpy()
    else:
        pred_3d_np = pred_3d

    if torch.is_tensor(gt_labels_3d):
        gt_labels_np = gt_labels_3d.detach().cpu().numpy()
    else:
        gt_labels_np = gt_labels_3d

    N = len(gt_labels_np)

    # 2. 处理不同的输入格式
    if pred_3d_np.ndim == 2:
        # 2D输入，需要shape_3d
        if shape_3d is None:
            raise ValueError("当pred_3d是2D时，必须提供shape_3d参数")
        H, W, O = shape_3d
        pred_3d_np = pred_3d_np.reshape(N, H, W, O)
    elif pred_3d_np.ndim == 4:
        # 4D输入，根据dim_order判断
        if dim_order == "OHW":
            # [N, O, H, W] -> [N, H, W, O]
            pred_3d_np = pred_3d_np.transpose(0, 2, 3, 1)
        elif dim_order == "HWO":
            # 已经是 [N, H, W, O]，不需要转换
            pass
        else:
            raise ValueError(f"dim_order必须是'HWO'或'OHW'，got {dim_order}")

        N, H, W, O = pred_3d_np.shape
    else:
        raise ValueError(f"pred_3d维度必须是2D或4D，got {pred_3d_np.ndim}D")

    # 3. Marginalize旋转维度，得到2D平面概率图
    pred_2d_np = pred_3d_np.sum(axis=-1)  # [N, H, W]

    # 4. 将GT标签从3D转换为2D（忽略旋转维度）
    # gt_labels_3d 是基于 [H, W, O] flatten的索引
    gt_coords_3d = np.array([np.unravel_index(int(label), (H, W, O)) for label in gt_labels_np])
    gt_h = gt_coords_3d[:, 0]  # [N]
    gt_w = gt_coords_3d[:, 1]  # [N]
    # 忽略方向: gt_coords_3d[:, 2]

    # 转换为2D的flatten索引
    gt_labels_2d = gt_h * W + gt_w  # [N]

    # 5. 如果输入是torch tensor，返回torch tensor
    if is_torch:
        pred_2d = torch.from_numpy(pred_2d_np)
    else:
        pred_2d = pred_2d_np

    return pred_2d, gt_labels_2d, (H, W)


def compute_2d_plane_accuracy(pred_3d, gt_labels_3d, shape_3d=None, dim_order="HWO", k_values=[1, 2, 3, 4, 9, 16]):
    """
    计算2D平面定位的Top-K准确率（忽略方向维度）

    Args:
        pred_3d: np.array or torch.Tensor
                 - 如果是2D: shape [N, H*W*O]，需要提供shape_3d
                 - 如果是4D: shape [N, H, W, O] 或 [N, O, H, W]，根据dim_order判断
        gt_labels_3d: np.array, shape [N]
                      3D GT标签（flatten索引，基于[H, W, O]顺序）
        shape_3d: tuple or None, (H, W, O)
                  仅当pred_3d是2D时需要提供
        dim_order: str, "HWO" 或 "OHW"
                   指定pred_3d的维度顺序（仅对4D输入有效）
        k_values: list, 要计算的K值列表

    Returns:
        dict: 包含各种2D平面准确率指标
    """
    # 1. 使用辅助函数将3D转换为2D
    pred_2d, gt_labels_2d, (H, W) = convert_3d_to_2d_predictions(
        pred_3d, gt_labels_3d, shape_3d, dim_order
    )

    # 2. 转换为numpy进行计算
    if torch.is_tensor(pred_2d):
        pred_2d_np = pred_2d.detach().cpu().numpy()
    else:
        pred_2d_np = pred_2d

    N = pred_2d_np.shape[0]

    # 3. 解析GT的2D坐标（用于计算距离误差）
    gt_h = gt_labels_2d // W
    gt_w = gt_labels_2d % W

    # 4. Flatten 2D概率图
    pred_2d_flat = pred_2d_np.reshape(N, H * W)  # [N, H*W]

    # 6. 计算Top-K准确率
    sorted_indices = np.argsort(pred_2d_flat, axis=-1)[:, ::-1]  # [N, H*W] 降序

    results = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]  # [N, K]
        correct = np.any(top_k_indices == gt_labels_2d[:, None], axis=1)  # [N]
        accuracy = correct.mean() * 100
        results[f'2D_top{k}_acc'] = accuracy

    # 7. 计算排名和距离误差
    ranks = []
    distances = []

    for i in range(N):
        # 排名
        rank = np.where(sorted_indices[i] == gt_labels_2d[i])[0][0] + 1
        ranks.append(rank)

        # 预测位置（概率最大的位置）
        pred_idx_flat = sorted_indices[i, 0]
        pred_h, pred_w = np.unravel_index(pred_idx_flat, (H, W))

        # 欧氏距离误差（像素）
        dist = np.sqrt((pred_h - gt_h[i]) ** 2 + (pred_w - gt_w[i]) ** 2)
        distances.append(dist)

    ranks = np.array(ranks)
    distances = np.array(distances)

    results['2D_mean_rank'] = ranks.mean()
    results['2D_median_rank'] = np.median(ranks)
    results['2D_mean_distance_error'] = distances.mean()
    results['2D_median_distance_error'] = np.median(distances)

    return results


def compute_2d_neighbors_recall(pred_pdf, gt_labels, len_neighbors=2,
                                 k_values=None, shape_2d=None, shape_3d=None,
                                 dim_order="HWO", title="2D邻域Recall"):
    """
    计算2D平面定位的邻域recall

    这个函数计算预测的邻域cell中是否包含GT位置，用于评估定位精度的容错性。
    支持直接传入3D预测（会自动边缘化旋转维度），也支持直接传入2D预测。

    Args:
        pred_pdf: np.array or torch.Tensor
                  - 2D输入: shape [N, H*W]，需要提供shape_2d
                  - 3D输入: shape [N, H, W] （已经边缘化过旋转维度）
                  - 4D输入: shape [N, H, W, O] 或 [N, O, H, W]，需要提供shape_3d或dim_order
        gt_labels: np.array or torch.Tensor, shape [N]
                   - 如果pred_pdf是2D/3D: 2D GT标签（flatten索引，基于H*W）或 [N, 2] 的2D坐标
                   - 如果pred_pdf是4D: 3D GT标签（flatten索引，基于[H, W, O]），会自动转换为2D
        len_neighbors: int, 邻域长度（例如2表示2x2=4个邻域，3表示3x3=9个邻域）
        k_values: list of int, 要计算的k值列表
                  如果为None，则自动生成 [1, 4, 9, ..., len_neighbors^2]
        shape_2d: tuple or None, (H, W)
                  仅当pred_pdf是2D时需要提供
        shape_3d: tuple or None, (H, W, O)
                  仅当pred_pdf是4D (包含旋转维度) 时需要提供
        dim_order: str, "HWO" 或 "OHW"
                   当pred_pdf是4D时，指定维度顺序
        title: str, 打印结果的标题

    Returns:
        dict: 包含 recall@k 的字典

    Example:
        >>> # 示例1: 直接使用2D预测
        >>> pred_2d = np.random.rand(100, 40, 30)  # [N, H, W]
        >>> gt_labels_2d = np.random.randint(0, 1200, 100)  # [N]
        >>> results = compute_2d_neighbors_recall(pred_2d, gt_labels_2d, len_neighbors=3)

        >>> # 示例2: 从3D预测自动转换
        >>> pred_3d = np.random.rand(100, 40, 30, 12)  # [N, H, W, O]
        >>> gt_labels_3d = np.random.randint(0, 14400, 100)  # [N] (基于H*W*O)
        >>> results = compute_2d_neighbors_recall(
        ...     pred_3d, gt_labels_3d, len_neighbors=3,
        ...     shape_3d=(40, 30, 12)
        ... )
    """
    # 1. 检查是否为3D/4D输入（包含旋转维度），如果是则转换为2D
    if (torch.is_tensor(pred_pdf) and pred_pdf.ndim == 4) or \
       (isinstance(pred_pdf, np.ndarray) and pred_pdf.ndim == 4):
        # 4D输入，需要边缘化旋转维度
        pred_pdf_2d, gt_labels_2d, (H, W) = convert_3d_to_2d_predictions(
            pred_pdf, gt_labels, shape_3d, dim_order
        )
    else:
        # 2D或3D输入，直接使用
        pred_pdf_2d = pred_pdf
        gt_labels_2d = gt_labels
        H, W = None, None  # 稍后会确定

    # 2. 数据准备 - 转换为torch tensor
    if not torch.is_tensor(pred_pdf_2d):
        pred_pdf_2d = torch.from_numpy(pred_pdf_2d)
    if not torch.is_tensor(gt_labels_2d):
        gt_labels_2d = torch.from_numpy(gt_labels_2d)

    # 3. 处理输入格式
    if pred_pdf_2d.ndim == 2:
        # [N, H*W] 格式，需要reshape
        if shape_2d is None:
            raise ValueError("当pred_pdf是2D时，必须提供shape_2d参数 (H, W)")
        H, W = shape_2d
        N = pred_pdf_2d.shape[0]
        pred_pdf_2d = pred_pdf_2d.reshape(N, H, W)
    elif pred_pdf_2d.ndim == 3:
        # [N, H, W] 格式，直接使用
        N, H, W = pred_pdf_2d.shape
    else:
        raise ValueError(f"pred_pdf_2d维度必须是2D或3D，got {pred_pdf_2d.ndim}D")

    # 4. 处理GT标签
    if gt_labels_2d.ndim == 1:
        # 一维索引 [N]，保持不变
        gt_labels_flat = gt_labels_2d.long()
    elif gt_labels_2d.ndim == 2 and gt_labels_2d.shape[1] == 2:
        # 二维坐标 [N, 2]，转换为一维索引
        gt_labels_flat = gt_labels_2d[:, 0] * W + gt_labels_2d[:, 1]
        gt_labels_flat = gt_labels_flat.long()
    else:
        raise ValueError(f"gt_labels_2d格式不正确，应该是[N]或[N,2]，got shape {gt_labels_2d.shape}")

    # 5. 计算每个样本的邻域cell索引
    # 使用与训练代码一致的邻域计算逻辑
    device = pred_pdf_2d.device

    # 找到和最大的 len_neighbors x len_neighbors 区域的左上角坐标
    # 使用卷积找到每个样本的最优邻域
    kernel = torch.ones((len_neighbors, len_neighbors), dtype=torch.float32, device=device)
    input_4d = pred_pdf_2d.unsqueeze(1)  # [N, 1, H, W]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]

    # 卷积计算每个 n x n 窗口的和
    sum_map = torch.nn.functional.conv2d(input_4d, kernel_4d, padding=0)  # [N, 1, H', W']
    sum_map_squeezed = sum_map.squeeze(1)  # [N, H', W']

    out_height, out_width = sum_map_squeezed.shape[1], sum_map_squeezed.shape[2]
    sum_map_flat = sum_map_squeezed.view(N, -1)

    # 找到最大和的位置（左上角坐标）
    _, top_left_flat_indices = torch.max(sum_map_flat, dim=1)  # [N]
    top_left_rows = top_left_flat_indices // out_width
    top_left_cols = top_left_flat_indices % out_width
    id_toplefts = torch.stack([top_left_rows, top_left_cols], dim=1)  # [N, 2]

    # 生成邻域内所有点的相对偏移
    row_offsets, col_offsets = torch.meshgrid(
        torch.arange(len_neighbors, device=device),
        torch.arange(len_neighbors, device=device),
        indexing='ij'
    )
    offsets = torch.stack([row_offsets.flatten(), col_offsets.flatten()], dim=1)  # [n*n, 2]

    # 计算所有邻域cell的坐标
    id_neighbors_2d = id_toplefts.unsqueeze(1) + offsets.unsqueeze(0)  # [N, n*n, 2]

    # 转换为一维索引
    id_neighbors_flat = id_neighbors_2d[..., 0] * W + id_neighbors_2d[..., 1]  # [N, n*n]

    # 6. 计算recall@k
    if k_values is None:
        # 自动生成k值：1, 4, 9, ..., n^2
        max_k = len_neighbors ** 2
        k_values = [k for k in [1, 4, 9, 16, 25] if k <= max_k]
        if max_k not in k_values:
            k_values.append(max_k)

    # 转为numpy进行计算（更方便）
    id_neighbors_np = id_neighbors_flat.cpu().numpy()  # [N, n*n]
    gt_labels_np = gt_labels_flat.cpu().numpy()  # [N]

    recall_dict = {}
    print(f"\n{title}:")
    print(f"  邻域大小: {len_neighbors}×{len_neighbors} = {len_neighbors**2} cells")
    print(f"  样本数: {N}")

    for k in k_values:
        if k > len_neighbors ** 2:
            print(f"  ⚠️  警告: k={k} 超过邻域总数 {len_neighbors**2}，跳过")
            continue

        # 检查GT是否在top-k邻域中
        top_k_neighbors = id_neighbors_np[:, :k]  # [N, k]
        correct = np.any(top_k_neighbors == gt_labels_np[:, None], axis=1)  # [N]
        recall = correct.mean() * 100

        recall_dict[f'recall@{k}'] = recall
        print(f"  Recall@{k}: {recall:.2f}%")

    return recall_dict


def compute_rotation_accuracy_at_gt_position(pred_3d, gt_labels_3d, shape_3d=None, dim_order="HWO", k_values=[1, 2, 3]):
    """
    在2D平面GT位置处，评估旋转方向的预测准确率

    Args:
        pred_3d: np.array or torch.Tensor
                 - 如果是2D: shape [N, H*W*O]，需要提供shape_3d
                 - 如果是4D: shape [N, H, W, O] 或 [N, O, H, W]，根据dim_order判断
        gt_labels_3d: np.array, shape [N]
                      3D GT标签（flatten索引，基于[H, W, O]顺序）
        shape_3d: tuple or None, (H, W, O)
                  仅当pred_3d是2D时需要提供
        dim_order: str, "HWO" 或 "OHW"
                   指定pred_3d的维度顺序（仅对4D输入有效）
        k_values: list, 要计算的K值列表，默认 [1, 2, 3]

    Returns:
        dict: 包含旋转方向预测的各种准确率指标
    """
    # 1. 数据准备
    if torch.is_tensor(pred_3d):
        pred_3d_np = pred_3d.detach().cpu().numpy()
    else:
        pred_3d_np = pred_3d

    if torch.is_tensor(gt_labels_3d):
        gt_labels_np = gt_labels_3d.detach().cpu().numpy()
    else:
        gt_labels_np = gt_labels_3d

    N = len(gt_labels_np)

    # 2. 处理不同的输入格式
    if pred_3d_np.ndim == 2:
        # 2D输入，需要shape_3d
        if shape_3d is None:
            raise ValueError("当pred_3d是2D时，必须提供shape_3d参数")
        H, W, O = shape_3d
        pred_3d_np = pred_3d_np.reshape(N, H, W, O)
    elif pred_3d_np.ndim == 4:
        # 4D输入，根据dim_order判断
        if dim_order == "OHW":
            # [N, O, H, W] -> [N, H, W, O]
            pred_3d_np = pred_3d_np.transpose(0, 2, 3, 1)
        elif dim_order == "HWO":
            # 已经是 [N, H, W, O]，不需要转换
            pass
        else:
            raise ValueError(f"dim_order必须是'HWO'或'OHW'，got {dim_order}")

        N, H, W, O = pred_3d_np.shape
    else:
        raise ValueError(f"pred_3d维度必须是2D或4D，got {pred_3d_np.ndim}D")

    # 3. 解析GT标签得到 (h, w, o) 坐标
    gt_coords_3d = np.array([np.unravel_index(int(label), (H, W, O)) for label in gt_labels_np])
    gt_h = gt_coords_3d[:, 0]  # [N]
    gt_w = gt_coords_3d[:, 1]  # [N]
    gt_o = gt_coords_3d[:, 2]  # [N]

    # 4. 在GT的2D位置处提取旋转方向的概率分布
    rot_probs_at_gt_pos = np.zeros((N, O))
    for i in range(N):
        rot_probs_at_gt_pos[i] = pred_3d_np[i, gt_h[i], gt_w[i], :]  # [O]

    # 5. 计算Top-K准确率
    sorted_indices = np.argsort(rot_probs_at_gt_pos, axis=-1)[:, ::-1]  # [N, O] 降序

    results = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]  # [N, K]
        correct = np.any(top_k_indices == gt_o[:, None], axis=1)  # [N]
        accuracy = correct.mean() * 100
        results[f'Rot_top{k}_acc'] = accuracy

    # 6. 计算角度距离误差（考虑循环性）
    pred_o = sorted_indices[:, 0]  # 预测的旋转索引 [N]

    # 计算角度（度）
    angle_per_bin = 360.0 / O
    pred_angles = pred_o * angle_per_bin
    gt_angles = gt_o * angle_per_bin

    # 计算角度差（考虑循环，取最小角度差）
    angle_diffs = np.abs(pred_angles - gt_angles)
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)  # 处理跨0度的情况

    results['Rot_mean_angle_error'] = angle_diffs.mean()
    results['Rot_median_angle_error'] = np.median(angle_diffs)

    # 7. 计算排名
    ranks = []
    for i in range(N):
        rank = np.where(sorted_indices[i] == gt_o[i])[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    results['Rot_mean_rank'] = ranks.mean()
    results['Rot_median_rank'] = np.median(ranks)

    return results

def print_accuracy_results(results, title="3D定位准确率"):
    """打印准确率结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for key, value in results.items():
        if 'acc' in key:
            print(f"{key}: {value:.2f}%")
        elif 'rank' in key:
            print(f"{key}: {value:.2f}")
        elif 'angle' in key:
            # 角度误差用度数显示
            print(f"{key}: {value:.2f}°")
        elif 'distance' in key or 'error' in key:
            print(f"{key}: {value:.2f}")
    print(f"{'='*60}\n")


# === 使用示例 ===
# 设置 tolerance=0 表示必须完全重合，=2 表示允许2像素误差
# diagnose_rotation_conditional(pred_filtered, gt_label, layout='OHW', spatial_tolerance=1.5)

# 使用示例 (假设你有一个 batch 的数据)
# diagnose_rotation_at_gt(pred_vol_softmax, gt_indices_3d)

# === 使用示例 ===
if __name__ == "__main__":
    exp_dir = '/home/data/zwk/pyproj_neuloc_v0/trainers/exps/stage3_neural_proxy_nr40_nc30_r36_85_weightmaxCL_best/seq_loc_results'
    p2pred3d = os.path.join(exp_dir,'pred_3d_ep003_nr40_nc30_nrot36_Ttrain32.85_Ttest32.85.npz')
    data = np.load(p2pred3d, allow_pickle=True)
    shape = data['n_coarse_3d']
    gt_label = data['q_label_3d_all']
    pred_3d = data['pred_pdf_3d_all']
    pred_3d = torch.from_numpy(pred_3d.reshape(-1,*shape))
    p2filtered = os.path.join(exp_dir,'pred_3d_ep002_nr40_nc30_nrot36_Ttrain0.05_Ttest0.05_2dTop9Acc91.4.pt')
    pred_3d_filtered = torch.load(p2filtered)
    gt_label_shaped = np.unravel_index(gt_label,shape)

    # === 计算2D平面定位精度 ===
    print("\n" + "="*60)
    print("计算2D平面定位准确率...")
    print("="*60)
    # 对滤波前的预测计算2D准确率
    results_2d_before = compute_2d_plane_accuracy(
        pred_3d,
        gt_label,
        shape_3d=shape,
        k_values=[1, 2, 3, 4, 9, 16]
    )
    print_accuracy_results(results_2d_before, title="2D平面定位准确率 (滤波前)")

    # 对滤波后的预测计算2D准确率
    results_2d_after = compute_2d_plane_accuracy(
        pred_3d_filtered,
        gt_label,
        shape_3d=shape,
        k_values=[1, 2, 3, 4, 9, 16]
    )
    print_accuracy_results(results_2d_after, title="2D平面定位准确率 (滤波后)")

    # === 计算在GT位置处的旋转准确率 ===
    print("\n" + "="*60)
    print("计算在GT 2D位置处的旋转方向准确率...")
    print("="*60)

    # 对滤波前的预测计算旋转准确率
    results_rot_before = compute_rotation_accuracy_at_gt_position(
        pred_3d,
        gt_label,
        shape_3d=shape,
        k_values=[1, 2, 3]
    )
    print_accuracy_results(results_rot_before, title="旋转方向准确率@GT位置 (滤波前)")

    # 对滤波后的预测计算旋转准确率
    results_rot_after = compute_rotation_accuracy_at_gt_position(
        pred_3d_filtered,
        gt_label,
        shape_3d=shape,
        dim_order='HWO',
        k_values=[1, 2, 3]
    )
    print_accuracy_results(results_rot_after, title="旋转方向准确率@GT位置 (滤波后)")

    # === 计算3D定位精度（可选）===
    # 如果你也想看3D的准确率对比
    # pred_3d_flat = pred_3d.reshape(pred_3d.shape[0], -1).numpy()
    # pred_3d_filtered_flat = pred_3d_filtered.reshape(pred_3d_filtered.shape[0], -1).numpy()
    # results_3d_before = compute_top_k_accuracy(pred_3d_flat, gt_label, k_values=[1, 4, 9, 16, 50])
    # results_3d_after = compute_top_k_accuracy(pred_3d_filtered_flat, gt_label, k_values=[1, 4, 9, 16, 50])
    # print_accuracy_results(results_3d_before, title="3D定位准确率 (滤波前)")
    # print_accuracy_results(results_3d_after, title="3D定位准确率 (滤波后)")

    # === 运行生成视频 ===
    name = os.path.basename(p2pred3d).replace('.npz','.mp4')
    create_localization_video_with_gt(pred_3d.permute(0, 3, 1, 2), gt_label, os.path.join(exp_dir,name))
    # create_localization_video_with_gt(pred_3d_filtered.permute(0, 3, 1, 2), gt_label,os.path.join(exp_dir, "pred_filtered_alpha0.25.mp4"),fps=20)