import matplotlib.pyplot as plt
import numpy as np


def visualize_angle_distribution(pred_deg, gt_deg, save_path=None):
    """
    bias_correction: 假如我们要手动修正偏差，填入计算出的度数 (比如 75)
    """
    # 转换数据为 numpy
    if hasattr(pred_deg, 'cpu'): pred_deg = pred_deg.cpu().numpy()
    if hasattr(gt_deg, 'cpu'): gt_deg = gt_deg.cpu().numpy()

    # 统一转换到 [0, 360) 区间，方便画直方图比较
    pred_360 = (pred_deg % 360 + 360) % 360
    gt_360 = (gt_deg % 360 + 360) % 360

    plt.figure(figsize=(15, 6))

    # --- 图1: 线性直方图 (Linear Histogram) ---
    plt.subplot(1, 2, 1)
    # GT 分布
    plt.hist(gt_360, bins=72, range=(0, 360), color='green', alpha=0.5, label='Ground Truth', density=True)
    # Pred 分布
    plt.hist(pred_360, bins=72, range=(0, 360), color='blue', alpha=0.5, label='Prediction', density=True)

    plt.xlabel('Angle (0-360 degrees)')
    plt.ylabel('Density')
    plt.title('Angle Distribution Comparison (Linear)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 图2: 极坐标直方图 (Polar/Rose Plot) ---
    # 这种图能完美展示角度的周期性和方向性
    ax = plt.subplot(1, 2, 2, projection='polar')

    # 转换成弧度用于极坐标绘图
    bins_rad = np.linspace(0.0, 2 * np.pi, 37)  # 36个bin，每个10度

    # 绘制 GT (绿色)
    hist_gt, _ = np.histogram(np.deg2rad(gt_360), bins=bins_rad)
    ax.bar(bins_rad[:-1], hist_gt, width=(2 * np.pi) / 36, bottom=0.0, color='green', alpha=0.4, label='GT')

    # 绘制 Pred (蓝色)
    hist_pred, _ = np.histogram(np.deg2rad(pred_360), bins=bins_rad)
    ax.bar(bins_rad[:-1], hist_pred, width=(2 * np.pi) / 36, bottom=0.0, color='blue', alpha=0.4, label='Pred')

    # 设置极坐标方向 (0度朝北，顺时针) - 依照地图习惯
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    plt.title('Angle Distribution (Polar View)')
    plt.legend(loc='lower right')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

