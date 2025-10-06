import numpy as np
import matplotlib.pyplot as plt
import torch

def calculate_peak_saliency(distance_map, normalize=False, dist2negexp = True,beta=1.0):
    if type(distance_map) == torch.Tensor:
        distance_map = distance_map.detach().cpu().numpy()

    if normalize:
        min_val = np.min(distance_map)
        max_val = np.max(distance_map)
        # 避免除以零
        if (max_val - min_val) > 1e-9:
            distance_map = (distance_map - min_val) / (max_val - min_val)
        # 如果所有值都一样，就变成全零图
        else:
            distance_map = np.zeros_like(distance_map)

    # 我们仍然可以将其转换为置信度图来计算
    if dist2negexp:
        confidence_map = np.exp(-beta * distance_map)

    peak_confidence =  np.max(distance_map)
    # 计算除了峰值以外的平均置信度
    mean_confidence = np.mean(confidence_map)
    # 峰值与平均值的比率
    saliency = peak_confidence / (mean_confidence + 1e-9)
    return saliency

from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图工具包
def visualize_response_map_3d(response_map: np.ndarray,
                              title: str = "Query Response Distribution (3D Surface)",
                              cmap: str = 'coolwarm',
                              normalize_xy: bool = True):
    """
    将2D响应矩阵可视化为3D曲面图。

    Args:
        response_map (np.ndarray): 要可视化的2D NumPy数组，形状为 (H, W)。
                                   Z轴的值将是这个矩阵中的值。
        title (str, optional): 图表的标题。
        cmap (str, optional): 用于曲面着色的颜色映射。'coolwarm' 或 'viridis' 效果很好。
        normalize_xy (bool, optional): 是否将X,Y坐标轴归一化到[-1, 1]范围，
                                     以模仿您提供的参考图样式。Defaults to True.
    """
    if type(response_map) == torch.Tensor:
        response_map = response_map.detach().cpu().numpy()

    if response_map.ndim != 2:
        raise ValueError(f"response_map 必须是2D数组，但收到了 {response_map.ndim}D 数组。")

    H, W = response_map.shape

    # 1. 创建 X, Y 坐标网格
    # np.meshgrid 是创建3D图X,Y坐标的关键
    if normalize_xy:
        # 将坐标归一化到 [-1, 1] 区间
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
    else:
        # 使用原始的像素/索引坐标
        x = np.arange(W)
        y = np.arange(H)

    X, Y = np.meshgrid(x, y)

    # 2. 创建 3D 图形和坐标轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. 绘制 3D 曲面
    # rstride 和 cstride 控制网格线的稀疏程度，1表示最密集
    surf = ax.plot_surface(X, Y, response_map, cmap='coolwarm',
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True)  # 将线宽设为0，或者直接删除linewidth和edgecolor
    # 4. 自定义图表
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.set_zlabel("Response Value", fontsize=12)

    # 设置Z轴的范围，可以根据你的数据进行调整
    # ax.set_zlim(-0.5, 1.0)

    # 调整视角 (elevation, azimuth)
    ax.view_init(elev=30, azim=-60)

    # 5. 添加颜色条
    fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1, label="Response Value")

    plt.tight_layout()
    plt.show()


def visualize_response_map(response_map: np.ndarray,
                           ground_truth_idx: tuple = None,
                           mark_extreme: str = None,  # 'max' or 'min'
                           title: str = "Query Response Distribution",
                           cmap: str = 'viridis'):
    """
    可视化一个2D响应矩阵，并可选地标记真值点和响应的最大/最小值点。
    当真值点和极值点重合时，使用不同的标记符号和/或微调文本位置以区分。

    Args:
        response_map (np.ndarray): 要可视化的2D NumPy数组，表示查询在地图上的响应分布。
                                   形状应为 (H, W)。
        ground_truth_idx (tuple, optional): 真值点在 response_map 中的 (行, 列) 索引。
                                            例如 (y, x)。如果为 None，则不绘制真值点。
                                            Defaults to None.
        mark_extreme (str, optional): 指定要标记的极值类型。
                                      - 'max': 标记最大响应值的位置。
                                      - 'min': 标记最小响应值的位置。
                                      - None: 不标记任何极值。
                                      Defaults to None.
        title (str, optional): 图表的标题。Defaults to "Query Response Distribution".
        cmap (str, optional): 用于绘制热力图的颜色映射。Defaults to 'viridis'.
    """
    if response_map.ndim != 2:
        raise ValueError(f"response_map 必须是2D数组，但收到了 {response_map.ndim}D 数组。")

    plt.figure(figsize=(9, 8))
    plt.imshow(response_map, cmap=cmap, origin='lower', interpolation='nearest')

    cbar = plt.colorbar(label="Response Value")
    cbar.ax.tick_params(labelsize=10)

    # 存储标记点的坐标，用于判断是否重合
    gt_pos = None
    extreme_pos = None

    # 标记真值点
    if ground_truth_idx is not None:
        if not isinstance(ground_truth_idx, tuple) or len(ground_truth_idx) != 2:
            raise ValueError("ground_truth_idx 必须是包含 (行, 列) 的元组。")

        gt_y, gt_x = ground_truth_idx
        gt_pos = (gt_x, gt_y)  # 存储为 (x, y) 格式

        if not (0 <= gt_y < response_map.shape[0] and 0 <= gt_x < response_map.shape[1]):
            print(f"警告: 真值点索引 {ground_truth_idx} 超出了矩阵范围 {response_map.shape}，将不绘制。")
        else:
            # 原始 GT 标记为红色星号
            plt.plot(gt_x, gt_y, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5,
                     label=f'Ground Truth ({gt_x}, {gt_y})', zorder=5)
            plt.text(gt_x + 0.5, gt_y + 0.8, 'GT', color='white', fontsize=12, ha='left', va='center',
                     fontweight='bold', zorder=6)

    # 标记最大/最小值
    extreme_label = None
    if mark_extreme in ['max', 'min']:
        if mark_extreme == 'max':
            extreme_idx_flat = np.argmax(response_map)
            marker_color = 'magenta'  # 颜色不变
            marker_style = 'X'  # 标记符号：Max 用 'X'
            text_str = 'Max'
        else:  # mark_extreme == 'min'
            extreme_idx_flat = np.argmin(response_map)
            marker_color = 'cyan'  # 颜色不变
            marker_style = 'D'  # 标记符号：Min 用菱形 'D'
            text_str = 'Min'

        extreme_y, extreme_x = np.unravel_index(extreme_idx_flat, response_map.shape)
        extreme_pos = (extreme_x, extreme_y)  # 存储为 (x, y) 格式
        extreme_label = f'{text_str} ({extreme_x}, {extreme_y})'

        # 判断 GT 和极值点是否重合或非常接近
        is_coincident = False
        if gt_pos is not None and extreme_pos is not None:
            # 可以定义一个小的容忍度，这里简单用完全重合判断
            if gt_pos == extreme_pos:
                is_coincident = True

        # 绘制极值点标记
        if is_coincident:
            # 如果重合，使用一个组合标记，或者调整位置
            # 这里我们选择在同一个位置绘制，但使用不同的 marker style
            # 文本则稍微错开，以显示两者
            # 为了让GT和极值都能被清晰看到，这里让极值标记稍微小一点，或调整zorder
            plt.plot(extreme_x, extreme_y, marker_style, color=marker_color, markersize=12,
                     markeredgecolor='white', markeredgewidth=1.5, label=extreme_label, zorder=4)  # 稍微低一点的zorder
            # 文本位置调整，确保不完全遮挡 GT 文本
            plt.text(extreme_x - 0.5, extreme_y + 0.8, text_str,
                     color=marker_color, fontsize=12, ha='right', va='center', fontweight='bold', zorder=6)
        else:
            # 不重合时正常绘制
            plt.plot(extreme_x, extreme_y, marker_style, color=marker_color, markersize=12,
                     markeredgecolor='white', markeredgewidth=1.5, label=extreme_label, zorder=5)
            # 文本位置默认放在标记右侧
            plt.text(extreme_x + 0.5, extreme_y + 0.5, text_str,
                     color='white', fontsize=12, ha='left', va='center', fontweight='bold', zorder=6)

    # 统一显示图例
    if ground_truth_idx is not None or extreme_label is not None:
        plt.legend(loc='upper right', fontsize=10, facecolor='lightgray', framealpha=0.8)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("X-coordinate (Column Index)", fontsize=12)
    plt.ylabel("Y-coordinate (Row Index)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7, color='lightgray')  # 调整网格线颜色
    plt.tight_layout()
    plt.show()


# --- 示例使用 ---
if __name__ == "__main__":
    # 示例1: 随机生成的响应矩阵，带真值点和最大值
    response_matrix_1 = np.random.rand(20, 30) * 10  # 乘以10让数值范围更大
    gt_index_1 = (5, 12)
    visualize_response_map(response_matrix_1, ground_truth_idx=gt_index_1,
                           mark_extreme='max',
                           title="Random Response Map (GT + Max)", cmap='viridis')

    # 示例2: 模拟一个峰值响应的地图，真值点在峰值附近，同时标记最小值
    H, W = 40, 50
    response_matrix_2 = np.zeros((H, W))
    peak_y, peak_x = 20, 25

    for y in range(H):
        for x in range(W):
            response_matrix_2[y, x] = np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / (2 * 5 ** 2))

    # 在某个角落添加一个非常低的值，以便标记最小值
    response_matrix_2[0, 0] = -0.5

    gt_index_2 = (peak_y + 1, peak_x - 2)
    visualize_response_map(response_matrix_2, ground_truth_idx=gt_index_2,
                           mark_extreme='min',
                           title="Gaussian Response Map (GT + Min)", cmap='magma')

    # 示例3: 仅标记最大值
    response_matrix_3 = np.random.rand(15, 15) * 5
    visualize_response_map(response_matrix_3, mark_extreme='max',
                           title="Random Response Map (Only Max)", cmap='hot')

    # 示例4: 仅标记最小值
    response_matrix_4 = np.random.rand(10, 10) - 0.5  # 包含负值
    visualize_response_map(response_matrix_4, mark_extreme='min',
                           title="Random Response Map (Only Min)", cmap='cividis')

    # 示例5: 同时标记真值点、最大值和最小值（需要对函数进行少量调整，见下方说明）
    # 但由于 mark_extreme 只能选择 'max' 或 'min'，直接使用会报错
    # 如果要同时显示，需要将 mark_extreme 扩展为列表或在内部处理两个
    # 这里为了演示，我们先只标记最大值
    response_matrix_5 = np.random.rand(25, 25)
    response_matrix_5[5, 5] = 10  # 制造一个最大值
    response_matrix_5[20, 20] = -5  # 制造一个最小值
    gt_idx_5 = (10, 10)
    visualize_response_map(response_matrix_5, ground_truth_idx=gt_idx_5, mark_extreme='max',
                           title="Custom Response Map (GT + Max)", cmap='jet')