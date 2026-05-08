import plotly.graph_objects as go
import numpy as np
import torch

def visualize_3d_field(volume,
                       gt_coord=None,  # [x, y, z] 物理坐标 (或者 None)
                       resolution=1.0,  # 体素到物理坐标的缩放因子
                       better='max',  # 'max' (越大越好) or 'min' (越小越好)
                       mark_pred=True,  # 是否自动标记预测出的最优点
                       save_path=None):
    """
    通用3D标量场可视化函数 (Point Cloud 风格)
    """

    # --- 1. 数据清洗 ---
    if torch.is_tensor(volume): volume = volume.detach().cpu().numpy()
    if volume.ndim == 4: volume = volume[0]  # Handle (1, D, H, W)

    # 获取原始数据统计
    v_min, v_max = volume.min(), volume.max()

    # --- 2. 计算可视化用的"重要性" (Importance) ---
    # Importance 用于控制 透明度(Opacity) 和 大小(Size)
    # 无论原始值是距离还是相似度，我们都希望"最好的点" Importance = 1

    eps = 1e-6
    if better == 'max':
        # 归一化到 [0, 1]
        importance = (volume - v_min) / (v_max - v_min + eps)
        colorscale = 'Jet'  # 红=高(好)
        pred_idx = np.unravel_index(np.argmax(volume), volume.shape)
    else:  # better == 'min'
        # 距离越小越好，反转归一化
        importance = 1.0 - (volume - v_min) / (v_max - v_min + eps)
        colorscale = 'Jet_r'  # 翻转色标，蓝=高(坏), 红=低(好)
        pred_idx = np.unravel_index(np.argmin(volume), volume.shape)

    # --- 3. 过滤背景噪声 ---
    # 只显示 Importance > 0.15 的点，让图像更干净
    mask = importance > 0.15
    z_idx, y_idx, x_idx = np.where(mask)

    if len(z_idx) == 0:
        print("Warning: Field is uniform or empty.")
        return

    # 提取符合条件的数值
    val_points = volume[mask]
    imp_points = importance[mask]

    # 坐标转换 (Index -> Physical)
    # 假设 volume 顺序为 [Z, Y, X] (常见深度学习习惯) 或 [D, H, W]
    # 如果你的顺序不同，请调整这里的对应关系
    px = x_idx * resolution
    py = y_idx * resolution
    pz = z_idx * resolution

    # --- 4. 绘图 ---
    fig = go.Figure()

    # A. 标量场云图 (Cloud)
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='markers',
        marker=dict(
            size=imp_points,  # 重要性越高，点越大
            sizeref=imp_points.max() / 5.0,
            sizemode='diameter',
            color=val_points,  # 颜色对应原始物理值
            colorscale=colorscale,
            opacity=imp_points * 0.8,  # 重要性越高，越不透明
            line=dict(width=0)
        ),
        name='Field Value',
        hovertemplate='X:%{x:.1f}<br>Y:%{y:.1f}<br>Z:%{z:.1f}<br>Val:%{marker.color:.3f}<extra></extra>'
    ))

    # B. 标记预测点 (Pred)
    if mark_pred:
        # 获取预测点的物理坐标
        pred_z, pred_y, pred_x = pred_idx
        pred_phys = np.array([pred_x, pred_y, pred_z]) * resolution

        pred_val = volume[pred_idx]

        fig.add_trace(go.Scatter3d(
            x=[pred_phys[0]], y=[pred_phys[1]], z=[pred_phys[2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='cross', line=dict(width=2, color='white')),
            name='Prediction',
            text=[f'Pred: {pred_val:.2f}'],
            textposition="top center"
        ))

    # C. 标记真值点 (GT)
    if gt_coord is not None:
        # 假设传入的 gt_coord 已经是物理坐标 [x, y, z]
        gt_x, gt_y, gt_z = gt_coord

        fig.add_trace(go.Scatter3d(
            x=[gt_x], y=[gt_y], z=[gt_z],
            mode='markers+text',
            marker=dict(size=10, color='green', symbol='diamond', line=dict(width=2, color='white')),
            name='Ground Truth',
            text=['GT'],
            textposition="top center"
        ))

        # 可选：画一条连接 Pred 和 GT 的线，直观显示误差
        if mark_pred:
            fig.add_trace(go.Scatter3d(
                x=[pred_phys[0], gt_x], y=[pred_phys[1], gt_y], z=[pred_phys[2], gt_z],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                name='Error Line',
                hoverinfo='skip'
            ))

    # --- 5. 布局设置 ---
    fig.update_layout(
        title=f"3D Field Visualization (Better={better})",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            aspectmode='data'  # 保持物理比例
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        width=1000, height=800
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")

    return fig