import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def visualize_coarse_partition_3d(
        ranges,# [4, 2]
        n_coarse,# [4]
        coord_gt,
        candidates,
        energies,
        title="Coarse Space Partition Visualization"
):
    """
    可视化 3D 粗空间划分 (NR, NC, Rot)。

    Args:
        sampler: SubspaceSampler 实例
        coord_gt: Ground Truth 坐标 [4] (nr, nc, rot, scale)
        candidates: 候选点坐标列表 [N, 4]
        energies: 候选点对应的能量值 [N]
        title: 图表标题
    """

    # 1. 数据预处理 (转为 Numpy)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    gt = to_numpy(coord_gt)
    cands = to_numpy(candidates)
    en = to_numpy(energies)

    # 提取前三个维度的数据
    # 注意：这里我们设定 X轴=NC(Width), Y轴=NR(Height), Z轴=Rot
    # 这样符合通常图像平面 X-宽, Y-高的直觉
    gt_xyz = np.array([gt[1], gt[0], gt[2]])
    cands_xyz = np.stack([cands[:, 1], cands[:, 0], cands[:, 2]], axis=1)

    # 2. 获取空间范围信息
    # ranges = sampler.coord_ranges  # [4, 2]
    # n_coarse = sampler.n_coarse  # [4]

    nr_range = ranges[0]
    nc_range = ranges[1]
    rot_range = ranges[2]

    n_nr, n_nc, n_rot = n_coarse[0], n_coarse[1], n_coarse[2]

    # 3. 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 4. 绘制候选点 (Scatter)
    # 能量越低越好 -> 使用反转色阶，让低能量显示为暖色/高亮色
    img = ax.scatter(
        cands_xyz[:, 0], cands_xyz[:, 1], cands_xyz[:, 2],
        c=en, cmap='viridis_r', s=20, alpha=0.8, label='Candidates'
    )
    cbar = plt.colorbar(img, ax=ax, shrink=0.6)
    cbar.set_label('Energy (Lower is Better)')

    # 5. 绘制 Ground Truth
    ax.scatter(
        gt_xyz[0], gt_xyz[1], gt_xyz[2],
        c='red', marker='*', s=200, edgecolors='black', zorder=10, label='Ground Truth'
    )

    # 6. 绘制粗划分网格 (Coarse Grid Lines)
    # 计算刻度位置
    nr_ticks = np.linspace(nr_range[0], nr_range[1], n_nr + 1)
    nc_ticks = np.linspace(nc_range[0], nc_range[1], n_nc + 1)
    rot_ticks = np.linspace(rot_range[0], rot_range[1], n_rot + 1)

    # 绘制分割平面 (用虚线表示)
    # X planes (NC 固定)
    for x in nc_ticks:
        ax.plot([x, x], [nr_range[0], nr_range[1]], [rot_range[0], rot_range[0]], 'k-', alpha=0.1)  # 底面投影
        ax.plot([x, x], [nr_range[0], nr_range[0]], [rot_range[0], rot_range[1]], 'k-', alpha=0.1)  # 侧面投影

    # Y planes (NR 固定)
    for y in nr_ticks:
        ax.plot([nc_range[0], nc_range[1]], [y, y], [rot_range[0], rot_range[0]], 'k-', alpha=0.1)
        ax.plot([nc_range[0], nc_range[0]], [y, y], [rot_range[0], rot_range[1]], 'k-', alpha=0.1)

    # Z planes (Rot 固定)
    for z in rot_ticks:
        ax.plot([nc_range[0], nc_range[1]], [nr_range[0], nr_range[0]], [z, z], 'k-', alpha=0.1)
        ax.plot([nc_range[0], nc_range[0]], [nr_range[0], nr_range[1]], [z, z], 'k-', alpha=0.1)

    # 7. 高亮 GT 所在的粗子空间 (Highlight the Coarse Voxel containing GT)
    # 计算 GT 所在的 bin 索引
    bin_nr = int((gt[0] - nr_range[0]) / (nr_range[1] - nr_range[0]) * n_nr)
    bin_nc = int((gt[1] - nc_range[0]) / (nc_range[1] - nc_range[0]) * n_nc)
    bin_rot = int((gt[2] - rot_range[0]) / (rot_range[1] - rot_range[0]) * n_rot)

    # 边界保护
    bin_nr = np.clip(bin_nr, 0, n_nr - 1)
    bin_nc = np.clip(bin_nc, 0, n_nc - 1)
    bin_rot = np.clip(bin_rot, 0, n_rot - 1)

    # 获取该 bin 的 3D 边界
    box_nr = [nr_ticks[bin_nr], nr_ticks[bin_nr + 1]]
    box_nc = [nc_ticks[bin_nc], nc_ticks[bin_nc + 1]]
    box_rot = [rot_ticks[bin_rot], rot_ticks[bin_rot + 1]]

    # 绘制该 Voxel 的线框
    def draw_box(ax, x_range, y_range, z_range, color='red', style='--'):
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        # 绘制长方体边框有点繁琐，这里用简单连线模拟
        pts = np.array([
            [x_range[0], y_range[0], z_range[0]],
            [x_range[1], y_range[0], z_range[0]],
            [x_range[1], y_range[1], z_range[0]],
            [x_range[0], y_range[1], z_range[0]],
            [x_range[0], y_range[0], z_range[1]],
            [x_range[1], y_range[0], z_range[1]],
            [x_range[1], y_range[1], z_range[1]],
            [x_range[0], y_range[1], z_range[1]],
        ])
        # 连接底面
        ax.plot(pts[:4, 0], pts[:4, 1], pts[:4, 2], color, linestyle=style, alpha=0.5)
        ax.plot([pts[0, 0], pts[3, 0]], [pts[0, 1], pts[3, 1]], [pts[0, 2], pts[3, 2]], color, linestyle=style,
                alpha=0.5)
        # 连接顶面
        ax.plot(pts[4:, 0], pts[4:, 1], pts[4:, 2], color, linestyle=style, alpha=0.5)
        ax.plot([pts[4, 0], pts[7, 0]], [pts[4, 1], pts[7, 1]], [pts[4, 2], pts[7, 2]], color, linestyle=style,
                alpha=0.5)
        # 连接柱子
        for i in range(4):
            ax.plot([pts[i, 0], pts[i + 4, 0]], [pts[i, 1], pts[i + 4, 1]], [pts[i, 2], pts[i + 4, 2]], color,
                    linestyle=style, alpha=0.5)

    draw_box(ax, box_nc, box_nr, box_rot, color='red', style='--')

    # 8. 设置轴标签和范围
    ax.set_xlabel('NC (Col/Width)')
    ax.set_ylabel('NR (Row/Height)')
    ax.set_zlabel('Rot (Radians)')

    ax.set_xlim(nc_range)
    ax.set_ylim(nr_range)
    ax.set_zlim(rot_range)

    # 翻转 NR 轴 (图像坐标系中，Y轴向下通常对应数值增大，但在3D Plot中通常希望原点在左下或左上)
    # 如果希望符合图像直觉 (Row 0 在上)，可以反转 Y 轴
    ax.invert_yaxis()

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_coarse_partition_interactive(
        ranges,  # [4, 2]
        n_coarse,  # [4]
        coord_gt,
        candidates,
        energies,
        mode='min',
        save_path="coarse_partition_pred_vs_gt.html",
        title=None
):
    """
    生成可交互HTML，同时高亮 GT(红) 和 预测最优值(绿)，并显示完整网格。
    """
    if title is None:
        title = f"Interactive Partition: GT(Red) vs Pred Top1 {mode.upper()}(Green)"

    # 1. 数据转换
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    gt = to_numpy(coord_gt)
    cands = to_numpy(candidates)
    en = to_numpy(energies)

    # 提取坐标: X=NC(Width), Y=NR(Height), Z=Rot
    x_cands, y_cands, z_cands = cands[:, 1], cands[:, 0], cands[:, 2]
    x_gt, y_gt, z_gt = gt[1], gt[0], gt[2]

    # 2. 获取范围和刻度
    nr_range, nc_range, rot_range = ranges[0], ranges[1], ranges[2]
    n_nr, n_nc, n_rot = n_coarse[0], n_coarse[1], n_coarse[2]

    nr_ticks = np.linspace(nr_range[0], nr_range[1], n_nr + 1)
    nc_ticks = np.linspace(nc_range[0], nc_range[1], n_nc + 1)
    rot_ticks = np.linspace(rot_range[0], rot_range[1], n_rot + 1)

    traces = []

    # ================= 内部辅助函数：绘制 Voxel 线框 =================
    def _get_box_trace(coord_x, coord_y, coord_z, color, name, line_width=4):
        """绘制单个 Voxel 的高亮实线框"""
        bin_nr = int((coord_y - nr_range[0]) / (nr_range[1] - nr_range[0]) * n_nr)
        bin_nc = int((coord_x - nc_range[0]) / (nc_range[1] - nc_range[0]) * n_nc)
        bin_rot = int((coord_z - rot_range[0]) / (rot_range[1] - rot_range[0]) * n_rot)

        bin_nr = np.clip(bin_nr, 0, n_nr - 1)
        bin_nc = np.clip(bin_nc, 0, n_nc - 1)
        bin_rot = np.clip(bin_rot, 0, n_rot - 1)

        y_min, y_max = nr_ticks[bin_nr], nr_ticks[bin_nr + 1]
        x_min, x_max = nc_ticks[bin_nc], nc_ticks[bin_nc + 1]
        z_min, z_max = rot_ticks[bin_rot], rot_ticks[bin_rot + 1]

        box_x, box_y, box_z = [], [], []
        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        for s, e in edges:
            box_x.extend([corners[s][0], corners[e][0], None])
            box_y.extend([corners[s][1], corners[e][1], None])
            box_z.extend([corners[s][2], corners[e][2], None])

        return go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode='lines',
            name=name,
            line=dict(color=color, width=line_width),
            hoverinfo='skip'
        )

    # ================= 3. 绘制候选点 =================
    colorscale = 'Viridis_r' if mode == 'min' else 'Viridis'
    trace_cands = go.Scatter3d(
        x=x_cands, y=y_cands, z=z_cands,
        mode='markers',
        name='Candidates',
        marker=dict(
            size=3, color=en, colorscale=colorscale, opacity=0.6,
            showscale=True, colorbar=dict(title=f"Energy", x=0.8)
        ),
        hovertemplate='<b>Cand</b><br>NC: %{x:.3f}<br>NR: %{y:.3f}<br>Rot: %{z:.3f}<br>Val: %{marker.color:.4f}<extra></extra>'
    )
    traces.append(trace_cands)

    # ================= 4. 预测点高亮 (Green) =================
    if mode == 'min':
        best_idx = np.argmin(en)
    else:
        best_idx = np.argmax(en)

    x_pred, y_pred, z_pred = x_cands[best_idx], y_cands[best_idx], z_cands[best_idx]

    # 4.1 绿色十字标记
    traces.append(go.Scatter3d(
        x=[x_pred], y=[y_pred], z=[z_pred],
        mode='markers', name=f'Pred Best',
        marker=dict(symbol='cross', size=12, color='lime', line=dict(width=3, color='darkgreen')),
        hovertemplate=f'<b>Pred</b><br>Val: {en[best_idx]:.4f}<extra></extra>'
    ))
    # 4.2 绿色 Voxel 框
    traces.append(_get_box_trace(x_pred, y_pred, z_pred, color='lime', name='Pred Voxel'))

    # ================= 5. GT 高亮 (Red) =================
    # 5.1 红色菱形标记
    traces.append(go.Scatter3d(
        x=[x_gt], y=[y_gt], z=[z_gt],
        mode='markers', name='Ground Truth',
        marker=dict(symbol='diamond', size=10, color='red', line=dict(width=2, color='black')),
        hovertemplate='<b>GT</b><extra></extra>'
    ))
    # 5.2 红色 Voxel 框
    traces.append(_get_box_trace(x_gt, y_gt, z_gt, color='red', name='GT Voxel'))

    # ================= 6. 恢复完整的网格背景 =================
    grid_x, grid_y, grid_z = [], [], []

    def add_line(p1, p2):
        grid_x.extend([p1[0], p2[0], None])
        grid_y.extend([p1[1], p2[1], None])
        grid_z.extend([p1[2], p2[2], None])

    # X 平面线 (固定 X，画 Y-Z 平面上的框)
    for x in nc_ticks:
        add_line((x, nr_range[0], rot_range[0]), (x, nr_range[1], rot_range[0]))  # 竖线
        add_line((x, nr_range[0], rot_range[1]), (x, nr_range[1], rot_range[1]))  # 竖线
        add_line((x, nr_range[0], rot_range[0]), (x, nr_range[0], rot_range[1]))  # 横线
        add_line((x, nr_range[1], rot_range[0]), (x, nr_range[1], rot_range[1]))  # 横线

        # 补充中间的 Y 刻度线
        for y in nr_ticks:
            add_line((x, y, rot_range[0]), (x, y, rot_range[1]))

    # Y 平面线 (固定 Y，画 X-Z 平面上的框)
    for y in nr_ticks:
        add_line((nc_range[0], y, rot_range[0]), (nc_range[1], y, rot_range[0]))
        add_line((nc_range[0], y, rot_range[1]), (nc_range[1], y, rot_range[1]))

        # 补充中间的 X 刻度线
        for x in nc_ticks:
            add_line((x, y, rot_range[0]), (x, y, rot_range[1]))

    # Z 平面线 (固定 Z，画 X-Y 平面上的框 - 最重要的底面和顶面网格)
    for z in rot_ticks:
        for x in nc_ticks:
            add_line((x, nr_range[0], z), (x, nr_range[1], z))
        for y in nr_ticks:
            add_line((nc_range[0], y, z), (nc_range[1], y, z))

    trace_grid = go.Scatter3d(
        x=grid_x, y=grid_y, z=grid_z,
        mode='lines',
        name='Coarse Grid',
        line=dict(color='gray', width=1, dash='dot'),  # 虚线，更美观
        hoverinfo='skip',
        opacity=0.2,  # 稍微清楚一点
        visible=True  # [重要修复] 默认显示
    )
    traces.append(trace_grid)

    # ================= 7. Layout =================
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='NC (Width) →', range=[nc_range[0], nc_range[1]], backgroundcolor="rgb(245,245,245)"),
            # Y轴反转
            yaxis=dict(title='NR (Height) ↓', range=[nr_range[1], nr_range[0]], backgroundcolor="rgb(245,245,245)"),
            zaxis=dict(title='Rot (Rad)', range=[rot_range[0], rot_range[1]], backgroundcolor="rgb(240,240,240)"),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.7)
        ),
        scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.1, z=2.2)),
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig = go.Figure(data=traces, layout=layout)
    print(f"正在保存 HTML 到: {save_path} ...")
    fig.write_html(save_path)
    print(f"✓ 保存成功！网格线已恢复。")

# ================= 测试代码 =================
if __name__ == "__main__":
    # 模拟数据环境 (与之前相同)
    class MockDataset:
        nr2sample_min, nr2sample_max = -1.0, 1.0
        nc2sample_min, nc2sample_max = -1.0, 1.0
        satimgsize_scale_to_refm_boundary = np.array([0.8, 1.2])


    mock_dataset = MockDataset()
    sampler = SubspaceSampler(mock_dataset, n_coarse=(4, 4, 6, 1))

    coord_gt = torch.tensor([0.2, -0.5, 1.0, 1.0])
    candidates = torch.randn(200, 4) * 0.4 + coord_gt  # 随机撒点
    energies = torch.norm(candidates[:, :3] - coord_gt[:3], dim=1)  # 模拟能量

    visualize_coarse_partition_interactive(sampler, coord_gt, candidates, energies)