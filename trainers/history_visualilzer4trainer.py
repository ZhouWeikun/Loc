
def visualize_udf_gradient_flow(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                                n_samples_per_dim=15, delta=0.15,
                                z_amplification=1.0,
                                adaptive_z_scale=True,
                                save_path="vis_results/udf_flow_raw.html", show_plot=False):
    """
    可视化 UDF 梯度场（Raw数值版本）。
    特点：
    1. 坐标轴显示真实的 Row/Col/Rot 数值。
    2. 使用 Layout 强制视觉为正方体，避免因数值范围差异(0.1 vs 3.14)导致图形畸变。
    3. 包含自适应 Z 轴梯度放大。
    """
    import numpy as np
    import os
    import torch
    import torch.nn.functional as TF
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 0. Eval 模式
    for model in self.param2optimize.values():
        model.eval()
    for model in self.param2freeze.values():
        model.eval()

    # 1. 数据准备
    if query_feat is None or gt_coord_4d is None:
        try:
            uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        except StopIteration:
            uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        uav_img = uav_img[0].to(self.device).unsqueeze(0)
        gt_coord_4d = uav_coords_4d[0].to(self.device)
        query_feat = self._get_feats_fm_imgs(uav_img)

    # 2. 确保 Scale
    if scale_fixed is None:
        scale_fixed = gt_coord_4d[3].item()

    # 3. 构建采样网格 (使用原始物理范围)
    nr_center, nc_center, rot_center, scale_val = gt_coord_4d

    # RC 范围：Center +/- Delta
    nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
    nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)

    # Rot 范围：Center +/- PI/2 (这里范围很大，与 RC 的 0.15 相比差距巨大)
    rot_range = torch.linspace(rot_center - torch.pi / 2, rot_center + torch.pi / 2, n_samples_per_dim,
                               device=self.device)

    grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

    # [N, 4]
    coords_sampled_4d = torch.stack([
        grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
        torch.full_like(grid_nr.flatten(), scale_fixed)
    ], dim=-1)

    # 开启梯度
    coords_sampled_4d.requires_grad_(True)

    # 4. 前向传播
    # 归一化用于输入网络 (coords_sampled_4d 保持 Raw)
    coords_sampled_6d = self.coord_normer.raw_to_norm(coords_sampled_4d, append_linear_rot=True)

    grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
    feats_grid_raw = self._get_feats_fm_grid(grid_input)
    coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
    feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
    feats_grid = TF.normalize(feats_grid, dim=-1)

    # MetricNet
    feats_grid_exp = feats_grid.unsqueeze(0)
    coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])
    N = coords_sampled_4d.shape[0]
    query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
    coords_enc_exp = coords_encoded_metric.unsqueeze(0)

    dist_pred_raw = self.metric_net(query_feat_exp, feats_grid_exp, coords_enc_exp)
    dist_pred = self.softplus(dist_pred_raw) if self.use_softplus else dist_pred_raw
    dist_pred = dist_pred.squeeze(0)

    # 5. 计算梯度
    grad_outputs = torch.ones_like(dist_pred)
    gradients = torch.autograd.grad(dist_pred, coords_sampled_4d, grad_outputs, create_graph=False)[0]

    # 6. 梯度处理
    grad_r = -gradients[:, 0]
    grad_c = -gradients[:, 1]
    grad_rot = -gradients[:, 2]

    # 统计与自适应放大
    print(f"\n[Gradient Stats (Raw Coordinates)]")
    grad_rc_mean = (grad_r.abs().mean() + grad_c.abs().mean()) / 2
    grad_rot_mean = grad_rot.abs().mean()
    print(f"Mean Abs Grad (R/C): {grad_rc_mean:.6f}")
    print(f"Mean Abs Grad (Rot): {grad_rot_mean:.6f}")

    if adaptive_z_scale:
        # 防止除零
        if grad_rot_mean < 1e-9:
            z_amplification = 100.0  # 兜底
        else:
            z_amplification = (grad_rc_mean / grad_rot_mean).item()
        print(f"Adaptive Z-Amp Factor: {z_amplification:.2f}")

    # 应用放大
    grad_rot_amplified = grad_rot * z_amplification

    # 合成向量用于显示 (注意：这里只是为了定箭头的方向)
    grad_vec_vis = torch.stack([grad_r, grad_c, grad_rot_amplified], dim=1)
    grad_vec_norm = TF.normalize(grad_vec_vis, dim=1)  # 归一化模长，只保留方向

    # 7. 转 Numpy 准备绘图
    # === 核心修改：直接使用原始坐标 ===
    coords_np = coords_sampled_4d.detach().cpu().numpy()
    dist_pred_np = dist_pred.detach().cpu().numpy()
    grad_vec_np = grad_vec_norm.detach().cpu().numpy()

    # GT 位置 (原始值)
    gt_r, gt_c, gt_rot = nr_center.item(), nc_center.item(), rot_center.item()

    # 预测最佳位置
    min_idx = np.argmin(dist_pred_np)
    pred_best_coord = coords_np[min_idx]

    # 8. 绘图 (Plotly)
    try:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        # 箭头大小计算
        # 因为坐标轴并未归一化，RC跨度小(0.3)，Rot跨度大(3.0)。
        # 如果用 absolute 大小，箭头在 Rot 轴看起来会很短，在 RC 轴看起来很长。
        # 这里我们取 RC 的步长作为基准，因为我们更关心 RC 平面的定位精度。
        step_r = (coords_np[:, 0].max() - coords_np[:, 0].min()) / n_samples_per_dim
        cone_size = step_r * 2.0  # 稍微大一点

        # (1) 梯度场
        fig.add_trace(go.Cone(
            x=coords_np[:, 0],
            y=coords_np[:, 1],
            z=coords_np[:, 2],  # 使用原始 Rot 值
            u=grad_vec_np[:, 0],
            v=grad_vec_np[:, 1],
            w=grad_vec_np[:, 2],
            sizemode="absolute",
            sizeref=cone_size,
            anchor="tail",
            colorscale='Jet',
            showscale=True,
            colorbar=dict(title='Grad Dir'),
            opacity=0.8,
            name='Gradient Flow',
            hovertemplate='<b>Pos</b>: %{x:.2f}, %{y:.2f}, %{z:.2f}<br><b>Dist</b>: %{customdata:.4f}<extra></extra>',
            customdata=dist_pred_np
        ))

        # (2) GT
        fig.add_trace(go.Scatter3d(
            x=[gt_r], y=[gt_c], z=[gt_rot],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
            name='GT'
        ))

        # (3) Pred Best
        fig.add_trace(go.Scatter3d(
            x=[pred_best_coord[0]], y=[pred_best_coord[1]], z=[pred_best_coord[2]],
            mode='markers',
            marker=dict(size=8, color='yellow', symbol='x', line=dict(width=2, color='black')),
            name='Pred Min',
            hovertext=f"Min Dist: {dist_pred_np.min():.4f}"
        ))

        # (4) Layout 设置 【这是关键】
        # 因为 Rot (±3.14) 的数值范围比 RC (±0.15) 大得多，
        # 如果不强制 AspectRatio，Z轴会拉得极长。
        # 我们强制 x:y:z = 1:1:1，让它们在视觉上构成一个正方体盒子。
        fig.update_layout(
            title=f"UDF Gradient (Raw Coords, Z-Amp=x{z_amplification:.1f})",
            scene=dict(
                xaxis_title='Row (0-1)',
                yaxis_title='Col (0-1)',
                zaxis_title='Rot (Rad)',
                # === 强制视觉比例 ===
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                # 这样 Plotly 会自动缩放轴的刻度间距，使得 0.3 的长度和 6.0 的长度在屏幕上一样长
            ),
            height=700,
            width=900,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 可视化(Raw)已保存至: {save_path}")

        if show_plot:
            fig.show()

    except ImportError:
        print("⚠️ Plotly missing.")

def visualize_udf_field_3d(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                           n_samples_per_dim=32, delta=0.15,
                           rot_span=None,  # <--- 新增：手动控制Z轴范围，默认为 PI
                           save_path="vis_results/udf_field_3d_raw.html", show_plot=False):
    """
    可视化 UDF 标量场 (Raw数值 + 可控旋转范围 + 详细统计)。
    """
    import numpy as np
    import os
    import torch
    import torch.nn.functional as TF
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 0. Eval 模式
    for model in self.param2optimize.values():
        model.eval()
    for model in self.param2freeze.values():
        model.eval()

    # 1. 数据准备
    if query_feat is None or gt_coord_4d is None:
        try:
            uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        except StopIteration:
            uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        uav_img = uav_img[0].to(self.device).unsqueeze(0)
        gt_coord_4d = uav_coords_4d[0].to(self.device)
        query_feat = self._get_feats_fm_imgs(uav_img)

    if scale_fixed is None:
        scale_fixed = gt_coord_4d[3].item()

    # 2. 构建采样网格
    nr_center, nc_center, rot_center, scale_val = gt_coord_4d

    # 默认旋转范围设为 PI (用户要求)
    if rot_span is None:
        rot_span = torch.pi

    # RC 范围：局部窗口
    nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
    nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)

    # Rot 范围：Center +/- (span / 2)
    rot_range = torch.linspace(rot_center - rot_span / 2, rot_center + rot_span / 2, n_samples_per_dim,
                               device=self.device)

    grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

    coords_sampled_4d = torch.stack([
        grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
        torch.full_like(grid_nr.flatten(), scale_fixed)
    ], dim=-1)

    # 3. 推理 (Forward)
    coords_sampled_6d = self.coord_normer.raw_to_norm(coords_sampled_4d, append_linear_rot=True)

    with torch.no_grad():
        grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
        feats_grid_raw = self._get_feats_fm_grid(grid_input)

        coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
        feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
        feats_grid = TF.normalize(feats_grid, dim=-1)
        feats_grid_exp = feats_grid.unsqueeze(0)

        coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])
        N = coords_sampled_4d.shape[0]
        query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
        coords_enc_exp = coords_encoded_metric.unsqueeze(0)

        dist_pred_raw = self.metric_net(query_feat_exp, feats_grid_exp, coords_enc_exp)
        dist_pred = self.softplus(dist_pred_raw) if self.use_softplus else dist_pred_raw
        dist_pred = dist_pred.squeeze(0)

        # 计算 GT UDF
        gt_coord_6d = self.coord_normer.raw_to_norm(gt_coord_4d.unsqueeze(0), append_linear_rot=True)
        udf_gt = self.udf_compter_5d.compute_udf_matrix_from_norm(
            gt_coord_6d[:, :5], coords_sampled_6d[:, :5]
        ).squeeze(0)

    # 4. 数据转换 (Raw Values)
    coords_np = coords_sampled_4d.cpu().numpy()
    dist_pred_np = dist_pred.cpu().numpy()
    udf_gt_np = udf_gt.cpu().numpy()

    # 打印统计信息 (Requirement 3)
    print(f"\n{'=' * 40}")
    print(f"UDF Field Statistics (Scale Fixed: {scale_fixed:.4f})")
    print(f"Sampling Range: RC (+/- {delta}), Rot (+/- {rot_span / 2:.2f})")
    print(f"{'-' * 40}")
    print(f"GT UDF   | Min: {udf_gt_np.min():.4f} | Max: {udf_gt_np.max():.4f} | Mean: {udf_gt_np.mean():.4f}")
    print(
        f"Pred UDF | Min: {dist_pred_np.min():.4f} | Max: {dist_pred_np.max():.4f} | Mean: {dist_pred_np.mean():.4f}")
    print(
        f"Abs Diff | Min: {np.abs(dist_pred_np - udf_gt_np).min():.4f} | Max: {np.abs(dist_pred_np - udf_gt_np).max():.4f}")
    print(f"{'=' * 40}\n")

    # 归一化颜色
    def normalize_data(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    dist_pred_norm_color = normalize_data(dist_pred_np)
    udf_gt_norm_color = normalize_data(udf_gt_np)
    shape_error = np.abs(dist_pred_norm_color - udf_gt_norm_color)

    # 关键点
    best_idx = np.argmin(dist_pred_np)
    coord_pred_best = coords_np[best_idx]
    gt_r, gt_c, gt_rot = nr_center.item(), nc_center.item(), rot_center.item()

    # 打印位置信息
    print(f"{'=' * 40}")
    print(f"Position Information")
    print(f"{'-' * 40}")
    print(f"GT Position:")
    print(f"  Row: {gt_r:.6f}, Col: {gt_c:.6f}, Rot: {gt_rot:.6f} rad ({gt_rot * 180 / np.pi:.2f}°)")
    print(f"\nPredicted Min Position:")
    print \
        (f"  Row: {coord_pred_best[0]:.6f}, Col: {coord_pred_best[1]:.6f}, Rot: {coord_pred_best[2]:.6f} rad ({coord_pred_best[2] * 180 / np.pi:.2f}°)")
    print(f"  Min UDF Value: {dist_pred_np[best_idx]:.6f}")
    print(f"\nPosition Error:")
    rc_error = np.sqrt((coord_pred_best[0] - gt_r )* *2 + (coord_pred_best[1] - gt_c )* *2)
    rot_error_rad = np.abs(coord_pred_best[2] - gt_rot)
    # Handle angle wrap-around for rotation error
    rot_error_rad = np.minimum(rot_error_rad, 2 * np.pi - rot_error_rad)
    print(f"  RC Distance: {rc_error:.6f}")
    print(f"  Rotation Error: {rot_error_rad:.6f} rad ({rot_error_rad * 180 / np.pi:.2f}°)")
    print(f"{'=' * 40}\n")

    # 5. 绘图
    try:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('GT UDF (Raw Coords)', 'Pred UDF (Raw Coords)', 'Shape Error (Norm Diff)'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        marker_common = dict(size=3, opacity=0.5)

        # --- Subplot 1: GT Field ---
        fig.add_trace(go.Scatter3d(
            x=coords_np[:, 0], y=coords_np[:, 1], z=coords_np[:, 2],
            mode='markers',
            marker=dict(**marker_common, color=udf_gt_norm_color, colorscale='Viridis',
                        colorbar=dict(title='GT Norm', x=0.3)),
            hovertemplate='<b>GT Dist</b>: %{customdata:.4f}<br>Rot: %{z:.2f}<extra></extra>',
            customdata=udf_gt_np,
            name='GT Field'
        ), row=1, col=1)

        # GT Marker (Subplot 1)
        fig.add_trace(go.Scatter3d(
            x=[gt_r], y=[gt_c], z=[gt_rot],
            mode='markers', marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
            name='GT Position'
        ), row=1, col=1)

        # --- Subplot 2: Pred Field ---
        fig.add_trace(go.Scatter3d(
            x=coords_np[:, 0], y=coords_np[:, 1], z=coords_np[:, 2],
            mode='markers',
            marker=dict(**marker_common, color=dist_pred_norm_color, colorscale='Viridis',
                        colorbar=dict(title='Pred Norm', x=0.65)),
            hovertemplate='<b>Pred Dist</b>: %{customdata:.4f}<br>Rot: %{z:.2f}<extra></extra>',
            customdata=dist_pred_np,
            name='Pred Field'
        ), row=1, col=2)

        # GT Marker (Subplot 2 - Comparison)
        fig.add_trace(go.Scatter3d(
            x=[gt_r], y=[gt_c], z=[gt_rot],
            mode='markers', marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
            showlegend=False, name='GT Position'
        ), row=1, col=2)

        # Pred Best Marker
        fig.add_trace(go.Scatter3d(
            x=[coord_pred_best[0]], y=[coord_pred_best[1]], z=[coord_pred_best[2]],
            mode='markers', marker=dict(size=10, color='yellow', symbol='x', line=dict(width=2, color='black')),
            name='Pred Min'
        ), row=1, col=2)

        # --- Subplot 3: Error ---
        fig.add_trace(go.Scatter3d(
            x=coords_np[:, 0], y=coords_np[:, 1], z=coords_np[:, 2],
            mode='markers',
            marker=dict(**marker_common, color=shape_error, colorscale='Hot',
                        colorbar=dict(title='Shape Err', x=1.0)),
            hovertemplate='<b>Err</b>: %{marker.color:.3f}<extra></extra>',
            name='Shape Error'
        ), row=1, col=3)

        # GT Marker (Subplot 3 - Reference)
        fig.add_trace(go.Scatter3d(
            x=[gt_r], y=[gt_c], z=[gt_rot],
            mode='markers', marker=dict(size=12, color='blue', symbol='diamond', line=dict(width=2, color='white')),
            showlegend=False, name='GT Position'
        ), row=1, col=3)

        # 6. Layout 设置
        layout_scene = dict(
            xaxis_title='Row (Raw)',
            yaxis_title='Col (Raw)',
            zaxis_title='Rot (Raw)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)  # 强制 1:1:1 视觉比例
        )

        fig.update_layout(
            title=f'UDF Scalar Field (Range={rot_span / np.pi:.1f}π, Scale={scale_fixed:.2f})',
            height=600,
            scene=layout_scene, scene2=layout_scene, scene3=layout_scene
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 标量场可视化已保存至: {save_path}")

        if show_plot:
            fig.show()

    except ImportError:
        print("⚠️ Plotly missing.")


    def visualize_isosurfaces(self, query_feat=None, gt_coord_4d=None,
                              n_samples_per_dim=50, delta=0.2, rot_span=torch.pi,
                              surface_count=5, opacity=0.3,
                              save_path="vis_results/udf_isosurface.html", show_plot=False,
                              use_train_uav=False):
        """
        [新功能] UDF等值面可视化 (Isosurface)
        修正点：
        1. caps参数修复：分别指定x,y,z的show属性。
        2. 视角修正：强制使用立方体视角 (aspectmode='cube')，防止图像变形。
        """
        import numpy as np
        import os
        import torch
        import torch.nn.functional as TF
        import plotly.graph_objects as go

        # 0. 模式切换
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        # 1. 数据准备
        if query_feat is None or gt_coord_4d is None:
            if use_train_uav and hasattr(self, 'uav_dataloader_train'):
                dataloader = self.uav_dataloader_train
                tag = "train"
            elif hasattr(self, 'uav_dataloader_test'):
                dataloader = self.uav_dataloader_test
                tag = "test"
            else:
                raise AttributeError("未找到可用的UAV数据加载器。")

            try:
                uav_img, uav_coords_4d = next(iter(dataloader))
            except StopIteration:
                uav_img, uav_coords_4d = next(iter(dataloader))

            uav_img = uav_img[0].to(self.device).unsqueeze(0)
            gt_coord_4d = uav_coords_4d[0].to(self.device)
            query_feat = self._get_feats_fm_imgs(uav_img)
            print(f"可视化数据来源: {tag}")

        scale_fixed = gt_coord_4d[3].item()
        nr_center, nc_center, rot_center, _ = gt_coord_4d

        # 2. 构建采样网格
        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)
        rot_range = torch.linspace(rot_center - rot_span / 2, rot_center + rot_span / 2, n_samples_per_dim,
                                   device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        coords_sampled_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_fixed)
        ], dim=-1)

        # 3. 推理 UDF
        coords_sampled_6d = self.coord_normer.raw_to_norm(coords_sampled_4d, append_linear_rot=True)

        with torch.no_grad():
            grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
            feats_grid_raw = self._get_feats_fm_grid(grid_input)
            coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
            feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
            feats_grid = TF.normalize(feats_grid, dim=-1)
            feats_grid_exp = feats_grid.unsqueeze(0)

            coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])
            N = coords_sampled_4d.shape[0]
            query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
            coords_enc_exp = coords_encoded_metric.unsqueeze(0)

            dist_pred_raw = self.metric_net(query_feat_exp, feats_grid_exp, coords_enc_exp)
            dist_pred = self.softplus(dist_pred_raw) if self.use_softplus else dist_pred_raw
            dist_pred = dist_pred.squeeze(0)

        # 4. 转 Numpy
        X = coords_sampled_4d[:, 0].cpu().numpy()
        Y = coords_sampled_4d[:, 1].cpu().numpy()
        Z = coords_sampled_4d[:, 2].cpu().numpy()
        V = dist_pred.cpu().numpy()

        min_v, max_v = V.min(), V.max()
        print(f"Isosurface 统计: Min UDF={min_v:.6f}, Max UDF={max_v:.6f}")

        # 5. 绘图
        gt_r, gt_c, gt_rot = nr_center.item(), nc_center.item(), rot_center.item()
        best_idx = np.argmin(V)
        pred_r, pred_c, pred_rot = X[best_idx], Y[best_idx], Z[best_idx]

        fig = go.Figure()

        # Isosurface Trace
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=min_v,
            isomax=max_v,
            surface_count=surface_count,
            colorscale='Plasma',
            opacity=opacity,
            # [修正] Caps 必须分开写
            caps=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False)
            ),
            showscale=True,
            colorbar=dict(title='UDF Value'),
            name='UDF Levels'
        ))

        # GT Point
        fig.add_trace(go.Scatter3d(
            x=[gt_r], y=[gt_c], z=[gt_rot],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
            name='GT 位置'
        ))

        # Pred Min Point
        fig.add_trace(go.Scatter3d(
            x=[pred_r], y=[pred_c], z=[pred_rot],
            mode='markers',
            marker=dict(size=8, color='yellow', symbol='x', line=dict(width=2, color='black')),
            name=f'Pred Min ({min_v:.4f})'
        ))

        # [核心修正] Layout 设置为立方体
        fig.update_layout(
            title=f'UDF Isosurfaces (Scale={scale_fixed:.2f}, Min={min_v:.4f})',
            scene=dict(
                xaxis_title='NR (Row)',
                yaxis_title='NC (Col)',
                zaxis_title='Rotation (Rad)',
                # aspectmode='data' 会导致长宽比随数据变化
                # aspectmode='cube' 强制让 X:Y:Z 的视觉比例为 1:1:1
                aspectmode='cube'
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=50)  # 减少白边
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 等值面可视化已保存: {save_path}")

        if show_plot:
            fig.show()


    def visualize_udf_combined(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                               n_samples_per_dim=32, delta=0.15, rot_span=torch.pi,
                               z_amplification=1.0, adaptive_z_scale=True,
                               save_path="vis_results/udf_combined.html", show_plot=False,
                               use_train_uav=False, use_train_mode=False):
        """
        合并的UDF可视化函数：在一个页面中显示UDF场值和梯度场。
        (已修复：优化梯度场显示参数，解决毛刺和崩溃问题，保留NR/NC标签)
        """
        import numpy as np
        import os
        import torch
        import torch.nn.functional as TF
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 0. 设置模型模式
        if not use_train_mode:
            for model in self.param2optimize.values():
                model.eval()
            for model in self.param2freeze.values():
                model.eval()
            print("使用Eval模式进行可视化")
        else:
            for model in self.param2optimize.values():
                model.train()
            for model in self.param2freeze.values():
                model.train()
            print("使用Train模式进行可视化（保留dropout等训练行为）")

        # 1. 数据准备
        if query_feat is None or gt_coord_4d is None:
            if use_train_uav and hasattr(self, 'uav_dataloader_train'):
                dataloader = self.uav_dataloader_train
                dataset_name = "训练集"
            elif hasattr(self, 'uav_dataloader_test'):
                dataloader = self.uav_dataloader_test
                dataset_name = "训练集" if use_train_uav else "测试集"
            else:
                raise AttributeError("未找到可用的UAV数据加载器。")
            print(f"可视化使用dataset={dataset_name}")

            try:
                uav_img, uav_coords_4d = next(iter(dataloader))
            except StopIteration:
                uav_img, uav_coords_4d = next(iter(dataloader))

            uav_img = uav_img[0].to(self.device).unsqueeze(0)
            gt_coord_4d = uav_coords_4d[0].to(self.device)
            query_feat = self._get_feats_fm_imgs(uav_img)
            print(f"使用{dataset_name}的UAV图像进行可视化")

        if scale_fixed is None:
            scale_fixed = gt_coord_4d[3].item()

        # 2. 构建采样网格
        nr_center, nc_center, rot_center, scale_val = gt_coord_4d

        if rot_span is None:
            rot_span = torch.pi

        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)
        rot_range = torch.linspace(rot_center - rot_span / 2, rot_center + rot_span / 2, n_samples_per_dim,
                                   device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        coords_sampled_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_fixed)
        ], dim=-1)

        # 3. 推理UDF场值
        coords_sampled_6d = self.coord_normer.raw_to_norm(coords_sampled_4d, append_linear_rot=True)

        with torch.no_grad():
            grid_input = torch.cat([coords_sampled_6d[:, :2], coords_sampled_6d[:, -1:]], dim=-1)
            feats_grid_raw = self._get_feats_fm_grid(grid_input)

            coords_encoded_stage2 = self.pos_encoder_grid(coords_sampled_6d[:, :5])
            feats_grid = self.grid_mlp(inputs=feats_grid_raw, condition_features=coords_encoded_stage2)
            feats_grid = TF.normalize(feats_grid, dim=-1)
            feats_grid_exp = feats_grid.unsqueeze(0)

            coords_encoded_metric = self.pos_encoder_metric(coords_sampled_6d[:, :5])
            N = coords_sampled_4d.shape[0]
            query_feat_exp = query_feat.unsqueeze(1).expand(1, N, -1)
            coords_enc_exp = coords_encoded_metric.unsqueeze(0)

            dist_pred_raw = self.metric_net(query_feat_exp, feats_grid_exp, coords_enc_exp)
            dist_pred = self.softplus(dist_pred_raw) if self.use_softplus else dist_pred_raw
            dist_pred = dist_pred.squeeze(0)

        # 4. 计算梯度场
        coords_sampled_4d_grad = coords_sampled_4d.clone().requires_grad_(True)
        coords_sampled_6d_grad = self.coord_normer.raw_to_norm(coords_sampled_4d_grad, append_linear_rot=True)

        grid_input_grad = torch.cat([coords_sampled_6d_grad[:, :2], coords_sampled_6d_grad[:, -1:]], dim=-1)
        feats_grid_raw_grad = self._get_feats_fm_grid(grid_input_grad)
        coords_encoded_stage2_grad = self.pos_encoder_grid(coords_sampled_6d_grad[:, :5])
        feats_grid_grad = self.grid_mlp(inputs=feats_grid_raw_grad, condition_features=coords_encoded_stage2_grad)
        feats_grid_grad = TF.normalize(feats_grid_grad, dim=-1)

        feats_grid_exp_grad = feats_grid_grad.unsqueeze(0)
        coords_encoded_metric_grad = self.pos_encoder_metric(coords_sampled_6d_grad[:, :5])
        coords_enc_exp_grad = coords_encoded_metric_grad.unsqueeze(0)

        dist_pred_raw_grad = self.metric_net(query_feat_exp, feats_grid_exp_grad, coords_enc_exp_grad)
        dist_pred_grad = self.softplus(dist_pred_raw_grad) if self.use_softplus else dist_pred_raw_grad
        dist_pred_grad = dist_pred_grad.squeeze(0)

        # 计算梯度
        grad_outputs = torch.ones_like(dist_pred_grad)
        gradients = torch.autograd.grad(dist_pred_grad, coords_sampled_4d_grad, grad_outputs, create_graph=False)[0]

        grad_r = -gradients[:, 0]
        grad_c = -gradients[:, 1]
        grad_rot = -gradients[:, 2]

        # 自适应Z轴放大
        grad_rc_mean = (grad_r.abs().mean() + grad_c.abs().mean()) / 2
        grad_rot_mean = grad_rot.abs().mean()

        if adaptive_z_scale:
            if grad_rot_mean < 1e-9:
                z_amplification = 100.0
            else:
                z_amplification = (grad_rc_mean / grad_rot_mean).item()
            print(f"自适应Z轴放大因子: {z_amplification:.2f}")

        grad_rot_amplified = grad_rot * z_amplification
        grad_vec_vis = torch.stack([grad_r, grad_c, grad_rot_amplified], dim=1)
        grad_vec_norm = TF.normalize(grad_vec_vis, dim=1)

        # 5. 数据转换
        coords_np = coords_sampled_4d.detach().cpu().numpy()
        dist_pred_np = dist_pred.detach().cpu().numpy()
        grad_vec_np = grad_vec_norm.detach().cpu().numpy()

        # 归一化颜色
        def normalize_data(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        dist_pred_norm_color = normalize_data(dist_pred_np)

        # 关键点
        best_idx = np.argmin(dist_pred_np)
        coord_pred_best = coords_np[best_idx]
        gt_r, gt_c, gt_rot = nr_center.item(), nc_center.item(), rot_center.item()

        # 打印统计信息
        print(f"\n{'=' * 60}")
        print(f"UDF场可视化统计")
        print(f"{'-' * 60}")
        print(
            f"预测UDF | Min: {dist_pred_np.min():.4f} | Max: {dist_pred_np.max():.4f} | Mean: {dist_pred_np.mean():.4f}")
        print(f"GT位置  | NR: {gt_r:.6f}, NC: {gt_c:.6f}, Rot: {gt_rot:.6f} rad")
        print(
            f"预测位置 | NR: {coord_pred_best[0]:.6f}, NC: {coord_pred_best[1]:.6f}, Rot: {coord_pred_best[2]:.6f} rad")
        rc_error = np.sqrt((coord_pred_best[0] - gt_r) ** 2 + (coord_pred_best[1] - gt_c) ** 2)
        print(f"位置误差 | RC_Dist: {rc_error:.6f}")
        print(f"{'=' * 60}\n")

        # 6. 绘图
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('预测UDF场值', 'UDF梯度流场'),
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
            )

            marker_common = dict(size=3, opacity=0.5)

            # === 左图: UDF场值 ===
            fig.add_trace(go.Scatter3d(
                x=coords_np[:, 0], y=coords_np[:, 1], z=coords_np[:, 2],
                mode='markers',
                marker=dict(**marker_common, color=dist_pred_norm_color, colorscale='Viridis',
                            colorbar=dict(title='Pred UDF', x=0.45)),
                hovertemplate='<b>Pred Dist</b>: %{customdata:.4f}<br>NR: %{x:.4f}<br>NC: %{y:.4f}<br>Rot: %{z:.4f}<extra></extra>',
                customdata=dist_pred_np,
                name='预测UDF场'
            ), row=1, col=1)

            # GT标记（左图）
            fig.add_trace(go.Scatter3d(
                x=[gt_r], y=[gt_c], z=[gt_rot],
                mode='markers', marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
                name='GT位置',
                hovertemplate='<b>GT位置</b><br>NR: %{x:.6f}<br>NC: %{y:.6f}<br>Rot: %{z:.4f}<extra></extra>'
            ), row=1, col=1)

            # 预测最佳位置标记（左图）
            fig.add_trace(go.Scatter3d(
                x=[coord_pred_best[0]], y=[coord_pred_best[1]], z=[coord_pred_best[2]],
                mode='markers', marker=dict(size=10, color='yellow', symbol='x', line=dict(width=2, color='black')),
                name='预测位置',
                customdata=[dist_pred_np[best_idx]],
                hovertemplate='<b>预测位置</b><br>NR: %{x:.6f}<br>NC: %{y:.6f}<br>Rot: %{z:.4f}<br>Min Dist: %{customdata:.4f}<extra></extra>'
            ), row=1, col=1)

            # === 右图: 梯度场 ===
            # 计算合适的参考尺寸
            step_r = (coords_np[:, 0].max() - coords_np[:, 0].min()) / n_samples_per_dim

            # [修正]: 使用单位向量 (grad_vec_np)，不手动乘倍数
            # 通过调整 sizeref 来控制显示大小。sizeref 越小，图标越大。
            # 原版是 step_r * 2.0 (导致太小)，现在改为 step_r * 0.5 (适中偏大)
            cone_scale_factor = step_r * 8.

            fig.add_trace(go.Cone(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                u=grad_vec_np[:, 0],  # 保持单位向量
                v=grad_vec_np[:, 1],
                w=grad_vec_np[:, 2],
                sizemode="absolute",
                sizeref=cone_scale_factor,  # 关键修改：用 sizeref 控制大小
                anchor="tail",
                colorscale='Jet',
                showscale=True,
                colorbar=dict(title='梯度方向', x=1.0),
                opacity=0.8,
                name='梯度流',
                hovertemplate='<b>Grid</b><br>NR: %{x:.2f}<br>NC: %{y:.2f}<br>Rot: %{z:.2f}<br>Dist: %{customdata:.4f}<extra></extra>',
                customdata=dist_pred_np
            ), row=1, col=2)

            # GT标记（右图）
            fig.add_trace(go.Scatter3d(
                x=[gt_r], y=[gt_c], z=[gt_rot],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='black')),
                showlegend=False, name='GT位置',
                hovertemplate='<b>GT位置</b><br>NR: %{x:.6f}<br>NC: %{y:.6f}<br>Rot: %{z:.4f}<extra></extra>'
            ), row=1, col=2)

            # 预测最佳位置标记（右图）
            fig.add_trace(go.Scatter3d(
                x=[coord_pred_best[0]], y=[coord_pred_best[1]], z=[coord_pred_best[2]],
                mode='markers',
                marker=dict(size=8, color='yellow', symbol='x', line=dict(width=2, color='black')),
                showlegend=False, name='预测位置',
                customdata=[dist_pred_np[best_idx]],
                hovertemplate='<b>预测位置</b><br>NR: %{x:.6f}<br>NC: %{y:.6f}<br>Rot: %{z:.4f}<br>Min Dist: %{customdata:.4f}<extra></extra>'
            ), row=1, col=2)

            # Layout设置: 轴标题改为 NR / NC
            layout_scene = dict(
                xaxis_title='NR',
                yaxis_title='NC',
                zaxis_title='Rot (rad)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            )

            fig.update_layout(
                title=f'UDF场可视化 (Rot范围={rot_span / np.pi:.1f}π, Scale={scale_fixed:.2f}, Z放大={z_amplification:.1f}x)',
                height=600,
                width=1400,
                scene=layout_scene,
                scene2=layout_scene
            )

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                print(f"✅ 合并可视化已保存至: {save_path}")

            if show_plot:
                fig.show()

        except ImportError:
            print("⚠️ Plotly未安装，无法生成可视化。")
