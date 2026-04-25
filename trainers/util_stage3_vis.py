import os
import numpy as np
import torch
import torch.nn.functional as TF


def _parse_epoch_from_ckpt_path(ckpt_path):
        if not ckpt_path:
            return 'current'
        basename = os.path.basename(str(ckpt_path))
        if basename.startswith('epoch') and basename.endswith('.pth'):
            return basename.replace('epoch', '').replace('.pth', '')
        return 'current'


def _resolve_analyze_energy_save_context(self, plot_mode='both'):
        """
        Resolve save_dir/epoch for analyze_energy_field according to plot mode.

        - both: prefer Stage 3 checkpoint context
        - ingp: prefer Stage 2 checkpoint context derived from Stage 3 opts
        """
        plot_mode = str(plot_mode or 'both').strip().lower()
        if plot_mode not in {'ingp', 'both'}:
            raise ValueError(f"plot_mode must be 'ingp' or 'both', got {plot_mode}")

        fallback_dir = None
        fallback_epoch = 'current'
        if hasattr(self, '_get_dir2save'):
            try:
                fallback_dir, fallback_epoch = self._get_dir2save(ret_epoch=True)
            except Exception as exc:
                print(f"⚠️  fallback _get_dir2save failed in analyze_energy_field: {exc}")

        stage3_ckpt_path = self._get_stage3_checkpoint_path() if hasattr(self, '_get_stage3_checkpoint_path') else None

        if plot_mode == 'both':
            if stage3_ckpt_path:
                save_dir = os.path.join(os.path.dirname(stage3_ckpt_path), 'loc_results')
                os.makedirs(save_dir, exist_ok=True)
                return save_dir, _parse_epoch_from_ckpt_path(stage3_ckpt_path)
            if fallback_dir is not None:
                return fallback_dir, fallback_epoch
            raise RuntimeError("Unable to resolve save directory for plot_mode='both'.")

        stage2_ckpt_path = None
        if hasattr(self, '_get_stage2_checkpoint_path'):
            try:
                stage2_ckpt_path = self._get_stage2_checkpoint_path(stage3_ckpt_path)
            except TypeError:
                stage2_ckpt_path = self._get_stage2_checkpoint_path()
            except Exception as exc:
                print(f"⚠️  resolve Stage 2 checkpoint failed in analyze_energy_field: {exc}")
        if stage2_ckpt_path:
            save_dir = os.path.join(os.path.dirname(stage2_ckpt_path), 'loc_results')
            os.makedirs(save_dir, exist_ok=True)
            return save_dir, _parse_epoch_from_ckpt_path(stage2_ckpt_path)

        print("⚠️  plot_mode='ingp' 未解析到Stage 2 checkpoint，回退到当前默认保存目录。")
        if fallback_dir is not None:
            return fallback_dir, fallback_epoch
        raise RuntimeError("Unable to resolve save directory for plot_mode='ingp'.")

def visualize_energy_of_coords(
            self,
            coords_samples,
            energys,
            coord_gt,
            coord_pred=None,
            save_path=None,
            mode='min'  # 新增参数: 'min' 或 'max'
    ):
        """
        生成极简版可交互3D可视化 (HTML)

        Args:
            coords_samples: [M, 4] 采样点坐标
            energys: [M] 采样点对应的值
            coord_gt: [4] GT坐标
            coord_pred: [4] (可选) 预测坐标
            save_path: (str) 保存路径
            mode: (str) 'min' 表示寻找能量最小值，'max' 表示寻找最大值 (当 coord_pred 为 None 时生效)
        """
        import plotly.graph_objects as go
        import numpy as np
        import os
        import torch
        import time

        # 1. 数据转换
        def to_numpy(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        coords_np = to_numpy(coords_samples)
        energys_np = to_numpy(energys)
        gt_np = to_numpy(coord_gt)

        # 2. 准备坐标
        X = coords_np[:, 1]  # NC
        Y = coords_np[:, 0]  # NR
        Z = np.rad2deg(coords_np[:, 2])  # Rot

        # 3. 创建 Figure
        fig = go.Figure()

        # --- Layer 1: 所有采样点 ---
        # 根据 mode 调整 colorscale 可能视觉效果更好，这里保持 Hot_r
        # 如果是 'max' 模式 (如相似度)，通常数值越大越好，可能更适合 'Viridis'
        colorscale = 'Hot_r' if mode == 'min' else 'Viridis'

        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(
                size=3,
                color=energys_np,
                colorscale=colorscale,
                opacity=0.6,
                colorbar=dict(title='Value', thickness=20)
            ),
            text=[f"Val: {e:.4f}" for e in energys_np],
            name='Samples'
        ))

        # --- Layer 2: GT ---
        fig.add_trace(go.Scatter3d(
            x=[gt_np[1]], y=[gt_np[0]], z=[np.rad2deg(gt_np[2])],
            mode='markers+text',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='GT',
            text=['GT'],
            textposition="top center"
        ))

        # --- Layer 3: Pred (或 Best Candidate) ---
        if coord_pred is not None:
            pred_np = to_numpy(coord_pred)
            label = "Pred"
        else:
            # === 这里增加了 mode 判断逻辑 ===
            if mode == 'max':
                best_idx = energys_np.argmax()
                label = "Max Value"
            else:
                best_idx = energys_np.argmin()
                label = "Min Energy"

            pred_np = coords_np[best_idx]

        fig.add_trace(go.Scatter3d(
            x=[pred_np[1]], y=[pred_np[0]], z=[np.rad2deg(pred_np[2])],
            mode='markers+text',
            marker=dict(size=6, color='red', symbol='x'),
            name=label,
            text=[f'{label}'],
            textposition="top center"
        ))

        # 4. 设置布局
        err_2d = np.linalg.norm(pred_np[:2] - gt_np[:2])

        fig.update_layout(
            title=f"3D Distribution ({label} Error 2D: {err_2d:.2f})",
            width=1000, height=800,
            scene=dict(
                xaxis_title='NC (Col)',
                yaxis_title='NR (Row)',
                zaxis_title='Rotation (deg)',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 5. 保存
        if save_path:
            final_path = save_path
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
        else:
            save_dir = self._get_dir2save() if hasattr(self, '_get_dir2save') else "vis_results"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = f"{time.time():.5f}".replace('.', '_')
            final_path = os.path.join(save_dir, f'vis_3d_{timestamp}.html')

        fig.write_html(final_path)
        print(f"✅ 3D可视化(HTML)已保存: {final_path}")

def _compute_energy_field_local(
            self,
            query_feat,
            gt_coord_4d,
            scale_fixed=None,
            n_samples_per_dim=32,
            delta=0.1,
            rot_span=torch.pi,
            show_grad_field=True,
            adaptive_z_scale=False,
            argmode='min',
            energy_backend='ingp',
            chunk_size=4096,
    ):
        """
        生成局部能量/概率场，返回绘制所需的张量与元信息。
        """
        import torch.nn.functional as TF

        nr_center, nc_center, rot_center, _ = gt_coord_4d
        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_samples_per_dim, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_samples_per_dim, device=self.device)
        rot_range = torch.linspace(rot_center - rot_span / 2, rot_center + rot_span / 2, n_samples_per_dim,
                                   device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        coords_sampled_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_fixed)
        ], dim=-1)

        if show_grad_field:
            coords_sampled_4d.requires_grad_(True)

        energy_backend = energy_backend.lower()
        if energy_backend not in ('ingp', 'projector'):
            raise ValueError(f"energy_backend must be 'ingp' or 'projector', got {energy_backend}")

        if energy_backend == 'ingp':
            metric_out = self._compute_metric_from_ingp(
                query_feats=query_feat,
                ref_points=coords_sampled_4d,
                coord_space='raw',
                chunk_size=chunk_size,
                metric='dist',
            ).squeeze(0)  # [N]
        else:
            # 使用 projector 路径计算概率，能量定义为 1 - prob
            prob_chunks = []
            total = coords_sampled_4d.shape[0]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                coords_chunk = coords_sampled_4d[start:end]
                prob_chunk = self._compute_metric_from_query_and_points(
                    query_feats=query_feat,
                    ref_points=coords_chunk,
                    temperature=self.energy_temperature,
                    metric='possibility',
                    coord_space='raw',
                    chunk_size=None,  # 这里按外层 chunk 手动控制
                    feat_type='projector',
                )  # [1, chunk]
                prob_chunks.append(prob_chunk)
            prob_all = torch.cat(prob_chunks, dim=1).squeeze(0)  # [N]
            metric_out = 1.0 - prob_all  # 概率越大，能量越小

        argmode = argmode.lower()
        if argmode not in ('min', 'max'):
            raise ValueError(f"argmode must be 'min' or 'max', got {argmode}")

        if argmode == 'min':
            energy_pred = metric_out
            prob_pred = (2 - metric_out).clamp(min=0) / 2  # 小距离 → 大概率
            best_reducer = torch.argmin
            grad_dir = -1.0
        else:
            energy_pred = -metric_out  # 取负后“越近越大”，便于 argmax
            e_min, e_max = energy_pred.min(), energy_pred.max()
            prob_pred = (energy_pred - e_min) / (e_max - e_min + 1e-8)
            best_reducer = torch.argmax
            grad_dir = 1.0

        z_amplification = 1.0
        grad_vec_norm = None
        if show_grad_field:
            grad_outputs = torch.ones_like(energy_pred)
            gradients = torch.autograd.grad(energy_pred, coords_sampled_4d, grad_outputs, create_graph=False)[0]

            grad_r = grad_dir * gradients[:, 0]
            grad_c = grad_dir * gradients[:, 1]
            grad_rot = grad_dir * gradients[:, 2]

            grad_rc_mean = (grad_r.abs().mean() + grad_c.abs().mean()) / 2
            grad_rot_mean = grad_rot.abs().mean()
            if adaptive_z_scale:
                if grad_rot_mean < 1e-9:
                    z_amplification = 100.0
                else:
                    z_amplification = (grad_rc_mean / grad_rot_mean).item()
                grad_rot_amplified = grad_rot * z_amplification
            else:
                grad_rot_amplified = grad_rot
                z_amplification = 1.0

            grad_vec_vis = torch.stack([grad_r, grad_c, grad_rot_amplified], dim=1)
            grad_vec_norm = TF.normalize(grad_vec_vis, dim=1)

        best_idx = best_reducer(energy_pred).item()

        return {
            "coords_sampled_4d": coords_sampled_4d.detach(),
            "energy_pred": energy_pred.detach(),
            "prob_pred": prob_pred.detach(),
            "grad_vec_norm": grad_vec_norm.detach() if grad_vec_norm is not None else None,
            "z_amplification": z_amplification,
            "best_idx": best_idx,
            "centers": (nr_center.item(), nc_center.item(), rot_center.item()),
            "scale_fixed": scale_fixed,
        }

def _render_energy_field_local(
            self,
            field_data,
            n_samples_per_dim,
            surface_min_ratio,
            save_path,
            show_plot,
            argmode,
    ):
        import numpy as np
        import os
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        coords_sampled_4d = field_data["coords_sampled_4d"]
        energy_pred = field_data["energy_pred"]
        grad_vec_norm = field_data["grad_vec_norm"]
        z_amplification = field_data["z_amplification"]
        best_idx = field_data["best_idx"]
        gt_r, gt_c, gt_rot = field_data["centers"]
        scale_fixed = field_data["scale_fixed"]

        coords_np = coords_sampled_4d.cpu().numpy()
        X = coords_np[:, 0]
        Y = coords_np[:, 1]
        Z = coords_np[:, 2]

        V = energy_pred.cpu().numpy()
        G_vec = grad_vec_norm.cpu().numpy() if grad_vec_norm is not None else None

        min_v, max_v = V.min(), V.max()
        print(f"📊 统计: Min Energy={min_v:.5f}, Max Energy={max_v:.5f}, Z-Scale={z_amplification:.1f}x (argmode={argmode})")

        pred_r, pred_c, pred_rot = X[best_idx], Y[best_idx], Z[best_idx]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f'场值散点 (Min={min_v:.2f})', '梯度流场 (Descent)', '等值面结构 (Geometry)'),
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.02
        )

        step = 1 if n_samples_per_dim <= 32 else 2
        mask = slice(None, None, step)

        fig.add_trace(go.Scatter3d(
            x=X[mask], y=Y[mask], z=Z[mask],
            mode='markers',
            marker=dict(size=3, opacity=0.4, color=V[mask], colorscale='Viridis',
                        colorbar=dict(title='Energy', x=0.28, len=0.5)),
            hovertemplate='Energy: %{marker.color:.4f}<extra></extra>',
            name='Energy Cloud'
        ), row=1, col=1)

        if grad_vec_norm is not None:
            step_r = (X.max() - X.min()) / n_samples_per_dim
            cone_scale = step_r * 8.0

            fig.add_trace(go.Cone(
                x=X[mask], y=Y[mask], z=Z[mask],
                u=G_vec[mask, 0], v=G_vec[mask, 1], w=G_vec[mask, 2],
                sizemode="absolute", sizeref=cone_scale, anchor="tail",
                colorscale='Jet', showscale=False, opacity=0.7,
                name='Gradients'
            ), row=1, col=2)

        val_range = max_v - min_v
        split_val = min_v + val_range * surface_min_ratio

        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=min_v,
            isomax=split_val,
            surface_count=2,
            colorscale='Plasma',
            opacity=0.2,
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            showscale=False,
            name='Inner Core',
            hovertemplate='Core Energy: %{value:.4f}<extra></extra>'
        ), row=1, col=3)

        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z,
            value=V,
            isomin=split_val,
            isomax=max_v,
            surface_count=3,
            colorscale='Viridis',
            opacity=0.15,
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            colorbar=dict(title='Energy Level', x=1.0, len=0.5),
            name='Outer Shell',
            hovertemplate='Global Energy: %{value:.4f}<extra></extra>'
        ), row=1, col=3)

        for col in [1, 2, 3]:
            fig.add_trace(go.Scatter3d(
                x=[gt_r], y=[gt_c], z=[gt_rot],
                mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
                showlegend=(col == 1), name='GT'
            ), row=1, col=col)
            fig.add_trace(go.Scatter3d(
                x=[pred_r], y=[pred_c], z=[pred_rot],
                mode='markers', marker=dict(size=6, color='yellow', symbol='x'),
                showlegend=(col == 1), name='Best'
            ), row=1, col=col)

        scene_layout = dict(
            xaxis_title='NR', yaxis_title='NC', zaxis_title='Rot',
            aspectmode='cube'
        )

        fig.update_layout(
            title=f'Energy Field Comprehensive Analysis (Scale={scale_fixed:.2f})',
            height=600, width=1600,
            scene1=scene_layout,
            scene2=scene_layout,
            scene3=scene_layout,
            margin=dict(l=10, r=10, b=10, t=60)
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✅ 能量场可视化已保存: {save_path}")

        if show_plot:
            fig.show()

def visualize_energy_field_local(self, query_feat=None, gt_coord_4d=None, scale_fixed=None,
                                     n_samples_per_dim=32, delta=0.1, rot_span=torch.pi,
                                     surface_min_ratio=0.2, adaptive_z_scale=False,
                                     save_path="vis_results/energy_field.html", show_plot=False,
                                     use_train_uav=False, show_grad_field=True, argmode='min',
                                     energy_backend='ingp', chunk_size=4096):
        """
        能量场综合可视化：拆分为生成场与绘制两个子步骤。
        """
        import torch

        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        if query_feat is None or gt_coord_4d is None:
            if use_train_uav and hasattr(self, 'uav_dataloader_train'):
                dataloader = self.uav_dataloader_train
                tag = "Train"
            elif hasattr(self, 'uav_dataloader_test'):
                dataloader = self.uav_dataloader_test
                tag = "Test"
            else:
                raise AttributeError("未找到可用的UAV数据加载器。")

            try:
                uav_img, uav_coords_4d = next(iter(dataloader))
            except StopIteration:
                uav_img, uav_coords_4d = next(iter(dataloader))

            uav_img = uav_img[0].to(self.device).unsqueeze(0)
            gt_coord_4d = uav_coords_4d[0].to(self.device)
            query_feat = self._get_feats_fm_imgs(uav_img)
            print(f"可视化数据来源: {tag} Set")

        if scale_fixed is None:
            scale_fixed = gt_coord_4d[3].item()

        field_data = self._compute_energy_field_local(
            query_feat=query_feat,
            gt_coord_4d=gt_coord_4d,
            scale_fixed=scale_fixed,
            n_samples_per_dim=n_samples_per_dim,
            delta=delta,
            rot_span=rot_span,
            show_grad_field=show_grad_field,
            adaptive_z_scale=adaptive_z_scale,
            argmode=argmode,
            energy_backend=energy_backend,
            chunk_size=chunk_size,
        )

        self._render_energy_field_local(
            field_data=field_data,
            n_samples_per_dim=n_samples_per_dim,
            surface_min_ratio=surface_min_ratio,
            save_path=save_path,
            show_plot=show_plot,
            argmode=argmode,
        )

def analyze_feat_freq_band(self, n_points_per_subspace=1, use_fine=False,vis=False):
        """
        获取当前网格下的所有采样点（用于后续特征频域分析）。

        Args:
            n_points_per_subspace: 每个粗子空间采样的点数
            use_fine: 是否展开细粒度采样

        Returns:
            coords_all: [1, N, 4] 物理坐标
            candidate_labels: [1, N, 4] 对应索引标签
        """
        if not hasattr(self, "subspace_sampler"):
            raise AttributeError("subspace_sampler 未初始化，先确保已运行 _init_datasets/_load_checkpoints 创建采样器。")

        coords_all, candidate_labels = self.subspace_sampler.sample_all_subspaces_gpu(
            n_points_per_subspace=n_points_per_subspace,
            use_fine=use_fine,
            rand_offset=False,
        )
        # 恢复为多维 shape: [NR, NC, Rot, Scale, P, 4] / [NR, NC, Rot, Scale, P]
        n_coarse = tuple(self.subspace_sampler.n_coarse.tolist())
        coords_multi = coords_all.view(*n_coarse, n_points_per_subspace, 4)
        labels_multi = candidate_labels.view(*n_coarse, n_points_per_subspace)
        print(f"采样完成: flat coords={coords_all.shape}, multi coords={coords_multi.shape}")

        rot_id = min(18, n_coarse[2] - 1)
        coords_2d = coords_multi[:, :, rot_id, 0, 0, :]

        # 2D网格细采样（固定 rot/scale），在每个 coarse cell 内生成 fine_grid×fine_grid 细分点
        fine_grid = 2
        delta_nr, delta_nc = self.subspace_sampler.coarse_bin_sizes[:2] * 0.5
        nr_lin = torch.linspace(-delta_nr, delta_nr, fine_grid, device=coords_2d.device)
        nc_lin = torch.linspace(-delta_nc, delta_nc, fine_grid, device=coords_2d.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing='ij')  # [g, g]
        nr_grid = nr_grid.unsqueeze(0).unsqueeze(0)
        nc_grid = nc_grid.unsqueeze(0).unsqueeze(0)
        nr_fine = coords_2d[..., 0:1, None, None] + nr_grid
        nc_fine = coords_2d[..., 1:2, None, None] + nc_grid
        rot_fine = torch.ones_like(nr_fine) * coords_2d[..., 2:3, None, None]
        scale_fine = torch.ones_like(nr_fine) * coords_2d[..., 3:4, None, None]
        # coords_2d_fine = torch.stack([nr_fine, nc_fine, rot_fine, scale_fine], dim=-1)  # [NR,NC,g,g,4]
        g = fine_grid
        coords_2d_fine = torch.stack([nr_fine, nc_fine, rot_fine, scale_fine], dim=-1)  # [NR,NC,1,g,g,4]
        coords_2d_fine = coords_2d_fine.squeeze(2)  # [NR,NC,g,g,4]
        coords_2d_fine = coords_2d_fine.permute(0, 2, 1, 3, 4).contiguous()  # [NR,g,NC,g,4]
        coords_2d_fine = coords_2d_fine.view(coords_2d.shape[0] * g, coords_2d.shape[1] * g, 4)  # [NR*g,NC*g,4]

        with torch.no_grad():
            feats_ingpg_2d = self._get_feats_fm_INGP(coords_2d_fine.view(-1, 4), coord_mode='raw').reshape(
                *coords_2d_fine.shape[:2], -1)
            feats_projector_2d = self.projector(
                feats_ingpg_2d.view(-1, feats_ingpg_2d.shape[-1])
            ).reshape(*coords_2d_fine.shape[:2], -1)

        from scripts.analysis.util_fft_analyse import analyse_feature_frequency
        res = analyse_feature_frequency(
            feats_ingpg_2d, feats_projector_2d,
            cdf_tau=0.95, hf_frac=0.33, eps=1e-12,
            norm="ortho", channel_norm=True, return_radial=True
        )
        # 打印（保持原格式）
        mF, mZ, d = res["metrics_F"], res["metrics_Z"], res["delta"]
        print(f"[INGP feat space ] fc={mF['fc']:.3f}, f95={mF['f95']:.1f}, hf_ratio={mF['hf_ratio']:.5f} (f0 bin={mF['f0_bin']})")
        print(f"[Proj feat space ] fc={mZ['fc']:.3f}, f95={mZ['f95']:.1f}, hf_ratio={mZ['hf_ratio']:.5f} (f0 bin={mZ['f0_bin']})")
        print(f"Delta fc   : {d['fc']:+.3f}")
        print(f"Delta f95  : {d['f95']:+.1f}")
        print(f"HF ratio Z/F: {d['hf_ratio_Z_over_F']:.3f}")


        if vis:
            res = analyse_feature_frequency(
                feats_ingpg_2d, feats_projector_2d,
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12,
                norm="ortho", channel_norm=True, return_radial=True,wo_DC=False
            )
            P_F, P_Z = res["P_F"], res["P_Z"]
            eps = 1e-12
            dir2save,epoch = self._get_dir2save(ret_epoch=True)

            import matplotlib.pyplot as plt
            from matplotlib import cm
            P_F_np = torch.fft.fftshift(P_F).cpu().numpy()
            P_Z_np = torch.fft.fftshift(P_Z).cpu().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            vmin = min(np.log(P_F_np + eps).min(), np.log(P_Z_np + eps).min())
            vmax = max(np.log(P_F_np + eps).max(), np.log(P_Z_np + eps).max())
            im0 = axs[0].imshow(np.log(P_F_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[0].set_title("P_F (INGP) log spectrum")
            im1 = axs[1].imshow(np.log(P_Z_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[1].set_title("P_Z (Projector) log spectrum")
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(os.path.join(dir2save,f'feat_space_fft_w_dc_ep{epoch}.png'))
            print(f'已保存'+os.path.join(dir2save,f'feat_space_fft_w_dc_ep{epoch}.png'))

def analyze_energy_field(self, n_nr=128, n_nc=128, use_train_uav=True, local_zoom_wh=None, vis=False,
                                       use_vis_ref=False, chunk_size_vis=1024, analyse_fft=False, query_id=20,
                                       plot_mode='both', plot_contour_setting=None):
        """
        随机取一帧，固定 rot/scale 与 coord_q 一致，在 nr/nc 平面均匀采样，计算 INGP 相似度/距离场。

        plot_mode:
            - 'ingp': 仅绘制 INGP 地形图，保存目录优先跟随 Stage 2 checkpoint
            - 'both': 同时绘制 INGP + Projector，保存目录优先跟随 Stage 3 checkpoint

        plot_contour_setting:
            传给 vis_featmap.plot_contour 的可选字典。适合在外部直接调图像风格，
            例如 with_flow / cmap / n_fill_levels / n_line_levels / contour_line_*。
        """
        plot_mode = str(plot_mode or 'both').strip().lower()
        if plot_mode not in {'ingp', 'both'}:
            raise ValueError(f"plot_mode must be 'ingp' or 'both', got {plot_mode}")
        if plot_contour_setting is not None and not isinstance(plot_contour_setting, dict):
            raise TypeError(f"plot_contour_setting must be a dict or None, got {type(plot_contour_setting).__name__}")

        dataset = self.uav_dataset_train if use_train_uav else self.uav_dataset_test
        if dataset is None or len(dataset) == 0:
            raise ValueError("数据集为空或未初始化，无法抽样进行分析。")

        idx = query_id
        img, coord_q = dataset[idx]
        img = img.unsqueeze(0).to(self.device)
        coord_q = coord_q.to(self.device)

        feat_vis = self._get_feats_fm_imgs(img)  # [1, C]

        global_nr_min = float(self.sat_dataset.nr2sample_min)
        global_nr_max = float(self.sat_dataset.nr2sample_max)
        global_nc_min = float(self.sat_dataset.nc2sample_min)
        global_nc_max = float(self.sat_dataset.nc2sample_max)

        if local_zoom_wh is not None:
            w_ratio_nr, w_ratio_nc = local_zoom_wh
            span_nr = global_nr_max - global_nr_min
            span_nc = global_nc_max - global_nc_min
            half_nr = (span_nr * w_ratio_nr) / 2
            half_nc = (span_nc * w_ratio_nc) / 2

            center_nr = coord_q[0].item()
            center_nc = coord_q[1].item()

            start_nr = max(global_nr_min, center_nr - half_nr)
            end_nr = min(global_nr_max, center_nr + half_nr)
            start_nc = max(global_nc_min, center_nc - half_nc)
            end_nc = min(global_nc_max, center_nc + half_nc)

            if vis:
                print(f"[Analyze] Local Zoom Enabled: Center=({center_nr:.2f}, {center_nc:.2f})")
                print(f"          Range NR: [{start_nr:.2f}, {end_nr:.2f}], NC: [{start_nc:.2f}, {end_nc:.2f}]")
        else:
            start_nr, end_nr = global_nr_min, global_nr_max
            start_nc, end_nc = global_nc_min, global_nc_max
        local_zoom_wh = 'global' if local_zoom_wh is None else local_zoom_wh

        nr_lin = torch.linspace(start_nr, end_nr, n_nr, device=self.device)
        nc_lin = torch.linspace(start_nc, end_nc, n_nc, device=self.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing='ij')

        rot_grid = torch.full_like(nr_grid, coord_q[2])
        scale_grid = torch.full_like(nr_grid, coord_q[3])

        coords_grid = torch.stack([nr_grid, nc_grid, rot_grid, scale_grid], dim=-1)
        coords_flat = coords_grid.view(-1, 4)

        span_nr = max(end_nr - start_nr, 1e-12)
        span_nc = max(end_nc - start_nc, 1e-12)
        gt_nr_f = (coord_q[0].item() - start_nr) / span_nr * (n_nr - 1)
        gt_nc_f = (coord_q[1].item() - start_nc) / span_nc * (n_nc - 1)
        gt_nr_idx = int(np.clip(round(gt_nr_f), 0, n_nr - 1))
        gt_nc_idx = int(np.clip(round(gt_nc_f), 0, n_nc - 1))

        energy_visencoder = None
        energy_ingp = None
        energy_projector = None
        contour_kwargs = {
            'crop_size': 0,
            'with_flow': False,
            'flow_mode': 'ascent',
            'unified_scale': False,
            'cmap': ["#ffffff", "#3a3a3a"],
            'n_fill_levels': 24,
            'n_line_levels': 32,
            'contour_line_color': "#6a6a6a",
            'contour_line_width': 0.8,
            'contour_line_alpha': 0.6,
        }
        if plot_contour_setting:
            reserved_keys = {'dist_ingp', 'dist_proj', 'gt_coords', 'save_path'}
            overlap = reserved_keys.intersection(plot_contour_setting.keys())
            if overlap:
                raise ValueError(
                    "plot_contour_setting contains reserved keys managed by analyze_energy_field: "
                    + ", ".join(sorted(overlap))
                )
            contour_kwargs.update(plot_contour_setting)
        with torch.no_grad():
            if use_vis_ref:
                feat_q_vis = TF.normalize(feat_vis, dim=-1)
                dist_chunks = []
                for start in range(0, coords_flat.shape[0], chunk_size_vis):
                    end = min(start + chunk_size_vis, coords_flat.shape[0])
                    satimgs_refs_chunk = self.sat_dataset.crop_satimg_by_4d_coords(coords_flat[start:end].cpu()).to(self.device)
                    feats_ref_chunk = TF.normalize(
                        self._get_feats_fm_imgs(satimgs_refs_chunk), dim=-1
                    )
                    dist_chunk = torch.norm(feats_ref_chunk - feat_q_vis, dim=-1)
                    dist_chunks.append(dist_chunk)

                dist_visencoder = torch.cat(dist_chunks, dim=0).reshape(*coords_grid.shape[:2])
                energy_visencoder = torch.exp(-dist_visencoder)
            else:
                dist_ingp = self._compute_metric_from_query_and_points(
                    metric='dist',
                    feat_type='ingp',
                    query_feats=feat_vis,
                    ref_points=coords_flat,
                    temperature=self.energy_temperature,
                ).reshape(*coords_grid.shape[:2])
                energy_ingp = torch.exp(-dist_ingp)

                if plot_mode == 'both':
                    dist_projector = self._compute_metric_from_query_and_points(
                        metric='dist',
                        feat_type='projector',
                        query_feats=feat_vis,
                        ref_points=coords_flat,
                        temperature=self.energy_temperature,
                    ).reshape(*coords_grid.shape[:2])
                    energy_projector = torch.exp(-dist_projector)

        if vis:
            set_name = "train" if use_train_uav else "test"
            dir2save, epoch = _resolve_analyze_energy_save_context(self, plot_mode=plot_mode)
            suffix = f'hw{local_zoom_wh[0]:.2f}' if local_zoom_wh != 'global' else local_zoom_wh

            from vis_featmap import plot_contour, vis_griddata_in_3d_surface_interactive

            if energy_ingp is not None and plot_mode == 'ingp':
                contour_p2save = os.path.join(dir2save, f'contour_{set_name}_id{idx}_ns{n_nr}_{suffix}_ingp.png')
                plot_contour(
                    dist_ingp=energy_ingp.detach().cpu(),
                    dist_proj=None,
                    gt_coords=(gt_nr_idx, gt_nc_idx),
                    save_path=contour_p2save,
                    **contour_kwargs,
                )
                print(f'已保存' + contour_p2save)
            elif energy_ingp is not None and energy_projector is not None and plot_mode == 'both':
                contour_p2save = os.path.join(dir2save, f'contour_{set_name}_id{idx}_ns{n_nr}_{suffix}_ingp&projector.png')
                plot_contour(
                    dist_ingp=energy_ingp.detach().cpu(),
                    dist_proj=energy_projector.detach().cpu(),
                    gt_coords=(gt_nr_idx, gt_nc_idx),
                    save_path=contour_p2save,
                    **contour_kwargs,
                )
                print(f'已保存' + contour_p2save)

            if energy_projector is not None and plot_mode == 'both':
                projector_path2save = os.path.join(dir2save, f'ep{epoch}_energy_projector_{set_name}_id{idx}_ns{n_nr}_{suffix}.html')
                vis_griddata_in_3d_surface_interactive(
                    energy_projector,
                    p2save=projector_path2save,
                    colorscale='RdBu_r',
                    show_axis_info=True,
                )
                print(f'已保存' + projector_path2save)

            if energy_ingp is not None:
                ingp_path2save = os.path.join(dir2save, f'ep{epoch}_energy_ingp_{set_name}_id{idx}_nr{n_nr}_{suffix}.html')
                vis_griddata_in_3d_surface_interactive(
                    energy_ingp,
                    p2save=ingp_path2save,
                    colorscale='RdBu_r',
                    show_axis_info=True,
                )
                print(f'已保存' + ingp_path2save)

            if energy_visencoder is not None:
                visencoder_path2save = os.path.join(dir2save, f'ep{epoch}_energy_visencoder_{set_name}_id{idx}_ns{n_nr}_{suffix}.html')
                vis_griddata_in_3d_surface_interactive(
                    energy_visencoder,
                    p2save=visencoder_path2save,
                    colorscale='RdBu_r',
                    show_axis_info=False,
                )
                print(f'已保存' + visencoder_path2save)

        if analyse_fft:
            if energy_ingp is None or energy_projector is None:
                print(f"⚠️  analyse_fft requires both INGP and projector fields; skip for plot_mode='{plot_mode}' or use_vis_ref={use_vis_ref}.")
                return

            from scripts.analysis.util_fft_analyse import analyse_feature_frequency
            res = analyse_feature_frequency(
                energy_ingp[..., None], energy_projector[..., None],
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12, wo_DC=True,
                norm="ortho", channel_norm=True, return_radial=True
            )
            mF, mZ, d = res["metrics_F"], res["metrics_Z"], res["delta"]
            print(f"[INGP energy space] fc={mF['fc']:.3f}, f95={mF['f95']:.1f}, hf_ratio={mF['hf_ratio']:.5f} (f0 bin={mF['f0_bin']})")
            print(f"[Proj energy space] fc={mZ['fc']:.3f}, f95={mZ['f95']:.1f}, hf_ratio={mZ['hf_ratio']:.5f} (f0 bin={mZ['f0_bin']})")
            print(f"Delta fc   : {d['fc']:+.3f}")
            print(f"Delta f95  : {d['f95']:+.1f}")
            print(f"HF ratio Z/F: {d['hf_ratio_Z_over_F']:.3f}")

            res = analyse_feature_frequency(
                energy_ingp[..., None], energy_projector[..., None],
                cdf_tau=0.95, hf_frac=0.33, eps=1e-12, wo_DC=False,
                norm="ortho", channel_norm=True, return_radial=True
            )
            dir2save, epoch = _resolve_analyze_energy_save_context(self, plot_mode=plot_mode)
            P_F, P_Z = res["P_F"], res["P_Z"]
            eps = 1e-12

            import matplotlib.pyplot as plt
            from matplotlib import cm
            P_F_np = torch.fft.fftshift(P_F).cpu().numpy()
            P_Z_np = torch.fft.fftshift(P_Z).cpu().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            vmin = min(np.log(P_F_np + eps).min(), np.log(P_Z_np + eps).min())
            vmax = max(np.log(P_F_np + eps).max(), np.log(P_Z_np + eps).max())
            axs[0].imshow(np.log(P_F_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[0].set_title("P_F (INGP) log spectrum")
            axs[1].imshow(np.log(P_Z_np + eps), cmap=cm.viridis, interpolation="bilinear", vmin=vmin, vmax=vmax)
            axs[1].set_title("P_Z (Projector) log spectrum")
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(os.path.join(dir2save, f'energy_space_fft_w_dc_ep{epoch}.png'))
            print(f'已保存' + os.path.join(dir2save, f'energy_space_fft_w_dc_ep{epoch}.png'))
