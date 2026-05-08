import os

import numpy as np
import torch
import torch.nn.functional as TF


class Stage3EnergyFieldAnalyzer:
    """Compute and render Stage-3 response/energy fields around a query."""

    def __init__(self, trainer):
        self.trainer = trainer

    @staticmethod
    def _parse_epoch_from_ckpt_path(ckpt_path):
        if not ckpt_path:
            return "current"
        basename = os.path.basename(str(ckpt_path))
        if basename.startswith("epoch") and basename.endswith(".pth"):
            return basename.replace("epoch", "").replace(".pth", "")
        return "current"

    def resolve_save_context(self, plot_mode="both"):
        trainer = self.trainer
        plot_mode = str(plot_mode or "both").strip().lower()
        if plot_mode not in {"ingp", "both"}:
            raise ValueError(f"plot_mode must be 'ingp' or 'both', got {plot_mode}")

        fallback_dir = None
        fallback_epoch = "current"
        if hasattr(trainer, "_get_dir2save"):
            try:
                fallback_dir, fallback_epoch = trainer._get_dir2save(ret_epoch=True)
            except Exception as exc:
                print(f"WARNING: fallback _get_dir2save failed in analyze_energy_field: {exc}")

        stage3_ckpt_path = trainer._get_stage3_checkpoint_path() if hasattr(trainer, "_get_stage3_checkpoint_path") else None

        if plot_mode == "both":
            if stage3_ckpt_path:
                save_dir = os.path.join(os.path.dirname(stage3_ckpt_path), "loc_results")
                os.makedirs(save_dir, exist_ok=True)
                return save_dir, self._parse_epoch_from_ckpt_path(stage3_ckpt_path)
            if fallback_dir is not None:
                return fallback_dir, fallback_epoch
            raise RuntimeError("Unable to resolve save directory for plot_mode='both'.")

        stage2_ckpt_path = None
        if hasattr(trainer, "_get_stage2_checkpoint_path"):
            try:
                stage2_ckpt_path = trainer._get_stage2_checkpoint_path(stage3_ckpt_path)
            except TypeError:
                stage2_ckpt_path = trainer._get_stage2_checkpoint_path()
            except Exception as exc:
                print(f"WARNING: resolve Stage 2 checkpoint failed in analyze_energy_field: {exc}")
        if stage2_ckpt_path:
            save_dir = os.path.join(os.path.dirname(stage2_ckpt_path), "loc_results")
            os.makedirs(save_dir, exist_ok=True)
            return save_dir, self._parse_epoch_from_ckpt_path(stage2_ckpt_path)

        print("WARNING: plot_mode='ingp' did not resolve a Stage 2 checkpoint; falling back to default save dir.")
        if fallback_dir is not None:
            return fallback_dir, fallback_epoch
        raise RuntimeError("Unable to resolve save directory for plot_mode='ingp'.")

    def resolve_query(self, query_id=20, use_train_uav=True):
        trainer = self.trainer
        dataset = trainer.uav_dataset_train if use_train_uav else trainer.uav_dataset_test
        if dataset is None or len(dataset) == 0:
            raise ValueError("数据集为空或未初始化，无法抽样进行分析。")

        img, coord_q = dataset[int(query_id)]
        img = img.unsqueeze(0).to(trainer.device)
        coord_q = coord_q.to(trainer.device)
        feat_vis = trainer._get_feats_fm_imgs(img)
        return {
            "query_id": int(query_id),
            "image": img,
            "coord_q": coord_q,
            "feat_vis": feat_vis,
            "set_name": "train" if use_train_uav else "test",
        }

    def build_grid(self, coord_q, n_nr=128, n_nc=128, local_zoom_wh=None, verbose=False):
        trainer = self.trainer
        global_nr_min = float(trainer.sat_dataset.nr2sample_min)
        global_nr_max = float(trainer.sat_dataset.nr2sample_max)
        global_nc_min = float(trainer.sat_dataset.nc2sample_min)
        global_nc_max = float(trainer.sat_dataset.nc2sample_max)

        if local_zoom_wh is not None:
            w_ratio_nr, w_ratio_nc = local_zoom_wh
            span_nr = global_nr_max - global_nr_min
            span_nc = global_nc_max - global_nc_min
            half_nr = (span_nr * float(w_ratio_nr)) / 2
            half_nc = (span_nc * float(w_ratio_nc)) / 2

            center_nr = float(coord_q[0].item())
            center_nc = float(coord_q[1].item())
            start_nr = max(global_nr_min, center_nr - half_nr)
            end_nr = min(global_nr_max, center_nr + half_nr)
            start_nc = max(global_nc_min, center_nc - half_nc)
            end_nc = min(global_nc_max, center_nc + half_nc)

            if verbose:
                print(f"[Analyze] Local Zoom Enabled: Center=({center_nr:.2f}, {center_nc:.2f})")
                print(f"          Range NR: [{start_nr:.2f}, {end_nr:.2f}], NC: [{start_nc:.2f}, {end_nc:.2f}]")
        else:
            start_nr, end_nr = global_nr_min, global_nr_max
            start_nc, end_nc = global_nc_min, global_nc_max

        nr_lin = torch.linspace(start_nr, end_nr, int(n_nr), device=trainer.device)
        nc_lin = torch.linspace(start_nc, end_nc, int(n_nc), device=trainer.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing="ij")
        rot_grid = torch.full_like(nr_grid, coord_q[2])
        scale_grid = torch.full_like(nr_grid, coord_q[3])
        coords_grid = torch.stack([nr_grid, nc_grid, rot_grid, scale_grid], dim=-1)

        span_nr = max(end_nr - start_nr, 1e-12)
        span_nc = max(end_nc - start_nc, 1e-12)
        gt_nr_f = (float(coord_q[0].item()) - start_nr) / span_nr * (int(n_nr) - 1)
        gt_nc_f = (float(coord_q[1].item()) - start_nc) / span_nc * (int(n_nc) - 1)
        gt_index = (
            int(np.clip(round(gt_nr_f), 0, int(n_nr) - 1)),
            int(np.clip(round(gt_nc_f), 0, int(n_nc) - 1)),
        )

        return {
            "nr_lin": nr_lin,
            "nc_lin": nc_lin,
            "coords_grid": coords_grid,
            "coords_flat": coords_grid.view(-1, 4),
            "raw_extent": (start_nc, end_nc, end_nr, start_nr),
            "raw_bounds": {
                "start_nr": start_nr,
                "end_nr": end_nr,
                "start_nc": start_nc,
                "end_nc": end_nc,
            },
            "gt_index": gt_index,
            "gt_raw": (float(coord_q[0].item()), float(coord_q[1].item())),
            "local_zoom_label": "global" if local_zoom_wh is None else local_zoom_wh,
        }

    def compute_fields(self, query, grid, plot_mode="both", use_vis_ref=False, chunk_size_vis=1024):
        trainer = self.trainer
        plot_mode = str(plot_mode or "both").strip().lower()
        if plot_mode not in {"ingp", "both"}:
            raise ValueError(f"plot_mode must be 'ingp' or 'both', got {plot_mode}")

        feat_vis = query["feat_vis"]
        coords_flat = grid["coords_flat"]
        coords_grid = grid["coords_grid"]

        energy_visencoder = None
        energy_ingp = None
        energy_projector = None

        with torch.no_grad():
            if use_vis_ref:
                feat_q_vis = TF.normalize(feat_vis, dim=-1)
                dist_chunks = []
                for start in range(0, coords_flat.shape[0], int(chunk_size_vis)):
                    end = min(start + int(chunk_size_vis), coords_flat.shape[0])
                    satimgs_refs_chunk = trainer.sat_dataset.crop_satimg_by_4d_coords(coords_flat[start:end].cpu()).to(trainer.device)
                    feats_ref_chunk = TF.normalize(trainer._get_feats_fm_imgs(satimgs_refs_chunk), dim=-1)
                    dist_chunk = torch.norm(feats_ref_chunk - feat_q_vis, dim=-1)
                    dist_chunks.append(dist_chunk)
                dist_visencoder = torch.cat(dist_chunks, dim=0).reshape(*coords_grid.shape[:2])
                energy_visencoder = torch.exp(-dist_visencoder)
            else:
                dist_ingp = trainer._compute_metric_from_query_and_points(
                    metric="dist",
                    feat_type="ingp",
                    query_feats=feat_vis,
                    ref_points=coords_flat,
                    temperature=trainer.energy_temperature,
                ).reshape(*coords_grid.shape[:2])
                energy_ingp = torch.exp(-dist_ingp)

                if plot_mode == "both":
                    dist_projector = trainer._compute_metric_from_query_and_points(
                        metric="dist",
                        feat_type="projector",
                        query_feats=feat_vis,
                        ref_points=coords_flat,
                        temperature=trainer.energy_temperature,
                    ).reshape(*coords_grid.shape[:2])
                    energy_projector = torch.exp(-dist_projector)

        return {
            "energy_visencoder": energy_visencoder,
            "energy_ingp": energy_ingp,
            "energy_projector": energy_projector,
        }

    @staticmethod
    def default_contour_kwargs():
        return {
            "crop_size": 0,
            "with_flow": False,
            "flow_mode": "ascent",
            "unified_scale": False,
            "cmap": ["#ffffff", "#3a3a3a"],
            "n_fill_levels": 24,
            "n_line_levels": 32,
            "contour_line_color": "#6a6a6a",
            "contour_line_width": 0.8,
            "contour_line_alpha": 0.6,
        }

    @staticmethod
    def _merge_contour_kwargs(plot_contour_setting):
        contour_kwargs = Stage3EnergyFieldAnalyzer.default_contour_kwargs()
        if plot_contour_setting:
            reserved_keys = {"dist_ingp", "dist_proj", "gt_coords", "save_path"}
            overlap = reserved_keys.intersection(plot_contour_setting.keys())
            if overlap:
                raise ValueError(
                    "plot_contour_setting contains reserved keys managed by analyze_energy_field: "
                    + ", ".join(sorted(overlap))
                )
            contour_kwargs.update(plot_contour_setting)
        return contour_kwargs

    @staticmethod
    def _suffix_from_zoom(local_zoom_label):
        if local_zoom_label == "global":
            return "global"
        return f"hw{float(local_zoom_label[0]):.2f}"

    @staticmethod
    def default_map_plot_setting():
        return {
            "draw_heatmap": True,
            "draw_contour": True,
            "cmap": "magma",
            "heatmap_alpha": 0.55,
            "n_fill_levels": 64,
            "contour_alpha": 0.65,
            "contour_color": "#f7f0da",
            "contour_line_width": 0.55,
            "n_line_levels": 24,
            "background_alpha": 1.0,
            "show_gt_marker": True,
            "gt_marker": "*",
            "gt_marker_size": 150,
            "gt_marker_facecolor": "#f4e84a",
            "gt_marker_edgecolor": "#1f1f1f",
            "gt_marker_linewidth": 1.0,
            "show_axis": False,
            "show_frame": True,
            "frame_line_width": 0.8,
            "frame_color": "#303030",
            "title": None,
            "title_fontsize": 10,
            "fig_width": 5.2,
            "dpi": 300,
            "pad_inches": 0.02,
            "transparent": False,
        }

    @staticmethod
    def _merge_map_plot_setting(map_plot_setting=None, legacy_plot_kwargs=None):
        cfg = Stage3EnergyFieldAnalyzer.default_map_plot_setting()
        legacy_plot_kwargs = dict(legacy_plot_kwargs or {})
        map_plot_setting = dict(map_plot_setting or {})

        # Keep old direct keyword calls working while encouraging map_plot_setting.
        legacy_aliases = {
            "heatmap_alpha",
            "contour_alpha",
            "cmap",
            "n_fill_levels",
            "n_line_levels",
            "dpi",
            "show_gt_marker",
            "title",
            "transparent",
        }
        unknown = set(legacy_plot_kwargs) - legacy_aliases
        if unknown:
            raise TypeError("Unsupported map plot kwargs: " + ", ".join(sorted(unknown)))
        cfg.update(legacy_plot_kwargs)
        cfg.update(map_plot_setting)

        if not cfg["draw_heatmap"] and not cfg["draw_contour"]:
            raise ValueError("At least one of draw_heatmap or draw_contour must be True.")
        return cfg

    @staticmethod
    def default_surface_plot_setting():
        return {
            "colorscale": "RdBu_r",
            "show_axis_info": True,
            "visencoder_show_axis_info": False,
            "width": 900,
            "height": 700,
            "colorbar_title": "Value",
            "hovertemplate": None,
            "z_aspect": 0.45,
            "include_plotlyjs": "cdn",
            "clean_scene": False,
        }

    @staticmethod
    def _merge_surface_plot_setting(surface_plot_setting=None):
        cfg = Stage3EnergyFieldAnalyzer.default_surface_plot_setting()
        surface_plot_setting = dict(surface_plot_setting or {})
        unknown = set(surface_plot_setting) - set(cfg)
        if unknown:
            raise TypeError("Unsupported surface plot setting keys: " + ", ".join(sorted(unknown)))
        cfg.update(surface_plot_setting)
        return cfg

    @staticmethod
    def _surface_plot_kwargs(surface_plot_cfg, show_axis_info):
        return {
            "colorscale": surface_plot_cfg["colorscale"],
            "show_axis_info": bool(show_axis_info),
            "width": int(surface_plot_cfg["width"]),
            "height": int(surface_plot_cfg["height"]),
            "colorbar_title": surface_plot_cfg["colorbar_title"],
            "hovertemplate": surface_plot_cfg["hovertemplate"],
            "z_aspect": float(surface_plot_cfg["z_aspect"]),
            "include_plotlyjs": surface_plot_cfg["include_plotlyjs"],
            "clean_scene": bool(surface_plot_cfg["clean_scene"]),
        }

    def render_default_outputs(self, result, plot_mode="both", plot_contour_setting=None, surface_plot_setting=None):
        from vis_featmap import plot_contour, vis_griddata_in_3d_surface_interactive

        plot_mode = str(plot_mode or "both").strip().lower()
        contour_kwargs = self._merge_contour_kwargs(plot_contour_setting)
        surface_cfg = self._merge_surface_plot_setting(surface_plot_setting)
        dir2save, epoch = self.resolve_save_context(plot_mode=plot_mode)
        set_name = result["query"]["set_name"]
        idx = result["query"]["query_id"]
        n_nr = int(result["grid"]["coords_grid"].shape[0])
        suffix = self._suffix_from_zoom(result["grid"]["local_zoom_label"])

        energy_ingp = result["fields"]["energy_ingp"]
        energy_projector = result["fields"]["energy_projector"]
        energy_visencoder = result["fields"]["energy_visencoder"]
        saved_paths = []

        if energy_ingp is not None and plot_mode == "ingp":
            contour_p2save = os.path.join(dir2save, f"contour_{set_name}_id{idx}_ns{n_nr}_{suffix}_ingp.png")
            plot_contour(
                dist_ingp=energy_ingp.detach().cpu(),
                dist_proj=None,
                gt_coords=result["grid"]["gt_index"],
                save_path=contour_p2save,
                **contour_kwargs,
            )
            print("已保存" + contour_p2save)
            saved_paths.append(contour_p2save)
        elif energy_ingp is not None and energy_projector is not None and plot_mode == "both":
            contour_p2save = os.path.join(dir2save, f"contour_{set_name}_id{idx}_ns{n_nr}_{suffix}_ingp&projector.png")
            plot_contour(
                dist_ingp=energy_ingp.detach().cpu(),
                dist_proj=energy_projector.detach().cpu(),
                gt_coords=result["grid"]["gt_index"],
                save_path=contour_p2save,
                **contour_kwargs,
            )
            print("已保存" + contour_p2save)
            saved_paths.append(contour_p2save)

        if energy_projector is not None and plot_mode == "both":
            path = os.path.join(dir2save, f"ep{epoch}_energy_projector_{set_name}_id{idx}_ns{n_nr}_{suffix}.html")
            vis_griddata_in_3d_surface_interactive(
                energy_projector,
                p2save=path,
                **self._surface_plot_kwargs(surface_cfg, surface_cfg["show_axis_info"]),
            )
            print("已保存" + path)
            saved_paths.append(path)

        if energy_ingp is not None:
            path = os.path.join(dir2save, f"ep{epoch}_energy_ingp_{set_name}_id{idx}_nr{n_nr}_{suffix}.html")
            vis_griddata_in_3d_surface_interactive(
                energy_ingp,
                p2save=path,
                **self._surface_plot_kwargs(surface_cfg, surface_cfg["show_axis_info"]),
            )
            print("已保存" + path)
            saved_paths.append(path)

        if energy_visencoder is not None:
            path = os.path.join(dir2save, f"ep{epoch}_energy_visencoder_{set_name}_id{idx}_ns{n_nr}_{suffix}.html")
            vis_griddata_in_3d_surface_interactive(
                energy_visencoder,
                p2save=path,
                **self._surface_plot_kwargs(surface_cfg, surface_cfg["visencoder_show_axis_info"]),
            )
            print("已保存" + path)
            saved_paths.append(path)

        result.setdefault("saved_paths", []).extend(saved_paths)
        return saved_paths

    def render_map_contour(
            self,
            result,
            field_name="energy_ingp",
            save_path=None,
            background_img=None,
            background_extent=None,
            map_plot_setting=None,
            **legacy_plot_kwargs,
    ):
        """Render a map-proportional contour/heatmap in raw NR/NC coordinates."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        field = result["fields"].get(field_name)
        if field is None:
            raise ValueError(f"Field '{field_name}' is not available in result.")

        plot_cfg = self._merge_map_plot_setting(map_plot_setting, legacy_plot_kwargs)

        data = field.detach().cpu().numpy() if hasattr(field, "detach") else np.asarray(field)
        nr = result["grid"]["nr_lin"].detach().cpu().numpy()
        nc = result["grid"]["nc_lin"].detach().cpu().numpy()
        xx, yy = np.meshgrid(nc, nr)

        extent = background_extent if background_extent is not None else result["grid"]["raw_extent"]
        width = abs(float(extent[1]) - float(extent[0]))
        height = abs(float(extent[2]) - float(extent[3]))
        fig_w = float(plot_cfg["fig_width"])
        fig_h = max(2.0, fig_w * height / max(width, 1e-12))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=int(plot_cfg["dpi"]))

        if background_img is not None:
            if hasattr(background_img, "detach"):
                background_img = background_img.detach().cpu().numpy()
            background_img = np.asarray(background_img)
            if background_img.ndim == 3 and background_img.shape[0] in {1, 3, 4}:
                background_img = np.moveaxis(background_img, 0, -1)
            ax.imshow(background_img, extent=extent, origin="upper", alpha=float(plot_cfg["background_alpha"]))

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
            if np.isclose(vmin, vmax):
                pad = max(abs(vmin) * 0.05, 1e-6)
                vmin, vmax = vmin - pad, vmax + pad

        if plot_cfg["draw_heatmap"]:
            fill_levels = np.linspace(vmin, vmax, max(3, int(plot_cfg["n_fill_levels"])))
            ax.contourf(
                xx,
                yy,
                data,
                levels=fill_levels,
                cmap=plot_cfg["cmap"],
                alpha=float(plot_cfg["heatmap_alpha"]),
            )
        if plot_cfg["draw_contour"]:
            line_levels = np.linspace(vmin, vmax, max(3, int(plot_cfg["n_line_levels"])))
            ax.contour(
                xx,
                yy,
                data,
                levels=line_levels,
                colors=plot_cfg["contour_color"],
                linewidths=float(plot_cfg["contour_line_width"]),
                alpha=float(plot_cfg["contour_alpha"]),
            )

        if plot_cfg["show_gt_marker"]:
            gt_nr, gt_nc = result["grid"]["gt_raw"]
            ax.scatter(
                [gt_nc],
                [gt_nr],
                marker=plot_cfg["gt_marker"],
                s=float(plot_cfg["gt_marker_size"]),
                facecolor=plot_cfg["gt_marker_facecolor"],
                edgecolor=plot_cfg["gt_marker_edgecolor"],
                linewidth=float(plot_cfg["gt_marker_linewidth"]),
                zorder=8,
            )

        if plot_cfg["title"]:
            ax.set_title(str(plot_cfg["title"]), fontsize=float(plot_cfg["title_fontsize"]))
        ax.set_xlim(float(extent[0]), float(extent[1]))
        ax.set_ylim(float(extent[2]), float(extent[3]))
        ax.set_aspect("equal", adjustable="box")
        if not plot_cfg["show_axis"]:
            ax.set_xticks([])
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(bool(plot_cfg["show_frame"]))
            spine.set_linewidth(float(plot_cfg["frame_line_width"]))
            spine.set_color(plot_cfg["frame_color"])
        fig.tight_layout(pad=0.02)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                save_path,
                dpi=int(plot_cfg["dpi"]),
                bbox_inches="tight",
                pad_inches=float(plot_cfg["pad_inches"]),
                transparent=bool(plot_cfg["transparent"]),
            )
            print("已保存" + save_path)
            result.setdefault("saved_paths", []).append(save_path)
        plt.close(fig)
        return fig

    def analyze_fft(self, result, plot_mode="both"):
        energy_ingp = result["fields"]["energy_ingp"]
        energy_projector = result["fields"]["energy_projector"]
        if energy_ingp is None or energy_projector is None:
            print(f"WARNING: analyse_fft requires both INGP and projector fields; skip for plot_mode='{plot_mode}'.")
            return None

        from scripts.analysis.util_fft_analyse import analyse_feature_frequency

        res = analyse_feature_frequency(
            energy_ingp[..., None], energy_projector[..., None],
            cdf_tau=0.95, hf_frac=0.33, eps=1e-12, wo_DC=True,
            norm="ortho", channel_norm=True, return_radial=True,
        )
        mF, mZ, d = res["metrics_F"], res["metrics_Z"], res["delta"]
        print(f"[INGP energy space] fc={mF['fc']:.3f}, f95={mF['f95']:.1f}, hf_ratio={mF['hf_ratio']:.5f} (f0 bin={mF['f0_bin']})")
        print(f"[Proj energy space] fc={mZ['fc']:.3f}, f95={mZ['f95']:.1f}, hf_ratio={mZ['hf_ratio']:.5f} (f0 bin={mZ['f0_bin']})")
        print(f"Delta fc   : {d['fc']:+.3f}")
        print(f"Delta f95  : {d['f95']:+.1f}")
        print(f"HF ratio Z/F: {d['hf_ratio_Z_over_F']:.3f}")

        res_with_dc = analyse_feature_frequency(
            energy_ingp[..., None], energy_projector[..., None],
            cdf_tau=0.95, hf_frac=0.33, eps=1e-12, wo_DC=False,
            norm="ortho", channel_norm=True, return_radial=True,
        )
        dir2save, epoch = self.resolve_save_context(plot_mode=plot_mode)
        P_F, P_Z = res_with_dc["P_F"], res_with_dc["P_Z"]
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
        save_path = os.path.join(dir2save, f"energy_space_fft_w_dc_ep{epoch}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print("已保存" + save_path)
        result.setdefault("saved_paths", []).append(save_path)
        return res

    def analyze(
            self,
            n_nr=128,
            n_nc=128,
            use_train_uav=True,
            local_zoom_wh=None,
            vis=False,
            use_vis_ref=False,
            chunk_size_vis=1024,
            analyse_fft=False,
            query_id=20,
            plot_mode="both",
            plot_contour_setting=None,
            render_map_contour=False,
            map_contour_setting=None,
            map_plot_setting=None,
            surface_plot_setting=None,
    ):
        query = self.resolve_query(query_id=query_id, use_train_uav=use_train_uav)
        grid = self.build_grid(query["coord_q"], n_nr=n_nr, n_nc=n_nc, local_zoom_wh=local_zoom_wh, verbose=vis)
        fields = self.compute_fields(
            query=query,
            grid=grid,
            plot_mode=plot_mode,
            use_vis_ref=use_vis_ref,
            chunk_size_vis=chunk_size_vis,
        )
        result = {
            "query": query,
            "grid": grid,
            "fields": fields,
            "plot_mode": plot_mode,
            "saved_paths": [],
        }

        if vis:
            self.render_default_outputs(
                result,
                plot_mode=plot_mode,
                plot_contour_setting=plot_contour_setting,
                surface_plot_setting=surface_plot_setting,
            )

        if render_map_contour:
            setting = dict(map_contour_setting or {})
            if map_plot_setting is not None:
                setting["map_plot_setting"] = map_plot_setting
            if "save_path" not in setting:
                dir2save, _ = self.resolve_save_context(plot_mode=plot_mode)
                suffix = self._suffix_from_zoom(grid["local_zoom_label"])
                setting["save_path"] = os.path.join(
                    dir2save,
                    f"map_contour_{query['set_name']}_id{query['query_id']}_ns{n_nr}_{suffix}.png",
                )
            self.render_map_contour(result, **setting)

        if analyse_fft:
            self.analyze_fft(result, plot_mode=plot_mode)

        return result
