import numpy as np
import torch


def _log_runtime_config(self, message):
    if getattr(self, "logger", None) is not None:
        self.logger.info(message)
    else:
        print(message)


def _estimate_stage3_grid_shape_from_overlap(self, overlap):
    sat_dataset = self.sat_dataset
    crop_size = float(sat_dataset.satimgsize2crop_mean)
    overlap = float(overlap)

    if hasattr(sat_dataset, "estimate_grid_shape_from_overlap"):
        return sat_dataset.estimate_grid_shape_from_overlap(
            size2clip=crop_size,
            overlap=overlap,
        )

    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"stage3_overlap must be in [0, 1), got {overlap}")

    step_size_px = int(crop_size * (1.0 - overlap))
    if step_size_px <= 0:
        raise ValueError(
            f"overlap={overlap:.6f} is too dense for crop_size={crop_size:.3f}; "
            "it makes the pixel step become 0."
        )

    row_centers = (
        np.arange(max(int((sat_dataset.satmap_h - crop_size) / step_size_px), 0), dtype=np.float32)
        * step_size_px
        + crop_size / 2.0
    ) / float(sat_dataset.satmap_hw_max)
    col_centers = (
        np.arange(max(int((sat_dataset.satmap_w - crop_size) / step_size_px), 0), dtype=np.float32)
        * step_size_px
        + crop_size / 2.0
    ) / float(sat_dataset.satmap_hw_max)

    nr_min, nr_max = sat_dataset.nr2sample_range
    nc_min, nc_max = sat_dataset.nc2sample_range
    row_mask = (row_centers >= nr_min) & (row_centers <= nr_max)
    col_mask = (col_centers >= nc_min) & (col_centers <= nc_max)

    return {
        "overlap": overlap,
        "crop_size_px": crop_size,
        "step_size_px": step_size_px,
        "grid_rows": int(row_mask.sum()),
        "grid_cols": int(col_mask.sum()),
    }


def _resolve_stage3_sampling_config(self):
    opt = self.opt
    sat_dataset = self.sat_dataset

    explicit_n_coarse = getattr(opt, "n_coarse", None)
    explicit_n_fine = getattr(opt, "n_fine_per_coarse", None)

    if explicit_n_coarse is not None:
        n_coarse = tuple(int(v) for v in self._ensure_param_sequence(explicit_n_coarse, int))
        if len(n_coarse) != 4:
            raise ValueError(f"n_coarse must be length-4, got {n_coarse}")
        layout_source = "opt.n_coarse"
        stage3_overlap = getattr(opt, "stage3_overlap", None)
    else:
        stage3_overlap = float(
            getattr(
                opt,
                "stage3_overlap",
                getattr(opt, "subspace_overlap", getattr(opt, "val_overlap", 0.5)),
            )
        )
        grid_info = _estimate_stage3_grid_shape_from_overlap(self, stage3_overlap)
        n_rot = int(getattr(opt, "stage3_n_rot", getattr(opt, "n_rot", 36)))
        n_scale = int(getattr(opt, "stage3_n_scale", getattr(opt, "n_scale", 1)))
        n_coarse = (
            int(grid_info["grid_rows"]),
            int(grid_info["grid_cols"]),
            n_rot,
            n_scale,
        )
        layout_source = f"dataset_overlap={stage3_overlap:.4f}"

    if explicit_n_fine is not None:
        n_fine_per_coarse = tuple(int(v) for v in self._ensure_param_sequence(explicit_n_fine, int))
        if len(n_fine_per_coarse) != 4:
            raise ValueError(f"n_fine_per_coarse must be length-4, got {n_fine_per_coarse}")
    else:
        n_fine_per_coarse = (1, 1, 1, 1)

    sigma_nrc_factor = float(getattr(opt, "stage3_sigma_nrc_factor", 1.0))
    sigma_rot_factor = float(getattr(opt, "stage3_sigma_rot_factor", 0.65))
    sigma_scale_factor = float(getattr(opt, "stage3_sigma_scale_factor", 0.65))

    gs_sigma_nrc = float(
        getattr(opt, "gs_sigma_nrc", float(sat_dataset.halfimg_radius_nrc) * sigma_nrc_factor)
    )

    rot_bin_width = float((2.0 * torch.pi) / max(int(n_coarse[2]), 1))
    gs_sigma_radrot = float(
        getattr(opt, "gs_sigma_radrot", rot_bin_width * sigma_rot_factor)
    )

    scale_boundary = np.asarray(sat_dataset.satimgsize_scale_to_refm_boundary, dtype=np.float32)
    scale_log_span = float(np.log(scale_boundary[1] / scale_boundary[0])) if scale_boundary[0] > 0 else 0.0
    if hasattr(opt, "gs_sigma_logscale"):
        gs_sigma_logscale = float(getattr(opt, "gs_sigma_logscale"))
    elif int(n_coarse[3]) > 1 and scale_log_span > 0:
        gs_sigma_logscale = float((scale_log_span / int(n_coarse[3])) * sigma_scale_factor)
    else:
        gs_sigma_logscale = float(getattr(opt, "stage3_sigma_logscale_default", 0.1))

    self.n_coarse = n_coarse
    self.n_fine_per_coarse = n_fine_per_coarse
    self.gs_sigma_nrc = gs_sigma_nrc
    self.gs_sigma_radrot = gs_sigma_radrot
    self.gs_sigma_logscale = gs_sigma_logscale
    self.stage3_overlap = stage3_overlap

    _log_runtime_config(
        self,
        "Stage3 sampler配置: "
        f"source={layout_source}, "
        f"n_coarse={self.n_coarse}, "
        f"n_fine_per_coarse={self.n_fine_per_coarse}, "
        f"gs_sigma_nrc={self.gs_sigma_nrc:.6f}, "
        f"gs_sigma_radrot={self.gs_sigma_radrot:.6f}, "
        f"gs_sigma_logscale={self.gs_sigma_logscale:.6f}",
    )


def _init_stage3_test_runtime(self, use_train_uav=False):
    self._init_datasets(create_train_loader=False)
    self._resolve_stage3_sampling_config()

    from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor
    self.coord_normer = CoordsNormProcessor(self.sat_dataset)

    from trainer_depends.datasets.util_core_subspace_sampler import SubspaceSampler
    self.subspace_sampler = SubspaceSampler(
        sat_dataset=self.sat_dataset,
        n_coarse=self.n_coarse,
        n_fine_per_coarse=self.n_fine_per_coarse,
    )

    self.uav_dataloader_test = torch.utils.data.DataLoader(
        self.uav_dataset_test,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    if use_train_uav:
        self.uav_dataloader_train = torch.utils.data.DataLoader(
            self.uav_dataset_train,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )


def _init_stage3_train_runtime(self):
    opt = self.opt

    self._init_datasets(create_train_loader=False)
    self._resolve_stage3_sampling_config()

    self.sat_dataloader = torch.utils.data.DataLoader(
        self.sat_dataset,
        batch_size=opt.batchsize_sat,
        num_workers=opt.num_worker,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(opt.num_worker > 0),
    )

    self.uav_dataloader_train = torch.utils.data.DataLoader(
        self.uav_dataset_train,
        batch_size=opt.batchsize_uav,
        num_workers=opt.num_worker,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(opt.num_worker > 0),
    )

    self.uav_dataloader_test = torch.utils.data.DataLoader(
        self.uav_dataset_test,
        batch_size=opt.batchsize_uav,
        num_workers=opt.num_worker,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(opt.num_worker > 0),
    )

    from trainer_depends.datasets.util_core_coords_translater import CoordsNormProcessor
    self.coord_normer = CoordsNormProcessor(self.sat_dataset)

    from trainer_depends.utils.util_gaussian_importance_sampler import NormalizedGaussianSampler
    self.normed_sigmas = self.coord_normer.get_linear_sigmas(
        self.gs_sigma_nrc,
        self.gs_sigma_radrot,
        self.gs_sigma_logscale,
    )
    self.gs_sampler = NormalizedGaussianSampler(self.normed_sigmas, device=self.device)

    from trainer_depends.datasets.util_core_subspace_sampler import SubspaceSampler
    self.n_points_per_subspace = getattr(opt, 'n_points_per_subspace', 1)
    self.subspace_sampler = SubspaceSampler(
        sat_dataset=self.sat_dataset,
        n_coarse=self.n_coarse,
        n_fine_per_coarse=self.n_fine_per_coarse,
    )
