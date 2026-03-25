import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as TF
import tqdm

from trainer_depends.datasets.util_core_subspace_sampler import SubspaceSampler


@dataclass
class Stage1ReferenceGalleryLayoutConfig:
    mode: str = "overlap"
    n_bins_4d: object = None
    overlap: float = 0.5
    n_rot: int = 1
    n_scale: int = 1
    scale_mode: str = "linear"
    rot_values: object = None


@dataclass
class Stage1ReferenceGalleryFeatureConfig:
    chunk_size_vis: int = 1024
    normalize_feats: bool = True
    build_faiss: bool = True
    show_progress: bool = True


@dataclass
class Stage1ReferenceGalleryDownsampleConfig:
    mode: str = "stride"
    stride_4d: tuple = (1, 1, 1, 1)
    interp_target: str = "coords"
    build_faiss: bool = True


class Stage1ReferenceGalleryBank:
    """
    A standalone gallery manager for Stage 1 reference banks.

    Responsibilities:
    - build gallery coords from either `n_bins_4d` or `overlap + n_rot + n_scale`
    - optionally extract visual features and build a FAISS index
    - save/load gallery artifacts for reuse
    """

    def __init__(self, sat_dataset, trainer=None, device=None):
        self.sat_dataset = sat_dataset
        self.trainer = trainer
        self.device = device or getattr(trainer, "device", torch.device("cpu"))

        self.layout_cfg = None
        self.meta = {}
        self.coords_gallery = None
        self.feats_gallery = None
        self.faiss_index = None

    # ------------------------------------------------------------------
    # Config / metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mode(mode):
        mode = str(mode).strip().lower()
        if mode not in ("n_bins_4d", "overlap"):
            raise ValueError(f"layout mode must be 'n_bins_4d' or 'overlap', got {mode}")
        return mode

    @staticmethod
    def _normalize_scale_mode(scale_mode):
        scale_mode = str(scale_mode).strip().lower()
        if scale_mode not in ("linear", "log"):
            raise ValueError(f"scale_mode must be 'linear' or 'log', got {scale_mode}")
        return scale_mode

    @staticmethod
    def _to_python(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _get_scale_boundary(self):
        if hasattr(self.sat_dataset, "satimgsize_scale_to_ref_m_boundary"):
            return np.asarray(self.sat_dataset.satimgsize_scale_to_ref_m_boundary, dtype=np.float32)
        if hasattr(self.sat_dataset, "satimgsize_scale_to_refm_boundary"):
            return np.asarray(self.sat_dataset.satimgsize_scale_to_refm_boundary, dtype=np.float32)
        raise AttributeError("sat_dataset has no scale boundary attribute.")

    def _get_scale_mean(self):
        if hasattr(self.sat_dataset, "satimgsize_scale_to_ref_m_mean"):
            return float(self.sat_dataset.satimgsize_scale_to_ref_m_mean)
        if hasattr(self.sat_dataset, "satimgsize_scale_to_refm_mean"):
            return float(self.sat_dataset.satimgsize_scale_to_refm_mean)
        boundary = self._get_scale_boundary()
        return float(boundary.mean())

    def _validate_layout_cfg(self, layout_cfg):
        cfg = layout_cfg if isinstance(layout_cfg, Stage1ReferenceGalleryLayoutConfig) else Stage1ReferenceGalleryLayoutConfig(**layout_cfg)
        cfg.mode = self._normalize_mode(cfg.mode)
        cfg.scale_mode = self._normalize_scale_mode(cfg.scale_mode)

        if cfg.mode == "n_bins_4d":
            n_bins = np.asarray(cfg.n_bins_4d, dtype=np.int32).reshape(-1)
            if n_bins.size != 4 or (n_bins <= 0).any():
                raise ValueError("n_bins_4d must be length-4 with positive entries.")
            cfg.n_bins_4d = tuple(int(x) for x in n_bins.tolist())
            cfg.rot_values = None
            return cfg

        if not (0 <= float(cfg.overlap) < 1):
            raise ValueError("overlap must be in [0, 1).")
        if cfg.rot_values is not None:
            rot_values = np.asarray(cfg.rot_values, dtype=np.float32).reshape(-1)
            if rot_values.size == 0:
                raise ValueError("rot_values must be non-empty when provided.")
            cfg.rot_values = rot_values.tolist()
            cfg.n_rot = int(rot_values.size)
        else:
            cfg.rot_values = None
            cfg.n_rot = int(cfg.n_rot)
        cfg.n_scale = int(cfg.n_scale)
        if cfg.n_rot <= 0:
            raise ValueError("n_rot must be > 0.")
        if cfg.n_scale <= 0:
            raise ValueError("n_scale must be > 0.")
        return cfg

    def _validate_downsample_cfg(self, downsample_cfg):
        cfg = downsample_cfg if isinstance(downsample_cfg, Stage1ReferenceGalleryDownsampleConfig) else (
            Stage1ReferenceGalleryDownsampleConfig(**downsample_cfg)
        )
        cfg.mode = str(cfg.mode).strip().lower()
        if cfg.mode != "stride":
            raise ValueError(f"Only stride downsampling is supported now, got {cfg.mode}")
        stride_4d = np.asarray(cfg.stride_4d, dtype=np.float64).reshape(-1)
        if stride_4d.size != 4 or (stride_4d < 1).any():
            raise ValueError("stride_4d must be length-4 with entries >= 1.")
        cfg.stride_4d = tuple(float(v) for v in stride_4d.tolist())
        cfg.interp_target = str(cfg.interp_target).strip().lower()
        if cfg.interp_target not in ("coords", "features"):
            raise ValueError(f"interp_target must be 'coords' or 'features', got {cfg.interp_target}")
        return cfg

    def _estimate_grid_shape_from_overlap(self, overlap):
        if hasattr(self.sat_dataset, "estimate_grid_shape_from_overlap"):
            return self.sat_dataset.estimate_grid_shape_from_overlap(
                size2clip=float(self.sat_dataset.satimgsize2crop_mean),
                overlap=overlap,
            )

        overlap = float(overlap)
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1).")

        crop_size = float(self.sat_dataset.satimgsize2crop_mean)
        step_size_px = int(crop_size * (1.0 - overlap))
        if step_size_px <= 0:
            raise ValueError(
                f"overlap={overlap:.6f} is too dense for crop_size={crop_size:.3f}; "
                "it makes the pixel step become 0."
            )

        n_rowsteps = int((self.sat_dataset.satmap_h - crop_size) / step_size_px)
        n_colsteps = int((self.sat_dataset.satmap_w - crop_size) / step_size_px)
        row_centers = (
            np.arange(max(n_rowsteps, 0), dtype=np.float32) * step_size_px + crop_size / 2.0
        ) / float(self.sat_dataset.satmap_hw_max)
        col_centers = (
            np.arange(max(n_colsteps, 0), dtype=np.float32) * step_size_px + crop_size / 2.0
        ) / float(self.sat_dataset.satmap_hw_max)

        nr_min, nr_max = self.sat_dataset.nr2sample_range
        nc_min, nc_max = self.sat_dataset.nc2sample_range
        row_mask = (row_centers >= nr_min) & (row_centers <= nr_max)
        col_mask = (col_centers >= nc_min) & (col_centers <= nc_max)
        row_centers_valid = row_centers[row_mask]
        col_centers_valid = col_centers[col_mask]
        grid_rows = int(row_mask.sum())
        grid_cols = int(col_mask.sum())
        return {
            "overlap": overlap,
            "crop_size_px": crop_size,
            "step_size_px": step_size_px,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "row_centers_nrc_range": (
                float(row_centers_valid[0]),
                float(row_centers_valid[-1]),
            ) if row_centers_valid.size > 0 else (None, None),
            "col_centers_nrc_range": (
                float(col_centers_valid[0]),
                float(col_centers_valid[-1]),
            ) if col_centers_valid.size > 0 else (None, None),
        }

    def estimate_overlap_from_n_bins_4d(self, n_bins_4d):
        n_bins_arr = np.asarray(n_bins_4d, dtype=np.int32).reshape(-1)
        if n_bins_arr.size != 4 or (n_bins_arr <= 0).any():
            raise ValueError("n_bins_4d must be length-4 with positive entries.")

        target_rows = int(n_bins_arr[0])
        target_cols = int(n_bins_arr[1])
        if hasattr(self.sat_dataset, "estimate_overlap_from_grid_shape"):
            return float(self.sat_dataset.estimate_overlap_from_grid_shape(
                size2clip=float(self.sat_dataset.satimgsize2crop_mean),
                grid_rows=target_rows,
                grid_cols=target_cols,
                allow_approx=True,
            ))

        crop_size = float(self.sat_dataset.satimgsize2crop_mean)

        best_match = None
        max_step_px = max(1, int(crop_size))
        for step_size_px in range(1, max_step_px + 1):
            overlap = max(0.0, 1.0 - step_size_px / crop_size)
            grid_info = self._estimate_grid_shape_from_overlap(overlap)
            grid_rows = int(grid_info["grid_rows"])
            grid_cols = int(grid_info["grid_cols"])

            score = abs(grid_rows - target_rows) + abs(grid_cols - target_cols)
            candidate = (score, abs(grid_rows - target_rows), abs(grid_cols - target_cols), overlap)
            if best_match is None or candidate < best_match:
                best_match = candidate
            if score == 0:
                return overlap

        if best_match is None:
            raise ValueError(f"Failed to estimate overlap from n_bins_4d={tuple(int(v) for v in n_bins_arr)}")
        return float(best_match[-1])

    @classmethod
    def resolve_layout_name_info(cls, sat_dataset, layout_cfg):
        bank = cls(sat_dataset=sat_dataset)
        cfg = bank._validate_layout_cfg(layout_cfg)

        if cfg.mode == "overlap":
            grid_info = bank._estimate_grid_shape_from_overlap(cfg.overlap)
            n_bins_4d = (
                int(grid_info["grid_rows"]),
                int(grid_info["grid_cols"]),
                int(cfg.n_rot),
                int(cfg.n_scale),
            )
            overlap = float(cfg.overlap)
        else:
            n_bins_4d = tuple(int(v) for v in cfg.n_bins_4d)
            overlap = bank.estimate_overlap_from_n_bins_4d(n_bins_4d)

        return {
            "mode": cfg.mode,
            "overlap": float(overlap),
            "n_bins_4d": n_bins_4d,
            "scale_mode": cfg.scale_mode,
        }

    @classmethod
    def estimate_layout_summary(cls, sat_dataset, layout_cfg):
        bank = cls(sat_dataset=sat_dataset)
        cfg = bank._validate_layout_cfg(layout_cfg)

        if cfg.mode == "overlap":
            grid_info = bank._estimate_grid_shape_from_overlap(cfg.overlap)
            n_bins_4d = (
                int(grid_info["grid_rows"]),
                int(grid_info["grid_cols"]),
                int(cfg.n_rot),
                int(cfg.n_scale),
            )
            overlap = float(cfg.overlap)
        else:
            n_bins_4d = tuple(int(v) for v in cfg.n_bins_4d)
            overlap = float(bank.estimate_overlap_from_n_bins_4d(n_bins_4d))
            grid_info = bank._estimate_grid_shape_from_overlap(overlap)

        total_points_2d = int(grid_info["grid_rows"]) * int(grid_info["grid_cols"])
        return {
            "scene_name": getattr(sat_dataset, "name", None),
            "mode": cfg.mode,
            "overlap": float(overlap),
            "crop_size_px": int(round(float(grid_info["crop_size_px"]))),
            "step_size_px": int(grid_info["step_size_px"]),
            "grid_rows": int(grid_info["grid_rows"]),
            "grid_cols": int(grid_info["grid_cols"]),
            "n_rot": int(n_bins_4d[2]),
            "n_scale": int(n_bins_4d[3]),
            "total_points_2d": total_points_2d,
            "total_points_4d": int(np.prod(n_bins_4d)),
            "gallery_scale": float(bank._get_scale_mean()),
            "n_bins_4d": n_bins_4d,
            "scale_mode": cfg.scale_mode,
            "row_centers_nrc_range": grid_info["row_centers_nrc_range"],
            "col_centers_nrc_range": grid_info["col_centers_nrc_range"],
        }

    # ------------------------------------------------------------------
    # Coordinate builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_uniform_rot_values(n_rot):
        n_rot = int(n_rot)
        if n_rot <= 0:
            raise ValueError("n_rot must be > 0.")
        centers = (torch.arange(n_rot, dtype=torch.float32) + 0.5) * (2 * np.pi / n_rot) - np.pi
        return centers

    def _build_scale_values(self, n_scale, scale_mode):
        scale_values, satimgsize_values = self.sat_dataset.mk_sacle_levels(
            n_level=int(n_scale),
            scale_mode=scale_mode,
        )
        return scale_values.to(torch.float32), satimgsize_values.to(torch.float32)

    def _build_coords_from_bins(self, cfg):
        sampler = SubspaceSampler(
            self.sat_dataset,
            n_coarse=cfg.n_bins_4d,
            n_fine_per_coarse=(1, 1, 1, 1),
        )
        coords_gallery = sampler.sample_uniform_grid_by_bins(
            cfg.n_bins_4d,
            device=torch.device("cpu"),
            scale_mode=cfg.scale_mode,
        )
        n_nr, n_nc, n_rot, n_scale = [int(x) for x in cfg.n_bins_4d]
        meta = {
            "mode": "n_bins_4d",
            "n_bins_4d": cfg.n_bins_4d,
            "n_rot": n_rot,
            "n_scale": n_scale,
            "scale_mode": cfg.scale_mode,
            "gallery_scale_mean": self._get_scale_mean(),
            "gallery_scale_boundary": self._get_scale_boundary().tolist(),
            "gallery_has_rot": n_rot > 1,
        }
        return coords_gallery, meta

    def _build_coords_from_overlap(self, cfg):
        crop_size = float(self.sat_dataset.satimgsize2crop_mean)
        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=crop_size,
            overlap=float(cfg.overlap),
            only_nrcs=True,
        )
        nrcs_flat = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
        if cfg.rot_values is not None:
            rot_vals = torch.as_tensor(cfg.rot_values, dtype=torch.float32)
        else:
            rot_vals = self._build_uniform_rot_values(cfg.n_rot)
        scale_vals, satimgsize_values = self._build_scale_values(cfg.n_scale, cfg.scale_mode)

        n_pos = int(nrcs_flat.shape[0])
        n_rot = int(rot_vals.numel())
        n_scale = int(scale_vals.numel())

        nrc_rep = nrcs_flat[:, None, None, :].repeat(1, n_rot, n_scale, 1)
        rot_rep = rot_vals[None, :, None, None].repeat(n_pos, 1, n_scale, 1)
        scale_rep = scale_vals[None, None, :, None].repeat(n_pos, n_rot, 1, 1)
        coords_gallery = torch.cat([nrc_rep, rot_rep, scale_rep], dim=-1).reshape(-1, 4)

        meta = {
            "mode": "overlap",
            "overlap": float(cfg.overlap),
            "crop_size_px": float(crop_size),
            "grid_rows": int(nrcs_gallery.shape[0]),
            "grid_cols": int(nrcs_gallery.shape[1]),
            "n_rot": n_rot,
            "n_scale": n_scale,
            "scale_mode": cfg.scale_mode,
            "rot_values": rot_vals.tolist(),
            "scale_values": scale_vals.tolist(),
            "satimgsize_values": satimgsize_values.tolist(),
            "n_bins_4d": (int(nrcs_gallery.shape[0]), int(nrcs_gallery.shape[1]), n_rot, n_scale),
            "gallery_scale_mean": self._get_scale_mean(),
            "gallery_scale_boundary": self._get_scale_boundary().tolist(),
            "gallery_has_rot": n_rot > 1,
        }
        return coords_gallery, meta

    def build_coords(self, layout_cfg):
        cfg = self._validate_layout_cfg(layout_cfg)
        self.layout_cfg = cfg

        if cfg.mode == "n_bins_4d":
            coords_gallery, meta = self._build_coords_from_bins(cfg)
        else:
            coords_gallery, meta = self._build_coords_from_overlap(cfg)

        self.coords_gallery = coords_gallery.to(torch.float32).cpu()
        self.feats_gallery = None
        self.faiss_index = None
        self.meta = {
            "layout_cfg": asdict(cfg),
            "scene_name": getattr(self.sat_dataset, "name", None),
            "n_points": int(self.coords_gallery.shape[0]),
            **{k: self._to_python(v) for k, v in meta.items()},
        }
        return self.coords_gallery

    def _require_grid_shape(self):
        n_bins_4d = self.meta.get("n_bins_4d", None)
        if n_bins_4d is None:
            raise ValueError("Gallery metadata does not contain n_bins_4d, cannot downsample.")
        n_bins_4d = tuple(int(v) for v in n_bins_4d)
        if np.prod(n_bins_4d) != int(self.coords_gallery.shape[0]):
            raise ValueError(
                f"Gallery size {self.coords_gallery.shape[0]} does not match n_bins_4d product {np.prod(n_bins_4d)}."
            )
        return n_bins_4d

    @staticmethod
    def _has_effective_rotation(rot_values, atol=1e-6):
        rot_arr = np.asarray(rot_values, dtype=np.float32).reshape(-1)
        if rot_arr.size == 0:
            return False
        return bool(rot_arr.size > 1 or np.any(np.abs(rot_arr) > float(atol)))

    def _summarize_axis_meta_from_coords(self, coords_gallery, n_bins_4d):
        coords_grid = coords_gallery.reshape(*n_bins_4d, 4)
        row_values = coords_grid[:, 0, 0, 0, 0].to(torch.float32).cpu().tolist()
        col_values = coords_grid[0, :, 0, 0, 1].to(torch.float32).cpu().tolist()
        rot_values = coords_grid[0, 0, :, 0, 2].to(torch.float32).cpu().tolist()
        scale_values = coords_grid[0, 0, 0, :, 3].to(torch.float32).cpu().tolist()

        satimgsize_values = None
        if hasattr(self.sat_dataset, "scale_ref_m") and hasattr(self.sat_dataset, "geo_res_m"):
            satimgsize_factor = float(self.sat_dataset.scale_ref_m) / float(self.sat_dataset.geo_res_m)
            satimgsize_values = (
                coords_grid[0, 0, 0, :, 3].to(torch.float32).cpu() * satimgsize_factor
            ).tolist()

        gallery_scale_mean = None
        gallery_scale_boundary = None
        if scale_values:
            scale_min = float(min(scale_values))
            scale_max = float(max(scale_values))
            gallery_scale_mean = float(np.mean(scale_values))
            gallery_scale_boundary = [scale_min, scale_max]

        return {
            "row_centers_nrc_range": (
                float(row_values[0]),
                float(row_values[-1]),
            ) if row_values else (None, None),
            "col_centers_nrc_range": (
                float(col_values[0]),
                float(col_values[-1]),
            ) if col_values else (None, None),
            "rot_values": rot_values,
            "scale_values": scale_values,
            "satimgsize_values": satimgsize_values,
            "gallery_scale_mean": gallery_scale_mean,
            "gallery_scale_boundary": gallery_scale_boundary,
            "gallery_has_rot": self._has_effective_rotation(rot_values),
        }

    @staticmethod
    def _is_effectively_integer(value, atol=1e-6):
        return bool(abs(float(value) - round(float(value))) <= float(atol))

    @classmethod
    def _is_integer_stride_4d(cls, stride_4d, atol=1e-6):
        stride_arr = np.asarray(stride_4d, dtype=np.float64).reshape(-1)
        return bool(np.all(np.abs(stride_arr - np.rint(stride_arr)) <= float(atol)))

    @staticmethod
    def _build_axis_positions(n_src, stride):
        n_src = int(n_src)
        stride = float(stride)
        if n_src <= 0:
            raise ValueError("n_src must be > 0.")
        if n_src == 1:
            return np.asarray([0.0], dtype=np.float64)

        last_index = float(n_src - 1)
        positions = np.arange(0.0, last_index + 1e-8, stride, dtype=np.float64)
        if positions.size == 0:
            positions = np.asarray([0.0], dtype=np.float64)
        if abs(float(positions[-1]) - last_index) > 1e-6:
            positions = np.concatenate([positions, np.asarray([last_index], dtype=np.float64)], axis=0)
        return positions

    @staticmethod
    def _interp_axis_linear(axis_values, target_positions):
        axis_values = np.asarray(axis_values, dtype=np.float64).reshape(-1)
        src_positions = np.arange(axis_values.size, dtype=np.float64)
        out = np.interp(target_positions, src_positions, axis_values)
        return torch.as_tensor(out, dtype=torch.float32)

    @staticmethod
    def _interp_axis_log(axis_values, target_positions):
        axis_values = np.asarray(axis_values, dtype=np.float64).reshape(-1)
        axis_values = np.clip(axis_values, 1e-12, None)
        src_positions = np.arange(axis_values.size, dtype=np.float64)
        out = np.exp(np.interp(target_positions, src_positions, np.log(axis_values)))
        return torch.as_tensor(out, dtype=torch.float32)

    @staticmethod
    def _wrap_angle_rad(angle_values):
        angle_values = np.asarray(angle_values, dtype=np.float64)
        return (angle_values + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _interp_axis_rotation_periodic(cls, axis_values, target_positions):
        axis_values = np.asarray(axis_values, dtype=np.float64).reshape(-1)
        if axis_values.size == 1:
            return torch.as_tensor(axis_values, dtype=torch.float32)
        src_positions = np.arange(axis_values.size, dtype=np.float64)
        unwrapped = np.unwrap(axis_values)
        out = np.interp(target_positions, src_positions, unwrapped)
        out = cls._wrap_angle_rad(out)
        return torch.as_tensor(out, dtype=torch.float32)

    def _extract_axis_values(self, coords_gallery, n_bins_4d):
        coords_grid = coords_gallery.reshape(*n_bins_4d, 4)
        return {
            "row": coords_grid[:, 0, 0, 0, 0].to(torch.float32).cpu().numpy(),
            "col": coords_grid[0, :, 0, 0, 1].to(torch.float32).cpu().numpy(),
            "rot": coords_grid[0, 0, :, 0, 2].to(torch.float32).cpu().numpy(),
            "scale": coords_grid[0, 0, 0, :, 3].to(torch.float32).cpu().numpy(),
        }

    @staticmethod
    def _build_coords_from_axis_values(row_values, col_values, rot_values, scale_values):
        row_values = torch.as_tensor(row_values, dtype=torch.float32)
        col_values = torch.as_tensor(col_values, dtype=torch.float32)
        rot_values = torch.as_tensor(rot_values, dtype=torch.float32)
        scale_values = torch.as_tensor(scale_values, dtype=torch.float32)

        row_grid, col_grid, rot_grid, scale_grid = torch.meshgrid(
            row_values,
            col_values,
            rot_values,
            scale_values,
            indexing="ij",
        )
        return torch.stack([row_grid, col_grid, rot_grid, scale_grid], dim=-1).reshape(-1, 4)

    def _populate_downsample_child_meta(self, child_bank, cfg, source_n_bins_4d, child_n_bins_4d, keep_layout_cfg):
        child_bank.meta = dict(self.meta)
        child_bank.meta["source_n_points"] = int(self.coords_gallery.shape[0])
        child_bank.meta["n_points"] = int(child_bank.coords_gallery.shape[0])
        child_bank.meta["downsample_cfg"] = asdict(cfg)
        child_bank.meta["source_n_bins_4d"] = list(source_n_bins_4d)
        child_bank.meta["n_bins_4d"] = [int(v) for v in child_n_bins_4d]
        child_bank.meta["n_rot"] = int(child_n_bins_4d[2])
        child_bank.meta["n_scale"] = int(child_n_bins_4d[3])

        axis_meta = self._summarize_axis_meta_from_coords(child_bank.coords_gallery, child_n_bins_4d)
        child_bank.meta["row_centers_nrc_range"] = axis_meta["row_centers_nrc_range"]
        child_bank.meta["col_centers_nrc_range"] = axis_meta["col_centers_nrc_range"]
        child_bank.meta["rot_values"] = axis_meta["rot_values"]
        child_bank.meta["scale_values"] = axis_meta["scale_values"]
        if axis_meta["satimgsize_values"] is not None:
            child_bank.meta["satimgsize_values"] = axis_meta["satimgsize_values"]
        else:
            child_bank.meta.pop("satimgsize_values", None)
        if axis_meta["gallery_scale_mean"] is not None:
            child_bank.meta["gallery_scale_mean"] = axis_meta["gallery_scale_mean"]
        if axis_meta["gallery_scale_boundary"] is not None:
            child_bank.meta["gallery_scale_boundary"] = axis_meta["gallery_scale_boundary"]
        child_bank.meta["gallery_has_rot"] = axis_meta["gallery_has_rot"]

        if keep_layout_cfg and self.layout_cfg is not None:
            child_bank.layout_cfg = Stage1ReferenceGalleryLayoutConfig(**asdict(self.layout_cfg))
            if child_bank.layout_cfg.n_bins_4d is not None:
                child_bank.layout_cfg.n_bins_4d = tuple(child_bank.meta["n_bins_4d"])
            else:
                child_bank.layout_cfg.n_rot = int(child_n_bins_4d[2])
                child_bank.layout_cfg.n_scale = int(child_n_bins_4d[3])
                child_bank.layout_cfg.rot_values = list(axis_meta["rot_values"]) if axis_meta["rot_values"] else None
            child_bank.meta["layout_cfg"] = asdict(child_bank.layout_cfg)
        else:
            child_bank.layout_cfg = None
            child_bank.meta["layout_cfg"] = None

    def _downsample_by_integer_stride(self, cfg, n_bins_4d):
        stride_nr, stride_nc, stride_rot, stride_scale = [int(round(v)) for v in cfg.stride_4d]

        grid_indices = torch.arange(int(self.coords_gallery.shape[0]), dtype=torch.long).reshape(*n_bins_4d)
        selected_indices = grid_indices[
            ::stride_nr,
            ::stride_nc,
            ::stride_rot,
            ::stride_scale,
        ].reshape(-1)

        child_bank = Stage1ReferenceGalleryBank(
            sat_dataset=self.sat_dataset,
            trainer=self.trainer,
            device=self.device,
        )
        child_bank.coords_gallery = self.coords_gallery[selected_indices].clone()
        child_n_bins_4d = (
            int(len(range(0, n_bins_4d[0], stride_nr))),
            int(len(range(0, n_bins_4d[1], stride_nc))),
            int(len(range(0, n_bins_4d[2], stride_rot))),
            int(len(range(0, n_bins_4d[3], stride_scale))),
        )
        self._populate_downsample_child_meta(
            child_bank=child_bank,
            cfg=cfg,
            source_n_bins_4d=n_bins_4d,
            child_n_bins_4d=child_n_bins_4d,
            keep_layout_cfg=True,
        )

        if self.feats_gallery is not None:
            child_bank.feats_gallery = self.feats_gallery[selected_indices].clone()
            if cfg.build_faiss:
                child_bank.build_faiss_index()

        return child_bank

    def _downsample_by_interp_coords(self, cfg, n_bins_4d):
        if cfg.interp_target == "features":
            raise NotImplementedError("Feature interpolation downsampling is not implemented yet.")

        axis_values = self._extract_axis_values(self.coords_gallery, n_bins_4d)
        row_positions = self._build_axis_positions(n_bins_4d[0], cfg.stride_4d[0])
        col_positions = self._build_axis_positions(n_bins_4d[1], cfg.stride_4d[1])
        rot_positions = self._build_axis_positions(n_bins_4d[2], cfg.stride_4d[2])
        scale_positions = self._build_axis_positions(n_bins_4d[3], cfg.stride_4d[3])

        row_values = self._interp_axis_linear(axis_values["row"], row_positions)
        col_values = self._interp_axis_linear(axis_values["col"], col_positions)
        rot_values = self._interp_axis_rotation_periodic(axis_values["rot"], rot_positions)
        if str(self.meta.get("scale_mode", getattr(self.layout_cfg, "scale_mode", "linear"))).strip().lower() == "log":
            scale_values = self._interp_axis_log(axis_values["scale"], scale_positions)
        else:
            scale_values = self._interp_axis_linear(axis_values["scale"], scale_positions)

        child_bank = Stage1ReferenceGalleryBank(
            sat_dataset=self.sat_dataset,
            trainer=self.trainer,
            device=self.device,
        )
        child_bank.coords_gallery = self._build_coords_from_axis_values(
            row_values=row_values,
            col_values=col_values,
            rot_values=rot_values,
            scale_values=scale_values,
        ).cpu()
        child_n_bins_4d = (
            int(row_values.numel()),
            int(col_values.numel()),
            int(rot_values.numel()),
            int(scale_values.numel()),
        )
        self._populate_downsample_child_meta(
            child_bank=child_bank,
            cfg=cfg,
            source_n_bins_4d=n_bins_4d,
            child_n_bins_4d=child_n_bins_4d,
            keep_layout_cfg=False,
        )
        child_bank.meta["downsample_interp_kind"] = "coords"
        child_bank.feats_gallery = None
        child_bank.faiss_index = None
        return child_bank

    def downsample(self, downsample_cfg):
        if self.coords_gallery is None:
            raise ValueError("No gallery loaded. Build or load a gallery before downsampling.")

        cfg = self._validate_downsample_cfg(downsample_cfg)
        n_bins_4d = self._require_grid_shape()
        if self._is_integer_stride_4d(cfg.stride_4d):
            return self._downsample_by_integer_stride(cfg, n_bins_4d)
        return self._downsample_by_interp_coords(cfg, n_bins_4d)

    # ------------------------------------------------------------------
    # Feature / index helpers
    # ------------------------------------------------------------------
    def _require_trainer(self):
        if self.trainer is None:
            raise ValueError("trainer is required for feature extraction or FAISS index building.")

    def build_features(self, feature_cfg=None):
        self._require_trainer()
        if self.coords_gallery is None:
            raise ValueError("build_coords(...) must be called before build_features(...).")

        cfg = feature_cfg if isinstance(feature_cfg, Stage1ReferenceGalleryFeatureConfig) else (
            Stage1ReferenceGalleryFeatureConfig(**feature_cfg) if feature_cfg is not None else Stage1ReferenceGalleryFeatureConfig()
        )

        feats_gallery_list = []
        gallery_has_rot = bool(self.meta.get("gallery_has_rot", False))
        chunk_starts = range(0, self.coords_gallery.shape[0], cfg.chunk_size_vis)
        if cfg.show_progress:
            chunk_starts = tqdm.tqdm(
                chunk_starts,
                desc=f"[Gallery Features] {self.meta.get('scene_name', 'unknown')}",
                leave=False,
            )
        with torch.no_grad():
            for start in chunk_starts:
                end = min(start + cfg.chunk_size_vis, self.coords_gallery.shape[0])
                coords_chunk = self.coords_gallery[start:end]
                satimgs_refs = self._crop_gallery_images(
                    coords_chunk=coords_chunk,
                    apply_rotation=gallery_has_rot,
                    chunk_size=cfg.chunk_size_vis,
                )
                satimgs_refs = satimgs_refs.to(self.trainer.device)
                feats_ref = self.trainer._get_feats_fm_imgs(satimgs_refs)
                if cfg.normalize_feats:
                    feats_ref = TF.normalize(feats_ref, dim=-1)
                feats_gallery_list.append(feats_ref.detach().cpu())

                # Large galleries should not keep intermediate tensors on GPU.
                del satimgs_refs
                del feats_ref

        self.feats_gallery = torch.cat(feats_gallery_list, dim=0)
        self.meta["feature_cfg"] = asdict(cfg)
        self.meta["feat_dim"] = int(self.feats_gallery.shape[1])
        if cfg.build_faiss:
            self.build_faiss_index()
        return self.feats_gallery

    def build_faiss_index(self):
        if self.feats_gallery is None:
            raise ValueError("build_features(...) must be called before build_faiss_index().")
        import faiss

        feat_dim = int(self.feats_gallery.shape[1])
        self.faiss_index = faiss.IndexFlatL2(feat_dim)
        self.faiss_index.add(self.feats_gallery.numpy())
        self.meta["faiss_index_type"] = "IndexFlatL2"
        return self.faiss_index

    def _crop_gallery_images(self, coords_chunk, apply_rotation, chunk_size):
        if hasattr(self.sat_dataset, "crop_satimg_by_4d_coords_fast"):
            return self.sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_chunk,
                apply_rotation=apply_rotation,
                chunk_size=chunk_size,
            )
        if hasattr(self.sat_dataset, "crop_satimg_by_4d_coords"):
            return self.sat_dataset.crop_satimg_by_4d_coords(
                coords_chunk,
                apply_rotation=apply_rotation,
            )
        raise AttributeError("sat_dataset has neither crop_satimg_by_4d_coords_fast nor crop_satimg_by_4d_coords.")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, save_dir, save_feats=True, save_meta=True):
        if self.coords_gallery is None:
            raise ValueError("No gallery to save. Call build_coords(...) first.")
        os.makedirs(save_dir, exist_ok=True)

        coords_path = os.path.join(save_dir, "coords_gallery.pt")
        torch.save(self.coords_gallery, coords_path)

        if save_feats and self.feats_gallery is not None:
            feats_path = os.path.join(save_dir, "feats_gallery.pt")
            torch.save(self.feats_gallery, feats_path)

        if save_meta:
            meta_path = os.path.join(save_dir, "gallery_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, save_dir, sat_dataset, trainer=None, build_faiss=False):
        bank = cls(sat_dataset=sat_dataset, trainer=trainer)

        coords_path = os.path.join(save_dir, "coords_gallery.pt")
        meta_path = os.path.join(save_dir, "gallery_meta.json")
        feats_path = os.path.join(save_dir, "feats_gallery.pt")

        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"Missing gallery coords file: {coords_path}")
        bank.coords_gallery = torch.load(coords_path, map_location="cpu")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                bank.meta = json.load(f)
            layout_cfg = bank.meta.get("layout_cfg", None)
            if layout_cfg is not None:
                bank.layout_cfg = Stage1ReferenceGalleryLayoutConfig(**layout_cfg)
        else:
            bank.meta = {"n_points": int(bank.coords_gallery.shape[0])}

        if os.path.exists(feats_path):
            bank.feats_gallery = torch.load(feats_path, map_location="cpu")
            if build_faiss:
                bank.build_faiss_index()

        return bank

    def summary(self):
        return {
            "scene_name": self.meta.get("scene_name", None),
            "mode": self.meta.get("mode", None),
            "overlap": self.meta.get("overlap", None),
            "n_points": self.meta.get("n_points", 0),
            "n_bins_4d": self.meta.get("n_bins_4d", None),
            "gallery_shape": self.meta.get("n_bins_4d", None),
            "n_rot": self.meta.get("n_rot", None),
            "n_scale": self.meta.get("n_scale", None),
            "scale_mode": self.meta.get("scale_mode", None),
            "layout_cfg": self.meta.get("layout_cfg", None),
            "crop_size_px": self.meta.get("crop_size_px", None),
            "grid_rows": self.meta.get("grid_rows", None),
            "grid_cols": self.meta.get("grid_cols", None),
            "downsample_cfg": self.meta.get("downsample_cfg", None),
            "has_feats": self.feats_gallery is not None,
            "has_index": self.faiss_index is not None,
        }
