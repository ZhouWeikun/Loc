import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tqdm


@dataclass
class Stage2ReferenceGalleryLayoutConfig:
    mode: str = "rc_rot_scale"
    n_bins_4d: object = None
    overlap: float = 0.5
    fixed_rot: float = 0.0
    fixed_scale: object = None
    delta_rot_deg: float = 10.0
    n_scales: int = 1
    scale_mode: str = "linear"


@dataclass
class Stage2ReferenceGalleryFeatureConfig:
    chunk_size_coords: int = 512
    normalize_feats: bool = True
    build_faiss: bool = True
    show_progress: bool = True


class Stage2ReferenceGalleryBank:
    """
    Stage 2 gallery manager for Grid HashFit features.

    Responsibilities:
    - build gallery coords for rc / rc+rot / rc+scale / rc+rot+scale layouts
    - extract Stage 2 features from coords via the trainer
    - optionally build a FAISS index
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

    @staticmethod
    def _normalize_mode(mode):
        mode = str(mode).strip().lower()
        valid_modes = ("rc", "rc_rot", "rc_scale", "rc_rot_scale", "n_bins_4d")
        if mode not in valid_modes:
            raise ValueError(f"layout mode must be one of {valid_modes}, got {mode}")
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

    @staticmethod
    def _has_effective_rotation(rot_values, atol=1e-6):
        rot_arr = np.asarray(rot_values, dtype=np.float32).reshape(-1)
        if rot_arr.size == 0:
            return False
        return bool(rot_arr.size > 1 or np.any(np.abs(rot_arr) > float(atol)))

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

    def _resolve_fixed_scale(self, fixed_scale):
        if fixed_scale is None:
            return float(self._get_scale_mean())
        fixed_scale = float(fixed_scale)
        if fixed_scale <= 0:
            raise ValueError("fixed_scale must be > 0.")
        return fixed_scale

    @staticmethod
    def _validate_delta_rot(delta_rot_deg):
        delta_rot_deg = float(delta_rot_deg)
        if delta_rot_deg <= 0:
            raise ValueError("delta_rot_deg must be > 0.")
        n_rot = int(round(360.0 / delta_rot_deg))
        if n_rot <= 0 or abs(n_rot * delta_rot_deg - 360.0) > 1e-4:
            raise ValueError("delta_rot_deg must evenly divide 360 degrees.")
        return delta_rot_deg

    def _validate_layout_cfg(self, layout_cfg):
        cfg = layout_cfg if isinstance(layout_cfg, Stage2ReferenceGalleryLayoutConfig) else (
            Stage2ReferenceGalleryLayoutConfig(**layout_cfg)
        )
        cfg.mode = self._normalize_mode(cfg.mode)
        cfg.scale_mode = self._normalize_scale_mode(cfg.scale_mode)

        if cfg.mode == "n_bins_4d":
            n_bins = np.asarray(cfg.n_bins_4d, dtype=np.int32).reshape(-1)
            if n_bins.size != 4 or (n_bins <= 0).any():
                raise ValueError("n_bins_4d must be length-4 with positive entries.")
            cfg.n_bins_4d = tuple(int(v) for v in n_bins.tolist())
            cfg.fixed_rot = float(cfg.fixed_rot)
            cfg.fixed_scale = None if cfg.fixed_scale is None else float(cfg.fixed_scale)
            cfg.delta_rot_deg = float(cfg.delta_rot_deg)
            cfg.n_scales = int(cfg.n_bins_4d[3])
            return cfg

        if not (0 <= float(cfg.overlap) < 1):
            raise ValueError("overlap must be in [0, 1).")

        cfg.fixed_rot = float(cfg.fixed_rot)
        cfg.fixed_scale = None if cfg.fixed_scale is None else float(cfg.fixed_scale)
        if cfg.fixed_scale is not None and cfg.fixed_scale <= 0:
            raise ValueError("fixed_scale must be > 0 when provided.")

        if cfg.mode in ("rc_rot", "rc_rot_scale"):
            cfg.delta_rot_deg = self._validate_delta_rot(cfg.delta_rot_deg)
        else:
            cfg.delta_rot_deg = float(cfg.delta_rot_deg)

        cfg.n_scales = int(cfg.n_scales)
        if cfg.mode in ("rc_scale", "rc_rot_scale") and cfg.n_scales <= 0:
            raise ValueError("n_scales must be > 0 for scale-aware layouts.")
        if cfg.mode in ("rc", "rc_rot") and cfg.n_scales <= 0:
            cfg.n_scales = 1
        return cfg

    @staticmethod
    def _build_rot_values(delta_rot_deg):
        n_rot = int(round(360.0 / float(delta_rot_deg)))
        rot_values_deg = -180.0 + torch.arange(n_rot, dtype=torch.float32) * float(delta_rot_deg)
        return torch.deg2rad(rot_values_deg)

    @staticmethod
    def _build_uniform_axis_values(axis_min, axis_max, n_bins):
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError("n_bins must be > 0.")
        axis_min = float(axis_min)
        axis_max = float(axis_max)
        if axis_max <= axis_min:
            raise ValueError(f"axis_max must be > axis_min, got ({axis_min}, {axis_max})")
        bin_size = (axis_max - axis_min) / float(n_bins)
        return axis_min + (torch.arange(n_bins, dtype=torch.float32) + 0.5) * bin_size

    @staticmethod
    def _build_uniform_rot_values(n_rot):
        n_rot = int(n_rot)
        if n_rot <= 0:
            raise ValueError("n_rot must be > 0.")
        centers = (torch.arange(n_rot, dtype=torch.float32) + 0.5) * (2.0 * np.pi / float(n_rot)) - np.pi
        return ((centers + np.pi) % (2.0 * np.pi)) - np.pi

    def _build_scale_values(self, n_scales, scale_mode):
        scale_values, satimgsize_values = self.sat_dataset.mk_sacle_levels(
            n_level=int(n_scales),
            scale_mode=scale_mode,
        )
        return scale_values.to(torch.float32), satimgsize_values.to(torch.float32)

    def _build_rc_grid(self, satimgsize2crop, overlap):
        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=float(satimgsize2crop),
            overlap=float(overlap),
            only_nrcs=True,
        )
        grid_shape = tuple(int(v) for v in nrcs_gallery.shape[:2])
        nrcs_flat = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
        return nrcs_flat, grid_shape

    @staticmethod
    def _all_grid_shapes_same(grid_shapes):
        if not grid_shapes:
            return False
        first_shape = tuple(int(v) for v in grid_shapes[0])
        return all(tuple(int(v) for v in shape) == first_shape for shape in grid_shapes)

    def _build_shared_overlap_rc_grid(self, overlap):
        # Use one reference crop size for overlap-mode layouts so rc is independent of the scale axis.
        ref_scale = float(self._get_scale_mean())
        ref_crop_px = ref_scale * float(self.sat_dataset.scale_ref_m) / float(self.sat_dataset.geo_res_m)
        nrcs_flat, grid_shape = self._build_rc_grid(satimgsize2crop=ref_crop_px, overlap=overlap)
        row_values = nrcs_flat[:, 0].reshape(grid_shape[0], grid_shape[1])[:, 0]
        col_values = nrcs_flat[:, 1].reshape(grid_shape[0], grid_shape[1])[0, :]
        meta = {
            "shared_rc_grid": True,
            "rc_grid_reference_scale": ref_scale,
            "rc_grid_reference_crop_px": float(ref_crop_px),
            "row_centers_nrc_range": (
                float(row_values[0].item()),
                float(row_values[-1].item()),
            ),
            "col_centers_nrc_range": (
                float(col_values[0].item()),
                float(col_values[-1].item()),
            ),
        }
        return nrcs_flat, grid_shape, meta

    def _build_coords_from_n_bins(self, cfg):
        n_nr, n_nc, n_rot, n_scale = [int(v) for v in cfg.n_bins_4d]
        nr_min, nr_max = self.sat_dataset.nr2sample_range
        nc_min, nc_max = self.sat_dataset.nc2sample_range

        row_values = self._build_uniform_axis_values(nr_min, nr_max, n_nr)
        col_values = self._build_uniform_axis_values(nc_min, nc_max, n_nc)
        rot_values = self._build_uniform_rot_values(n_rot)
        scale_values, satimgsize_values = self._build_scale_values(n_scale, cfg.scale_mode)

        rr, cc, rot_grid, scale_grid = torch.meshgrid(
            row_values,
            col_values,
            rot_values,
            scale_values.to(torch.float32),
            indexing="ij",
        )
        coords_gallery = torch.stack([rr, cc, rot_grid, scale_grid], dim=-1).reshape(-1, 4)

        fixed_rot = float(rot_values[0].item()) if n_rot == 1 else None
        fixed_scale = float(scale_values[0].item()) if n_scale == 1 else None
        delta_rot_deg = None if n_rot <= 1 else float(360.0 / n_rot)

        meta = {
            "fixed_scale": fixed_scale,
            "fixed_rot": fixed_rot,
            "delta_rot_deg": delta_rot_deg,
            "rot_values": rot_values.tolist(),
            "scale_values": scale_values.tolist(),
            "satimgsize_values": satimgsize_values.tolist(),
            "crop_sizes_px": satimgsize_values.tolist(),
            "grid_shapes_rc": [[int(n_nr), int(n_nc)] for _ in range(n_scale)],
            "gallery_shape": [int(n_nr), int(n_nc), int(n_rot), int(n_scale)],
            "n_bins_4d": [int(n_nr), int(n_nc), int(n_rot), int(n_scale)],
            "n_rot": int(n_rot),
            "n_scale": int(n_scale),
            "is_regular_grid_4d": True,
            "row_centers_nrc_range": (
                float(row_values[0].item()),
                float(row_values[-1].item()),
            ),
            "col_centers_nrc_range": (
                float(col_values[0].item()),
                float(col_values[-1].item()),
            ),
        }
        return coords_gallery, meta

    def _build_rc_coords(self, cfg):
        fixed_scale = self._resolve_fixed_scale(cfg.fixed_scale)
        satimgsize2crop = fixed_scale * float(self.sat_dataset.scale_ref_m) / float(self.sat_dataset.geo_res_m)
        nrcs_flat, grid_shape = self._build_rc_grid(satimgsize2crop=satimgsize2crop, overlap=cfg.overlap)

        coords_gallery = torch.cat(
            [
                nrcs_flat,
                torch.full((nrcs_flat.shape[0], 1), float(cfg.fixed_rot), dtype=torch.float32),
                torch.full((nrcs_flat.shape[0], 1), float(fixed_scale), dtype=torch.float32),
            ],
            dim=-1,
        )

        meta = {
            "fixed_scale": float(fixed_scale),
            "fixed_rot": float(cfg.fixed_rot),
            "rot_values": [float(cfg.fixed_rot)],
            "scale_values": [float(fixed_scale)],
            "crop_sizes_px": [float(satimgsize2crop)],
            "grid_shapes_rc": [list(grid_shape)],
            "gallery_shape": [int(grid_shape[0]), int(grid_shape[1]), 1, 1],
            "n_bins_4d": [int(grid_shape[0]), int(grid_shape[1]), 1, 1],
            "n_rot": 1,
            "n_scale": 1,
            "is_regular_grid_4d": True,
        }
        return coords_gallery, meta

    def _build_rc_rot_coords(self, cfg):
        fixed_scale = self._resolve_fixed_scale(cfg.fixed_scale)
        satimgsize2crop = fixed_scale * float(self.sat_dataset.scale_ref_m) / float(self.sat_dataset.geo_res_m)
        nrcs_flat, grid_shape = self._build_rc_grid(satimgsize2crop=satimgsize2crop, overlap=cfg.overlap)
        rot_values = self._build_rot_values(cfg.delta_rot_deg)

        nrcs_expanded = nrcs_flat.unsqueeze(1).expand(-1, rot_values.numel(), -1)
        rot_expanded = rot_values[None, :, None].expand(nrcs_flat.shape[0], -1, 1)
        scale_expanded = torch.full(
            (nrcs_flat.shape[0], rot_values.numel(), 1),
            float(fixed_scale),
            dtype=torch.float32,
        )
        coords_gallery = torch.cat([nrcs_expanded, rot_expanded, scale_expanded], dim=-1).reshape(-1, 4)

        meta = {
            "fixed_scale": float(fixed_scale),
            "fixed_rot": None,
            "delta_rot_deg": float(cfg.delta_rot_deg),
            "rot_values": rot_values.tolist(),
            "scale_values": [float(fixed_scale)],
            "crop_sizes_px": [float(satimgsize2crop)],
            "grid_shapes_rc": [list(grid_shape)],
            "gallery_shape": [int(grid_shape[0]), int(grid_shape[1]), int(rot_values.numel()), 1],
            "n_bins_4d": [int(grid_shape[0]), int(grid_shape[1]), int(rot_values.numel()), 1],
            "n_rot": int(rot_values.numel()),
            "n_scale": 1,
            "is_regular_grid_4d": True,
        }
        return coords_gallery, meta

    def _build_rc_scale_coords(self, cfg):
        scale_values, satimgsize_values = self._build_scale_values(cfg.n_scales, cfg.scale_mode)
        nrcs_flat, grid_shape, shared_rc_meta = self._build_shared_overlap_rc_grid(cfg.overlap)
        n_pos = int(nrcs_flat.shape[0])
        n_scale = int(scale_values.numel())

        nrc_rep = nrcs_flat[:, None, :].expand(-1, n_scale, -1)
        rot_rep = torch.full((n_pos, n_scale, 1), float(cfg.fixed_rot), dtype=torch.float32)
        scale_rep = scale_values[None, :, None].expand(n_pos, -1, 1)
        coords_gallery = torch.cat([nrc_rep, rot_rep, scale_rep], dim=-1).reshape(-1, 4)

        grid_shapes = [list(grid_shape) for _ in range(n_scale)]
        n_bins_4d = [int(grid_shape[0]), int(grid_shape[1]), 1, int(n_scale)]
        gallery_shape = list(n_bins_4d)

        meta = {
            "fixed_scale": None,
            "fixed_rot": float(cfg.fixed_rot),
            "rot_values": [float(cfg.fixed_rot)],
            "scale_values": scale_values.tolist(),
            "satimgsize_values": satimgsize_values.tolist(),
            "crop_sizes_px": satimgsize_values.tolist(),
            "grid_shapes_rc": grid_shapes,
            "gallery_shape": gallery_shape,
            "n_bins_4d": n_bins_4d,
            "n_rot": 1,
            "n_scale": int(n_scale),
            "is_regular_grid_4d": True,
            **shared_rc_meta,
        }
        return coords_gallery, meta

    def _build_rc_rot_scale_coords(self, cfg):
        scale_values, satimgsize_values = self._build_scale_values(cfg.n_scales, cfg.scale_mode)
        rot_values = self._build_rot_values(cfg.delta_rot_deg)
        nrcs_flat, grid_shape, shared_rc_meta = self._build_shared_overlap_rc_grid(cfg.overlap)
        n_pos = int(nrcs_flat.shape[0])
        n_rot = int(rot_values.numel())
        n_scale = int(scale_values.numel())

        nrc_rep = nrcs_flat[:, None, None, :].expand(-1, n_rot, n_scale, -1)
        rot_rep = rot_values[None, :, None, None].expand(n_pos, -1, n_scale, 1)
        scale_rep = scale_values[None, None, :, None].expand(n_pos, n_rot, -1, 1)
        coords_gallery = torch.cat([nrc_rep, rot_rep, scale_rep], dim=-1).reshape(-1, 4)

        grid_shapes = [list(grid_shape) for _ in range(n_scale)]
        n_bins_4d = [int(grid_shape[0]), int(grid_shape[1]), int(n_rot), int(n_scale)]
        gallery_shape = list(n_bins_4d)

        meta = {
            "fixed_scale": None,
            "fixed_rot": None,
            "delta_rot_deg": float(cfg.delta_rot_deg),
            "rot_values": rot_values.tolist(),
            "scale_values": scale_values.tolist(),
            "satimgsize_values": satimgsize_values.tolist(),
            "crop_sizes_px": satimgsize_values.tolist(),
            "grid_shapes_rc": grid_shapes,
            "gallery_shape": gallery_shape,
            "n_bins_4d": n_bins_4d,
            "n_rot": int(n_rot),
            "n_scale": int(n_scale),
            "is_regular_grid_4d": True,
            **shared_rc_meta,
        }
        return coords_gallery, meta

    def build_coords(self, layout_cfg):
        cfg = self._validate_layout_cfg(layout_cfg)
        self.layout_cfg = cfg

        if cfg.mode == "n_bins_4d":
            coords_gallery, meta = self._build_coords_from_n_bins(cfg)
        elif cfg.mode == "rc":
            coords_gallery, meta = self._build_rc_coords(cfg)
        elif cfg.mode == "rc_rot":
            coords_gallery, meta = self._build_rc_rot_coords(cfg)
        elif cfg.mode == "rc_scale":
            coords_gallery, meta = self._build_rc_scale_coords(cfg)
        else:
            coords_gallery, meta = self._build_rc_rot_scale_coords(cfg)

        scale_values = meta.get("scale_values", [])
        gallery_scale_boundary = None
        gallery_scale_mean = None
        if scale_values:
            gallery_scale_boundary = [float(min(scale_values)), float(max(scale_values))]
            gallery_scale_mean = float(np.mean(scale_values))

        self.coords_gallery = coords_gallery.to(torch.float32).cpu()
        self.feats_gallery = None
        self.faiss_index = None
        self.meta = {
            "layout_cfg": asdict(cfg),
            "scene_name": getattr(self.sat_dataset, "name", None),
            "mode": cfg.mode,
            "overlap": None if cfg.mode == "n_bins_4d" else float(cfg.overlap),
            "scale_mode": cfg.scale_mode,
            "n_points": int(self.coords_gallery.shape[0]),
            "gallery_has_rot": self._has_effective_rotation(meta.get("rot_values", [])),
            "gallery_scale_mean": gallery_scale_mean,
            "gallery_scale_boundary": gallery_scale_boundary,
            **{k: self._to_python(v) for k, v in meta.items()},
        }
        return self.coords_gallery

    def _require_trainer(self):
        if self.trainer is None:
            raise ValueError("trainer is required for Stage 2 feature extraction or FAISS index building.")

    def build_features(self, feature_cfg=None):
        self._require_trainer()
        if self.coords_gallery is None:
            raise ValueError("build_coords(...) must be called before build_features(...).")

        cfg = feature_cfg if isinstance(feature_cfg, Stage2ReferenceGalleryFeatureConfig) else (
            Stage2ReferenceGalleryFeatureConfig(**feature_cfg) if feature_cfg is not None else Stage2ReferenceGalleryFeatureConfig()
        )

        chunk_starts = range(0, self.coords_gallery.shape[0], int(cfg.chunk_size_coords))
        if cfg.show_progress:
            chunk_starts = tqdm.tqdm(
                chunk_starts,
                desc=f"[Stage2 Gallery Features] {self.meta.get('scene_name', 'unknown')}",
                leave=False,
            )

        feats_gallery_list = []
        with torch.no_grad():
            for start in chunk_starts:
                end = min(start + int(cfg.chunk_size_coords), self.coords_gallery.shape[0])
                coords_chunk = self.coords_gallery[start:end]
                feats_chunk = self.trainer._extract_stage2_feats_from_coords_chunk(
                    coords_chunk,
                    normalize=bool(cfg.normalize_feats),
                )
                feats_gallery_list.append(feats_chunk.detach().cpu())

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
                bank.layout_cfg = Stage2ReferenceGalleryLayoutConfig(**layout_cfg)
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
            "gallery_shape": self.meta.get("gallery_shape", None),
            "n_rot": self.meta.get("n_rot", None),
            "n_scale": self.meta.get("n_scale", None),
            "scale_mode": self.meta.get("scale_mode", None),
            "layout_cfg": self.meta.get("layout_cfg", None),
            "fixed_scale": self.meta.get("fixed_scale", None),
            "fixed_rot": self.meta.get("fixed_rot", None),
            "delta_rot_deg": self.meta.get("delta_rot_deg", None),
            "crop_sizes_px": self.meta.get("crop_sizes_px", None),
            "grid_shapes_rc": self.meta.get("grid_shapes_rc", None),
            "shared_rc_grid": self.meta.get("shared_rc_grid", None),
            "rc_grid_reference_scale": self.meta.get("rc_grid_reference_scale", None),
            "rc_grid_reference_crop_px": self.meta.get("rc_grid_reference_crop_px", None),
            "has_feats": self.feats_gallery is not None,
            "has_index": self.faiss_index is not None,
        }
