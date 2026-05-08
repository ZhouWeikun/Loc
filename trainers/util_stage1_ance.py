from dataclasses import dataclass

import numpy as np
import torch
import tqdm

from trainer_depends.miners import MultiSceneANCEMiner, SatGalleryProvider, SceneNegMasker
from trainer_depends.utils.util_uav_image_transform import warp_uav_imgs


@dataclass
class Stage1ANCEConfig:
    enabled: bool = False
    backend: str = "faiss"
    metric: str = "l2"
    use_gpu_index: bool = False
    top_k: int = 1024
    n_neg: int = 16
    refresh_epoch: int = 2
    gallery_chunk_size: int = 1024
    overlap: float = 0.5
    rot_rad_resolution: object = np.float32(np.pi / 18.0)
    rot_list: object = None
    consider_scale: bool = False
    ref_wo_rot_var: bool = False
    ref_wo_scale_var: bool = True
    query_rot2uniform: bool = False
    query_scale2uniform: bool = False

    @classmethod
    def from_opt(cls, opt):
        batchsize_sat = int(getattr(opt, "batchsize_sat", 0))
        batchsize_uav = max(1, int(getattr(opt, "batchsize_uav", 1)))
        return cls(
            enabled=bool(getattr(opt, "ance_enabled", False)),
            backend=getattr(opt, "ance_backend", "faiss"),
            metric=getattr(opt, "ance_metric", "l2"),
            use_gpu_index=bool(getattr(opt, "ance_use_gpu_index", False)),
            top_k=int(getattr(opt, "ance_top_k", 1024)),
            n_neg=int(getattr(opt, "ance_n_neg", batchsize_sat // batchsize_uav)),
            refresh_epoch=int(getattr(opt, "ance_refresh_epoch", 2)),
            gallery_chunk_size=int(getattr(opt, "ance_gallery_chunk_size", 1024)),
            overlap=float(getattr(opt, "ance_overlap", 0.5)),
            rot_rad_resolution=getattr(opt, "ance_rot_rad_resolution", np.float32(np.pi / 18.0)),
            rot_list=getattr(opt, "ance_rot_list", None),
            consider_scale=bool(getattr(opt, "ance_consider_scale", False)),
            ref_wo_rot_var=bool(getattr(opt, "ance_ref_wo_rot_var", False)),
            ref_wo_scale_var=bool(getattr(opt, "ance_ref_wo_scale_var", True)),
            query_rot2uniform=bool(getattr(opt, "ance_query_rot2uniform", False)),
            query_scale2uniform=bool(getattr(opt, "ance_query_scale2uniform", False)),
        )


class Stage1ANCEHelper:
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        self.opt = trainer.opt
        self.config = Stage1ANCEConfig.from_opt(self.opt)

        self.enabled = self.config.enabled
        self.miners = None
        self.gallery_info = {}
        self.last_refresh_epoch = None

    def initialize(self):
        if not self.enabled:
            return
        if not self.config.ref_wo_scale_var:
            raise NotImplementedError("ANCE gallery with ref_wo_scale_var=False is not implemented yet.")

        maskers_by_scene = {}
        for scene in self.opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.trainer.sat_datasets[scene_name]
            sigma = self.trainer.coord_normed_sigmas[scene_name]
            sigma_nrc = float(sigma[0].item())
            sigma_rot = float(sigma[2].item())
            sigma_scale_log = float(sigma[3].item())
            gs_factor = float(getattr(self.trainer, "gs_sigma2radius_factor", 2.0))
            radius_nrc = sigma_nrc * sat_dataset.halfimg_radius_nrc * gs_factor
            radius_rot = sigma_rot * gs_factor
            radius_scale_log = sigma_scale_log * gs_factor

            sat_dataset.ance_filter_radius = {
                "radius_nrc": radius_nrc,
                "radius_rot_rad": radius_rot,
                "radius_scale_log": radius_scale_log if self.config.consider_scale else None,
            }
            sat_dataset.ance_neg_radii = sat_dataset.ance_filter_radius

            maskers_by_scene[scene_name] = SceneNegMasker(
                radius_nrc=radius_nrc,
                radius_rot_rad=radius_rot,
                radius_scale_log=radius_scale_log if self.config.consider_scale else None,
            )

        self.miners = MultiSceneANCEMiner(
            backend=self.config.backend,
            use_gpu=self.config.use_gpu_index,
            metric=self.config.metric,
            maskers_by_scene=maskers_by_scene,
        )
        self.gallery_info = {}
        self.last_refresh_epoch = None

    def maybe_refresh(self, epoch):
        if not self.enabled:
            return
        if self.last_refresh_epoch is None or (int(epoch) - int(self.last_refresh_epoch)) >= self.config.refresh_epoch:
            self.refresh_gallery()
            self.last_refresh_epoch = int(epoch)

    def refresh_gallery(self, scene_name=None):
        if not self.enabled:
            return

        scene_names = [scene_name] if scene_name is not None else [s["name"] for s in self.opt.scenes_setting["scenes"]]
        for name in scene_names:
            sat_dataset = self.trainer.sat_datasets[name]
            provider = SatGalleryProvider(
                sat_dataset,
                overlap=self.config.overlap,
                ref_wo_rot_var=self.config.ref_wo_rot_var,
                ref_wo_scale_var=self.config.ref_wo_scale_var,
                rot_rad_resolution=self.config.rot_rad_resolution,
                rot_list=self.config.rot_list,
            )
            coords_gallery = provider.build_coords()
            feats_gallery = self._build_gallery_feats(name, sat_dataset, coords_gallery)
            self.miners.update_scene(name, feats_gallery, coords_gallery)

            n_rot = int(provider.rot_list.numel()) if getattr(provider, "rot_list", None) is not None else 1
            self.gallery_info[name] = {
                "scale": float(provider.ref_scale),
                "rot": float(provider.rot_list[0]) if n_rot == 1 else None,
                "n_rot": n_rot,
                "ref_wo_rot_var": bool(self.config.ref_wo_rot_var),
                "ref_wo_scale_var": bool(self.config.ref_wo_scale_var),
                "rot_rad_resolution": self.config.rot_rad_resolution,
                "rot_list": self.config.rot_list,
                "overlap": float(self.config.overlap),
                "size": int(coords_gallery.shape[0]),
            }
            if self.trainer.logger is not None:
                self.trainer.logger.info(f"[ANCE] refresh gallery {name}: {coords_gallery.shape[0]} pts")

    def prepare_batch(self, scene_name, sat_dataset, uavimgs, coords_uav):
        if not self.enabled:
            raise RuntimeError("ANCE helper is not enabled.")
        if self.miners is None or scene_name not in self.miners.miners:
            raise KeyError(f"Scene '{scene_name}' is not ready in ANCE miners.")

        gallery_info = self.gallery_info.get(scene_name, None)
        gallery_scale = (
            gallery_info["scale"]
            if gallery_info is not None
            else float(sat_dataset.satimgsize_scale_to_ref_m_mean)
        )

        uavimgs_aligned, coords_uav_aligned = self._maybe_align_queries(
            uavimgs=uavimgs,
            coords_uav=coords_uav,
            gallery_scale=gallery_scale,
        )

        with torch.no_grad():
            feats_q_mining = self.trainer._get_feats_fm_imgs(uavimgs_aligned).detach().cpu().numpy()

        coords_uav_neg = self.miners.mine(
            scene_name,
            feats_q_mining,
            coords_uav_aligned.detach().cpu(),
            top_k=self.config.top_k,
            n_neg=self.config.n_neg,
        )
        coords_uav_neg = self._jitter_neg_coords(coords_uav_neg, sat_dataset)
        coords_uav_neg_flat = coords_uav_neg.reshape(-1, coords_uav_neg.shape[-1])

        apply_rot_neg = (
            gallery_info.get("n_rot", 1) > 1
            if gallery_info is not None
            else (not self.config.ref_wo_rot_var)
        )
        satimgs_neg_flat = sat_dataset.crop_satimg_by_4d_coords_fast(
            coords_uav_neg_flat,
            apply_rotation=apply_rot_neg,
            chunk_size=self.config.gallery_chunk_size,
            random_satmap=True,
        )

        return {
            "uavimgs": uavimgs_aligned,
            "coords_uav": coords_uav_aligned,
            "satimgs_neg_flat": satimgs_neg_flat.to(self.device),
            "coords_uav_neg_flat": coords_uav_neg_flat.to(self.device),
        }

    def _build_gallery_feats(self, scene_name, sat_dataset, coords_gallery):
        feats_gallery_chunks = []
        apply_rot_gallery = not self.config.ref_wo_rot_var
        chunk_iter = tqdm.tqdm(
            range(0, coords_gallery.shape[0], self.config.gallery_chunk_size),
            desc=f"[ANCE] gallery {scene_name}",
            leave=False,
        )
        with torch.inference_mode():
            for start in chunk_iter:
                end = min(start + self.config.gallery_chunk_size, coords_gallery.shape[0])
                coords_chunk = coords_gallery[start:end]
                satimgs = sat_dataset.crop_satimg_by_4d_coords_fast(
                    coords_chunk,
                    apply_rotation=apply_rot_gallery,
                    chunk_size=self.config.gallery_chunk_size,
                    random_satmap=True,
                )
                satimgs = satimgs.to(self.device)
                if self.opt.autocast:
                    with torch.cuda.amp.autocast():
                        feats = self.trainer._get_feats_fm_imgs(satimgs)
                else:
                    feats = self.trainer._get_feats_fm_imgs(satimgs)
                feats_gallery_chunks.append(feats.detach().cpu())
                del satimgs, feats
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return torch.cat(feats_gallery_chunks, dim=0).numpy()

    def _maybe_align_queries(self, uavimgs, coords_uav, gallery_scale):
        rot_align = -coords_uav[:, 2] if self.config.query_rot2uniform else None
        scale_f = None
        if self.config.query_scale2uniform:
            scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)

        if not self.config.query_rot2uniform and not self.config.query_scale2uniform:
            return uavimgs, coords_uav

        uavimgs = warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)
        coords_uav = coords_uav.clone()
        if self.config.query_rot2uniform:
            coords_uav[:, 2] = 0
        if self.config.query_scale2uniform:
            coords_uav[:, 3] = gallery_scale
        return uavimgs, coords_uav

    def _jitter_neg_coords(self, coords, sat_dataset):
        if coords is None:
            return coords

        radii = getattr(sat_dataset, "ance_filter_radius", None) or getattr(sat_dataset, "ance_neg_radii", None)
        if not radii:
            return coords

        coords_t = coords if torch.is_tensor(coords) else torch.as_tensor(coords, dtype=torch.float32)
        coords_t = coords_t.clone()

        radius_nrc = float(radii.get("radius_nrc", 0.0) or 0.0)
        radius_rot = float(radii.get("radius_rot_rad", 0.0) or 0.0)
        radius_scale_log = radii.get("radius_scale_log", None)

        if radius_nrc > 0 and coords_t.shape[-1] >= 2:
            noise = (torch.rand_like(coords_t[..., :2]) * 2.0 - 1.0) * radius_nrc
            coords_t[..., :2] = coords_t[..., :2] + noise
            nr_min, nr_max = sat_dataset.nr2sample_range
            nc_min, nc_max = sat_dataset.nc2sample_range
            coords_t[..., 0] = coords_t[..., 0].clamp(min=float(nr_min), max=float(nr_max))
            coords_t[..., 1] = coords_t[..., 1].clamp(min=float(nc_min), max=float(nc_max))

        if radius_rot > 0 and coords_t.shape[-1] >= 3:
            noise = (torch.rand_like(coords_t[..., 2]) * 2.0 - 1.0) * radius_rot
            coords_t[..., 2] = coords_t[..., 2] + noise
            coords_t[..., 2] = (coords_t[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi

        if radius_scale_log is not None and coords_t.shape[-1] >= 4:
            radius_scale_log = float(radius_scale_log)
            noise = (torch.rand_like(coords_t[..., 3]) * 2.0 - 1.0) * radius_scale_log
            coords_t[..., 3] = coords_t[..., 3] * (1.0 + noise)
            s_min, s_max = sat_dataset.satimgsize_scale_to_ref_m_boundary
            coords_t[..., 3] = coords_t[..., 3].clamp(min=float(s_min), max=float(s_max))

        return coords_t
