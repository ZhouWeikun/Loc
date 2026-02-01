import numpy as np
import torch


class SatGalleryProvider:
    def __init__(self, sat_dataset, overlap=0.5,
                 ref_wo_rot_var=True, ref_wo_scale_var=True,
                 rot_rad_resolution=None, rot_list=None, scale_list=None):
        self.sat_dataset = sat_dataset
        self.overlap = overlap
        self.grid_hw = None

        if not ref_wo_scale_var:
            raise NotImplementedError("ref_wo_scale_var=False is not implemented in SatGalleryProvider.")

        if scale_list is not None and len(scale_list) > 1:
            raise NotImplementedError("multi-scale gallery is not implemented yet")

        self.ref_wo_rot_var = bool(ref_wo_rot_var)
        self.ref_wo_scale_var = bool(ref_wo_scale_var)

        # scale handling (single scale only)
        self.ref_scale = float(sat_dataset.satimgsize_scale_to_ref_m_mean)
        self._use_mean_crop = True

        # rot handling
        self.rot_list = None
        if self.ref_wo_rot_var:
            if rot_list is not None and len(rot_list) > 1:
                raise ValueError("ref_wo_rot_var=True expects no rot_list or a single rot.")
            if rot_rad_resolution is not None:
                raise ValueError("ref_wo_rot_var=True does not accept rot_rad_resolution.")
            self.rot_list = torch.tensor([0.0], dtype=torch.float32)
        else:
            if rot_list is None:
                if rot_rad_resolution is None:
                    raise ValueError("ref_wo_rot_var=False requires rot_list or rot_rad_resolution.")
                rot_rad_resolution = float(rot_rad_resolution)
                if rot_rad_resolution <= 0 or rot_rad_resolution > 2 * np.pi:
                    raise ValueError("rot_rad_resolution must be in (0, 2*pi].")
                self.rot_list = torch.arange(-np.pi, np.pi, rot_rad_resolution, dtype=torch.float32)
            else:
                rot_list_arr = np.asarray(rot_list, dtype=np.float32)
                if rot_list_arr.ndim != 1 or rot_list_arr.size == 0:
                    raise ValueError("rot_list must be a non-empty 1D list/array.")
                self.rot_list = torch.from_numpy(rot_list_arr)

    def build_coords(self):
        if self._use_mean_crop:
            satimgsize2crop = float(self.sat_dataset.satimgsize2crop_mean)
        else:
            satimgsize2crop = self.ref_scale * self.sat_dataset.scale_ref_m / self.sat_dataset.geo_res_m
            satimgsize2crop = float(np.clip(
                satimgsize2crop,
                self.sat_dataset.satimgsize2crop_boundary[0],
                self.sat_dataset.satimgsize2crop_boundary[1]
            ))

        nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
            size2clip=satimgsize2crop,
            overlap=self.overlap,
            only_nrcs=True
        )
        self.grid_hw = (nrcs_gallery.shape[0], nrcs_gallery.shape[1])
        nrcs_flat = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
        if self.rot_list is None or self.rot_list.numel() == 1:
            rot_gallery = torch.full((nrcs_flat.shape[0], 1), float(self.rot_list[0]), dtype=torch.float32)
            scale_gallery = torch.full((nrcs_flat.shape[0], 1), self.ref_scale, dtype=torch.float32)
            coords_gallery = torch.cat([nrcs_flat, rot_gallery, scale_gallery], dim=-1)
        else:
            n_rot = self.rot_list.numel()
            nrcs_rep = nrcs_flat[:, None, :].repeat(1, n_rot, 1).reshape(-1, 2)
            rot_rep = self.rot_list[None, :, None].repeat(nrcs_flat.shape[0], 1, 1).reshape(-1, 1)
            scale_rep = torch.full((nrcs_rep.shape[0], 1), self.ref_scale, dtype=torch.float32)
            coords_gallery = torch.cat([nrcs_rep, rot_rep, scale_rep], dim=-1)

        return coords_gallery
