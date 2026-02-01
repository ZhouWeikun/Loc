import os
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# from torchvision.transforms import InterpolationMode
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关闭限制
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
import  torchvision.transforms as T
#
from trainer_depends.utils.util_mk_data_transform import mk_pil_transform,mk_tensor_transform
from trainer_depends.utils.util_data_transform_with_params import mk_pil_transform_with_params, apply_augment_to_coords
from trainer_depends.utils.util_batch_rotation import batch_rotate_images_per_sample


def debug_plot_scale_hist(
        satimgsize_scale_to_ref_m,
        aff2d_corrected_mask,
        lower_bound,  # 建议去掉默认值，强制传入，避免误用
        upper_bound,  # 建议去掉默认值
        lower_label="lower",  # 新增参数：下界标签
        upper_label="upper",  # 新增参数：上界标签
        title=None,
        save_path=None,
):
    import matplotlib.pyplot as plt
    import numpy as np  # 确保导入numpy

    plt.figure(figsize=(6, 4))

    # 绘制直方图
    plt.hist(satimgsize_scale_to_ref_m, bins=80, alpha=0.6, label="all")
    plt.hist(satimgsize_scale_to_ref_m[aff2d_corrected_mask], bins=80, alpha=0.6, label="aff2d_corrected")

    # 绘制下界线 (动态 Label + 数值)
    if lower_bound is not None:
        plt.axvline(
            lower_bound,
            color="r",
            linestyle="--",
            linewidth=1,
            label=f"{lower_label} ({lower_bound:.2f})"  # 关键修改：显示标签名和数值
        )

    # 绘制上界线 (动态 Label + 数值)
    if upper_bound is not None:
        plt.axvline(
            upper_bound,
            color="g",
            linestyle="--",
            linewidth=1,
            label=f"{upper_label} ({upper_bound:.2f})"  # 关键修改：显示标签名和数值
        )

    plt.title(title or "satimgsize_scale_to_ref_m")
    plt.xlabel("scale (ref_m)")
    plt.ylabel("count")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def get_valid_range_mad(data, thresh=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # 为了与标准差尺度一致，通常乘以常数 1.4826 (如果假设近似正态)
    # 但在单纯去噪时，直接用 mad * thresh 也可以，或者如下标准化：
    modified_z_score = 0.6745 * (data - median) / mad

    mask = np.abs(modified_z_score) <= thresh
    # 返回基于mask后的数据的min/max，或者直接返回理论bound
    valid_data = data[mask]
    return valid_data.min(), valid_data.max()

def get_valid_range_iqr(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def _compute_scale_filter_from_df(df, geo_res_m, scale_ref_m=None):
    uav_h_cover_m = np.array(df['h_cover_m'])
    aff2d_corrected_mask = np.array(df['aff2d_corrected'])
    uav_h_cover_m_corrected = uav_h_cover_m[aff2d_corrected_mask]
    scale_ref_m = np.array((uav_h_cover_m_corrected).mean()//10+1) * 10 if scale_ref_m is None else scale_ref_m
    satimgsize_scale_to_ref_m_corrected = np.array(uav_h_cover_m_corrected) / scale_ref_m
    satimgsize_scale_to_ref_m = np.array(uav_h_cover_m / geo_res_m) * geo_res_m / scale_ref_m
    lower_bound, upper_bound = get_valid_range_mad(satimgsize_scale_to_ref_m_corrected)
    scale_mask = (satimgsize_scale_to_ref_m > lower_bound) * (satimgsize_scale_to_ref_m < upper_bound) * aff2d_corrected_mask
    return {
        "uav_h_cover_m": uav_h_cover_m,
        "aff2d_corrected_mask": aff2d_corrected_mask,
        "scale_ref_m": scale_ref_m,
        "satimgsize_scale_to_ref_m_corrected": satimgsize_scale_to_ref_m_corrected,
        "satimgsize_scale_to_ref_m": satimgsize_scale_to_ref_m,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "scale_mask": scale_mask,
    }

class SatDataset(object):
    def __init__(self,
                 p_satinfo_json,
                 imgsize2net = 224,
                 device='cpu',
                 scale_ref_m=None,
                 p_uav_geocsv=None,
                 return_pair=False,
                 name=None,
                 **kwargs,
                 ):
        # config device for cached tensors
        self.device = torch.device(device) if isinstance(device, str) else device
        if 'cuda' in str(self.device) and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        # read corresponding mate info
        self.p_satinfo_json = p_satinfo_json
        self.name = name if name is not None else os.path.splitext(os.path.basename(p_satinfo_json))[0]
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict= sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        self.epsg_code = int(self.satinfo_dict['epsg_code'])
        self.geo_res_m = 0.5 * (abs(self.satinfo_dict['x_resolution_m'])+abs(self.satinfo_dict['y_resolution_m']))

        # read the tifs
        self.satmaps,self.satmaps_tensor = [],[]
        for i,filepath in enumerate(self.satinfo_dict['filepaths']):
            satmap = Image.open(filepath)
            self.satmaps.append(satmap)  # a list of PILs with shape=hwc
            sat2tensor_transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.satinfo_dict['means_normalized'][i],
                                     std=self.satinfo_dict['stds_normalized'][i]),
            ]
            sat2tensor_transform = transforms.Compose(sat2tensor_transform)
            self.satmaps_tensor.append(sat2tensor_transform(satmap).to(self.device))

        # config the attrs about tifs
        self.n_satmaps = len(self.satmaps)
        self.satmap_h = self.satmaps[0].height
        self.satmap_w = self.satmaps[0].width
        self.satmap_hw_max = np.max([self.satmap_h, self.satmap_w])

        # config the transforms
        self.imgsize2net = imgsize2net
        self.sat_transform_train, self.sat_rotater = mk_tensor_transform(imgsize2net,rand_rot=True)
        self.scale_transform = T.Compose([transforms.Resize(self.imgsize2net,antialias=False)])

        # for defining the scale to sample
        self.p_uav_geocsv = p_uav_geocsv
        df = pd.read_csv(p_uav_geocsv)
        scale_info = _compute_scale_filter_from_df(df, self.geo_res_m, scale_ref_m)
        uav_h_cover_m = scale_info["uav_h_cover_m"]
        # aff2d_corrected_mask = scale_info["aff2d_corrected_mask"]
        self.scale_ref_m = scale_info["scale_ref_m"]
        # satimgsize_scale_to_ref_m = scale_info["satimgsize_scale_to_ref_m"]
        # lower_bound = scale_info["lower_bound"]
        # upper_bound = scale_info["upper_bound"]
        scale_mask = scale_info["scale_mask"]
        # without filtering:
        # scale_mask = aff2d_corrected_mask
        #
        # debug: visualize scale distribution (optional)
        # if kwargs.get("debug_scale_hist", False):
        # _debug_plot_scale_hist(
        #     satimgsize_scale_to_ref_m,
        #     aff2d_corrected_mask,
        #     lower_bound,
        #     upper_bound,
        #     title=f"satimgsize_scale_to_ref_m ({self.name})",
        #     save_path=kwargs.get("debug_scale_hist_path", None),
        # )

        # config the satimgsize2crop
        self.satimgsize_correspond2uav_list = uav_h_cover_m[scale_mask] / self.geo_res_m
        self.satimgsize2crop_mean = self.satimgsize_correspond2uav_list.mean()
        self.satimgsize2crop_boundary = np.array(
            [self.satimgsize_correspond2uav_list.min(), self.satimgsize_correspond2uav_list.max()])
        self.satimgsize_scale_to_ref_m_boundary = self.satimgsize2crop_boundary*self.geo_res_m/self.scale_ref_m
        self.satimgsize_scale_to_refm_boundary = self.satimgsize_scale_to_ref_m_boundary #为了兼容老版本调用
        self.satimgsize_scale_to_ref_m_list = (uav_h_cover_m/self.scale_ref_m)[scale_mask]
        self.satimgsize_scale_to_ref_m_mean = self.satimgsize_scale_to_ref_m_list.mean()

        #  define the range when sampling the satmap:
        self.satmap_edge_pixs = self.satimgsize2crop_boundary[1]+8
        self.nr2sample_min = self.satmap_edge_pixs / self.satmap_hw_max
        self.nc2sample_min = self.nr2sample_min
        self.nr2sample_max = (self.satmap_h - self.satmap_edge_pixs) / self.satmap_hw_max
        self.nc2sample_max = (self.satmap_w - self.satmap_edge_pixs) / self.satmap_hw_max
        self.nr_tiftop = 0  # the normalized row corresponding to the first row
        self.nc_tifleft = 0  # the normalized column corresponding to the first column
        self.nr2sample_range = [self.nr2sample_min, self.nr2sample_max]
        self.nc2sample_range = [self.nc2sample_min, self.nc2sample_max]
        self.nr2sample_h = self.nr2sample_max - self.nr2sample_min
        self.nc2sample_w = self.nc2sample_max - self.nc2sample_min

        self.coords_4d_bounds = {
            'nr': (self.nr2sample_min, self.nr2sample_max),
            'nc': (self.nc2sample_min, self.nc2sample_max),
            'rot': (-np.pi, np.pi),
            'scale': (self.satimgsize_scale_to_ref_m_boundary[0], self.satimgsize_scale_to_ref_m_boundary[1]),
        }
        self.coords_4d_limits = [
            self.coords_4d_bounds['nr'],
            self.coords_4d_bounds['nc'],
            self.coords_4d_bounds['rot'],
            self.coords_4d_bounds['scale'],
        ]

        # define the edge, and the normed_halfimg_radius_rc that defines positive samples
        self.halfimg_radius_nrc = self.satimgsize2crop_mean // 2. / self.satmap_hw_max
        self.halfimg_radius_meter = self.get_halfimg_radius_meter(self.satimgsize2crop_mean // 2)

        self.return_pair=return_pair

    """funcs about sampling satmap:"""
    def crop_satimg_by_nrc(self, nrc, satimgsize2crop=224, type='tensor', random_satmap=False):
        row = int((nrc[0] - self.nr_tiftop) * self.satmap_hw_max)
        col = int((nrc[1] - self.nc_tifleft) * self.satmap_hw_max)

        halfimg_width = satimgsize2crop / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width

        if type == 'tensor':
            if self.return_pair:
                if random_satmap and self.n_satmaps > 1:
                    satmaps_tensor = random.sample(self.satmaps_tensor, 2)
                else:
                    if self.n_satmaps > 1:
                        satmaps_tensor = self.satmaps_tensor[:2]
                    else:
                        satmaps_tensor = [self.satmaps_tensor[0], self.satmaps_tensor[0]]
                satimg0 = satmaps_tensor[0][:, int(row_begin):int(row_end), int(col_begin):int(col_end)]
                satimg1 = satmaps_tensor[1][:, int(row_begin):int(row_end), int(col_begin):int(col_end)]
                satimg = torch.stack([satimg0, satimg1])
            else:
                if random_satmap and self.n_satmaps > 1:
                    satmaps_tensor = random.choice(self.satmaps_tensor)
                else:
                    satmaps_tensor = self.satmaps_tensor[0]
                satimg = satmaps_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            if random_satmap and self.n_satmaps > 1:
                satmap = random.choice(self.satmaps)
            else:
                satmap = self.satmaps[0]
            satimg = satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))
        return satimg

    def mk_rand_nrcs(self, n_rand, return_tensor=False, dtype=np.float32):
        """
        生成随机的 normalized row/col 坐标
        Args:
            return_tensor: bool, 如果为 True，直接在 self.device 上生成 tensor，极大提高效率
        """
        if return_tensor:
            # === 纯 PyTorch 路径 (高效) ===
            # 直接在 GPU 上生成随机数，避免 CPU->GPU 传输
            rand_nrcs = torch.rand((n_rand, 2), device=self.device, dtype=torch.float32)
            nr = self.nr2sample_h * rand_nrcs[:, 0] + self.nr2sample_min
            nc = self.nc2sample_w * rand_nrcs[:, 1] + self.nc2sample_min
            rand_nrcs = torch.stack([nr, nc], dim=1)
        else:
            # === 原 NumPy 路径 (兼容) ===
            rand_nrcs = np.random.rand(n_rand, 2).astype(dtype)
            nr = self.nr2sample_h * rand_nrcs[:, 0] + self.nr2sample_min
            nc = self.nc2sample_w * rand_nrcs[:, 1] + self.nc2sample_min
            rand_nrcs = np.stack([nr, nc], axis=1)

        return rand_nrcs

    def mk_rand_coords_4d(self, n_rand, return_tensor=False, dtype=np.float32):
        """
        生成随机的 4D 坐标
        优化：如果 return_tensor=True，全程使用 PyTorch 操作
        """
        # 1. 生成随机 nrc (根据 flag 自动选择 numpy 或 tensor)
        rand_nrcs = self.mk_rand_nrcs(n_rand, return_tensor=return_tensor, dtype=dtype)

        scale_min = self.satimgsize_scale_to_ref_m_boundary[0]
        scale_max = self.satimgsize_scale_to_ref_m_boundary[1]

        if return_tensor:
            # === 全程 GPU Tensor 模式 ===
            rand_rots = torch.rand(n_rand, device=self.device, dtype=torch.float32)
            rand_rots = rand_rots * 2 * np.pi - np.pi
            rand_scales = torch.rand(n_rand, device=self.device, dtype=torch.float32)
            rand_scales = rand_scales * (scale_max - scale_min) + scale_min
            rand_coords_4d = torch.cat([
                rand_nrcs,  # [N, 2]
                rand_rots.unsqueeze(1),  # [N, 1]
                rand_scales.unsqueeze(1)  # [N, 1]
            ], dim=1)  # [N, 4]
        else:
            # === 全程 Numpy 模式 ===
            rand_rots = (np.random.rand(n_rand) * 2 * np.pi - np.pi).astype(dtype)
            rand_scales = (np.random.rand(n_rand) * (scale_max - scale_min) + scale_min).astype(dtype)
            rand_coords_4d = np.concatenate([
                rand_nrcs,
                rand_rots[:, np.newaxis],
                rand_scales[:, np.newaxis]
            ], axis=1)

        return rand_coords_4d

    def transfrom_georc_to_nrc(self, georc: np.ndarray, source_epsg_code = 2056, dtype=np.float32):
        from transform_raster_rcs import georc_to_raster_rc
        rows, cols = georc_to_raster_rc(
            geoy=georc[:, 0],
            geox=georc[:, 1],
            source_epsg_code=source_epsg_code,
            target_epsg_code=self.epsg_code,
            target_geotransform=self.geo_transform,
        )
        row_normed = rows / self.satmap_hw_max + self.nr_tiftop
        col_normed = cols / self.satmap_hw_max + self.nc_tifleft
        return np.stack([row_normed, col_normed], axis=-1).astype(dtype)

    def transfrom_nrc_to_georc(self, nrcs: np.ndarray,target_espg_code=2056, dtype=np.float32):
        from transform_raster_rcs import raster_rc_to_georc
        rcs = nrcs * self.satmap_hw_max + self.nr_tiftop
        geo_rs,geo_cs = raster_rc_to_georc(
        rows=rcs[:,0],
        cols=rcs[:,1],
        source_geotransform=self.geo_transform,
        source_epsg_code=self.epsg_code,
        target_epsg_code=target_espg_code
        )
        return np.stack([geo_rs,geo_cs], axis=-1).astype(dtype)

    def get_halfimg_radius_meter(self,pixel_offset):
        from rc_offset_to_meter import rc_offset_to_meters
        offset_x_m, offset_y_m = rc_offset_to_meters(
            offset_col=pixel_offset,
            offset_row=pixel_offset,
            geotransform=self.geo_transform,
            epsg_code=self.epsg_code,
        )
        meter_radius = 0.5*(np.abs(offset_x_m)+np.abs(offset_y_m))
        return meter_radius

    def crop_sat_unifrom(self, size2clip=224, overlap=0., only_nrcs=False):
        # get all sattiles:
        sat_rows = self.satmap_h
        sat_cols = self.satmap_w
        n_pix_perstep = int(size2clip * (1. - overlap))
        n_colsteps = 1. * (sat_cols - size2clip) / n_pix_perstep
        n_rowsteps = 1. * (sat_rows - size2clip) / n_pix_perstep

        r_ids_begin = np.linspace(start=0, stop=int(n_rowsteps - 1), num=int(n_rowsteps), dtype=np.int16)
        c_ids_begin = np.linspace(start=0, stop=int(n_colsteps - 1), num=int(n_colsteps), dtype=np.int16)
        rcs_begincoord = np.stack(np.meshgrid(c_ids_begin, r_ids_begin), axis=-1)[:, :, ::-1] * n_pix_perstep
        rcs_endcoord = rcs_begincoord + np.array([size2clip, size2clip])
        # rcs_girdcoord = np.concatenate([rcs_begincoord, rcs_endcoord], axis=-1)

        # rc2nrc:
        rcs_girdcoord_center = (rcs_begincoord + rcs_endcoord) / 2
        nrcs_girdcoord_center = rcs_girdcoord_center / self.satmap_hw_max

        # Ensure sampling falls within the valid range:
        nr_min, nr_max = self.nr2sample_range
        nc_min, nc_max = self.nc2sample_range
        row_centers = nrcs_girdcoord_center[:, 0, 0]
        col_centers = nrcs_girdcoord_center[0, :, 1]
        row_mask = (row_centers >= nr_min) & (row_centers <= nr_max)
        col_mask = (col_centers >= nc_min) & (col_centers <= nc_max)
        if row_mask.sum() == 0 or col_mask.sum() == 0:
            if only_nrcs:
                return nrcs_girdcoord_center[:0, :0]
            empty_tiles = torch.empty(
                (0, 0, 3, int(size2clip), int(size2clip)),
                device=self.satmaps_tensor[0].device
            )
            return empty_tiles, nrcs_girdcoord_center[:0, :0]
        rcs_begincoord = rcs_begincoord[row_mask][:, col_mask]
        rcs_endcoord = rcs_endcoord[row_mask][:, col_mask]
        # rcs_girdcoord_center = rcs_girdcoord_center[row_mask][:, col_mask]
        nrcs_girdcoord_center = nrcs_girdcoord_center[row_mask][:, col_mask]
        rcs_girdcoord = np.concatenate([rcs_begincoord, rcs_endcoord], axis=-1)

        if only_nrcs:
            return nrcs_girdcoord_center

        # self.satmap_tensor = self.satmap_tensor.cuda() if not self.satmap_tensor.is_cuda and torch.cuda.is_available() else self.satmap_tensor
        n, m, _ = rcs_girdcoord.shape
        # sat_tiles = torch.empty((n, m, 3, size2clip, size2clip), device=self.satmaps_tensor[0].device)
        sat_tiles = torch.empty((n, m, 3, int(size2clip), int(size2clip)), device=self.satmaps_tensor[0].device)
        # 向量化提取所有窗口
        for i in range(n):
            for j in range(m):
                rb, cb, re, ce = rcs_girdcoord[i, j]
                sat_tiles[i, j] = self.satmaps_tensor[0][:, int(rb):int(re), int(cb):int(ce)]  # [C, H, W]

        return sat_tiles, nrcs_girdcoord_center

    def mk_sacle_levels(self,n_level=3):
        # mk a scale_levels
        delta_scale = (self.satimgsize_scale_to_ref_m_boundary[1]-self.satimgsize_scale_to_ref_m_boundary[0])/n_level
        lower = torch.linspace(start=self.satimgsize_scale_to_ref_m_boundary[0],end=self.satimgsize_scale_to_ref_m_boundary[1]-delta_scale,steps=n_level,dtype=torch.float32)
        upper = torch.linspace(start=self.satimgsize_scale_to_ref_m_boundary[0]+delta_scale,end=self.satimgsize_scale_to_ref_m_boundary[1], steps=n_level,dtype=torch.float32)
        scale_radio_to_200m_list = 0.5*(lower+upper)
        satimgsize_list = scale_radio_to_200m_list*self.scale_ref_m/self.geo_res_m
        return scale_radio_to_200m_list,satimgsize_list

    def _get_base_grid(self, out_h, out_w, device, dtype):
        """crop_satimg_by_4d_coords_fast depends on this func"""
        if not hasattr(self, "_base_grid_cache"):
            self._base_grid_cache = {}
        key = (out_h, out_w, device, dtype)
        if key not in self._base_grid_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, out_h, device=device, dtype=dtype),
                torch.linspace(-1, 1, out_w, device=device, dtype=dtype),
                indexing='ij'
            )
            self._base_grid_cache[key] = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        return self._base_grid_cache[key]

    def crop_satimg_by_4d_coords_fast(self, coords_4d, apply_rotation=True, chunk_size=1024, random_satmap=False, id_satmap2sample=None):
        """
        Faster crop: fuse crop+rotation into a single grid_sample, supports per-sample scale/rot.

        Args:
            coords_4d: [N,4] or [4] in [nrc, rot(rad), scale]
            apply_rotation: bool
            chunk_size: split N to control memory
        """
        # Convert input to torch tensor on the same device as satmaps
        if not torch.is_tensor(coords_4d):
            coords_4d = torch.from_numpy(np.asarray(coords_4d))

        single_input = (coords_4d.ndim == 1)
        if single_input:
            coords_4d = coords_4d.unsqueeze(0)  # [1,4]

        device = self.satmaps_tensor[0].device
        coords_4d = coords_4d.to(device=device, dtype=torch.float32)

        nrc = coords_4d[:, :2]
        rot = coords_4d[:, 2]
        scale = coords_4d[:, 3]

        # center in pixel coords
        rows = (nrc[:, 0] - self.nr_tiftop) * self.satmap_hw_max
        cols = (nrc[:, 1] - self.nc_tifleft) * self.satmap_hw_max

        # per-sample crop size in pixels
        satimgsize2crop = scale * self.scale_ref_m / self.geo_res_m
        satimgsize2crop = satimgsize2crop.clamp(
            min=self.satimgsize2crop_boundary[0],
            max=self.satimgsize2crop_boundary[1]
        )
        half_w = satimgsize2crop / 2.0

        out_h = out_w = self.imgsize2net
        base_grid = self._get_base_grid(out_h, out_w, device=device, dtype=coords_4d.dtype)  # [H,W,2]
        base_grid = base_grid.unsqueeze(0)  # [1,H,W,2]

        # choose the satmap to be sampled
        if id_satmap2sample is not None:
            satmap_tensor = self.satmaps_tensor[int(id_satmap2sample)]
        elif random_satmap and self.n_satmaps > 1:
            satmap_tensor = random.choice(self.satmaps_tensor)
        else:
            satmap_tensor = self.satmaps_tensor[0]

        sat_h = self.satmap_h
        sat_w = self.satmap_w

        outputs = []
        n_total = coords_4d.shape[0]
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            b = end - start

            half_w_b = half_w[start:end].view(b, 1, 1)
            x = base_grid[..., 0] * half_w_b
            y = base_grid[..., 1] * half_w_b

            if apply_rotation:
                rot_b = rot[start:end]
                cos_v = torch.cos(rot_b).view(b, 1, 1)
                sin_v = torch.sin(rot_b).view(b, 1, 1)
                x_rot = cos_v * x + sin_v * y
                y_rot = -sin_v * x + cos_v * y
            else:
                x_rot, y_rot = x, y

            x_rot = x_rot + cols[start:end].view(b, 1, 1)
            y_rot = y_rot + rows[start:end].view(b, 1, 1)

            # normalize to [-1,1]
            x_norm = (x_rot / (sat_w - 1)) * 2 - 1
            y_norm = (y_rot / (sat_h - 1)) * 2 - 1
            sample_grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]

            sat_in = satmap_tensor.unsqueeze(0).expand(b, -1, -1, -1)
            satimgs = F.grid_sample(
                sat_in,
                sample_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )  # [B,C,H,W]
            outputs.append(satimgs)

        satimgs = torch.cat(outputs, dim=0)
        if single_input:
            satimgs = satimgs[0]

        return satimgs

    """funcs about getting item:"""
    def __getitem__(self,index):
        """关于输出4d坐标的数值范围："""
        sat_nrc_rand = self.mk_rand_nrcs(1)[0]

        # handling size/scale
        satimgsize2crop = np.clip(np.random.choice(self.satimgsize_correspond2uav_list) +  (np.random.rand() - 0.5) *\
                           (self.satimgsize2crop_boundary[1]- self.satimgsize2crop_boundary[0]) * 0.1,
                                  self.satimgsize2crop_boundary[0],self.satimgsize2crop_boundary[1])
        satimgsize_cover_ratio_to_refm = torch.tensor([satimgsize2crop*self.geo_res_m/self.scale_ref_m],dtype=torch.float32)

        # crop the satimg
        satimg_rand = self.crop_satimg_by_nrc(sat_nrc_rand, type='tensor',satimgsize2crop=satimgsize2crop)

        # resize to network input size (without rotation in sat_transform_train)
        satimg_rand = self.scale_transform(satimg_rand)  # [C, H, W]

        # handling rotation: use batch_rotate_images_per_sample for consistent rotation convention
        # 随机旋转角度 [-180, 180] 度，逆时针为正
        angle_deg = np.random.uniform(-180, 180)
        rad_roted = torch.tensor([np.deg2rad(angle_deg)], dtype=torch.float32)

        # 使用 batch_rotate_images_per_sample 进行旋转
        if self.return_pair:
            # satimg_rand: [2, C, H, W] -> 对两张图应用相同旋转角度 -> [2, C, H, W]
            satimg_rand = batch_rotate_images_per_sample(satimg_rand, [angle_deg, angle_deg])
        else:
            # satimg_rand: [C, H, W] -> [1, C, H, W] -> 旋转 -> [1, C, H, W] -> [C, H, W]
            satimg_rand = batch_rotate_images_per_sample(satimg_rand.unsqueeze(0), [angle_deg]).squeeze(0)

        # 组合成4D坐标 tensor，与 UAVDataset 对齐
        sat_nrc_rand_tensor = torch.from_numpy(sat_nrc_rand).to(torch.float32)  # [2]
        coords_4d = torch.cat([sat_nrc_rand_tensor, rad_roted, satimgsize_cover_ratio_to_refm], dim=-1)  # [4]
        # 如果 return_pair=True，需要扩展 coords_4d 以匹配 satimg_rand 的维度
        if self.return_pair:
            # satimg_rand: [2, C, H, W], 需要 coords_4d: [2, 4]
            coords_4d = coords_4d.unsqueeze(0).expand(2, -1)  # [4] -> [1, 4] -> [2, 4]

        return satimg_rand, coords_4d
        # debug
        # img2vis = self.denormalize_img(satimg_rand[1])
        # from matplotlib import pyplot as plt
        # plt.imshow(img2vis)
        # plt.show()
        # img2vis = self.crop_satimg_by_nrc(sat_nrc_rand,satimgsize2crop=satimgsize2crop, type='np')
        # from matplotlib import pyplot as plt
        # plt.imshow(img2vis)
        # plt.show()

    def __len__(self):
        return   int((self.satmap_h * self.satmap_w) / (self.satimgsize2crop_mean ** 2))


    """funcs about fine loc:"""
    def sample_sats_in_rect(self, nrc_topleft, nrc_buttonright, n2sample_h=128, n2sample_w=128, satimgsize2crop=224, type2clip='tensor'):
        halfimg_width = satimgsize2crop/2
        half_img_h = halfimg_width / self.satmap_hw_max
        half_img_w = half_img_h
        nrs_center = torch.linspace( nrc_topleft[0]+half_img_h, nrc_buttonright[0]-half_img_h, steps=n2sample_h)
        ncs_center = torch.linspace( nrc_topleft[1]+half_img_w, nrc_buttonright[1]-half_img_w, steps=n2sample_w)
        rows_center = (nrs_center*self.satmap_hw_max).to(torch.int32)
        cols_center = (ncs_center*self.satmap_hw_max).to(torch.int32)

        rows_begin = (rows_center - halfimg_width).to(torch.int32)
        rows_end = (rows_center + halfimg_width).to(torch.int32)
        cols_begin = (cols_center - halfimg_width).to(torch.int32)
        cols_end = (cols_center + halfimg_width).to(torch.int32)

        nrr, ncc = torch.meshgrid(nrs_center, ncs_center, indexing='ij')  # 'ij' 表示 y 行, x 列的顺序
        nrc_center_meshgrid = torch.stack([nrr, ncc]).permute(1, 2, 0)

        if type2clip == 'tensor':
            sat_tiles = torch.empty((n2sample_h, n2sample_w, 3, satimgsize2crop, satimgsize2crop),
                                    device=self.satmap_tensor.device)
            for i in range(rows_begin.shape[0]):
                for j in range(cols_begin.shape[0]):
                    rb, cb, re, ce = rows_begin[i], cols_begin[j], rows_end[i], cols_end[j]
                    sat_tiles[i, j] = self.satmap_tensor[:, rb:re, cb:ce]  # [C, H, W]
        else:
            sat_tiles = np.zeros((n2sample_h, n2sample_w, 3, satimgsize2crop, satimgsize2crop)).astype(np.float32)
            for i in range(rows_begin.shape[0]):
                for j in range(cols_begin.shape[0]):
                    rb, cb, re, ce = rows_begin[i], cols_begin[j], rows_end[i], cols_end[j]
                    sat_tiles[i, j] = self.satmap[:, rb:re, cb:ce]  # [C, H, W]

        return sat_tiles,nrc_center_meshgrid

    """funcs for debugging"""
    def denormalize_img(self,img_tensor):
        if img_tensor.device.type != 'cpu':
            img_tensor = img_tensor.cpu()
        img_np = img_tensor * torch.tensor(self.satinfo_dict['stds_normalized'][0])[:,None,None]+torch.tensor(self.satinfo_dict['means_normalized'][0])[:,None,None]
        img_np = img_np.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return img_np

    def crop_rect_satimg(self, nrc_topleft,nrc_rightbottom, type='tensor'):
        row_begin = int(nrc_topleft[0]*self.satmap_hw_max)
        col_begin = int(nrc_topleft[1]*self.satmap_hw_max)
        row_end = int(nrc_rightbottom[0]*self.satmap_hw_max)
        col_end = int(nrc_rightbottom[1]*self.satmap_hw_max)

        if type == 'tensor':
            satimg = self.satmap_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            satimg = self.satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))

        return satimg


class UAVDataset(object):
    def __init__(self,
                 p_uavinfo_json,
                 imgsize2net=224,
                 sat_dataset=None,
                 scale_ref_m=None,
                 stage='train',
                 use_augmentation=True,
                 name=None,
                 device='cpu',
                 dataset_name = 'visloc',
                 **kwargs,
                 ):
        # read corresponding uav imgs & mate info
        with open(p_uavinfo_json, "r") as f:
            self.uavinfo_dict = json.load(f)
        self.name = name if name is not None else os.path.splitext(os.path.basename(p_uavinfo_json))[0]
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type != 'cpu':
            raise ValueError("UAVDataset coords are kept on CPU; pass device='cpu' or leave default.")
        df = pd.read_csv(self.uavinfo_dict['uavimgs_geocsv_path'])
        self.uav_df = df

        if sat_dataset is not None:
            self.geo_res_m = sat_dataset.geo_res_m

        # filtering by the scale
        scale_ref_m_used = scale_ref_m
        if scale_ref_m_used is None and sat_dataset is not None and hasattr(sat_dataset, "scale_ref_m"):
            scale_ref_m_used = sat_dataset.scale_ref_m
        scale_info = _compute_scale_filter_from_df(df, self.geo_res_m, scale_ref_m_used)
        uav_h_cover_m = scale_info["uav_h_cover_m"]
        scale_mask = scale_info["scale_mask"]
        self.scale_ref_m = scale_info["scale_ref_m"]

        uav_names = self.uav_df['filename'][scale_mask]
        uavimgs_dir = self.uavinfo_dict['uavimgs_dir']
        if 'visloc' in dataset_name.lower():
            self.uavimg_paths = [os.path.join(uavimgs_dir, f'{int(name[3:7]):06d}', name) for name in uav_names]
        elif ('neuloc'  in dataset_name.lower()) or ('wingtra' in dataset_name.lower()):
            self.uavimg_paths = [os.path.join(uavimgs_dir, name) for name in uav_names]
        # keep lat/lon aligned with the filtered UAV list
        self.uav_latlons = np.stack([self.uav_df['latitude'], self.uav_df['longitude']], axis=1)[scale_mask]
        self.uav_rots = np.deg2rad(np.array(self.uav_df['rotdeg_fm_north_anticlock'][scale_mask]))
        self.uav_scales = np.array(uav_h_cover_m[scale_mask] / self.scale_ref_m)

        if 'geo_rc_epsg_code' in  self.uavinfo_dict.keys():
            self.epsg_code = int(self.uavinfo_dict['geo_rc_epsg_code'])
            self.uav_georcs = np.stack([self.uav_df[f'geo_row_proj{self.epsg_code}'],self.uav_df[f'geo_col_proj{self.epsg_code}']], axis=1)
            self.uav_georcs = self.uav_georcs[scale_mask]

        # if trans_georc2nrc_func is not None:
        #     self.uav_nrcs = trans_georc2nrc_func(self.uav_georcs,dtype=np.float32,source_epsg_code=self.epsg_code)
        if sat_dataset is not None:
            self.sat_dataset = sat_dataset
            self.uav_nrcs = sat_dataset.transfrom_georc_to_nrc(self.uav_georcs,dtype=np.float32,source_epsg_code=self.epsg_code)
            self.filter_by_sat_sampling_range(sat_dataset=sat_dataset)

        self.uav_nrcs_torch = torch.from_numpy(self.uav_nrcs).to(device=self.device, dtype=torch.float32)
        self.uav_rots_torch = torch.from_numpy(self.uav_rots).to(device=self.device, dtype=torch.float32)[...,None]
        self.uav_scales_torch = torch.from_numpy(self.uav_scales).to(device=self.device, dtype=torch.float32)[...,None]
        self.uav_coords_4d_torch = torch.concatenate([self.uav_nrcs_torch, self.uav_rots_torch, self.uav_scales_torch], dim=-1)
        self.split_uav_dataset()

        self.switch_stage(stage)

        # config transform for uavimgs
        self.imgsize2net = imgsize2net
        self.use_augmentation = use_augmentation

        # Test transform (无数据增强)
        self.uav_transform_test = mk_pil_transform(
            imgsize2net=self.imgsize2net,
            mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
            center_crop=True)

        # Train transform (根据use_augmentation决定)
        if self.use_augmentation:
            # 使用数据增强
            self.uav_transform_train = mk_pil_transform_with_params(
                mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
                imgsize2net=self.imgsize2net,
                rand_rot=True, #该参数会让坐标标签跟着改变
                rand_scale=True, scale_range=(0.95, 1.05), # scale_factor > 1 (图像放大，"zoom in")->scale 坐标会变小
                rand_affine=True, affine_para={'degrees': 0, 'translate': (0.0, 0.0), 'scale': (1., 1.), 'shear': 10}, #坐标标签不会改变，为了增加一定鲁棒性
                rand_erase=False, center_crop=True, rand_crop=False)
        else:
            # 不使用数据增强，直接使用test transform
            self.uav_transform_train = self.uav_transform_test


    """funcs about handling uavings:"""
    def split_uav_dataset(self, train_radio=0.9):
        # split the dataset for train/val/test
        n_train = int(len(self.uavimg_paths) * train_radio)
        self.n_train = n_train
        self.train_ratio = train_radio

        self.uavimg_paths_train = self.uavimg_paths[:n_train]
        self.uav_lonlats_train = self.uav_latlons[:n_train]

        self.uavimg_paths_test = self.uavimg_paths[n_train:]
        self.uav_lonlats_test = self.uav_latlons[n_train:]

        self.uav_coords_4d_torch_train = self.uav_coords_4d_torch[:n_train]
        self.uav_coords_4d_torch_test = self.uav_coords_4d_torch[n_train:]


    """funcs about get item:"""
    def _getitem_train(self,index):
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_train[index])
        uavimg_q = self.uav_transform_train(uavimg)

        # 检查是否使用数据增强
        if self.use_augmentation and hasattr(self.uav_transform_train, 'get_params'):
            # 使用数据增强：获取增强参数并应用到坐标
            params = self.uav_transform_train.get_params()
            coords_augmented = apply_augment_to_coords(
                self.uav_coords_4d_torch_train[index],
                rotation_deg=params['rotation_deg'],
                scale_factor=params['scale']
            )
            return uavimg_q, coords_augmented
        else:
            # 不使用数据增强：直接返回原始坐标
            coords = self.uav_coords_4d_torch_train[index]
            return uavimg_q, coords
        # # 2. Get coordinates
        # uav_rc = self.uav_nrcs_torch_train[index]
        # # 3. Get rotation and scale（直接索引，清晰明了）
        # uav_rot = self.uav_rots_torch_train[index]
        # uav_scale = self.uav_scales_torch_train[index]
        # return uavimg_q,uav_rc,uav_rot,uav_scale

    def _getitem_test(self,index):#todo:N从dataset_wingtra_4d.pyeeds to be refined according to external needs -> the test_func()
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_test[index])
        uavimg_q = self.uav_transform_test(uavimg)
        coords = self.uav_coords_4d_torch_test[index]
        return uavimg_q,coords
        # 2. Get coordinates
        # uav_rc = self.uav_nrcs_torch_test[index]
        # # 3. Get rotation and scale（直接索引，清晰明了）
        # uav_rot = self.uav_rots_torch_test[index]
        # uav_scale = self.uav_scales_torch_test[index]
        # return uavimg_q,uav_rc,uav_rot,uav_scale

    def switch_stage(self,stage='train'):
        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
            self.dataset_len = len(self.uavimg_paths_train)
        else:
            self._getitem = self._getitem_test
            self.dataset_len = len(self.uavimg_paths_test)

    def filter_by_sat_sampling_range(self, sat_dataset, include_scale=True, include_rot=False, eps=1e-6):
        """
        根据卫星图采样的归一化范围过滤 UAV 样本。

        Args:
            sat_dataset: SatDataset 实例，需包含 nr2sample_range / nc2sample_range 等字段。
            include_scale: 是否同时限制缩放比例（默认 True）。
            include_rot: 是否检查旋转范围（默认 False）。
            eps: 边界放宽的容差。

        Returns:
            dict: {'kept': 保留数量, 'dropped': 剔除数量}
        """
        if not hasattr(self, 'uav_nrcs'):
            raise ValueError("uav_nrcs 未初始化，请在构造 UAVDataset 时传入 trans_georc2nrc_func。")
        if not hasattr(sat_dataset, 'nr2sample_range') or not hasattr(sat_dataset, 'nc2sample_range'):
            raise ValueError("sat_dataset 缺少 nr2sample_range / nc2sample_range。")

        nr_min, nr_max = sat_dataset.nr2sample_range
        nc_min, nc_max = sat_dataset.nc2sample_range

        mask = (self.uav_nrcs[:, 0] >= nr_min - eps) & (self.uav_nrcs[:, 0] <= nr_max + eps)
        mask &= (self.uav_nrcs[:, 1] >= nc_min - eps) & (self.uav_nrcs[:, 1] <= nc_max + eps)

        if include_scale and hasattr(sat_dataset, 'satimgsize_scale_to_ref_m_boundary'):
            s_min, s_max = sat_dataset.satimgsize_scale_to_ref_m_boundary
            mask &= (self.uav_scales >= s_min - eps) & (self.uav_scales <= s_max + eps)

        if include_rot:
            if hasattr(sat_dataset, 'coords_4d_bounds') and 'rot' in sat_dataset.coords_4d_bounds:
                rot_min, rot_max = sat_dataset.coords_4d_bounds['rot']
            else:
                rot_min, rot_max = (-np.pi, np.pi)
            mask &= (self.uav_rots >= rot_min - eps) & (self.uav_rots <= rot_max + eps)

        n_before = len(mask)
        n_after = int(mask.sum())
        if n_after == n_before:
            return {'kept': n_after, 'dropped': 0}

        def _mask_list(seq):
            return [item for item, keep in zip(seq, mask) if keep]

        # 应用筛选
        self.uavimg_paths = _mask_list(self.uavimg_paths)
        self.uav_latlons = self.uav_latlons[mask]
        if hasattr(self, 'uav_georcs'):
            self.uav_georcs = self.uav_georcs[mask]
        self.uav_nrcs = self.uav_nrcs[mask]
        self.uav_rots = self.uav_rots[mask]
        self.uav_scales = self.uav_scales[mask]

        return {'kept': n_after, 'dropped': n_before - n_after}

    def __getitem__(self, index):
        return self._getitem(index)


    def __len__(self):
        return  self.dataset_len

    """funcs for debugging"""
    def denormalize_img(self,img_tensor):
        if img_tensor.device.type != 'cpu':
            img_tensor = img_tensor.cpu()
        img_np = img_tensor * torch.tensor(self.uavinfo_dict['std'])[:,None,None]+torch.tensor(self.uavinfo_dict['mean'])[:,None,None]
        img_np = img_np.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return img_np



if __name__ == '__main__':
    # import pandas as pd
    # p2csv = '/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv'
    # df=pd.read_csv(p2csv)
    # ratio = df['cover_ratio_to_200m*200m']
    # aff2d_corrected = df['aff2d_corrected']
    # # ratio = ratio[aff2d_corrected]
    # ratio = np.array(ratio)
    # x2 = 40000/384/576*ratio
    # x = np.sqrt(x2)
    # h_m = 384*x
    # w_m = 576*x
    # df['h_cover_m'] = h_m
    # df['w_cvoer_m'] = w_m
    # df.to_csv('/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv', index=False)

    sat_dataset = SatDataset(
        p_satinfo_json='/home/data/zwk/dataset_UAV-VisLoc/04/satellite04_epsg32650_res03m_multi_tifs.json',
        p_uav_geocsv='/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_geo_corrected.csv',
        imgsize2net=224,
    )
    uav_dataset = UAVDataset(
        p_uavinfo_json = '/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_metainfo.json',
        trans_georc2nrc_func = sat_dataset.transfrom_georc_to_nrc,
        sat_dataset=sat_dataset,
        # geo_res_m=0.3,
    )

    for i in range(5):
        try:
            data = uav_dataset[i]
        except Exception as e:
            print(f"加载样本 {i} 时出错: {e}")
            # 在这里中断或记录错误，然后去检查对应的文件或标注是否有问题
            break
