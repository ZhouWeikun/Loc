import os
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# from torchvision.transforms import InterpolationMode
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import random
import  torchvision.transforms as T
#
from train_img_encoder.datasets_custom.util_mk_data_transform import mk_pil_transform,mk_sat_tensor_transform


class SatDataset(object):
    def __init__(self,
                 p_satinfo_json,
                 imgsize2net = 224,
                 satimgsize2crop = 256,
                 scale_ref_m=200,
                 p_uav_geocsv=None,
                 **kwargs,
                 ):
        # read corresponding mate info
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
            self.satmaps_tensor.append(sat2tensor_transform(satmap))

        # config the attrs about tifs
        self.n_satmaps = len(self.satmaps)
        self.satmap_h = self.satmaps[0].height
        self.satmap_w = self.satmaps[0].width
        self.satmap_hw_max = np.max([self.satmap_h, self.satmap_w])

        #config the transforms
        self.imgsize2net = imgsize2net
        self.sat_transform_train, self.sat_rotater = mk_sat_tensor_transform(imgsize2net,rand_rot=True)
        self.scale_transform = T.Compose([transforms.Resize(self.imgsize2net,antialias=True)])

        # for defining the scale to sample
        self.scale_ref_m = scale_ref_m
        df = pd.read_csv(p_uav_geocsv)
        h_cover_m =  df['h_cover_m']
        aff2d_corrected_mask = df['aff2d_corrected']
        h_cover_m_corrected = h_cover_m[aff2d_corrected_mask]
        satimgsize_scale_to_200m_corrected = np.array(h_cover_m_corrected/self.geo_res_m)*self.geo_res_m/scale_ref_m
        lower_bound = np.percentile( satimgsize_scale_to_200m_corrected, 2)
        upper_bound = np.percentile( satimgsize_scale_to_200m_corrected, 99)
        satimgsize_scale_to_200m = np.array(h_cover_m / self.geo_res_m) * self.geo_res_m / scale_ref_m
        scale_mask = (satimgsize_scale_to_200m > lower_bound) * (satimgsize_scale_to_200m < upper_bound) * aff2d_corrected_mask

        # config the satimgsize2crop
        self.satimgsize_correspond2uav_list = np.array(h_cover_m[scale_mask] / self.geo_res_m)
        self.satimgsize2crop_mean = self.satimgsize_correspond2uav_list.mean()
        self.satimgsize2crop_boundary = np.array(
            [self.satimgsize_correspond2uav_list.min(), self.satimgsize_correspond2uav_list.max()])
        self.satimgsize_scale_to_200m_boundary = self.satimgsize2crop_boundary*self.geo_res_m/scale_ref_m
        self.satimgsize_scale_to_200m_list = satimgsize_scale_to_200m[scale_mask]
        self.satimgsize_scale_to_200m_mean = self.satimgsize_scale_to_200m_list.mean()
        # mk a scale_levels
        n_scales = 3
        delta_scale = (self.satimgsize_scale_to_200m_boundary[1]-self.satimgsize_scale_to_200m_boundary[0])/n_scales
        lower = torch.linspace(start=self.satimgsize_scale_to_200m_boundary[0],end=self.satimgsize_scale_to_200m_boundary[1]-delta_scale,steps=3,dtype=torch.float32)
        upper = torch.linspace(start=self.satimgsize_scale_to_200m_boundary[0]+delta_scale,end=self.satimgsize_scale_to_200m_boundary[1], steps=3,dtype=torch.float32)
        self.satimgsize_scale_to_200m_unif_list = 0.5*(lower+upper)
        self.satimgsize_unif_list = self.satimgsize_scale_to_200m_unif_list*scale_ref_m/self.geo_res_m

        #  define the range when sampling the satmap:
        self.satmap_edge_pixs = self.satimgsize2crop_boundary[1]+224
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

        # define the edge, and the normed_halfimg_radius_rc that defines positive samples
        self.halfimg_radius_nrc = self.satimgsize2crop_mean // 2. / self.satmap_hw_max
        self.halfimg_radius_meter = self.get_halfimg_radius_meter(satimgsize2crop // 2)

        self.return_pair=True


    """funcs about sampling satmap:"""
    def crop_satimg_by_nrc(self, nrc, satimgsize2crop=224, type='tensor'):
        row = int((nrc[0] - self.nr_tiftop) * self.satmap_hw_max)
        col = int((nrc[1] - self.nc_tifleft) * self.satmap_hw_max)

        halfimg_width = satimgsize2crop / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width

        if type == 'tensor':
            if self.return_pair:
                satmaps_tensor = random.sample(self.satmaps_tensor, 2)
                satimg0 = satmaps_tensor[0][:, int(row_begin):int(row_end), int(col_begin):int(col_end)]
                satimg1 = satmaps_tensor[1][:, int(row_begin):int(row_end), int(col_begin):int(col_end)]
                satimg = torch.stack([satimg0, satimg1])
            else:
                satmaps_tensor = random.choice(self.satmaps_tensor)
                satimg = satmaps_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            satmap = random.choice(self.satmaps)
            satimg = satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))
        return satimg

    def mk_rand_nrcs(self, n_rand, dtype=np.float32):
        rand_nrcs = np.random.rand(n_rand, 2).astype(dtype)
        nr = self.nr2sample_h * rand_nrcs[:, 0] + self.nr2sample_min
        nc = self.nc2sample_w * rand_nrcs[:, 1] + self.nc2sample_min
        rand_nrcs = np.stack([nr, nc], axis=1)
        return rand_nrcs

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
        from transform_raster_rcs import rc_offset_to_meters
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
        rcs_girdcoord = np.concatenate([rcs_begincoord, rcs_endcoord], axis=-1)

        # rc2nrc:
        rcs_girdcoord_center = (rcs_begincoord + rcs_endcoord) / 2
        nrcs_girdcoord_center = rcs_girdcoord_center / self.satmap_hw_max
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


    """funcs about getting item:"""
    def __getitem__(self,index):
        sat_nrc_rand = self.mk_rand_nrcs(1)[0]

        # handling size/scale
        satimgsize2crop = np.clip(np.random.choice(self.satimgsize_correspond2uav_list) +  (np.random.rand() - 0.5) *\
                           (self.satimgsize2crop_boundary[1]- self.satimgsize2crop_boundary[0]) * 0.1,
                                  self.satimgsize2crop_boundary[0],self.satimgsize2crop_boundary[1])
        satimgsize_len_ratio_to_200m = torch.tensor([satimgsize2crop*self.geo_res_m/self.scale_ref_m],dtype=torch.float32)

        # crop the satimg
        satimg_rand = self.crop_satimg_by_nrc(sat_nrc_rand, type='tensor',satimgsize2crop=satimgsize2crop)
        satimg_rand = self.sat_transform_train(satimg_rand)

        # hanling rot
        angle_roted = self.sat_rotater.angle
        rad_roted = torch.deg2rad(torch.tensor([angle_roted],dtype=torch.float32))
        # normalize the rad from [-pi,pi] to [0,2pi]
        return  satimg_rand,sat_nrc_rand,rad_roted,satimgsize_len_ratio_to_200m
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
                 scale_ref_m=200,
                 geo_res_m=0.3,
                 trans_georc2nrc_func=None,
                 **kwargs,
                 ):
        # read corresponding uav imgs & mate info
        with open(p_uavinfo_json, "r") as f:
            self.uavinfo_dict = json.load(f)
        df = pd.read_csv(self.uavinfo_dict['uavimgs_geocsv_path'])
        self.uav_df = df

        # filtering by the scale
        h_cover_m =  df['h_cover_m']
        aff2d_corrected_mask = df['aff2d_corrected']
        h_cover_m_corrected = h_cover_m[aff2d_corrected_mask]
        satimgsize_scale_to_200m_corrected = np.array(h_cover_m_corrected/geo_res_m)*geo_res_m/scale_ref_m
        lower_bound = np.percentile( satimgsize_scale_to_200m_corrected, 2)
        upper_bound = np.percentile( satimgsize_scale_to_200m_corrected, 99)
        satimgsize_scale_to_200m = np.array(h_cover_m / geo_res_m) * geo_res_m / scale_ref_m
        scale_mask = (satimgsize_scale_to_200m > lower_bound) * (satimgsize_scale_to_200m < upper_bound) * aff2d_corrected_mask

        uav_names = self.uav_df['filename'][scale_mask]
        uavimgs_dir = self.uavinfo_dict['uavimgs_dir']
        self.uavimg_paths = [os.path.join(uavimgs_dir, name) for name in uav_names]
        self.uav_latlons = np.stack([self.uav_df['latitude [decimal degrees]'], self.uav_df['longitude [decimal degrees]']], axis=1)
        self.uav_rot = np.deg2rad(np.array(self.uav_df['rotdeg_fm_north_anticlock'][scale_mask]))
        self.uav_rot_torch = torch.from_numpy(self.uav_rot).to(torch.float32)
        self.uav_sacle_ratio = np.array(h_cover_m[scale_mask] / scale_ref_m)
        self.uav_sacle_ratio_torch = torch.from_numpy(self.uav_sacle_ratio).to(torch.float32)

        if 'geo_rc_epsg_code' in  self.uavinfo_dict.keys():
            self.epsg_code = int(self.uavinfo_dict['geo_rc_epsg_code'])
            self.uav_georcs = np.stack([self.uav_df[f'geo_row_proj{self.epsg_code}'],self.uav_df[f'geo_col_proj{self.epsg_code}']], axis=1)

        if trans_georc2nrc_func is not None:
            self.uav_nrcs = trans_georc2nrc_func(self.uav_georcs,dtype=np.float32,source_epsg_code=self.epsg_code)

        self.split_uav_dataset()
        if kwargs['stage']!=None:
            self.switch_stage(kwargs['stage'])

        # config transform for uavimgs
        self.imgsize2net = imgsize2net
        self.uav_transform_train = mk_pil_transform(
            imgsize2net=self.imgsize2net,
            mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
            rand_affine=True, affine_para={'degrees': 0, 'translate': (0, 0), 'scale': (1.0, 1.0), 'shear': 5},
            rand_erase=True,center_crop=True,rand_crop=False)
        self.uav_transform_test = mk_pil_transform(
            imgsize2net=self.imgsize2net,
            mean=self.uavinfo_dict['mean'], std=self.uavinfod_ict['std'],
            center_crop=True)

    """funcs about handling uavings:"""
    def split_uav_dataset(self, train_radio=0.9):
        # split the dataset for train/val/test
        n_train = int(len(self.uavimg_paths) * train_radio)
        self.n_train = n_train

        self.uavimg_paths_train = self.uavimg_paths[:n_train]
        self.uav_lonlats_train = self.uav_latlons[:n_train]

        self.uavimg_paths_test = self.uavimg_paths[n_train:]
        self.uav_lonlats_test = self.uav_latlons[n_train:]

        if hasattr(self, 'uav_nrcs'):
            self.uav_nrcs_train = self.uav_nrcs[:n_train]
            self.uav_nrcs_test = self.uav_nrcs[n_train:]

    """funcs about get item:"""
    def _getitem_train(self,index):
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_train[index])
        uavimg_q = self.uav_transform_train(uavimg)
        uav_rc = self.uav_nrcs_train[index]
        return  uavimg_q,torch.tensor(uav_rc)


    def _getitem_test(self,index):#todo:Needs to be refined according to external needs -> the test_func()
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_test[index])
        uavimg_q = self.uav_transform_test(uavimg)
        uav_rc = self.uav_nrcs_test[index]

        return uavimg_q,torch.tensor(uav_rc)


    def switch_stage(self,stage='train'):
        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
            self.dataset_len = len(self.uavimg_paths_train)
        else:
            self._getitem = self._getitem_test
            self.dataset_len = len(self.uavimg_paths_test)


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
        p_satinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/blocks12_res03m.json',
        p_uav_geocsv='/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv',
        imgsize2net=224,
    )
    uav_dataset = UAVDataset(
        p_uavinfo_json = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info.json',
        trans_georc2nrc_func = sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
    )


    for i in range(5):
        try:
            data = sat_dataset[i]
        except Exception as e:
            print(f"加载样本 {i} 时出错: {e}")
            # 在这里中断或记录错误，然后去检查对应的文件或标注是否有问题
            break


