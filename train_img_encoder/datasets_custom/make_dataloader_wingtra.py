from torch.utils.data import Dataset
import os
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 #import after setting OPENCV_IO_MAX_IMAGE_PIXELS
# from torchvision.transforms import InterpolationMode
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # 取消限制（风险：不安全）
import pandas as pd
import numpy as np
import torch
from torchvision import transforms


def mk_transform(
        mean,
        std,
        imgsize2net=224,
        rand_affine = False,
        affine_para = None,
        rand_rot = False,
        rand_erase = False,
        color_jitter = False,
        center_crop = False,
        rand_crop = False,
        ):
    transform_list = [transforms.Resize(imgsize2net)]
    if center_crop:
        transform_list += [transforms.CenterCrop(imgsize2net)]
    if rand_crop:
        transform_list += [transforms.RandomCrop(imgsize2net)]
    if rand_rot:
        transform_list.append(transforms.RandomRotation(180, interpolation=3))
    if rand_affine:
        # 1. 为 RandomAffine 定义一套默认参数
        default_affine_params = {
            'degrees': 180,
            'translate': (0, 0),
            'scale': (1.0, 1.0),
            'shear': 5,
        }
        # 2. 如果用户提供了affine_para字典，用它来更新默认值
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)
            # 3. 使用字典解包(**)将参数传递给函数
        transform_list.append(
            transforms.RandomAffine(
                **default_affine_params,  # <--- 核心改动
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
        )
    if color_jitter:
        transform_list.append(
            transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                # transforms.RandomAutocontrast(p=0.3),
                # transforms.RandomGrayscale(p=0.3)
            ])
        )
    transform_list += [transforms.ToTensor()]
    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=1))
    transform_list +=[transforms.Normalize(mean=mean, std=std)]

    transform2ret = transforms.Compose(transform_list)
    return transform2ret


class SatDataset(object):
    def __init__(self,
                 p_satinfo_json,
                 imgsize2net = 224,
                 satimgsize2crop = 256,
                 **kwargs,
                 ):
        self.imgsize2net = imgsize2net

        # setting about satmap:
        # read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        self.epsg_code = int(self.satinfo_dict['epsg_code'])
        self.satmap = Image.open(self.satinfo_dict['filepath'])  # hwc
        self.satmap_h = self.satmap.height
        self.satmap_w = self.satmap.width
        self.satmap_hw_max = np.max([self.satmap_h, self.satmap_w])

        #  define the edge, and the normed_halfimg_radius_rc that defines positive samples
        self.satimgsize2crop = satimgsize2crop
        self.satmap_edge_pixs = satimgsize2crop // 2
        self.halfimg_radius_nrc = self.satimgsize2crop // 2. / self.satmap_hw_max  # about 30m
        self.halfimg_radius_meter = self.get_halfimg_radius_meter(satimgsize2crop // 2)

        #  define the range when sampling the satmap:
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

        # config transform for uavimgs
        sat2tensor_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean_normalized'],
                                 std=self.satinfo_dict['std_normalized']),
        ]
        self.sat2tensor_transform = transforms.Compose(sat2tensor_transform)
        self.satmap_tensor = self.sat2tensor_transform(self.satmap)
        self.sat_transform_train = mk_transform(
            imgsize2net=self.imgsize2net,
            mean=self.satinfo_dict['mean_normalized'], std=self.satinfo_dict['std_normalized'],
            rand_affine=True, affine_para={'degrees': 180, 'translate': (0, 0), 'scale': (1.0, 1.0), 'shear': 5},
            rand_erase=True)
        self.sat_transform_test = mk_transform(
            imgsize2net=self.imgsize2net,
            mean=self.satinfo_dict['mean_normalized'], std=self.satinfo_dict['std_normalized'])


    """funcs about sampling satmap:"""
    def crop_satimg_by_nrc(self, nrc, imgsize2crop=None, type='tensor'):
        row = int((nrc[0] - self.nr_tiftop) * self.satmap_hw_max)
        col = int((nrc[1] - self.nc_tifleft) * self.satmap_hw_max)

        halfimg_width = self.satimgsize2crop / 2 if imgsize2crop is None else imgsize2crop / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width

        if type == 'tensor':
            satimg = self.satmap_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            satimg = self.satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))

        return satimg


    def mk_rand_nrcs(self, n_rand, dtype=np.float32):
        rand_nrcs = np.random.rand(n_rand, 2)
        nr = self.nr2sample_h * rand_nrcs[:, 0] + self.nr2sample_min
        nc = self.nc2sample_w * rand_nrcs[:, 1] + self.nc2sample_min
        rand_nrcs = np.stack([nr, nc], axis=1)
        return rand_nrcs.astype(dtype)


    # def transform_latlon_to_nrc(self, lat_lons: np.ndarray, dtype=np.float32):
    #     from transform_raster_rcs import latlon_to_raster_rc
    #     rows, cols = latlon_to_raster_rc(
    #         lats=lat_lons[:, 0],
    #         lons=lat_lons[:, 1],
    #         target_geotransform=self.geo_transform,
    #         target_epsg_code=self.epsg_code,
    #     )
    #     col_normed = cols / self.satmap_hw_max + self.nc_tifleft
    #     row_normed = rows / self.satmap_hw_max + self.nr_tiftop
    #     return np.stack([row_normed, col_normed], axis=-1).astype(dtype)
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


    # def transform_nrc_to_georc(self, nrcs: np.ndarray, dtype=np.float32):
    #     from transform_raster_rcs import raster_rc_to_georc
    #     rcs = nrcs * self.satmap_hw_max + self.nr_tiftop
    #     r_geo,c_geo = raster_rc_to_georc(rcs[:,0],rcs[:,1],self.geo_transform,self.epsg_code)
    #     return np.stack([r_geo,c_geo], axis=-1).astype(dtype)
    def transform_nrc_to_georc(self, nrcs: np.ndarray,target_espg_code=2056, dtype=np.float32):
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


    def crop_sat_unifrom(self, size2clip=224, overlap=0.):
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

        # self.satmap_tensor = self.satmap_tensor.cuda() if not self.satmap_tensor.is_cuda and torch.cuda.is_available() else self.satmap_tensor
        n, m, _ = rcs_girdcoord.shape
        sat_tiles = torch.empty((n, m, 3, size2clip, size2clip), device=self.satmap_tensor.device)
        # 向量化提取所有窗口
        for i in range(n):
            for j in range(m):
                rb, cb, re, ce = rcs_girdcoord[i, j]
                sat_tiles[i, j] = self.satmap_tensor[:, rb:re, cb:ce]  # [C, H, W]

        return sat_tiles, nrcs_girdcoord_center



class UavDataset(object):
    def __init__(self,
                 p_uavinfo_json,
                 imgsize2net=224,
                 **kwargs,
                 ):
        # read corresponding uav imgs & mate info
        with open(p_uavinfo_json, "r") as f:
            self.uavinfo_dict = json.load(f)
        self.uav_df = pd.read_csv(self.uavinfo_dict['uavimgs_geocsv_path'])
        uav_names = self.uav_df['filename']
        self.uav_latlons = np.stack([self.uav_df['latitude [decimal degrees]'], self.uav_df['longitude [decimal degrees]']], axis=1)
        uavimgs_dir = self.uavinfo_dict['uavimgs_dir']
        self.uavimg_paths = [os.path.join(uavimgs_dir, name) for name in uav_names]
        self.split_uav_dataset()

        # config transform for uavimgs
        self.imgsize2net = imgsize2net
        # self.uav_transform_train = mk_transform(
        #     imgsize2net=self.imgsize2net,
        #     mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
        #     rand_affine=False,
        #     rand_crop=False,center_crop=True,
        #     rand_erase=False,) #this for debug
        self.uav_transform_train = mk_transform(
            imgsize2net=self.imgsize2net,
            mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
            rand_affine=True, affine_para={'degrees': 180, 'translate': (0, 0), 'scale': (1.0, 1.1), 'shear': 5},
            rand_crop=True,center_crop=False,
            rand_erase=True) #this for train
        self.uav_transform_test = mk_transform(
            imgsize2net=self.imgsize2net,
            mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std'],
            center_crop=True)


    """funcs about handling uavings:"""
    def split_uav_dataset(self, train_radio=0.9):
        # split the dataset for train/val/test
        n_train = int(len(self.uavimg_paths) * train_radio)

        self.uavimg_paths_train = self.uavimg_paths[:n_train]
        self.uav_lonlats_train = self.uav_latlons[:n_train]

        self.uavimg_paths_test = self.uavimg_paths[n_train:]
        self.uav_lonlats_test = self.uav_latlons[n_train:]

        if hasattr(self, 'uav_nrcs'):
            self.uav_nrcs_train = self.uav_nrcs[:n_train]
            self.uav_nrcs_test = self.uav_nrcs[n_train:]



class WingtraDataset(Dataset):
    def __init__(self,
                 sat_dataset = None,
                 uav_dataset = None,
                 n_rand2sample_per_pos=1,
                 stage='train',
                 **kwargs,
                 ):
        self.sat_dataset = sat_dataset
        self.uav_dataset = uav_dataset
        # self.uav_dataset.uav_nrcs = sat_dataset.transform_latlon_to_nrc(uav_dataset.uav_latlons,dtype=np.float32)
        self.uav_dataset.uav_nrcs = sat_dataset.transfrom_georc_to_nrc(uav_dataset.uav_latlons,source_epsg_code=4326,dtype=np.float32)
        self.uav_dataset.split_uav_dataset()
        self.n_rand2sample_per_pos = n_rand2sample_per_pos
        self.switch_stage(stage)

        #mission for crop satimg
        # sat_center_geocoord_list = []
        # for id,uav_nrc in enumerate(self.uav_dataset.uav_nrcs):
        #     sat_img = self.sat_dataset.crop_satimg_by_nrc(uav_nrc,imgsize2crop=1000,type='np')
        #     uav_name = os.path.basename(self.uav_dataset.uavimg_paths[id])
        #     sat_center_geocoord = self.sat_dataset.transform_nrc_to_georc(uav_nrc[None,:],target_espg_code=self.sat_dataset.epsg_code,dtype=np.float32)
        #     sat_center_geocoord_list.append(sat_center_geocoord)
        #     sat_img.save(f'/home/data/zwk/data_uavimgs_wingtra/Zurich/satimgs_h1k/{uav_name[:-4]}.png')
        # sat_center_geocoord_list = np.array(sat_center_geocoord_list).squeeze()
        # np.save(f'/home/data/zwk/data_uavimgs_wingtra/Zurich/satimgs_h1k_center_georc_espg{self.sat_dataset.epsg_code}.npy', sat_center_geocoord_list)


    """funcs about getting item:"""
    def _getitem_train(self, index):
        #1.select uavimg
        uavimg = Image.open(self.uav_dataset.uavimg_paths_train[index])
        uavimg_q = self.uav_dataset.uav_transform_train(uavimg)
        uav_nrc = self.uav_dataset.uav_nrcs_train[index]

        #2.select positive satimg, u can choose to use transfomr or not:
        # sat_rc_pos = uav_rc + (np.random.rand(2)-0.5) * self.halfimg_radius_rc * 0.5
        sat_nrc_pos = uav_nrc
        satimg = self.sat_dataset.crop_satimg_by_nrc(sat_nrc_pos,type='np')
        satimg_pos = self.sat_dataset.sat_transform_train(satimg)

        #3.select random satimgs,clip sat_img_tensor if necessary
        if self.n_rand2sample_per_pos > 0:
            sat_nrcs_rand = self.sat_dataset.mk_rand_nrcs(self.n_rand2sample_per_pos)
            satimgs_rand = torch.stack([self.sat_dataset.sat_transform_train(self.sat_dataset.crop_satimg_by_nrc(sat_nrc,type='np')) for sat_nrc in sat_nrcs_rand])
            return uavimg_q, satimg_pos, satimgs_rand, torch.tensor(uav_nrc), torch.tensor(sat_nrc_pos), torch.tensor(sat_nrcs_rand)
        else:
            return uavimg_q, satimg_pos, torch.tensor(uav_nrc), torch.tensor(sat_nrc_pos)

        # debug for vis:
        # dir2save = '/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_zurich_vis'
        # from matplotlib import pyplot as plt
        # plt.imshow(uavimg)
        # plt.savefig(f'{dir2save}/uav_zurich_id{index}_res05m.png')
        # plt.close()
        # plt.imshow(satimg)
        # plt.savefig(f'{dir2save}/sat_zurich_id{index}_res05m.png')

    def _getitem_test(self, index):
        uavimg = Image.open(self.uav_dataset.uavimg_paths_test[index])
        uavimg_q = self.uav_dataset.uav_transform_test(uavimg)
        uav_nrc = self.uav_dataset.uav_nrcs_test[index]
        return uavimg_q, uav_nrc


    def switch_stage(self,stage='train'):
        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
            self.dataset_len = len(self.uav_dataset.uavimg_paths_train)
        else:
            self._getitem = self._getitem_test
            self.dataset_len = len(self.uav_dataset.uavimg_paths_test)


    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return  self.dataset_len


    """funcs for debug:"""
    def denormalize_img(self, img, mode='sat'):
        if img.device.type != 'cpu':
            img = img.cpu()
        if mode == 'sat':
            img_np = img * torch.tensor(self.sat_dataset.satinfo_dict['std_normalized'])[:,None,None]+torch.tensor(self.sat_dataset.satinfo_dict['mean_normalized'])[:,None,None]
        else:
            img_np = img * torch.tensor(self.uav_dataset.uavinfo_dict['std'])[:, None, None] + torch.tensor(self.uav_dataset.uavinfo_dict['mean'])[:, None, None]
        img_np = img_np.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return img_np


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    uavimg_q, satimg_pos, satimgs_rand, uav_rc, sat_rc_pos, sat_rcs_rand =  zip(*batch)
    return torch.stack(uavimg_q),torch.stack(satimg_pos),torch.cat(satimgs_rand,dim=0),torch.stack(uav_rc),torch.stack(sat_rc_pos),torch.cat(sat_rcs_rand,dim=0)


def make_dataloader_wingtra(opt):
    sat_dataset = SatDataset(
        p_satinfo_json=opt.p_satinfo_json,
        satimgsize2crop=opt.satimgsize2crop,
        imgsize2net=opt.imgsize2net,
    )
    uav_dataset = UavDataset(
        p_uavinfo_json=opt.p_uavinfo_json,
        imgsize2net=opt.imgsize2net,
    )

    wingtra_dataset_train = WingtraDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        n_rand2sample_per_pos=opt.n_rand2sample_per_pos,
        stage='train',
    )

    wingtra_dataset_test = WingtraDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        n_rand2sample_per_pos=opt.n_rand2sample_per_pos,
        stage='test',
    )

    if opt.n_rand2sample_per_pos>0:
        dataloader_train = torch.utils.data.DataLoader(wingtra_dataset_train,
                                                       batch_size=opt.batchsize, num_workers=opt.num_worker,
                                                       pin_memory=True, shuffle=True, collate_fn=train_collate_fn,drop_last=False)
    else:
        dataloader_train = torch.utils.data.DataLoader(wingtra_dataset_train,
                                                       batch_size=opt.batchsize, num_workers=opt.num_worker,
                                                       pin_memory=True, shuffle=True, drop_last=False)

    dataloader_test = torch.utils.data.DataLoader(wingtra_dataset_test,
                                                  batch_size=opt.batchsize, num_workers=opt.num_worker,
                                                  pin_memory=True, shuffle=False)
    return dataloader_train, dataloader_test


if __name__ == "__main__":
    uav_dataset = UavDataset(
        p_uavinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info.json',
        stage='train',
    )
    sat_dataset = SatDataset(
        p_satinfo_json='/home/data/zwk/data_uavimgs_wingtra/Zurich/SWISSIMAGE2022_cover_uavimgs_proj2056_edgepix1024_res02m.json'
    )
    wingtra_dataset = WingtraDataset(
        uav_dataset=uav_dataset,
        sat_dataset=sat_dataset,
        n_rand2sample_per_pos=1,
    )







