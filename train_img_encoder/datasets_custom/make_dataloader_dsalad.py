from torch.utils.data import Dataset
import os

# import matplotlib
# matplotlib.use('TKAgg')

import pandas as pd
import json

# import faiss
# from sklearn.neighbors import NearestNeighbors
# from torchvision.transforms.v2 import Transform

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS#todo:
# from torchvision.transforms import InterpolationMode
from PIL import Image

from torchvision import transforms
import torch
import numpy as np

import multiprocessing
import sys

# 设置随机种子
np.random.seed(2025)  # 你可以选择任意整数作为种子

new_module_path = '/train_img_encoder/datasets_custom'
if new_module_path not in sys.path:
    sys.path.insert(0, new_module_path) # 将新路径插入到最前面，优先级最高
from datasets.autoaugment import ImageNetPolicy
from datasets.queryDataset import RandomErasing
# from datasets_custom.datasets import ImageNetPolicy
# from datasets_custom.datasets import RandomErasing

def _mean_fill_from_stats(mean_vals):
    if mean_vals is None:
        return 0
    mean_arr = np.asarray(mean_vals, dtype=float)
    if mean_arr.ndim == 0:
        val = float(mean_arr)
        return int(round(val * 255.0)) if val <= 1.0 else int(round(val))
    if mean_arr.max() <= 1.0:
        mean_arr = mean_arr * 255.0
    return tuple(int(round(x)) for x in mean_arr.tolist())

def mk_uav_transfroms_train(opt, uavimgs_info):
    transform_uav_list = []
    uav_fill = _mean_fill_from_stats(uavimgs_info.get('mean'))
    transform_uav_list += [transforms.Resize(opt.h, interpolation=3)] #缩放
    if "uav" in opt.rr:
        transform_uav_list += [transforms.RandomRotation(360, interpolation=3, fill=uav_fill)]  #旋转
    transform_uav_list += [
        transforms.CenterCrop(opt.h),  # 中心剪裁
        transforms.RandomHorizontalFlip(),  # 随机翻转
    ]
    if opt.DA:  # 针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list
    if "uav" in opt.ra:  # 随机仿射变换
        transform_uav_list = transform_uav_list +  [transforms.RandomAffine(180, fill=uav_fill)]
    if "uav" in opt.re:  # 随机擦除
        transform_uav_list = transform_uav_list +  [RandomErasing(probability=opt.erasing_p)]
    if "uav" in opt.cj:  # 随机颜色扰乱
        transform_uav_list = transform_uav_list +  [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)]
    transform_uav_list += [
        transforms.ToTensor(),
        transforms.Normalize(uavimgs_info['mean'], uavimgs_info['std'])
    ]
    return  transforms.Compose(transform_uav_list)

def mk_sat_transfroms_train(opt,  satimgs_info):
    transform_sat_list = []
    sat_fill = _mean_fill_from_stats(satimgs_info.get('mean'))
    # transform_sat_list += [transforms.Resize(opt.h, interpolation=3)]
    if "satellite" in opt.rr:
        transform_sat_list += [transforms.RandomRotation(360, interpolation=3, fill=sat_fill)]  # 旋转+缩放+中心剪裁到512正方形图像
    transform_sat_list += [
        transforms.RandomHorizontalFlip(),
    ]
    if "satellite" in opt.ra:
        transform_sat_list = transform_sat_list + [transforms.RandomAffine(180, fill=sat_fill)]
    if "satellite" in opt.re:
        transform_sat_list = transform_sat_list +  [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.cj:
        transform_sat_list = transform_sat_list +  [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)]

    transform_sat_list += [
        transforms.ToTensor(),
        transforms.Normalize(satimgs_info['mean'], satimgs_info['std'])
    ]
    return  transforms.Compose(transform_sat_list)

def mk_satensor_transfroms_train(opt):
    transform_sat_list = []
    # transform_sat_list += [transforms.Resize(opt.h, interpolation=3)]
    if "satellite" in opt.rr:
        transform_sat_list += [transforms.RandomRotation(360, interpolation=3, fill=0)]  # 旋转+缩放+中心剪裁到512正方形图像
    transform_sat_list += [
        transforms.RandomHorizontalFlip(),
    ]
    if "satellite" in opt.ra:
        transform_sat_list +=  [transforms.RandomAffine(180, fill=0)]
    if "satellite" in opt.re:
        # transform_sat_list = transform_sat_list +  [RandomErasing(probability=opt.erasing_p)]
        transform_sat_list += [transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random"  # 随机填充（在归一化范围内生成随机值）
        )]
    # if "satellite" in opt.cj:
    #     transform_sat_list = transform_sat_list +  [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)]

    return  transforms.Compose(transform_sat_list)

def mk_uav_transfroms_test(opt,uavimgs_info=None):
    transform_uav = transforms.Compose([
        transforms.Resize(opt.h, interpolation=3),
        transforms.CenterCrop(opt.h),
        transforms.ToTensor(),
        transforms.Normalize(uavimgs_info['mean'], uavimgs_info['std'])
    ])
    return transform_uav

def mk_sat_transfroms_test(opt,satimgs_info=None):
    transform_sat = transforms.Compose([
        # transforms.Resize(opt.h, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(satimgs_info['mean'], satimgs_info['std'])
    ])
    return transform_sat

def qlabel_fm_gallery_rc(sat_rcs_gallery,uav_rcs_query,halfimg_radius_rc):
    from sklearn.neighbors import NearestNeighbors
    from joblib import parallel_backend
    sat_knn = NearestNeighbors(n_jobs=-1)
    with parallel_backend('threading'):
        sat_knn.fit(sat_rcs_gallery)
    distrc_sats2uav, uav_labels_query = sat_knn.radius_neighbors(uav_rcs_query,radius=halfimg_radius_rc)
    return distrc_sats2uav,uav_labels_query

import geo_trans
class DatasetDsalad(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 p_uavinfo_json,
                 opt = None,
                 stage = 'train',
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        # read tif as np
        # self.sat_img = cv2.imread(self.satinfo_dict['tif_path']).astype(np.float32)[...,::-1] / 255.
        # self.sat_img_h = self.sat_img.shape[0]
        # self.sat_img_w = self.sat_img.shape[1]
        self.sat_img = Image.open(self.satinfo_dict['tif_path']) #hwc
        self.sat_img_h = self.sat_img.height
        self.sat_img_w = self.sat_img.width
        self.tif_size = np.max([self.sat_img_h,self.sat_img_w])

        #2.set other vals about satimg
        #  assign the vars about sampling imgs
        self.n_satrand_per_uav = opt.n_satrand_per_uav #n_satimgs to sample for every uavimg,loss_mat = [n_batchsize, n_batchsize*(n_satrand_per_uav+1)]
        # self.n_nav2sample = 32
        self.uavrcs_sampled = multiprocessing.Manager().list()  # 线程/进程安全的列表
        #  define the normed_halfimg_radius_rc that means positive samples
        self.satimgsize2net = 224
        self.halfimg_radius_rc = self.satimgsize2net // 2. / self.tif_size #about 30m
        self.halfimg_radius_met = self.get_halfimg_radius_meter()

        #  define the vals about how to clip
        self.img_edge_pixs = 224
        self.row_min_normalized = self.img_edge_pixs / self.tif_size
        self.col_min_normalized = self.row_min_normalized
        self.row_max_normalized = (self.sat_img_h - self.img_edge_pixs) / self.tif_size
        self.col_max_normalized = (self.sat_img_w - self.img_edge_pixs) / self.tif_size
        self.row_width_normed =  self.row_max_normalized -  self.row_min_normalized
        self.col_width_normed =  self.col_max_normalized -  self.col_min_normalized

        #3.read uavimgs' mate info
        with open(p_uavinfo_json, "r") as f:
            uav_infodict = json.load(f)
        self.uavinfo_dict = uav_infodict
        self.uav_df = pd.read_csv(uav_infodict['uavimgs_geocsv_path'])
        uav_names = self.uav_df['Name']
        self.uav_latlons = np.stack([self.uav_df['Latitude'],self.uav_df['Longitude']],axis=1)
        uavimgs_dir = self.uavinfo_dict['uavimgs_dir']
        # self.uavimg_paths = [os.path.join(uavimgs_dir,name) for name in uav_names]
        line = self.uav_df['Line']
        self.uavimg_paths = [os.path.join(uavimgs_dir,line[i],name) for i,name in enumerate(uav_names)] #assuming that the imgs are stored by lines
        self.uav_rcs = self.latlon_to_rowcol(self.uav_latlons)
        # self.east2uav_clockwise = self.uav_df['east2head_clockwise_fm_rcdiff'].values #the target_deg=-90deg
        # self.head2north_anticlockwise = self.east2uav_clockwise + 90
        if 'rotdeg_fm_north_anticlock' in self.uav_df.keys():
            self.rotdeg_fm_north_anticlock = self.uav_df['rotdeg_fm_north_anticlock'].values #the target_deg=-90deg

        #set vals about spliting training and testing sets
        self.split_uav_dataset()
        self.uav_transforms_train = mk_uav_transfroms_train(opt, self.satinfo_dict)
        self.sat_transforms_test = mk_sat_transfroms_test(opt,self.uavinfo_dict)
        self.sat_transforms_train = mk_satensor_transfroms_train(opt)
        self.uav_transforms_test = mk_uav_transfroms_test(opt, self.uavinfo_dict)
        self.sat_img_tensor = self.sat_transforms_test(self.sat_img)

        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
            self.dataset_len = len(self.uavimg_paths_train)
        else:
            self._getitem = self._getitem_test
            self.dataset_len = len(self.uavimg_paths_test)

    def get_halfimg_radius_meter(self):
        diff_lat =  self.satimgsize2net // 2. * np.abs(self.geo_transform[-1])
        diff_lon =  self.satimgsize2net // 2. * self.geo_transform[1]
        diff_met_lat = geo_trans.diff_lat_to_meter(diff_lat)
        diff_met_lon = geo_trans.diff_lon_to_meter(diff_lon,self.geo_transform[3])
        meter_radius = 0.5*(diff_met_lon+diff_met_lat)
        return meter_radius
    
    def split_uav_dataset(self,interval_sample=False):
        #split the dataset for train/val/test
        n_train = int(len(self.uavimg_paths) * 0.9)
        test_per_train = 5
        if interval_sample:
            self.uavimg_paths_train = [x for i, x in enumerate(self.uavimg_paths) if i % test_per_train != 0]
            self.uav_lonlats_train = [x for i, x in enumerate(self.uav_latlons) if i % test_per_train != 0]
            self.uav_rcs_train = torch.stack([x for i, x in enumerate(self.uav_rcs) if i % test_per_train != 0])

            self.uavimg_paths_test =  self.uavimg_paths[::test_per_train]
            self.uav_lonlats_test =  self.uav_latlons[::test_per_train]
            self.uav_rcs_test = self.uav_rcs[::test_per_train].contiguous()
        else:
            self.uavimg_paths_train = self.uavimg_paths[:n_train]
            self.uav_lonlats_train = self.uav_latlons[:n_train]
            self.uav_rcs_train = self.uav_rcs[:n_train]

            self.uavimg_paths_test = self.uavimg_paths[n_train:]
            self.uav_lonlats_test = self.uav_latlons[n_train:]
            self.uav_rcs_test = self.uav_rcs[n_train:]
        # _,uav_labels_test = self.sat_index.search(self.uav_rcs_test,1)
        # _,uav_labels_test = self.sat_knn.radius_neighbors(self.uav_rcs_test, radius=self.sat_halfimg_radius_rc_normalized)
        # self.uav_labels_test = uav_labels_test.squeeze()
        # _,uav_labels_train = self.sat_index.search(self.uav_rcs_train,1)
        # self.uav_labels_train = uav_labels_train.squeeze()
        # self.n_uav_test = len(self.uav_labels_test)
        # self.img_paths_test = self.uavimg_paths_test+self.sat_img_paths
        # self.lon_lats_test = torch.concat([self.uav_rcs_test,self.sat_rcs],dim=0)

    def set_sat_transform(self):
        """
        sat_transform for training and testing sets are the same
        """
        #config sat_transforms
        self.sat_transforms = transforms.Compose([
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),  # normalize value to [-1,1]
            # transforms.Resize((cliped_h, cliped_w), interpolation=3),
            # transforms.RandomRotation(180),
        ])

    def denormalize_satimg(self,satimg):
        satimgs_np = satimg * torch.tensor(self.satinfo_dict['std'])[:,None,None]+torch.tensor(self.satinfo_dict['mean'])[:,None,None]
        satimgs_np = satimgs_np.permute(1, 2, 0).numpy()
        satimgs_np = np.clip(satimgs_np * 255, 0, 255).astype(np.uint8)
        return satimgs_np

    def set_uav_transform_train(self):
        """
        sat_transform for training and testing sets are the same
        """
        #config sat_transforms
        self.uav_transforms_train = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.CenterCrop(self.satimgsize2net),
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std']),  # normalize value to [-1,1]
            # transforms.Resize((cliped_h, cliped_w), interpolation=3),
        ])
        #todo: setting the uav_transforms fm opts

    def latlon_to_rowcol(self, lat_lons: np.ndarray,):
        col = (lat_lons[..., 1] - self.geo_transform[0]) / self.geo_transform[1]
        col_normed = col / self.tif_size
        row = (lat_lons[..., 0] - self.geo_transform[3]) / self.geo_transform[-1]
        row_normed = row / self.tif_size
        return np.stack([row_normed, col_normed], axis=-1)

    def rowcol_to_latlon(self,sat_row_cols_normlized):
        if type(sat_row_cols_normlized) == torch.Tensor:
            sat_row_cols_normlized = sat_row_cols_normlized.detach().cpu().numpy()
        latlons = np.zeros_like(sat_row_cols_normlized)
        latlons[..., 1] = sat_row_cols_normlized[..., 1] * self.tif_size * self.geo_transform[1] + self.geo_transform[0]
        latlons[..., 0] = sat_row_cols_normlized[..., 0] * self.tif_size * self.geo_transform[-1] + self.geo_transform[3]
        return latlons

    def clip_satimg_fm_rc(self, rc, type = 'tensor'):
        # rc[0] = torch.clamp(rc[0], min=self.row_min_normalized, max=self.row_max_normalized)
        # rc[1] = torch.clamp(rc[1], min=self.col_min_normalized, max=self.col_max_normalized)

        row = int(rc[0]*self.tif_size)
        col = int(rc[1]*self.tif_size)
        col_begin = col - self.satimgsize2net / 2
        col_end = col + self.satimgsize2net / 2
        row_begin = row - self.satimgsize2net / 2
        row_end = row + self.satimgsize2net / 2

        if type =='tensor':
            sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            # sat_img = self.sat_img[int(row_begin):int(row_end),int(col_begin):int(col_end),:]  # chw for sat_img_tensor
            sat_img = self.sat_img.crop((int(col_begin),int(row_begin),int(col_end),int(row_end)))
        return sat_img

    def clip_satimg_fm_latlon(self, latlon):
        row = (latlon[0] - self.geo_transform[3]) / self.geo_transform[-1]
        col = (latlon[1] - self.geo_transform[0]) / self.geo_transform[1]
        row = int(row)
        col = int(col)

        col_begin = col - self.satimgsize2net / 2
        col_end = col + self.satimgsize2net / 2
        row_begin = row - self.satimgsize2net / 2
        row_end = row + self.satimgsize2net / 2

        sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        return sat_img

    def mk_a_randrc(self,uav_rc):
        r = self.row_min_normalized + np.random.rand()* self.row_width_normed
        c = self.col_min_normalized + np.random.rand()* self.col_width_normed
        while np.linalg.norm(uav_rc - np.array([r,c])) < self.halfimg_radius_rc:
            r = self.row_min_normalized + np.random.rand()* self.row_width_normed
            c = self.col_min_normalized + np.random.rand()* self.col_width_normed
        return np.array([r,c])

    def mk_randrcs(self):
        # randrcs = np.random.rand(self.n_satrand_per_uav*2,2)
        # r = self.row_min_normalized + randrcs[:,0] * self.row_width_normed
        # c = self.col_min_normalized + randrcs[:,1] * self.col_width_normed
        # randrcs = np.stack([r,c],axis=1)
        # distmat = np.linalg.norm(randrcs[:,None,:]-np.array(self.uavrcs_sampled)[None,...],axis=-1)
        # print(f'self.uavrcs_sampled.len={len(self.uavrcs_sampled)}')
        # mask = (distmat > self.halfimg_radius_rc).prod(axis=1)
        # return randrcs[mask][:self.n_satrand_per_uav]

        randrcs = np.random.rand(self.n_satrand_per_uav, 2)
        r = self.row_min_normalized + randrcs[:, 0] * self.row_width_normed
        c = self.col_min_normalized + randrcs[:, 1] * self.col_width_normed
        randrcs = np.stack([r, c], axis=1)
        return randrcs

    def index_uavid_fm_rc(self,rc):
        uav_id = np.argmin(np.linalg.norm([self.uav_rcs-rc],axis=-1)[0])
        return uav_id

    def clip_sat_unifrom(self,size2clip=224,overlap=0.):
        #get all sattiles:
        sat_rows = self.sat_img_h
        sat_cols = self.sat_img_w
        n_pix_perstep = int(size2clip*(1.-overlap))
        n_colsteps = 1. * (sat_cols-size2clip) / n_pix_perstep
        n_rowsteps = 1. * (sat_rows-size2clip) / n_pix_perstep

        r_ids_begin = np.linspace(start=0, stop=int(n_rowsteps-1), num=int(n_rowsteps),dtype=np.int16)
        c_ids_begin = np.linspace(start=0, stop=int(n_colsteps-1), num=int(n_colsteps),dtype=np.int16)
        rcs_begincoord = np.stack(np.meshgrid(c_ids_begin,r_ids_begin),axis=-1)[:,:,::-1] * n_pix_perstep
        rcs_endcoord = rcs_begincoord+np.array([size2clip,size2clip])
        rcs_girdcoord = np.concatenate([rcs_begincoord,rcs_endcoord],axis=-1)
        rcs_girdcoord_center = (rcs_begincoord+rcs_endcoord)/2
        rcs_girdcoord_center = rcs_girdcoord_center / self.tif_size

        # self.sat_img_tensor = self.sat_img_tensor.cuda() if not self.sat_img_tensor.is_cuda and torch.cuda.is_available() else self.sat_img_tensor
        n, m, _ = rcs_girdcoord.shape
        sat_tiles = torch.empty((n, m, 3, size2clip, size2clip), device=self.sat_img_tensor.device)
        # 向量化提取所有窗口
        for i in range(n):
            for j in range(m):
                rb, cb, re, ce = rcs_girdcoord[i, j]
                sat_tiles[i, j] = self.sat_img_tensor[:, rb:re, cb:ce] # [C, H, W]

        return sat_tiles,rcs_girdcoord_center

    def _getitem_train(self,index):
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_train[index])
        uavimg_q = self.uav_transforms_train(uavimg)
        uav_rc = self.uav_rcs_train[index]
        # self.uavrcs_sampled.append(uav_rc)

        #2.select positive satimg
        satimg_pos = self.clip_satimg_fm_rc(uav_rc)
        satimg_pos = self.sat_transforms_train(satimg_pos)
        #3.select random satimgs
        # clip sat_img_tensor
        satrcs_rand = self.mk_randrcs()
        satimgs_rand = torch.stack([self.clip_satimg_fm_rc(satrc) for satrc in satrcs_rand])
        satimgs_rand = self.sat_transforms_train(satimgs_rand)

        # exp23
        # from matplotlib import pyplot as plt
        # satimg_np = self.denormalize_satimg(satimgs_rand[0])
        # satimg_pos2vis = self.denormalize_satimg(satimg_pos)
        # plt.imshow(satimg_pos2vis)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad/datasets_custom/satimg_pos2vis.png')
        # satimg_pos_trans = self.sat_transforms_train(satimg_pos)
        # satimg_pos_trans2vis = self.denormalize_satimg(satimg_pos_trans)
        # plt.imshow(satimg_pos_trans2vis)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad/datasets_custom/satimg_pos_trans2vis.png')
        # plt.imshow(satimg_np)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad/datasets_custom/exp23.png')
        # satimgs_rand = np.stack([self.clip_satimg_fm_rc(satrc,type='np') for satrc in satrcs_rand])
        # satimgs_rand = self.sat_transforms_train(satimgs_rand)
        return uavimg_q,satimg_pos,satimgs_rand,torch.tensor(uav_rc),torch.tensor(satrcs_rand)

    def _getitem_test(self,index):#todo:Needs to be refined according to external needs -> the test_func()
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_test[index])
        uavimg_q = self.uav_transforms_test(uavimg)
        uav_rc = self.uav_rcs_test[index]

        return uavimg_q,uav_rc

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


def train_collate_fn(batch):#todo:this is ready for training, another for testing
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    uavimg_q, satimg_pos, satimgs_rand, uav_rc, satrcs_rand =  zip(*batch)
    return torch.stack(uavimg_q),torch.stack(satimg_pos),torch.concatenate(satimgs_rand,dim=0),torch.stack(uav_rc),torch.concatenate(satrcs_rand,dim=0)

def make_dataloader_train(opt):
    # custom Dataset
    image_dataset = DatasetDsalad(opt.p_satinfo_json,opt.p_uavinfo_json,opt) #Dataloader_University is a class that assign the __getitem__() func
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,num_workers=opt.num_worker, pin_memory=True,shuffle=True,collate_fn=train_collate_fn)
    return dataloader

def make_dataloader(opt, stage='train',dataset=None):
    if dataset is None:
        dataset = DatasetDsalad(opt.p_satinfo_json, opt.p_uavinfo_json, opt, stage)

    if stage=='train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,num_workers=opt.num_worker,
                                                 pin_memory=True,shuffle=True,collate_fn=train_collate_fn)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, num_workers=opt.num_worker,
                                                 pin_memory=True, shuffle=False)
    return dataloader

if __name__=="__main__":
    p_satinfo_json = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/satimgs_xiangan_xmu_info.json'
    p_uavinfo_json = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json'
    p_uavloc_csv = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/uavimgs_geoloc_xiangan_xmu_dji.csv'
    from train_img_encoder.trainer_org import get_parse
    opt = get_parse()
    stage ='train'
    dataset = DatasetDsalad(p_satinfo_json,p_uavinfo_json,opt,stage)
    dataset.split_sat_unifrom(overlap=0.5)
    for item in dataset:
        print('x')
