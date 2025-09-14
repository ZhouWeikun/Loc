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

# 设置随机种子
np.random.seed(2025)  # 你可以选择任意整数作为种子
from .datasets.autoaugment import ImageNetPolicy
from .datasets.queryDataset import RandomErasing

def mk_uav_transfroms_train(opt, uavimgs_info):
    transform_uav_list = []
    transform_uav_list += [transforms.Resize(opt.h, interpolation=3)] #缩放
    if "uav" in opt.rr:
        transform_uav_list += [transforms.RandomRotation(180, interpolation=3)]  #旋转
    transform_uav_list += [
        transforms.CenterCrop(opt.h),  # 中心剪裁
        transforms.RandomHorizontalFlip(),  # 随机翻转
    ]
    if opt.DA:  # 针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list
    if "uav" in opt.ra:  # 随机仿射变换
        transform_uav_list = transform_uav_list +  [transforms.RandomAffine(180)]
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
    # transform_sat_list += [transforms.Resize(opt.h, interpolation=3)]
    if "satellite" in opt.rr:
        transform_sat_list += [transforms.RandomRotation(360, interpolation=3)]  # 旋转+缩放+中心剪裁到512正方形图像
    transform_sat_list += [
        transforms.RandomHorizontalFlip(),
    ]
    if "satellite" in opt.ra:
        transform_sat_list = transform_sat_list + [transforms.RandomAffine(180)]
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
        transform_sat_list += [transforms.RandomRotation(180, interpolation=3)]  # 旋转+缩放+中心剪裁到512正方形图像
    transform_sat_list += [
        transforms.RandomHorizontalFlip(),
    ]
    if "satellite" in opt.ra:
        transform_sat_list +=  [transforms.RandomAffine(180)]
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

from sklearn.neighbors import NearestNeighbors
from joblib import parallel_backend


def qurey_label_fm_gallery_rc(sat_rcs_gallery,uav_rcs_query,halfimg_radius_rc):
    # gallery_clean = np.ascontiguousarray(np.asarray(sat_rcs_gallery, dtype=np.float64))
    # query_clean = np.ascontiguousarray(np.asarray(uav_rcs_query, dtype=np.float64))
    #
    # # 初始化模型
    # sat_knn = NearestNeighbors(n_jobs=1)
    #
    # # 首先，用全部 gallery 数据 fit
    # try:
    #     print("--- 正在使用全部 gallery 数据进行 fit... ---")
    #     sat_knn.fit(gallery_clean)
    #     print("Fit 成功！现在开始逐行诊断 query 数据...")
    # except Exception as e:
    #     print(f"错误：Fit 过程就已失败！请检查 gallery 数据。错误信息: {e}")
    #
    # print("\n--- 开始诊断：逐行测试 query 数据 ---")
    # for i in range(len(query_clean)):
    #     row_to_query = query_clean[i:i + 1]  # 保持数据是二维的
    #     try:
    #         if i % 100 == 0:  # 每100行打印一次进度
    #             print(f"正在测试 Query 行 {i}...")
    #
    #         # 这就是在那段 helper 代码内部执行的调用
    #         sat_knn.radius_neighbors(row_to_query, radius=halfimg_radius_rc)
    #
    #     except Exception as e:
    #         print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #         print(f"!!! 问题定位：错误在 QUERY 数据的第 {i} 行触发 !!!")
    #         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #         print(f"\n导致问题的查询数据是: {query_clean[i]}")
    #         print(f"\n底层错误信息: {e}")
    #         print("\n完整的错误追溯信息:")
    #         traceback.print_exc()
    #         # 找到了问题，停止循环
    #         break
    # else:  # 如果 for 循环正常结束（没有 break），则执行这里
    #     print("\n--- 诊断完成：所有 query 数据均测试通过，未引发错误 ---")

    sat_knn = NearestNeighbors(n_jobs=-1)
    with parallel_backend('threading'):
        sat_knn.fit(sat_rcs_gallery)
    distrc_sats2uav, uav_labels_query = sat_knn.radius_neighbors(uav_rcs_query,radius=float(halfimg_radius_rc))

    # # --- 1. 数据准备 (Faiss 偏好 float32) ---
    # gallery_data = np.asarray(sat_rcs_gallery, dtype=np.float32)
    # query_data = np.asarray(uav_rcs_query, dtype=np.float32)
    # # 获取数据的维度
    # d = gallery_data.shape[1]
    #
    # import faiss
    # # --- 2. 构建 Faiss 索引 ---
    # # IndexFlatL2 是最基础的索引，它进行精确的L2距离（欧氏距离）搜索
    # # 这与 NearestNeighbors 的默认行为完全相同
    # index = faiss.IndexFlatL2(d)
    #
    # # 3. 执行半径搜索 (【关键修正】)
    # # 使用 faiss.range_search() 函数，而不是 index 的方法
    # # L2 距离的平方根是欧氏距离，所以半径的平方是距离的阈值
    # # Faiss 的 range_search 使用的是距离的平方，所以我们需要传入 radius*radius
    # radius_squared = halfimg_radius_rc * halfimg_radius_rc
    # lims, D, I = index.range_search(
    #     x=query_data,  # 查询向量
    #     thresh=radius_squared,  # 距离平方的阈值
    # )
    #
    # # --- 4. 解析结果 (使其格式与 scikit-learn 类似) ---
    # distances = []
    # indices = []
    # for i in range(len(query_data)):
    #     # 获取第 i 个查询点的结果在 D 和 I 中的切片范围
    #     start = lims[i]
    #     end = lims[i+1]
    #     distances.append(D[start:end])
    #     indices.append(I[start:end])

    return distrc_sats2uav,uav_labels_query

import geo_trans
class DatasetDsalad(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 p_uavinfo_json,
                 opt = None,
                 stage = 'train',
                 satimgsize2net = 224,
                 satimgsize2clip = 224,
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        # version 0,read tif as np:
        # self.sat_img = cv2.imread(self.satinfo_dict['tif_path']).astype(np.float32)[...,::-1] / 255.
        # self.tif_h= self.sat_img.shape[0]
        # self.tif_w = self.sat_img.shape[1]
        # version 1,read tif as Image:
        self.sat_img = Image.open(self.satinfo_dict['tif_path']) #hwc
        self.tif_h= self.sat_img.height
        self.tif_w = self.sat_img.width
        self.tif_hw_max = np.max([ self.tif_h, self.tif_w])

        #2.set other vals about satimg
        #  assign the vars about sampling imgs
        self.n_satrand_per_uav = opt.n_satrand_per_uav #n_satimgs to sample for every uavimg,loss_mat = [n_batchsize, n_batchsize*(n_satrand_per_uav+1)]
        self.uavrcs_sampled = multiprocessing.Manager().list()  # 线程/进程安全的列表
        #  define the normed_halfimg_radius_rc that means positive samples
        self.satimgsize2net = satimgsize2net
        self.halfimg_radius_rc = self.satimgsize2net // 2. / self.tif_hw_max #about 30m
        self.halfimg_radius_meter = self.get_halfimg_radius_meter()

        #  define the vals about how to clip,todo:change
        # self.tif_edge_pixs = 224
        # self.row_min_normalized = self.tif_edge_pixs / self.tif_hw_max
        # self.col_min_normalized = self.row_min_normalized
        # self.row_max_normalized = (self.tif_h- self.tif_edge_pixs) / self.tif_hw_max
        # self.col_max_normalized = (self.tif_w - self.tif_edge_pixs) / self.tif_hw_max
        # self.row_width_normed =  self.row_max_normalized -  self.row_min_normalized
        # self.col_width_normed =  self.col_max_normalized -  self.col_min_normalized
        
        self.satimgsize2clip = satimgsize2clip
        self.tif_edge_pixs = self.satimgsize2clip//2
        # define the sample range of normed row&col of satimg center,version 0:
        # The position of the inner boundary
        self.nr2sample_range = (0.5 - self.tif_h/self.tif_hw_max/2 + self.tif_edge_pixs / self.tif_hw_max,
                           0.5 + self.tif_h/self.tif_hw_max/2 - self.tif_edge_pixs / self.tif_hw_max)
        self.nc2sample_range = (0.5 - self.tif_w/self.tif_hw_max/2 + self.tif_edge_pixs / self.tif_hw_max,
                           0.5 + self.tif_w/self.tif_hw_max/2 - self.tif_edge_pixs / self.tif_hw_max)
        self.nr_h =  self.nr2sample_range[1]-self.nr2sample_range[0]
        self.nc_w = self.nc2sample_range[1] - self.nc2sample_range[0]
        # The position of the outer boundary
        self.nr_tiftop =  self.nr2sample_range[0] - self.tif_edge_pixs / self.tif_hw_max  #the normalized row corresponding to the first row
        self.nc_tifleft =  self.nc2sample_range[0] - self.tif_edge_pixs / self.tif_hw_max  #the normalized column corresponding to the first column

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

        self.uav_rcs = self.latlon_to_nrc(self.uav_latlons)
        if 'rotdeg_fm_north_anticlock' in uav_infodict.keys():
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

    # def latlon_to_rowcol(self, lat_lons: np.ndarray,):
    #     col = (lat_lons[..., 1] - self.geo_transform[0]) / self.geo_transform[1]
    #     col_normed = col / self.tif_hw_max
    #     row = (lat_lons[..., 0] - self.geo_transform[3]) / self.geo_transform[-1]
    #     row_normed = row / self.tif_hw_max
    #     return np.stack([row_normed, col_normed], axis=-1)

    def latlon_to_nrc(self, lat_lons: np.ndarray, dtype=np.float32):  # transfrom the latlon to the normalized coordinate sys of the sat_map
        col = (lat_lons[..., 1] - self.geo_transform[0]) / self.geo_transform[1]
        row = (lat_lons[..., 0] - self.geo_transform[3]) / self.geo_transform[-1]
        col_normed = col / self.tif_hw_max + self.nc_tifleft
        row_normed = row / self.tif_hw_max + self.nr_tiftop

        return np.stack([row_normed, col_normed], axis=-1).astype(dtype)

    def rowcol_to_latlon(self,sat_row_cols_normlized):
        if type(sat_row_cols_normlized) == torch.Tensor:
            sat_row_cols_normlized = sat_row_cols_normlized.detach().cpu().numpy()
        latlons = np.zeros_like(sat_row_cols_normlized)
        latlons[..., 1] = sat_row_cols_normlized[..., 1] * self.tif_hw_max * self.geo_transform[1] + self.geo_transform[0]
        latlons[..., 0] = sat_row_cols_normlized[..., 0] * self.tif_hw_max * self.geo_transform[-1] + self.geo_transform[3]
        return latlons

    # def clip_satimg_fm_rc(self, rc, type = 'tensor'):
    #     # rc[0] = torch.clamp(rc[0], min=self.row_min_normalized, max=self.row_max_normalized)
    #     # rc[1] = torch.clamp(rc[1], min=self.col_min_normalized, max=self.col_max_normalized)
    # 
    #     row = int(rc[0]*self.tif_hw_max)
    #     col = int(rc[1]*self.tif_hw_max)
    #     col_begin = col - self.satimgsize2net / 2
    #     col_end = col + self.satimgsize2net / 2
    #     row_begin = row - self.satimgsize2net / 2
    #     row_end = row + self.satimgsize2net / 2
    # 
    #     if type =='tensor':
    #         sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
    #     else:
    #         # sat_img = self.sat_img[int(row_begin):int(row_end),int(col_begin):int(col_end),:]
    #         sat_img = self.sat_img.crop((int(col_begin),int(row_begin),int(col_end),int(row_end)))
    #     return sat_img
    
    def clip_satimg_fm_nrc(self, nrc, type = 'tensor'):
        row = int((nrc[0]-self.nr_tiftop) * self.tif_hw_max)
        col = int((nrc[1]-self.nc_tifleft) * self.tif_hw_max)
        halfimg_width =  self.satimgsize2net / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width
        if col_end - col_begin < self.satimgsize2clip or row_end-row_begin < self.satimgsize2clip:
            print('error')

        if type =='tensor':
            sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            # sat_img = self.sat_img[i_nt(row_begin):int(row_end),int(colbegin):int(col_end),:]
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

    # def mk_a_randrc(self,uav_rc):
    #     r = self.row_min_normalized + np.random.rand()* self.row_width_normed
    #     c = self.col_min_normalized + np.random.rand()* self.col_width_normed
    #     while np.linalg.norm(uav_rc - np.array([r,c])) < self.halfimg_radius_rc:
    #         r = self.row_min_normalized + np.random.rand()* self.row_width_normed
    #         c = self.col_min_normalized + np.random.rand()* self.col_width_normed
    #     return np.array([r,c])
    
    def mk_a_rand_nrc(self):
        nr = self.nr_h * np.random.rand() + self.nr2sample_range[0]
        nc = self.nc_w * np.random.rand() + self.nc2sample_range[0]
        return np.array([nr,nc],dtype=np.float32)

    # def mk_randrcs(self,n_rand):
    #     randrcs = np.random.rand(n_rand, 2)
    #     r = self.row_min_normalized + randrcs[:, 0] * self.row_width_normed
    #     c = self.col_min_normalized + randrcs[:, 1] * self.col_width_normed
    #     randrcs = np.stack([r, c], axis=1)
    #     return randrcs
    
    def mk_rand_nrcs(self,n_rand):
        rand_nrcs = np.random.rand(n_rand, 2)
        nr = self.nr_h * rand_nrcs[:, 0] + self.nr2sample_range[0]
        nc = self.nc_w * rand_nrcs[:, 1] + self.nc2sample_range[0]
        rand_nrcs = np.stack([nr, nc], axis=1)
        return rand_nrcs

    def index_uavid_fm_rc(self,rc):
        uav_id = np.argmin(np.linalg.norm([self.uav_rcs-rc],axis=-1)[0])
        return uav_id

    def split_sat_unifrom(self,size2clip=224,overlap=0.,device=None):
        #get all sattiles:
        sat_rows = self.tif_h
        sat_cols = self.tif_w
        n_pix_perstep = int(size2clip*(1.-overlap))
        n_colsteps = 1. * (sat_cols-size2clip) / n_pix_perstep
        n_rowsteps = 1. * (sat_rows-size2clip) / n_pix_perstep

        r_ids_begin = np.linspace(start=0, stop=int(n_rowsteps-1), num=int(n_rowsteps),dtype=np.int16)
        c_ids_begin = np.linspace(start=0, stop=int(n_colsteps-1), num=int(n_colsteps),dtype=np.int16)
        rcs_begincoord = np.stack(np.meshgrid(c_ids_begin,r_ids_begin),axis=-1)[:,:,::-1] * n_pix_perstep
        rcs_endcoord = rcs_begincoord+np.array([size2clip,size2clip])
        rcs_girdcoord = np.concatenate([rcs_begincoord,rcs_endcoord],axis=-1)

        # rc2nrc:
        rcs_girdcoord_center = (rcs_begincoord+rcs_endcoord)/2
        nrcs_girdcoord_center = rcs_girdcoord_center / self.tif_hw_max + np.array([self.nr_tiftop,self.nc_tifleft])[None,None,:]
        
        #debug for vis
        # from matplotlib import pyplot as plt
        # plt.close('all')
        # plt.scatter(nrcs_girdcoord_center.reshape(-1,2)[:,1],nrcs_girdcoord_center.reshape(-1,2)[:,0])
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/gridcenter.jpg')

        self.sat_img_tensor = self.sat_img_tensor.to(device) #if not self.sat_img_tensor.is_cuda and torch.cuda.is_available() else self.sat_img_tensor
        n, m, _ = rcs_girdcoord.shape
        sat_tiles = torch.empty((n, m, 3, size2clip, size2clip), device=self.sat_img_tensor.device)
        # 向量化提取所有窗口
        for i in range(n):
            for j in range(m):
                rb, cb, re, ce = rcs_girdcoord[i, j]
                sat_tiles[i, j] = self.sat_img_tensor[:, rb:re, cb:ce] # [C, H, W]

        return sat_tiles,nrcs_girdcoord_center

    def _getitem_train(self,index):
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_train[index])
        uavimg_q = self.uav_transforms_train(uavimg)
        uav_rc = self.uav_rcs_train[index]

        #2.select positive satimg, u can choose to use transfomr or not:
        # sat_rc_pos = uav_rc + (np.random.rand(2)-0.5) * self.halfimg_radius_rc * 0.5
        sat_rc_pos = uav_rc
        satimg_pos = self.clip_satimg_fm_nrc(sat_rc_pos)
        #3.select random satimgs,clip sat_img_tensor
        sat_rcs_rand = self.mk_rand_nrcs(self.n_satrand_per_uav)
        satimgs_rand = torch.stack([self.clip_satimg_fm_nrc(sat_rc) for sat_rc in sat_rcs_rand])
        satimgs_rand = self.sat_transforms_train(satimgs_rand)

        #debug for vis
        # from matplotlib import pyplot as plt
        # plt.imshow(uavimg)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/uav_q.jpg')
        # plt.close()
        # satimg = self.denormalize_satimg(satimg_pos)
        # plt.imshow(satimg)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/sat_p.jpg')

        return uavimg_q,satimg_pos,satimgs_rand,torch.tensor(uav_rc),torch.tensor(sat_rc_pos),torch.tensor(sat_rcs_rand)

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
    uavimg_q, satimg_pos, satimgs_rand, uav_rc, sat_rc_pos, sat_rcs_rand =  zip(*batch)
    return torch.stack(uavimg_q),torch.stack(satimg_pos),torch.cat(satimgs_rand,dim=0),torch.stack(uav_rc),torch.stack(sat_rc_pos),torch.cat(sat_rcs_rand,dim=0)

def make_dataloader_train(opt):
    # custom Dataset
    image_dataset = DatasetDsalad(opt.p_satinfo_json,opt.p_uavinfo_json,opt) #Dataloader_University is a class that assign the __getitem__() func
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,num_workers=opt.num_worker, pin_memory=True,shuffle=True,collate_fn=train_collate_fn)
    return dataloader

def make_dataloader(opt, stage='train', dataset = None):
    if dataset is None:
        dataset = DatasetDsalad(opt.p_satinfo_json, opt.p_uavinfo_json, opt, stage)

    if stage=='train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,num_workers=opt.num_worker,
                                                 pin_memory=True,shuffle=True,collate_fn=train_collate_fn,drop_last=True)
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