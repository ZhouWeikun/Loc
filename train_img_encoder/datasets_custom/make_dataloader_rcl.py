import torchvision.transforms
from torch.utils.data import Dataset
import os

# import matplotlib
from matplotlib import pyplot as plt
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

# 设置随机种子
np.random.seed(2025)  # 你可以选择任意整数作为种子

class Dataset_RCL(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 p_uavinfo_json,
                 **kwargs,
                 ):
        #read corresponding satellite image & mate info
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
        # trans np to tensor
        # self.sat_img_tensor = (self.sat_img - self.satinfo_dict['mean']) / self.satinfo_dict['std']
        # self.sat_img_tensor = torch.from_numpy(self.sat_img_tensor).permute(2,0,1) #chw
        self.set_sat_transform()
        self.sat_img_tensor = self.sat_transforms(self.sat_img)

        #set other sat vals
        self.n_sat_random = 126
        #  define the normed_rc_radius that means positive samples
        self.satimgsize2net = 224
        self.rc_radius = self.satimgsize2net // 2. / self.tif_size #about 30m
        #  define the vals about how to clip
        self.img_edge_pixs = 224
        self.row_min_normalized = self.img_edge_pixs / self.tif_size
        self.col_min_normalized = self.row_min_normalized
        self.row_max_normalized = (self.sat_img_h - self.img_edge_pixs) / self.tif_size
        self.col_max_normalized = (self.sat_img_w - self.img_edge_pixs) / self.tif_size
        self.row_width_normed =  self.row_max_normalized -  self.row_min_normalized
        self.col_width_normed =  self.col_max_normalized -  self.col_min_normalized

        # self.col_min_normalized = self.img_edge_pixs / self.sat_img_w
        # self.col_max_normalized = (self.sat_img_w - self.img_edge_pixs) / self.sat_img_w

        #read uavimgs' mate info
        with open(p_uavinfo_json, "r") as f:
            uav_infodict = json.load(f)
        self.uavinfo_dict = uav_infodict
        self.uav_df = pd.read_csv(uav_infodict['uavimgs_geocsv_path'])
        uav_names = self.uav_df['Name']
        self.uav_latlons = np.stack([self.uav_df['Latitude'],self.uav_df['Longitude']],axis=1)
        uavimgs_dir = self.uavinfo_dict['uavimgs_dir']
        self.p_uavimgs = [os.path.join(uavimgs_dir,name) for name in uav_names]
        self.uav_rcs_normed = self.latlon_to_rowcol(self.uav_latlons)

        self.east2uav_clockwise = self.uav_df['east2head_clockwise_fm_rcdiff'].values #the target_deg=-90deg
        self.head2north_anticlockwise = self.east2uav_clockwise + 90
        self.set_uav_transform_train()

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

    def set_uav_transform_train(self):
        """
        sat_transform for training and testing sets are the same
        """
        #config sat_transforms
        self.uav_transforms_train = transforms.Compose([
            transforms.CenterCrop(self.satimgsize2net),
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std']),  # normalize value to [-1,1]
            # transforms.Resize((cliped_h, cliped_w), interpolation=3),
            # transforms.RandomRotation(180),
        ])

    def latlon_to_rowcol(self, lat_lons: np.ndarray,):
        col = (lat_lons[..., 1] - self.geo_transform[0]) / self.geo_transform[1]
        col_normed = col / self.tif_size
        row = (lat_lons[..., 0] - self.geo_transform[3]) / self.geo_transform[-1]
        row_normed = row / self.tif_size
        return np.stack([row_normed, col_normed], axis=-1)

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
        # if self.use_sat_transform:
        #     sat_img = self.sat_img[int(row_begin):int(row_end),int(col_begin):int(col_end),:]  # hwc for cv2
        #     sat_img = self.sat_transforms(sat_img)
        # else:
        #     sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end), int(col_begin):int(col_end),]  # chw for sat_img_tensor
        #     #todo:generating the sat_patches according to the overlap for testing
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

    def mk_rc_normed_random(self,uav_rc_normed):
        r = self.row_min_normalized + np.random.rand()* self.row_width_normed
        c = self.col_min_normalized + np.random.rand()* self.col_width_normed
        while np.linalg.norm(uav_rc_normed - np.array([r,c])) < self.rc_radius:
            r = self.row_min_normalized + np.random.rand()* self.row_width_normed
            c = self.col_min_normalized + np.random.rand()* self.col_width_normed
        return np.array([r,c])

    def index_uav_by_rc(self,rc):
        uav_id = np.argmin(np.linalg.norm([self.uav_rcs_normed-rc],axis=-1)[0])
        return uav_id


    def __getitem__(self, index):
        #1.select uavimg
        index= 229
        uavimg = Image.open(self.p_uavimgs[index])
        # uavimg = uavimg.rotate(self.head2north_anticlockwise[index])
        # plt.imshow(uavimg)
        # plt.show()
        uavimg_t = self.uav_transforms_train(uavimg)
        uav_latlon = self.uav_latlons[index]
        uav_rc_normed = self.latlon_to_rowcol(uav_latlon)
        #2.select positive satimg
        satimg_t_atuav = self.clip_satimg_fm_rc(torch.tensor(uav_rc_normed))
        # satimg_t_atuav = self.clip_satimg_fm_latlon(uav_latlon)
        # imgs2ret = torch.stack([uavimg_t,satimg_t_atuav])
        #3.randomly select satimg
        imgs2ret = [uavimg_t,satimg_t_atuav]
        rcs2ret = [uav_rc_normed,uav_rc_normed]
        for i in np.arange(0,self.n_sat_random):
            print(i)
            rc_random = self.mk_rc_normed_random(uav_rc_normed)
            satimg_t_random = self.clip_satimg_fm_rc(rc_random)
            imgs2ret.append(satimg_t_random)
            rcs2ret.append(rc_random)
            # imgs2ret = torch.concatenate([imgs2ret,satimg_t_random[None,...]])
            # imgs2ret = torch.concatenate([imgs2ret,])
        # imgs2ret = torch.concatenate([imgs2ret,torch.stack(satimgs_t_random)])
        # rcs_random = torch.concatenate([uav_rc_normed,uav_rc_normed,])

        imgs2ret = torch.stack(imgs2ret)
        rcs2ret = torch.from_numpy(np.stack(rcs2ret))
        return imgs2ret,rcs2ret

    def __len__(self):
        return len(self.p_uavimgs)


def make_dataloader_train(opt):
    # custom Dataset
    image_dataset = Dataset_RCL(opt.p_satinfo_json,opt.p_uavinfo_json) #Dataloader_University is a class that assign the __getitem__() func
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,num_workers=opt.num_worker, pin_memory=True,shuffle=True)
    return dataloader

if __name__=="__main__":
    p_satinfo_json = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/satimgs_xiangan_xmu_info.json'
    p_uavinfo_json = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json'
    p_uavloc_csv = '/home/data/zwk/dataset_xiangan/dataset_xmu_meta/uavimgs_geoloc_xiangan_xmu_dji.csv'
    dataset = Dataset_RCL(p_satinfo_json,p_uavinfo_json)
    for item in dataset:
        print('x')