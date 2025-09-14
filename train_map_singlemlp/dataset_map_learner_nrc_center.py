from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import geo_trans
import json
from torchvision import transforms
import torch

class SatMapDataset(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 satimgsize2clip=240,
                 satimgsize2net=224,
                 rand_rot_sat=False,
                 stage = 'train',
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        # version 0,read tif as np:
        # self.sat_img = cv2.imread(self.satinfo_dict['tif_path']).astype(np.float32)[...,::-1] / 255.
        #  self.tif_h = self.sat_img.shape[0]
        #  self.tif_w = self.sat_img.shape[1]
        # version 1,read tif as Image:
        self.sat_img = Image.open(self.satinfo_dict['tif_path']) #hwc
        self.tif_h = self.sat_img.height
        self.tif_w = self.sat_img.width
        self.tif_hw_max = np.max([ self.tif_h, self.tif_w])

        #2.set other vals about satimg
        #  define the normed_halfimg_radius_rc that means positive samples
        self.satimgsize2net = satimgsize2net
        self.halfimg_radius_rc = self.satimgsize2net // 2. / self.tif_hw_max #about 30m
        self.halfimg_radius_meter = self.get_halfimg_radius_meter()

        self.satimgsize2clip = satimgsize2clip
        self.tif_edge_pixs = satimgsize2clip/2

        #  define the sample range of normed row&col of satimg center,version 1:
        # self.nr2sample_min = self.tif_edge_pixs / self.tif_hw_max
        # self.nc2sample_min = self.nr2sample_min
        # self.nr2sample_max = (self.tif_h- self.tif_edge_pixs) / self.tif_hw_max
        # self.nc2sample_min = (self.tif_w - self.tif_edge_pixs) / self.tif_hw_max
        # self.nr2sample_range = [self.nr2sample_min, self.nr2sample_max]
        # self.nc2sample_range = [self.nc2sample_min, self.nc2sample_max]
        # self.nr2sample_h =  self.nr2sample_max - self.nr2sample_min
        # self.nc2sample_w =  self.nc2sample_max - self.nc2sample_min
        # self.nr_tiftop =  0  #the normalized row corresponding to the first row
        # self.nc_tifleft = 0  #the normalized column corresponding to the first column
        #  define the sample range of normed row&col of satimg center,version 2:
        self.nr2sample_range = (0.5 - self.tif_h/self.tif_hw_max/2 + self.tif_edge_pixs / self.tif_hw_max,
                           0.5 + self.tif_h/self.tif_hw_max/2 - self.tif_edge_pixs / self.tif_hw_max)
        self.nc2sample_range = (0.5 - self.tif_w/self.tif_hw_max/2 + self.tif_edge_pixs / self.tif_hw_max,
                           0.5 + self.tif_w/self.tif_hw_max/2 - self.tif_edge_pixs / self.tif_hw_max)
        self.nr2sample_h =  self.nr2sample_range[1]-self.nr2sample_range[0]
        self.nc2sample_w = self.nc2sample_range[1] - self.nc2sample_range[0]
        self.nr_tiftop =  self.nr2sample_range[0] - self.tif_edge_pixs / self.tif_hw_max  #the normalized row corresponding to the first row
        self.nc_tifleft =  self.nc2sample_range[0] - self.tif_edge_pixs / self.tif_hw_max  #the normalized column corresponding to the first column


        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),
        ]
        transform = transforms.Compose(transforms_list)
        self.sat_img_tensor = transform(self.sat_img)
        self.set_sat_transform()
        self.switch_stage(stage)


    def clip_satimg_fm_nrc(self, nrc, type = 'tensor'):
        row = int((nrc[0]-self.nr_tiftop) * self.tif_hw_max)
        col = int((nrc[1]-self.nc_tifleft) * self.tif_hw_max)

        halfimg_width =  self.satimgsize2net / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width

        if type =='tensor':
            sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            # sat_img = self.sat_img[i_nt(row_begin):int(row_end),int(colbegin):int(col_end),:]
            sat_img = self.sat_img.crop((int(col_begin),int(row_begin),int(col_end),int(row_end)))
        return sat_img


    def mk_a_rand_nrc(self):
        nr = self.nr_h * np.random.rand() + self.nr2sample_range[0]
        nc = self.nc_w * np.random.rand() + self.nc2sample_range[0]
        return np.array([nr,nc],dtype=np.float32)


    def mk_rand_nrcs(self,n_rand):
        rand_nrcs = np.random.rand(n_rand, 2)
        nr = self.nr_h * rand_nrcs[:, 0] + self.nr2sample_range[0]
        nc = self.nc_w * rand_nrcs[:, 1] + self.nc2sample_range[0]
        rand_nrcs = np.stack([nr, nc], axis=1)
        return rand_nrcs


    def mk_coord_grid(self, split_by='hw',ovrelap=0.5,hw=(8,10),delta_pixs=224,random=False,dtype=torch.float32):
        if not hasattr(self, 'meshgrid'):
            if split_by == 'hw':
                n_grid_h = hw[0]
                n_grid_w = hw[1]
            elif split_by == 'overlap':
                n_grid_h = int(self.tif_h / ((1-ovrelap)*self.satimgsize2clip))
                n_grid_w = int(self.tif_w / ((1-ovrelap)*self.satimgsize2clip))
            elif split_by == 'delta_pixs':
                n_grid_h = delta_pixs
                n_grid_w = delta_pixs

            y_coords = torch.linspace(self.nr2sample_range[0], self.nr2sample_range[1], steps=n_grid_h)
            x_coords = torch.linspace(self.nc2sample_range[0], self.nc2sample_range[1], steps=n_grid_w)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # 'ij' 表示 y 行, x 列的顺序
            self.meshgrid = torch.stack([yy, xx]).permute(1,2,0)
            self.n_grid_h = n_grid_h
            self.n_grid_w = n_grid_w
            self.grid_cell_radius = 0.25*(torch.diff(y_coords).mean() + torch.diff(x_coords).mean())
            #debug
            # from matplotlib import pyplot as plt
            # meshgrid = self.meshgrid.detach().numpy().reshape(-1,2)
            # plt.scatter(meshgrid[:,1], meshgrid[:,0], c='r')
            # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/grid.png')

        if random:
            rand_delta = (torch.rand(self.meshgrid.shape,dtype=dtype)-0.5)*torch.tensor([self.y_delta,self.x_delta],dtype=dtype)
            gred2ret = rand_delta + self.meshgrid
            return gred2ret
        else:
            return self.meshgrid

    def latlon_to_nrc(self, lat_lons: np.ndarray, dtype=np.float32):  # transfrom the latlon to the normalized coordinate sys of the sat_map
        col = (lat_lons[..., 1] - self.geo_transform[0]) / self.geo_transform[1]
        row = (lat_lons[..., 0] - self.geo_transform[3]) / self.geo_transform[-1]
        col_normed = col / self.tif_hw_max + self.nc_tifleft
        row_normed = row / self.tif_hw_max + self.nr_tiftop

        return np.stack([row_normed, col_normed], axis=-1).astype(dtype)


    def get_halfimg_radius_meter(self):
        diff_lat =  self.satimgsize2net // 2. * np.abs(self.geo_transform[-1])
        diff_lon =  self.satimgsize2net // 2. * self.geo_transform[1]
        diff_met_lat = geo_trans.diff_lat_to_meter(diff_lat)
        diff_met_lon = geo_trans.diff_lon_to_meter(diff_lon,self.geo_transform[3])
        meter_radius = 0.5*(diff_met_lon+diff_met_lat)
        return meter_radius


    def set_sat_transform(self,random_rot=False):
        """
        sat_transform for training and testing sets are the same
        """
        if self.satimgsize2clip != 224 :
            transforms_list = [ transforms.Resize((224,224), interpolation=3) ]
        else:
            transforms_list = []

        if random_rot:
            transforms_list.append(transforms.RandomRotation(180))

        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),
        ]

        self.sat_transform = transforms.Compose(transforms_list)


    def switch_stage(self, stage='train'):
        self.stage = stage
        if stage == 'train':
            self.dataset_len = int((self.tif_h * self.tif_w) / (self.satimgsize2clip ** 2) * 10)
        else:
            self.dataset_len = int((self.tif_h * self.tif_w) / (self.satimgsize2clip ** 2) * 2)


    def __getitem__(self,index):
        sat_nrc_rand = self.mk_a_rand_nrc()
        satimg_rand = self.sat_transform(self.clip_satimg_fm_nrc(sat_nrc_rand,type='np'))
        return sat_nrc_rand, satimg_rand

    def __len__(self):
        return  self.dataset_len


import pandas as pd
import os
class UAVDataset(Dataset):
    def __init__(self,
                 p_uavinfo_json,
                 uavimgsize2net=224,
                 stage='train',
                 uavimgsize2clip=224,
                 **kwargs,
                 ):
        #2.set other vals about satimg
        self.uavimgsize2net = uavimgsize2net

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
        # self.uav_rcs = self.mk_nrcs_fm_latlons(self.uav_latlons,satmap_dataset)
        # self.rotdeg_fm_north_anticlock = self.uav_df['rotdeg_fm_north_anticlock'].values #the target_deg=-90deg

        #set vals about spliting training and testing sets
        self.uav_transform_train = self.mk_uav_transform_train()
        self.uav_transforms_test = self.mk_uav_transfrom_test()
        self.split_uav_dataset()
        self.switch_stage(stage)


    def mk_uav_transform_train(self): #todo:modify to sample random
        uav_transform_train = transforms.Compose([
            transforms.Resize(self.uavimgsize2net, interpolation=3),
            transforms.RandomCrop(self.uavimgsize2net),
            transforms.RandomRotation(180),
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=self.uavinfo_dict['mean'], std=self.uavinfo_dict['std']),  # normalize value to [-1,1]
        ])
        return uav_transform_train


    def mk_uav_transfrom_test(self):
        transform_uav = transforms.Compose([
            transforms.Resize(self.uavimgsize2net, interpolation=3),
            transforms.CenterCrop(self.uavimgsize2net),
            transforms.ToTensor(),
            transforms.Normalize(self.uavinfo_dict['mean'], self.uavinfo_dict['std'])
        ])
        return transform_uav


    def split_uav_dataset(self, train_radio=0.9):
        #split the dataset for train/val/test
        n_train = int(len(self.uavimg_paths) * train_radio)

        self.uavimg_paths_train = self.uavimg_paths[:n_train]
        self.uav_lonlats_train = self.uav_latlons[:n_train]
        # self.uav_rcs_train = self.uav_rcs[:n_train]

        self.uavimg_paths_test = self.uavimg_paths[n_train:]
        self.uav_lonlats_test = self.uav_latlons[n_train:]
        # self.uav_rcs_test = self.uav_rcs[n_train:]


    def mk_nrcs_fm_latlons(self, satmap_dataset):
        self.uav_rcs = satmap_dataset.latlon_to_nrc(self.uav_latlons)

        # split_uav_dataset in nrcs
        n_train = int(len(self.uavimg_paths) * 0.9)
        self.uav_rcs_train = self.uav_rcs[:n_train]
        self.uav_rcs_test = self.uav_rcs[n_train:]


    def switch_stage(self,stage='train'):
        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
            self.dataset_len = len(self.uavimg_paths_train)
        else:
            self._getitem = self._getitem_test
            self.dataset_len = len(self.uavimg_paths_test)


    def _getitem_train(self,index):
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_train[index])
        uavimg_q = self.uav_transform_train(uavimg)
        uav_rc = self.uav_rcs_train[index]
        return   torch.tensor(uav_rc),uavimg_q


    def _getitem_test(self,index):#todo:Needs to be refined according to external needs -> the test_func()
        #1.select uavimg
        uavimg = Image.open(self.uavimg_paths_test[index])
        uavimg_q = self.uav_transforms_test(uavimg)
        uav_rc = self.uav_rcs_test[index]
        return torch.tensor(uav_rc),uavimg_q

    def __getitem__(self, index):
        return self._getitem(index)


    def __len__(self):
        return  self.dataset_len
