from torch.utils.data import Dataset
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None # 禁用检查，或者设置为一个足够大的整数，例如 1_000_000_000
import numpy as np
import geo_trans
import json
from torchvision import transforms
import torch
import cv2

SATE_LENGTH = 24576
TILE_LENGTH = 256


def sate2loc(tile_zoom, offset, tile_x, tile_y):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_center_x = (tile_pix * (tile_x+1/2+offset/TILE_LENGTH)) * 0.45
    loc_center_y = (tile_pix * (tile_y+1/2+offset/TILE_LENGTH)) * 0.45
    loc_tl_x = (tile_pix * (tile_x+offset/TILE_LENGTH)) * 0.45
    loc_tl_y = (tile_pix * (tile_y+offset/TILE_LENGTH)) * 0.45
    return loc_center_x, loc_center_y, loc_tl_x, loc_tl_y


def zoom_xys(loc_centers,zoom=None):
    """
    loc_center: with shape = [n,2]
    """
    if zoom == 6.6:
        scale = 1./0.45
    else:
        tile_pix = SATE_LENGTH / (2 ** zoom)
        scale = 256. / tile_pix * 1. / 0.45
    loc_centers = loc_centers * scale
    return loc_centers


def gen_centers2sample(p_sat_masked='/home/data/zwk/dataset_Game4Loc/satellite_4_masked.png',
                       size2crop=128,
                       ):
    satmap = cv2.imread(p_sat_masked)
    satmap = cv2.cvtColor(satmap, cv2.COLOR_BGR2RGB)
    satmap_hw_max = np.max([satmap.shape[:2]])
    n_grid_h = satmap.shape[0] // size2crop
    n_grid_w = satmap.shape[1] // size2crop

    nrs_bounary = torch.linspace(0, n_grid_h * size2crop / satmap_hw_max, steps=n_grid_h + 1)
    ncs_bounary = torch.linspace(0, n_grid_w * size2crop / satmap_hw_max, steps=n_grid_w + 1)
    nrs_center = (nrs_bounary[:-1] + nrs_bounary[1:]) / 2.
    ncs_center = (ncs_bounary[:-1] + ncs_bounary[1:]) / 2.

    yy_c, xx_c = torch.meshgrid(nrs_center, ncs_center, indexing='ij')  # 'ij' 表示 y 行, x 列的顺序
    nrc_center_meshgrid = torch.stack([yy_c, xx_c]).permute(1, 2, 0)
    delta_gird_h = torch.diff(nrs_bounary).mean()
    delta_gird_w = torch.diff(ncs_bounary).mean()
    yy_b, xx_b = torch.meshgrid(nrs_bounary, ncs_bounary, indexing='ij')
    nrc_boundary_meshgrid = torch.stack([yy_b, xx_b]).permute(1, 2, 0)

    sat_tiles = []
    for i in range(nrc_boundary_meshgrid.shape[0] - 1):
        for j in range(nrc_boundary_meshgrid.shape[1] - 1):
            rcb = (nrc_boundary_meshgrid[i, j] * satmap_hw_max).detach().numpy().astype(int)
            rce = (nrc_boundary_meshgrid[i + 1, j + 1] * satmap_hw_max).detach().numpy().astype(int)
            rb, cb, re, ce = rcb[0], rcb[1], rce[0], rce[1]
            sat_tiles.append(satmap[rb:re, cb:ce, :])
    sat_tiles = np.stack(sat_tiles) / 255.
    sat_tiles_tensor = torch.from_numpy(sat_tiles).float()
    isnot_sea_matrix = (sat_tiles_tensor.sum(dim=(1, 2, 3)) > 1e-3).reshape(n_grid_h, n_grid_w)

    dict2ret = {
        'nrc_center_meshgrid': nrc_center_meshgrid,
        'nrc_boundary_meshgrid': nrc_boundary_meshgrid,
        'isnot_sea_matrix': isnot_sea_matrix,
        'grid_cell_hw': (delta_gird_h, delta_gird_w),
    }
    return dict2ret


class SatMapDataset(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 imgsize2net=224,
                 rand_rot_sat=False,
                 stage = 'train',
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            self.sat_info_dict = json.load(f)
        self.imgsize2net = imgsize2net

        self.satmap = Image.open(self.sat_info_dict['path']) #hwc
        self.satmap_h= self.satmap.height
        self.satmap_w = self.satmap.width
        self.satmap_hw_max = np.max([ self.satmap_h, self.satmap_w])
        self.level2sample_min,self.level2sample_max = 6,7

        satmap2tenstor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.sat_info_dict['mean'], std=self.sat_info_dict['std']),
        ])
        self.satmap_tensor = satmap2tenstor_transform(self.satmap)
        self.set_sat_transform_train()

        self.grid_info_dict = gen_centers2sample()
        self.grid_centers = self.grid_info_dict['nrc_center_meshgrid']
        self.grid_mask = self.grid_info_dict['isnot_sea_matrix']
        self.grid_centers2sample = self.grid_info_dict['nrc_center_meshgrid'][self.grid_info_dict['isnot_sea_matrix']]
        self.grid_cell_hw = self.grid_info_dict['grid_cell_hw']
        self.grid_cell_radius = 0.25*(self.grid_cell_hw[0]+self.grid_cell_hw[1])


    def crop_satmap_fm_nrc(self,nrc,satimgsize2crop=256,type='tensor'):
        row = int(nrc.squeeze()[0] * self.satmap_hw_max)
        col = int(nrc.squeeze()[1] * self.satmap_hw_max)

        halfimg_width = satimgsize2crop / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width

        if type == 'tensor':
            sat_img = self.satmap_tensor[:, int(row_begin):int(row_end),
                      int(col_begin):int(col_end)]  # chw for sat_img_tensor

        else: #type == pil
            sat_img = self.satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))

        return sat_img


    def mk_rand_nrcs(self,n):
        rand_indices = torch.randint(low=0, high=self.grid_centers2sample.shape[0], size=(n,), device=self.grid_centers2sample.device)
        nrcs_center = self.grid_centers2sample[rand_indices]
        offsets_rand = (torch.rand(n, 2) - 0.5) * torch.tensor([self.grid_cell_hw[0], self.grid_cell_hw[1]])/2.
        return nrcs_center + offsets_rand


    def sample_sats_in_rect(self, nrc_topleft, nrc_buttonright, n2sample_h=128, n2sample_w=128, satimgsize2clip=224, type2clip='tensor'):
        halfimg_width = satimgsize2clip/2
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
            sat_tiles = torch.empty((n2sample_h, n2sample_w, 3, satimgsize2clip, satimgsize2clip),
                                    device=self.tif_img_tensor.device)
            for i in range(rows_begin.shape[0]):
                for j in range(cols_begin.shape[0]):
                    rb, cb, re, ce = rows_begin[i], cols_begin[j], rows_end[i], cols_end[j]
                    sat_tiles[i, j] = self.tif_img_tensor[:, rb:re, cb:ce]  # [C, H, W]
        else:
            sat_tiles = np.zeros((n2sample_h, n2sample_w, 3, satimgsize2clip, satimgsize2clip)).astype(np.float32)
            for i in range(rows_begin.shape[0]):
                for j in range(cols_begin.shape[0]):
                    rb, cb, re, ce = rows_begin[i], cols_begin[j], rows_end[i], cols_end[j]
                    sat_tiles[i, j] = self.tif_img[:, rb:re, cb:ce]  # [C, H, W]

        return sat_tiles,nrc_center_meshgrid


    def set_sat_transform_train(self):
        self.color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            # transforms.RandomAutocontrast(p=0.3),
            # transforms.RandomGrayscale(p=1.0)
            ])
        self.affine_transform = transforms.RandomAffine(
                degrees=180,  # 旋转范围 -45 到 +45 度
                translate=(0, 0),  # 水平和垂直平移最大 10%
                scale=(1.0, 1.0),  # 缩放范围 80% 到 120%
                shear=5,  # 剪切范围 -5 到 +5 度
                interpolation=transforms.InterpolationMode.BILINEAR,  # 双线性插值
                fill=0  # 填充黑色
            )
        self.rand_erasing = transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=1)
        self.da_transform = transforms.Compose([self.rand_erasing,self.color_transform,self.affine_transform])
        self.resize_transform = transforms.Resize(self.imgsize2net)
        # self.sat_transform_train = transforms.Compose([self.rand_erasing,self.color_transform,self.da_transform])
        self.sat_transform_train = transforms.Compose([self.resize_transform,self.affine_transform])


    def denormalize_satimg(self,satimg):
        # if satimg.device.type != 'cpu':
        #     satimg = satimg.cpu()
        satimgs_np = satimg * torch.tensor(self.sat_info_dict['std'])[:,None,None]+torch.tensor(self.sat_info_dict['mean'])[:,None,None]
        satimgs_np = satimgs_np.permute(1, 2, 0).numpy()
        satimgs_np = np.clip(satimgs_np * 255, 0, 255).astype(np.uint8)
        return satimgs_np


    def __getitem__(self,index):
        level2sample = np.random.rand()*(self.level2sample_max-self.level2sample_min) + self.level2sample_min
        # level2sample = 0.5*(level2sample_max+level2sample_min)
        tilesize2sample = SATE_LENGTH / (2 ** level2sample)
        nrc = self.mk_rand_nrcs(1)[0]
        satimg_croped = self.crop_satmap_fm_nrc(nrc,tilesize2sample)
        satimg_da = self.sat_transform_train(satimg_croped)

        return nrc, satimg_da


    def __len__(self):
        return  int((self.satmap_h * self.satmap_w) / (self.imgsize2net ** 2) * 4)


import pandas as pd
import os
class UAVDataset(Dataset):
    def __init__(self,
                 p_uavinfo_json,
                 p_pairs_json,
                 imgsize2net=224,
                 stage='train',
                 **kwargs,
                 ):
        with open(p_uavinfo_json, 'r') as f:
            self.uav_info_dict = json.load(f)
        with open(p_pairs_json, 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.imgsize2net = imgsize2net
        self.data_root = os.path.dirname(p_pairs_json)
        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()
        self.pairs_drone_xy = []
        self.pairs_drone_height = []
        self.drone_prefix_mask = []

        mode = 'pos_semipos'
        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            drone_xy = pair_drone2sate['drone_loc_x_y']
            drone_height = pair_drone2sate['drone_metadata']['height']
            # Training with Positive-only data or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']

            drone_img_file = os.path.join(self.data_root, drone_img_dir, drone_img_name)
            self.drone_prefix_mask.append(drone_img_name[:4] )

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(self.data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))
                self.pairs_drone_xy.append([drone_xy])
                self.pairs_drone_height.append([drone_height])

        self.pairs_drone_xy = np.stack(self.pairs_drone_xy).squeeze()
        self.satmap_hw_max = max([self.uav_info_dict['satmap_h'],self.uav_info_dict['satmap_w']])
        self.pairs_drone_nrc = zoom_xys(self.pairs_drone_xy, zoom=float(self.uav_info_dict['satmap_name'].split('_')[1][:-4])) / self.satmap_hw_max
        self.pairs_drone_nrc = np.stack([self.pairs_drone_nrc[:,1],self.pairs_drone_nrc[:,0]]).T
        self.pairs_drone_nrc = torch.tensor(self.pairs_drone_nrc).float()
        self.pairs_drone_height = np.stack(self.pairs_drone_height).squeeze()

        #filtering by the height:
        self.height_range = [100,200]
        self.zoom_range = [7,6]
        height_mask = (self.pairs_drone_height>self.height_range[0])*(self.pairs_drone_height <self.height_range[1])
        #filtering by the drone_name:
        uav_name_range = ['100_','200_']
        def extract_sort_key(row):
            # 从 drone 图像路径中提取文件名
            filename = row[0].split('/')[-1].replace('.png', '')
            mask = uav_name_range[0] in filename or uav_name_range[1] in filename
            return mask
        drone_prefix_mask = np.array([extract_sort_key(row) for row in self.pairs])
        mask =  drone_prefix_mask * height_mask

        self.pairs = np.array(self.pairs)[mask]
        self.pairs_drone_nrc = self.pairs_drone_nrc[mask]
        self.pairs_drone_height = self.pairs_drone_height[mask]
        half_tilesize = (SATE_LENGTH / (2 ** self.zoom_range[0]) + SATE_LENGTH / (2 ** self.zoom_range[1]))*0.25
        self.halfimg_radius_rc = half_tilesize / self.satmap_hw_max

        self.uav_transform_train = self.mk_uav_transform_train()
        self.uav_transform_test = self.mk_uav_transform_test()
        self.switch_stage(stage)

        #对uav_imgs进行重排序
        # def extract_sort_key(row):
        #     # 从 drone 图像路径中提取文件名
        #     filename = row[0].split('/')[-1].replace('.png', '')
        #     parts = filename.split('_')
        #     key = int(parts[0][0]) * 100000 + int(parts[-1])
        #     return key
        #
        # # 假设 self.pairs 是 numpy 数组，形状为 (n, 3)
        # sorted_pairs = np.array(sorted(self.pairs, key=extract_sort_key))


    def mk_uav_transform_train(self): #todo:modify to sample random
        uav_transform_train = transforms.Compose([
            transforms.RandomCrop( min([self.uav_info_dict['img_h'],self.uav_info_dict['img_h']])),
            transforms.Resize(self.imgsize2net, interpolation=3),
            transforms.RandomRotation(180),
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=self.uav_info_dict['mean'], std=self.uav_info_dict['std']),  # normalize value to [-1,1]
        ])
        return uav_transform_train


    def mk_uav_transform_test(self):
        transform_uav = transforms.Compose([
            transforms.CenterCrop(min([self.uav_info_dict['img_h'],self.uav_info_dict['img_h']])),
            transforms.Resize(self.imgsize2net, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(self.uav_info_dict['mean'], self.uav_info_dict['std'])
        ])
        return transform_uav


    def denormalize_uavimg(self,uavimgs):
        if uavimgs.device.type != 'cpu':
            uavimgs = uavimgs.cpu()
        uavimgs_np = uavimgs * torch.tensor(self.uav_info_dict['std'])[:,None,None]+torch.tensor(self.uav_info_dict['mean'])[:,None,None]
        uavimgs_np = uavimgs_np.permute(1, 2, 0).numpy()
        uavimgs_np = np.clip(uavimgs_np * 255, 0, 255).astype(np.uint8)
        return uavimgs_np


    def switch_stage(self,stage='train'):
        self.stage = stage
        if stage=='train':
            self._getitem = self._getitem_train
        else:
            self._getitem = self._getitem_test


    def _getitem_train(self,index):
        #1.select uavimg
        query_img_path, gallery_img_path, positive_weight = self.pairs[index]
        uavimg = Image.open(query_img_path)
        uavimg_q = self.uav_transform_train(uavimg)
        uav_nrc = self.pairs_drone_nrc[index]

        #debug,vis the uavimg
        # uav_da = self.denormalize_uavimg(uavimg_q)
        # from matplotlib import pyplot as plt
        # plt.imshow(uav_da)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/uav_da.png')
        # plt.close()
        return   uav_nrc,uavimg_q


    def _getitem_test(self,index):#todo:Needs to be refined according to external needs -> the test_func()
        #1.select uavimg
        query_img_path, gallery_img_path, positive_weight = self.pairs[index]
        uavimg = Image.open(query_img_path)
        uavimg_q = self.uav_transform_test(uavimg)
        uav_nrc = self.pairs_drone_nrc[index]
        return uav_nrc,uavimg_q


    def __getitem__(self, index):
        return self._getitem(index)


    def __len__(self):
        return  len(self.pairs)

if __name__ == '__main__':
    satmap = SatMapDataset(
        p_satinfo_json='/home/data/zwk/dataset_Game4Loc/satmap_info.json',
        satimgsize2clip=240,
        satimgsize2net=224,
    )
