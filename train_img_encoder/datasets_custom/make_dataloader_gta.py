import os,sys
# 获取当前脚本的绝对路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir # 假设 main.py 就是在项目根目录
# 将项目根目录添加到 Python 的模块搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 插入到最前面，优先搜索

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS#todo:
from torch.utils.data import Dataset
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 禁用检查，或者设置为一个足够大的整数，例如 1_000_000_000
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf

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


def gen_centers2sample(p_sat_masked='/home/data/zwk/dataset_Game4Loc/satellite_4_masked.png', size2crop=128,
                       vis=False):
    satmap = cv2.imread(p_sat_masked)
    satmap = cv2.cvtColor(satmap, cv2.COLOR_BGR2RGB)
    satmap_hw_max = np.max([satmap.shape[:2]])
    n_grid_h = satmap.shape[0] // size2crop
    n_grid_w = satmap.shape[1] // size2crop

    nrs_bounary = torch.linspace(0, n_grid_h * size2crop / satmap_hw_max, steps=n_grid_h + 1)
    ncs_bounary = torch.linspace(0, n_grid_w * size2crop / satmap_hw_max, steps=n_grid_w + 1)
    nrs_center = (nrs_bounary[:-1] + nrs_bounary[1:]) / 2.
    ncs_center = (ncs_bounary[:-1] + ncs_bounary[1:]) / 2.

    yy, xx = torch.meshgrid(nrs_center, ncs_center, indexing='ij')  # 'ij' 表示 y 行, x 列的顺序
    nrc_center_meshgrid = torch.stack([yy, xx]).permute(1, 2, 0)
    delta_gird_h = torch.diff(nrs_bounary).mean()
    delta_gird_w = torch.diff(ncs_center).mean()
    yy, xx = torch.meshgrid(nrs_bounary, ncs_bounary, indexing='ij')
    nrc_boundary_meshgrid = torch.stack([yy, xx]).permute(1, 2, 0)

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
    # sat_tiles_tensor_nosae = sat_tiles_tensor[sat_tiles_tensor.sum(dim=(1,2,3)) >1e-3]
    # channel_means = torch.mean(sat_tiles_tensor_nosae, dim=(0, 1, 2))
    # channel_stds = torch.std(sat_tiles_tensor_nosae, dim=(0, 1, 2))
    # nrcs_rand = (torch.rand(sat_tiles_tensor_nosae.shape[0],2)-0.5) * torch.tensor([delta_gird_h,delta_gird_w])

    if vis:
        # 2. 反归一化网格中心坐标
        # nrc_center_meshgrid 的 yy 是行 (y)，xx 是列 (x)
        # 反归一化后的 x 坐标 (列)
        x_centers_pixel = (nrc_center_meshgrid[:, :, 1] * satmap_hw_max).int().numpy()
        # 反归一化后的 y 坐标 (行)
        y_centers_pixel = (nrc_center_meshgrid[:, :, 0] * satmap_hw_max).int().numpy()
        H_patches, W_patches = isnot_sea_matrix.shape

        # 定义颜色 (BGR 格式)
        SEA_COLOR = (255, 0, 0)  # 蓝色 (BGR)
        LAND_COLOR = (0, 0, 255)  # 红色 (BGR)
        POINT_RADIUS = 5  # 圆点半径
        POINT_THICKNESS = -1  # -1 表示填充整个圆 (实心点)
        # 3. 遍历每个网格中心点并绘制
        for i in range(H_patches):
            for j in range(W_patches):
                center_x = x_centers_pixel[i, j]
                center_y = y_centers_pixel[i, j]

                # 获取当前网格块是否为海面的判断结果
                is_current_patch_sea = isnot_sea_matrix[i, j].item()  # .item() 将 tensor 转为 Python bool

                # 根据判断结果选择颜色
                color = SEA_COLOR if is_current_patch_sea else LAND_COLOR

                # 在图像上绘制圆点
                # cv2.circle(img, center, radius, color, thickness)
                cv2.circle(satmap, (center_x, center_y), POINT_RADIUS, color, POINT_THICKNESS)
                # print(f"绘制点 ({i},{j}) 在 ({center_x},{center_y}), 颜色: {color}") # 调试用

        line_color = (0, 0, 255)
        line_thickness = 1
        # 2. 绘制水平网格线
        for i in range(nrc_boundary_meshgrid.shape[0]):  # 遍历每一行顶点
            for j in range(nrc_boundary_meshgrid.shape[1] - 1):  # 遍历每一段水平线
                # 从 (i, j) 到 (i, j+1) 的水平线
                start_point = (nrc_boundary_meshgrid[i, j] * satmap_hw_max).detach().numpy().astype(int)
                end_point = (nrc_boundary_meshgrid[i, j + 1] * satmap_hw_max).detach().numpy().astype(int)
                cv2.line(satmap, start_point[::-1], end_point[::-1], line_color, line_thickness)
                # print(f"绘制水平线: {start_point} -> {end_point}") # 调试用

        # 3. 绘制垂直网格线
        for j in range(nrc_boundary_meshgrid.shape[1]):  # 遍历每一列顶点
            for i in range(nrc_boundary_meshgrid.shape[0] - 1):  # 遍历每一段垂直线
                # 从 (i, j) 到 (i+1, j) 的垂直线
                start_point = (nrc_boundary_meshgrid[i, j] * satmap_hw_max).detach().numpy().astype(int)
                end_point = (nrc_boundary_meshgrid[i + 1, j] * satmap_hw_max).detach().numpy().astype(int)
                cv2.line(satmap, start_point[::-1], end_point[::-1], line_color, line_thickness)
                # print(f"绘制垂直线: {start_point} -> {end_point}") # 调试用

        # 4. 保存结果图像
        cv2.imwrite('/home/data/zwk/dataset_Game4Loc/satellite_debug.png', satmap)

    return nrc_center_meshgrid[isnot_sea_matrix], nrc_boundary_meshgrid[:-1, :-1][
        isnot_sea_matrix], delta_gird_h, delta_gird_w


class DatasetGTA(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 p_uavinfo_json,
                 opt = None,
                 imgsize2net = 224,
                 satimgsize2crop = 256,
                 stage='train',
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        
        self.satmap = Image.open(self.satinfo_dict['path']) #hwc
        # self.satmap = self.satmap.convert('RGB')
        self.satmap_h= self.satmap.height
        self.satmap_w = self.satmap.width
        self.satmap_hw_max = np.max([ self.satmap_h, self.satmap_w])

        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),
        ]
        satimg_transform = transforms.Compose(transforms_list)
        self.satmap_tensor = satimg_transform(self.satmap)


        #2.set other vals about satimg
        #  define the normed_halfimg_radius_rc that means positive samples
        # opt.imgsize2net = imgsize2net
        # self.halfimg_radius_rc = opt.imgsize2net // 2. / self.satmap_hw_max #about 30m
        self.satimgsize2crop = satimgsize2crop
        self.n_rand2sample_per_pos = opt.n_rand2sample_per_pos

        #  define the vals about how to crop,version 1:
        # self.satmap_edge_pixs = 1290
        # self.nr2sample_min = self.satmap_edge_pixs / self.satmap_hw_max
        # self.nc2sample_min = self.nr2sample_min
        # self.nr2sample_max = (self.satmap_h - self.satmap_edge_pixs) / self.satmap_hw_max
        # self.nc2sample_max = (self.satmap_w - self.satmap_edge_pixs) / self.satmap_hw_max
        # self.nr_tiftop = 0  #the normalized row corresponding to the first row
        # self.nc_tifleft = 0  #the normalized column corresponding to the first column
        # self.nr2sample_range = [self.nr2sample_min, self.nr2sample_max]
        # self.nc2sample_range = [self.nc2sample_min, self.nc2sample_max]
        # self.nr2sample_h =  self.nr2sample_max - self.nr2sample_min
        # self.nc2sample_w =  self.nc2sample_max - self.nc2sample_min

        # prepare the grid_centers2sample
        self.nrc_centers2sample,self.nrc_boundaries2sample_begin,self.delta_grid2sample_h,self.delta_grid2sample_w = gen_centers2sample()

        #from GTADatasetTrain
        with open(p_uavinfo_json, 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = os.path.dirname(p_satinfo_json)
        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()
        self.pairs_drone_xy = []
        self.pairs_drone_height = []

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

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(self.data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))
                self.pairs_drone_xy.append([drone_xy])
                self.pairs_drone_height.append([drone_height])

                # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate['pair_pos_semipos_sate_img_list']
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.pairs_drone_xy = np.stack(self.pairs_drone_xy).squeeze()
        self.pairs_drone_nrc = zoom_xys(self.pairs_drone_xy, zoom=float(self.satinfo_dict['name'].split('_')[1][:-4])) / self.satmap_hw_max
        self.pairs_drone_nrc = np.stack([self.pairs_drone_nrc[:,1],self.pairs_drone_nrc[:,0]]).T
        self.pairs_drone_nrc = torch.tensor(self.pairs_drone_nrc).float()
        self.pairs_drone_height = np.stack(self.pairs_drone_height).squeeze()

        #cofig the transform
        from mk_transforms import mk_uav_transform_train,mk_satensor_transform_train,mk_uav_transform_test,mk_sat_transform_test
        self.uav_transform_train = mk_uav_transform_train(opt,self.satinfo_dict)
        self.sat_tensor_transform_train = mk_satensor_transform_train(opt)
        self.uav_transform_test = mk_uav_transform_test(opt,self.satinfo_dict)
        self.sat_transform_test = mk_sat_transform_test(opt,self.satinfo_dict)
        #reinforce the transform
        # self.color_transform = transforms.Compose([
        #     transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            # transforms.RandomAutocontrast(p=0.3),
            # transforms.RandomGrayscale(p=0.3)
            # ])
        # self.rand_erasing = transforms.RandomErasing(p=0.1, scale=(0.05, 0.1), ratio=(0.3, 3.3), value=1)
        # self.uav_transform_train = transforms.Compose([self.uav_transform_train,self.rand_erasing])
        # self.sat_transform_train = transforms.Compose([self.sat_tensor_transform_train,self.rand_erasing])
        # self.uav_transform_train = mk_gta_transform(opt.imgsize2net,self.satinfo_dict['mean'],self.satinfo_dict['std'])
        # self.sat_transform_train = mk_gta_transform(opt.imgsize2net,self.satinfo_dict['mean'],self.satinfo_dict['std'],p_rot=0)

        #filter the scale:
        self.height_range = [100,200]
        self.zoom_range = [7,6]
        height_mask = (self.pairs_drone_height>self.height_range[0])*(self.pairs_drone_height <self.height_range[1])
        self.pairs = np.array(self.pairs)[height_mask]
        self.pairs_drone_nrc = self.pairs_drone_nrc[height_mask]
        self.pairs_drone_height = self.pairs_drone_height[height_mask]
        half_tilesize = (SATE_LENGTH / (2 ** self.zoom_range[0]) + SATE_LENGTH / (2 ** self.zoom_range[1]))*0.25
        self.halfimg_radius_rc = half_tilesize / self.satmap_hw_max
        # height_levels = np.array([int(os.path.basename(n[0])[0]) for n in self.pairs])

        #debug, vis the rcs on satimg
        # xy_locs = zoom_xys(self.pairs_drone_xy, zoom=float(self.satinfo_dict['name'].split('_')[1][:-4]))
        # 确保坐标是 float 类型以便计算，然后转换为 int
        # satimg = cv2.imread('/home/data/zwk/dataset_Game4Loc/satellite_4.png')
        # satimg = cv2.cvtColor(satimg, cv2.COLOR_BGR2RGB)
        # xy_locs = zoom_xys(self.pairs_drone_xy, zoom=4)
        # points_pixel = np.zeros_like(self.pairs_drone_xy, dtype=int)
        # points_pixel[:, 0] = xy_locs[:, 0]
        # points_pixel[:, 1] = xy_locs[:, 1]
        #
        # # 3. 绘制每个点
        # color = (0, 255, 0)
        # radius = 5
        # thickness = -1
        # for point in points_pixel:
        #     center_x, center_y = point[0], point[1]
        #     cv2.circle(satimg, (center_x, center_y), radius, color, thickness)
        # cv2.imwrite('/home/data/zwk/dataset_Game4Loc/satimg_uav_locs.png', satimg)
        #
        #  # 4. 绘制一个zoom等级下的点
        # color = (0, 0, 255)
        # zoom_mask = [os.path.basename(drone_name[0])[0]=='1' for drone_name in self.pairs ]
        # points_pixel_zoom = points_pixel[zoom_mask]
        # for point in points_pixel_zoom:
        #     center_x, center_y = point[0], point[1]
        #     cv2.circle(satimg, (center_x, center_y), radius, color, thickness)
        # cv2.imwrite('/home/data/zwk/dataset_Game4Loc/satimg_uav_locs_test_1.png', satimg)


    def mk_a_rand_nrc(self):
        random_row_index = torch.randint(0, self.nrc_centers2sample.shape[0], (1,))
        nrc_rand = (torch.rand(2) - 0.5)* torch.tensor([self.delta_grid2sample_h, self.delta_grid2sample_w]) + self.nrc_centers2sample[random_row_index]
        return nrc_rand


    def mk_rand_nrcs(self,n):
        rand_indices = torch.randint(low=0, high=self.nrc_centers2sample.shape[0], size=(n,), device=self.nrc_centers2sample.device)
        nrcs_center = self.nrc_centers2sample[rand_indices]
        offsets_rand = (torch.rand(n, 2) - 0.5) * torch.tensor([self.delta_grid2sample_h, self.delta_grid2sample_w])/2.
        return nrcs_center + offsets_rand


    def crop_satmap_fm_nrc(self,nrc,satimgsize2crop=256,type='tensor'):
        row = int(nrc.squeeze()[0] * self.satmap_hw_max)
        col = int(nrc.squeeze()[1] * self.satmap_hw_max)

        halfimg_width = satimgsize2crop / 2
        col_begin = col - halfimg_width
        col_end = col + halfimg_width
        row_begin = row - halfimg_width
        row_end = row + halfimg_width
        # col_begin = int(max(0, col - halfimg_width))
        # col_end = int(min(self.satmap_w, col + halfimg_width)) # 限制在地图宽度内
        # row_begin = int(max(0, row - halfimg_width))
        # row_end = int(min(self.satmap_h, row + halfimg_width)) # 限制在地图高度内

        if type == 'tensor':
            sat_img = self.satmap_tensor[:, int(row_begin):int(row_end),
                      int(col_begin):int(col_end)]  # chw for sat_img_tensor
            # 检查原始图像尺寸是否为零或无效
            # if sat_img.size[-1] != sat_img.size[-2] or sat_img.size[-1] == 0:
            #     print(f"错误: 原始图像j剪裁尺寸无效!")
        else:
            # sat_img = self.tif_img[i_nt(row_begin):int(row_end),int(colbegin):int(col_end),:]
            sat_img = self.satmap.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))

        return sat_img


    def denormalize_img(self, satimg):
        satimgs_np = satimg * torch.tensor(self.satinfo_dict['std'])[:,None,None]+torch.tensor(self.satinfo_dict['mean'])[:,None,None]
        satimgs_np = satimgs_np.permute(1, 2, 0).numpy()
        satimgs_np = np.clip(satimgs_np * 255, 0, 255).astype(np.uint8)
        return satimgs_np


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, index):
        query_img_path, gallery_img_path, positive_weight = self.pairs[index]

        #handling uav_img
        #   using gta's transform:
        # img = cv2.imread(query_img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # uavimg_q = self.uav_transform_train(image=img)['image']
        #   using mine transform:
        uavimg_q = Image.open(query_img_path)
        uavimg_q = self.uav_transform_train(uavimg_q)
        uav_nrc = self.pairs_drone_nrc[index]
        drone_name = os.path.basename(self.pairs[index][0])

        #sample satimg_pos from nrc according the zoom_level
        sat_names = self.pairs_drone2sate_dict[drone_name]
        sat_zooms = [int(n[0]) for n in sat_names]
        # sat_zooms_filtered = []
        # for z in sat_zooms:
        #     if z in self.zoom_range:
        #         sat_zooms_filtered.append(z)
        level2sample_min,level2sample_max = (min(sat_zooms),max(sat_zooms))
        # level2sample = np.random.rand()*(level2sample_max-level2sample_min) + level2sample_min
        level2sample = 0.5*(level2sample_max+level2sample_min)
        # level2sample = random.choice(sat_zooms_filtered)
        tilesize2sample = SATE_LENGTH / (2 ** level2sample)
        #   using gta's transform:
        # satimg_pos = self.crop_satmap_fm_nrc(uav_nrc,tilesize2sample,type='pil')
        # satimg_pos = self.sat_transform_train(image=np.array(satimg_pos))['image']
        #   using mine transform:
        satimg_pos = self.crop_satmap_fm_nrc(uav_nrc,tilesize2sample,type='tensor')
        satimg_pos = self.sat_tensor_transform_train(satimg_pos)

        #debug:
        # satimg = self.denormalize_satimg(satimg_pos)
        # from matplotlib import pyplot as plt
        # plt.imshow(satimg)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_gta_vis/sat_pos_z6.png')
        # plt.close()
        # uavimg_q = Image.open(query_img_path)
        # plt.imshow(uavimg_q)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_gta_vis/uavimg_q.png')

        #sample satimgs_rand from nrc according the zoom_level
        sat_rcs_rand = self.mk_rand_nrcs(self.n_rand2sample_per_pos)
        satimgs_rand = torch.stack([self.crop_satmap_fm_nrc(sat_rc,tilesize2sample) for sat_rc in sat_rcs_rand])
        satimgs_rand = self.sat_tensor_transform_train(satimgs_rand)

        # vis for debug:
        # satimg2vis = self.denormalize_satimg(satimg_pos)
        # from matplotlib import pyplot as plt
        # uavimg_q = Image.open(query_img_path)
        # plt.imshow(uavimg_q)
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/exps_gta/uav_100_0001_0000000608.png')
        # plt.close()
        # plt.imshow(satimg2vis)
        # plt.savefig(f'/home/data/zwk/pyproj_DUAV_salad_6.4/exps_gta/sat_100_0001_0000000608_z{level2sample}.png')
        # plt.close()

        # return (uavimg_q,satimg_pos,satimgs_rand,torch.tensor(uav_nrc.copy(),dtype=torch.float32),torch.tensor(uav_nrc.copy(),dtype=torch.float32),torch.tensor(sat_rcs_rand,dtype=torch.float32))
        return uavimg_q,satimg_pos,satimgs_rand,uav_nrc,uav_nrc,sat_rcs_rand

def train_collate_fn(batch):#todo:this is ready for training, another for testing
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    uavimg_q, satimg_pos, satimgs_rand, uav_rc, sat_rc_pos, sat_rcs_rand =  zip(*batch)
    return torch.stack(uavimg_q),torch.stack(satimg_pos),torch.cat(satimgs_rand,dim=0),torch.stack(uav_rc),torch.stack(sat_rc_pos),torch.cat(sat_rcs_rand,dim=0)


def make_dataloader_gta(opt, stage='train', dataset = None):
    if dataset is None:
        dataset = DatasetGTA(opt.p_satinfo_json, opt.p_uavinfo_json, opt, stage)

    if stage=='train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,num_workers=opt.num_worker,
                                                 pin_memory=True,shuffle=True,collate_fn=train_collate_fn,drop_last=False)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, num_workers=opt.num_worker,
                                                 pin_memory=True, shuffle=False)
    return dataloader


# def eval(
#         model,
#         query_loader,
#         gallery_loader,
#         query_list,
#          ):


import tqdm
if __name__ == "__main__":
    p2cfg = '/home/data/zwk/pyproj_DUAV_salad_6.4/opts.yaml'
    config = OmegaConf.load(p2cfg)
    p_uavinfo_json = '/home/data/zwk/dataset_Game4Loc/cross-area-drone2sate-train.json'
    p_satinfo_json = '/home/data/zwk/dataset_Game4Loc/satmap_info.json'
    dataset = DatasetGTA(p_satinfo_json=p_satinfo_json,p_uavinfo_json=p_uavinfo_json,opt=config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.hardware_setting.batchsize, num_workers=config.hardware_setting.num_worker,
                                             pin_memory=True, shuffle=True, collate_fn=train_collate_fn,
                                             drop_last=False)
    for item in tqdm.tqdm(dataloader):
        len(item)
        pass