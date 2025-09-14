from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os,json
import numpy as np
import cv2
from make_dataloader_gta import sate2loc
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 禁用检查，或者设置为一个足够大的整数，例如 1_000_000_000


def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


class GTADatasetEval(Dataset):

    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=None,
                 height_range = [100,200],
                 filter_in_height = False,
                 zoom_range = [6,7],
                 filter_in_zoom = False,
                 ):
        super().__init__()

        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)

        self.images_path = []
        self.images_name = []
        self.images_center_loc_xy = []
        self.images_drone_height = []
        # For finer localization with image matching
        self.images_topleft_loc_xy = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        if view == 'drone':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                drone_height = pair_drone2sate['drone_metadata']['height']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.images_path.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.images_name.append(drone_img_name)
                    self.images_center_loc_xy.append((drone_loc_x_y[0], drone_loc_x_y[1]))
                    self.images_drone_height.append([drone_height])

            if filter_in_height:
                self.height_range = height_range
                self.images_drone_height = np.stack(self.images_drone_height)
                height_mask = ((self.images_drone_height > self.height_range[0]) * (
                            self.images_drone_height < self.height_range[1])).squeeze()
                self.images_drone_height = self.images_drone_height[height_mask]
                self.images_path = np.stack(self.images_path)[height_mask]
                self.images_name = np.stack(self.images_name)[height_mask]
                self.images_center_loc_xy = np.stack(self.images_center_loc_xy)[height_mask]

                key2del = []
                for k,v in self.pairs_drone2sate_dict.items():
                    if f'{height_range[0]}_' in k or f'{height_range[1]}_' in k:
                        if len(v)>0:
                            if int(v[0][0]) not in zoom_range:
                               key2del.append(k)
                    else:
                        key2del.append(k)

                for k in key2del:
                    self.pairs_drone2sate_dict.pop(k)


                zoom_mask = ~np.array([n in key2del for n in self.images_name])
                self.images_path = self.images_path[zoom_mask]
                self.images_name = self.images_name[zoom_mask]
                self.images_center_loc_xy = self.images_center_loc_xy[zoom_mask]


        elif view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    loc_center_x, loc_center_y, loc_topleft_x, loc_topleft_y = sate2loc(tile_zoom, offset, tile_x,
                                                                                        tile_y)
                    self.images_center_loc_xy.append((loc_center_x, loc_center_y))
                    self.images_topleft_loc_xy.append((loc_topleft_x, loc_topleft_y))

                if filter_in_zoom:
                    self.zoom_range = zoom_range
                    zoom_mask = [int(n[0]) in zoom_range for n in self.images_name]
                    zoom_mask = np.array(zoom_mask)
                    self.images_path = np.stack(self.images_path)[zoom_mask]
                    self.images_name = np.stack(self.images_name)[zoom_mask]
                    self.images_center_loc_xy = np.stack(self.images_center_loc_xy)[zoom_mask]
                    self.images_topleft_loc_xy = np.stack(self.images_topleft_loc_xy)[zoom_mask]

            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    loc_center_x, loc_center_y, loc_topleft_x, loc_topleft_y = sate2loc(tile_zoom, offset, tile_x,
                                                                                        tile_y)
                    self.images_center_loc_xy.append((loc_center_x, loc_center_y))
                    self.images_topleft_loc_xy.append((loc_topleft_x, loc_topleft_y))

        self.transforms = transforms



    # def __getitem__(self, index):
    #
    #     img_path = self.images_path[index]
    #
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # image transforms
    #     if self.transforms is not None:
    #         img = self.transforms(image=img)['image']
    #
    #     return img


    def __getitem__(self, index):
        img_path = self.images_path[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img


    def __len__(self):
        return len(self.images_name)
