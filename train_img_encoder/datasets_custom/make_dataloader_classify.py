from torch.utils.data import Dataset
import os
import pandas as pd
import json
# import faiss
# from sklearn.neighbors import NearestNeighbors
# from torchvision.transforms.v2 import Transform

# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS#todo:
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from torchvision import transforms
from datasets.autoaugment import ImageNetPolicy
import torch
from datasets.queryDataset import RandomErasing

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

def mk_transfroms_train(opt,uavimgs_info,satimgs_info):
    transform_uav_list = []
    transform_sat_list = []
    uav_fill = _mean_fill_from_stats(uavimgs_info.get('mean'))
    sat_fill = _mean_fill_from_stats(satimgs_info.get('mean'))

    transform_uav_list += [transforms.Resize(opt.h, interpolation=3)]
    transform_sat_list += [transforms.Resize(opt.h, interpolation=3)]
    
    if "uav" in opt.rr:
        transform_uav_list += [ transforms.RandomRotation(360, interpolation=3, fill=uav_fill)  ]  # 旋转+缩放+中心剪裁到512正方形图像
    if "satellite" in opt.rr:
        transform_sat_list += [ transforms.RandomRotation(360, interpolation=3, fill=sat_fill)  ]  # 旋转+缩放+中心剪裁到512正方形图像

    transform_uav_list += [
        transforms.CenterCrop(opt.h),
        transforms.RandomHorizontalFlip(),
    ]
    transform_sat_list += [
        transforms.RandomHorizontalFlip(),
    ]

    # config transform; 对sat&uav进行分别配置
    if opt.DA:  # 针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list

    if "uav" in opt.ra:  # 随机仿射变换
        transform_uav_list = transform_uav_list + \
                             [transforms.RandomAffine(180, fill=uav_fill)]
    if "satellite" in opt.ra:
        transform_sat_list = transform_sat_list + \
                                   [transforms.RandomAffine(180, fill=sat_fill)]

    if "uav" in opt.re:  # 随机擦除
        transform_uav_list = transform_uav_list + \
                             [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_sat_list = transform_sat_list + \
                                   [RandomErasing(probability=opt.erasing_p)]

    if "uav" in opt.cj:  # 随机颜色扰乱
        transform_uav_list = transform_uav_list + \
                             [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                     hue=0)]
    if "satellite" in opt.cj:
        transform_sat_list = transform_sat_list + \
                                   [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                           hue=0)]

    # config transform;  对sat&uav进行toTensor配置
    uav_last_aug = [
        transforms.ToTensor(),
        transforms.Normalize( uavimgs_info['mean'], uavimgs_info['std'])
    ]
    sat_last_aug = [
        transforms.ToTensor(),
        transforms.Normalize( satimgs_info['mean'], satimgs_info['std'])
    ]
    transform_uav_list += uav_last_aug
    transform_sat_list += sat_last_aug

    print(transform_uav_list)
    print(transform_sat_list)

    transforms_train = {
        'uav': transforms.Compose(transform_uav_list),
        'sat': transforms.Compose(transform_sat_list)
    }
    return transforms_train


def mk_transfroms_test(opt,uavimgs_info=None,satimgs_info=None):
    if uavimgs_info is not None:
        transform_uav = transforms.Compose([
            transforms.Resize(opt.h, interpolation=3),
            transforms.CenterCrop(opt.h),
            transforms.ToTensor(),
            transforms.Normalize(uavimgs_info['mean'], uavimgs_info['std'])
        ])
    else:
        transform_uav = None
    if satimgs_info is not None:
        transform_sat = transforms.Compose([
            transforms.Resize(opt.h, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(satimgs_info['mean'], satimgs_info['std'])
        ])
    else:
        transform_sat = None

    transforms_test = {
        'uav': transform_uav,
        'sat':transform_sat
    }
    return transforms_test


def read_dataset_meta_info(data_dir):
    labels = pd.read_csv(os.path.join(data_dir, 'geoloc.csv'))
    testing_radio = 0.1
    training_labels = labels.iloc[0:int(len(labels) * (1 - testing_radio))]
    testing_labels = labels.iloc[int(len(labels) * (1 - testing_radio)):]
    geolabel_dict = {"train": training_labels, "test": testing_labels, 'all':labels}

    # read meta info for uav and sat imgs
    with open(os.path.join(data_dir, 'uavimgs_info.json'), "r") as f:
        uavimg_infodict = json.load(f)
    with open(os.path.join(data_dir, 'satimgs_info.json'), "r") as f:
        satimg_infodict = json.load(f)
    imginfo_dict={'uav':uavimg_infodict,'sat':satimg_infodict}

    return geolabel_dict,imginfo_dict

class Dataset_uav_query(Dataset):
    def __init__(self,
                 opt,
                 **kwargs,
                 ):
        self.data_dir = opt.data_dir
        geolabel_dict, imginfo_dict = read_dataset_meta_info(opt.data_dir)
        self.labels = geolabel_dict['test']
        self.nclasses_train = len(geolabel_dict['train'])
        self.nclasses = len( self.labels )

        self.transform_dict = mk_transfroms_test(opt, uavimgs_info = imginfo_dict['uav'])

    def __getitem__(self, index):
        # item = self.label_stage[self.stage].iloc[index]
        item = self.labels.iloc[index]
        uav_img_path = os.path.join(self.data_dir,item['uav_path'])
        img_uav = self.transform_dict['uav'](Image.open(uav_img_path).convert("RGB"))
        
        return img_uav,index+self.nclasses_train

    def __len__(self):
        return self.nclasses

class Dataset_sat_gallary(Dataset):
    def __init__(self,
                 opt,
                 **kwargs,
                 ):
        self.data_dir = opt.data_dir
        geolabel_dict, imginfo_dict = read_dataset_meta_info(opt.data_dir)
        self.labels = geolabel_dict['all']
        self.nclasses = len(self.labels)

        self.transform_dict = mk_transfroms_test(opt, satimgs_info=imginfo_dict['sat'])

    def __getitem__(self, index):
        # item = self.label_stage[self.stage].iloc[index]
        item = self.labels.iloc[index]
        sat_img_path = os.path.join(self.data_dir,item['sat_path'])
        img_sat = self.transform_dict['sat'](Image.open(sat_img_path).convert("RGB"))

        return img_sat,index

    def __len__(self):
        return self.nclasses


class Dataset_uav2sat_classify(Dataset):
    def __init__(self,
                 opt,
                 stage = 'train',
                 imgsize2cnn=224,
                 **kwargs,
                 ):
        self.data_dir = opt.data_dir
        self.stage = stage

        #read label csv, split training and testing set
        labels = pd.read_csv(os.path.join( self.data_dir,'geoloc.csv'))
        testing_radio = 0.1
        training_labels = labels.iloc[0:int(len(labels)*(1-testing_radio))]
        testing_labels = labels.iloc[int(len(labels)*(1-testing_radio)):]
        self.labels_all = labels
        self.labels = training_labels if self.stage=='train' else testing_labels
        self.nclasses = len(self.labels)

        #read meta info for uav and sat imgs
        with open(os.path.join( self.data_dir,'satimgs_info.json'), "r") as f:
            satimg_infodict = json.load(f)
        with open(os.path.join( self.data_dir,'uavimgs_info.json'), "r") as f:
            uavimg_infodict = json.load(f)

        if self.stage == 'train':
            self.transform_dict = mk_transfroms_train(opt, uavimg_infodict, satimg_infodict)
        else:
            self.transform_dict = \
                (opt, uavimg_infodict, satimg_infodict)

    def __getitem__(self, index):
        # item = self.label_stage[self.stage].iloc[index]
        item = self.labels.iloc[index]
        uav_img_path = os.path.join(self.data_dir,item['uav_path'])
        sat_img_path = os.path.join(self.data_dir,item['sat_path'])
        img_uav = self.transform_dict['uav'](Image.open(uav_img_path).convert("RGB"))
        img_sat = self.transform_dict['sat'](Image.open(sat_img_path).convert("RGB"))
        # img_uav = self.uav_transform_stage[self.stage](cv2.imread(uav_img_path)[:,:,::-1].astype(np.float32) / 255.0)
        # img_sat = self.sat_transform_stage[self.stage](cv2.imread(sat_img_path)[:,:,::-1].astype(np.float32) / 255.0)
        return img_sat,img_uav,index

    def __len__(self):
        return self.nclasses

    def set_uav_transform(self,imgsize2cnn,img_mean,img_std):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(img_mean,img_std),
            transforms.Resize(imgsize2cnn,interpolation=InterpolationMode.BICUBIC),
            transforms.RandomRotation(180),
            transforms.CenterCrop((imgsize2cnn, imgsize2cnn)),
        ]
        self.uav_transform_train = transforms.Compose(transform_list)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(img_mean,img_std),
            transforms.Resize(imgsize2cnn,interpolation=InterpolationMode. BICUBIC),
            transforms.CenterCrop((imgsize2cnn, imgsize2cnn)),
        ]
        self.uav_transform_test = transforms.Compose(transform_list)
        self.uav_transform_stage = {'training': self.uav_transform_train,"testing":self.uav_transform_test}

    def set_sat_transform(self,imgsize2cnn,img_mean,img_std):
        self.sat_transform_train = transforms.Compose([
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=img_mean, std=img_std),  # normalize value to [-1,1]
            transforms.RandomRotation(180),
            transforms.Resize((imgsize2cnn, imgsize2cnn), interpolation=InterpolationMode. BICUBIC),
        ])
        self.sat_transform_test = transforms.Compose([
            transforms.ToTensor(), # turn numpy to tensor, reshape [h,w,c] to [c,h,w], scale [0,255] to [0,1] if type==np.uint8
            transforms.Normalize(mean=img_mean, std=img_std),  # normalize value to [-1,1]
            transforms.Resize((imgsize2cnn, imgsize2cnn), interpolation=InterpolationMode. BICUBIC),
        ])
        self.sat_transform_stage = {'training': self.sat_transform_train,"testing":self.sat_transform_test}

def collate_fn_train(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]

def make_dataloader_train(opt):
    # custom Dataset
    image_dataset =  Dataset_uav2sat_classify(opt,stage='train')  #Dataloader_University is a class that assign the __getitem__() func
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                             num_workers=opt.num_worker, pin_memory=True, collate_fn=collate_fn_train)
    return dataloader

def make_dataloader_test(opt,view='uav'):
    # custom Dataset
    if view == 'uav':
        image_dataset =  Dataset_uav_query(opt)  #Dataloader_University is a class that assign the __getitem__() func
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                                 num_workers=opt.num_worker, shuffle=False,pin_memory=True)
    else:
        image_dataset = Dataset_sat_gallary(opt)  # Dataloader_University is a class that assign the __getitem__() func
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                                 num_workers=opt.num_worker, shuffle=False,pin_memory=True)
    return dataloader


from numba import njit, prange
@njit(parallel=True)
def clip_satimg(sat_img_np, rcs_girdcoords, size2clip):
    patches = np.zeros(list(rcs_girdcoords.shape[:2]) + [3, size2clip, size2clip],
                          dtype=sat_img_np.sat_img_np.dtype)
    for i in prange(rcs_girdcoords):
        for j in prange(rcs_girdcoords[i]):
            rb, re, cb, ce = rcs_girdcoords[i, j]
            patches[i, j] = sat_img_np[:, rb:re, cb:ce]
    return patches

if __name__=="__main__":
    from history_cache.train_v1 import get_parse
    opt = get_parse()
    # dataset = Dataset_uav2sat_classify(
    # opt,
    # stage='train'
    # )
    # for  img_sat,img_uav,index in dataset:
    #     print(index)

    dataset = Dataset_uav_query(opt)
    for img,index in dataset:
        print(index)
