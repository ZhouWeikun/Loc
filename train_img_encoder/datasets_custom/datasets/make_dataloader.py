from torchvision import transforms

from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .autoaugment import ImageNetPolicy
import torch
from .queryDataset import RotateAndCrop, RandomCrop, RandomErasing

def make_dataloader(opt):
    #<---config transform
    # config transform;  对sat&uav进行共有配置
    transform_uav_list = []
    transform_satellite_list = []
    if "uav" in opt.rr:
        transform_uav_list.append(RotateAndCrop(0.5,output_size=(512,512))) #旋转+缩放+中心剪裁到512正方形图像
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(0.5,output_size=(512,512))) #旋转+缩放+中心剪裁到512正方形图像

    transform_uav_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3), #缩放到size2cnn，这一步可以省略
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_satellite_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    # config transform; 对sat&uav进行分别配置
    if opt.DA: #针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list

    if "uav" in opt.ra: #随机仿射变换
        transform_uav_list = transform_uav_list + \
            [transforms.RandomAffine(180)]
    if "satellite" in opt.ra:
        transform_satellite_list = transform_satellite_list + \
            [transforms.RandomAffine(180)]

    if "uav" in opt.re: #随机擦除
        transform_uav_list = transform_uav_list + \
            [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_satellite_list = transform_satellite_list + \
            [RandomErasing(probability=opt.erasing_p)]

    if "uav" in opt.cj: #随机颜色扰乱
        transform_uav_list = transform_uav_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
    if "satellite" in opt.cj:
        transform_satellite_list = transform_satellite_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]

    # config transform;  对sat&uav进行toTensor配置
    last_aug = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_uav_list += last_aug
    transform_satellite_list += last_aug
    transform_val_list += last_aug

    print(transform_uav_list)
    print(transform_satellite_list)

    data_transforms = {
        'train': transforms.Compose(transform_uav_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}
    # config transform--->

    # custom Dataset
    image_datasets = Dataloader_University(
        opt.data_dir, transforms=data_transforms)  #Dataloader_University is a class that assign the __getitem__() func
    samper = Sampler_University(
        image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    # dataset_sizes = {x: len(image_datasets) *
    #                  opt.sample_num for x in ['satellite', 'drone']}
    # class_names = image_datasets.cls_names
    return dataloader

from torchvision import datasets
import os
def make_dataset_test(opt):

    uav_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    sat_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = opt.data_dir.replace('train','test')
    image_datasets_query = {x: datasets.ImageFolder(os.path.join(data_dir, x), uav_transforms)
                            for x in ['query_drone']}

    image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir, x), sat_transforms)
                              for x in ['gallery_satellite']}

    image_datasets = {**image_datasets_query, **image_datasets_gallery}

    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize*6,shuffle=False, num_workers=opt.num_worker)
    #                for x in ['gallery_satellite', 'query_drone']}
    dataloader_uav = torch.utils.data.DataLoader(image_datasets['query_drone'], batch_size=opt.batchsize*6,shuffle=False, num_workers=opt.num_worker)
    dataloader_sat = torch.utils.data.DataLoader(image_datasets['gallery_satellite'], batch_size=opt.batchsize*10,shuffle=False, num_workers=opt.num_worker)

    return image_datasets, dataloader_uav,dataloader_sat