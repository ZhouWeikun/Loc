import numpy as np
from torchvision import transforms
from datasets.Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from datasets.autoaugment import ImageNetPolicy
import torch
from datasets.queryDataset import RotateAndCrop, RandomCrop, RandomErasing

def make_dataset(opt):
    # <---config transform
    # 1.
    transform_uav_list = []
    transform_satellite_list = []
    if "uav" in opt.rr:
        transform_uav_list.append(RotateAndCrop(1.0, output_size=(opt.h, opt.w)))  # 旋转+缩放+中心剪裁到正方形图像
    else:
        transform_uav_list += [
            transforms.CenterCrop(1080),
            transforms.Resize((opt.h, opt.w), interpolation=3),  # 缩放到size2cnn，这一步可以省略
        ]
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(1.0, output_size=(opt.h, opt.w)))  # 旋转+缩放+中心剪裁到正方形图像
    else:
        transform_satellite_list += [
            transforms.Resize((opt.h, opt.w), interpolation=3),  # 缩放到size2cnn，这一步可以省略
        ]

    # 2.
    transform_uav_list += [
        # transforms.Resize((opt.h, opt.w), interpolation=3), #缩放到size2cnn，这一步可以省略
        # transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_satellite_list += [
        # transforms.Resize((opt.h, opt.w), interpolation=3),
        # transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    # 3.
    if opt.DA: #针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list

    # 4.
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
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    dataset_sizes = {x: len(image_datasets) *
                     opt.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    return dataloaders, class_names, dataset_sizes

from torchvision import datasets
import os
def make_dataset_test(opt):

    uav_transforms = transforms.Compose([
        transforms.Resize((opt.h), interpolation=3), #todo:不能直接缩放，应先剪裁
        transforms.CenterCrop((opt.h, opt.w)),
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

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,shuffle=False, num_workers=opt.num_worker)
                   for x in ['gallery_satellite', 'query_drone']}
    return image_datasets, dataloaders

if __name__ == "__main__":
    from train_custom import get_parse
    opt = get_parse()

    transform_uav_list = []
    transform_satellite_list = []
    if "uav" in opt.rr:
        transform_uav_list.append(RotateAndCrop(1.0, output_size=(opt.h, opt.w)))  # 旋转+缩放+中心剪裁到正方形图像
    else:
        transform_uav_list.append([
            transforms.CenterCrop(1080),
            transforms.Resize((opt.h, opt.w), interpolation=3),  # 缩放到size2cnn，这一步可以省略
        ])
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(1.0, output_size=(opt.h, opt.w)))  # 旋转+缩放+中心剪裁到正方形图像
    else:
        transform_satellite_list.append([
            transforms.Resize((opt.h, opt.w), interpolation=3),  # 缩放到size2cnn，这一步可以省略
        ])

    # 2.
    transform_uav_list += [
        # transforms.Resize((opt.h, opt.w), interpolation=3), #缩放到size2cnn，这一步可以省略
        # transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_satellite_list += [
        # transforms.Resize((opt.h, opt.w), interpolation=3),
        # transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    # 3.
    if opt.DA:  # 针对uav_image的特殊配置
        transform_uav_list = [ImageNetPolicy()] + transform_uav_list

    # 4.
    if "uav" in opt.ra:  # 随机仿射变换
        transform_uav_list = transform_uav_list + \
                             [transforms.RandomAffine(180)]
    if "satellite" in opt.ra:
        transform_satellite_list = transform_satellite_list + \
                                   [transforms.RandomAffine(180)]

    if "uav" in opt.re:  # 随机擦除
        transform_uav_list = transform_uav_list + \
                             [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_satellite_list = transform_satellite_list + \
                                   [RandomErasing(probability=opt.erasing_p)]

    if "uav" in opt.cj:  # 随机颜色扰乱
        transform_uav_list = transform_uav_list + \
                             [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                     hue=0)]
    if "satellite" in opt.cj:
        transform_satellite_list = transform_satellite_list + \
                                   [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                           hue=0)]
    transform_uav = transforms.Compose(transform_uav_list)
    uav_img_path = "/home/data/zwk/dataset_DenseUAV/train/drone/002255/H80.JPG"
    sat_img_path = "/home/data/zwk/dataset_DenseUAV/train/satellite/002255/H100.tif"
    from PIL import Image
    uav_img =  Image.open(uav_img_path).convert("RGB")
    uav_img_trans = transform_uav(uav_img)
    from matplotlib import pyplot as plt
    import numpy as np
    plt.imshow(np.array(uav_img_trans))
    plt.show()