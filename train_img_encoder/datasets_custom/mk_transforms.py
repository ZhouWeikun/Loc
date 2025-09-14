import sys,os
# 获取当前脚本的绝对路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir # 假设 main.py 就是在项目根目录
# 将项目根目录添加到 Python 的模块搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 插入到最前面，优先搜索

import argparse
import omegaconf

from torchvision import transforms
from datasets.autoaugment import ImageNetPolicy
from datasets.queryDataset import RandomErasing

def mk_da_transform_list(config,mode='uav'):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    da_setting_list = da_setting.uav_da if mode == 'uav' else da_setting.sat_da

    transform_list = []
    if 'DA' in da_setting_list:  # 针对uav_image的特殊配置
        transform_list = [ImageNetPolicy()] + transform_list
    if 'ra' in da_setting_list:  # 随机仿射变换
        transform_list = transform_list + [transforms.RandomAffine(180)]
    if 're' in da_setting_list:  # 随机擦除
        transform_list = transform_list + [RandomErasing(probability=da_setting.erasing_p)]
    if 'cj' in da_setting_list:  # 随机颜色扰乱
        transform_list = transform_list + [
            transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)]

    return transform_list


def mk_uav_transform_train(config, uavimgs_info):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    transform_uav_list = []
    transform_uav_list += [transforms.Resize(da_setting.imgsize2net, interpolation=3)]  # 缩放
    transform_uav_list += [
        # transforms.CenterCrop(da_setting.imgsize2net),  # 中心剪裁
        transforms.RandomCrop(da_setting.imgsize2net),
        # transforms.RandomHorizontalFlip(),  # 随机翻转
    ]
    if 'rr' in da_setting.uav_da:
        transform_uav_list += [transforms.RandomRotation(180, interpolation=3)]  # 旋转

    transform_uav_list += mk_da_transform_list(da_setting, mode='uav')

    transform_uav_list += [
        transforms.ToTensor(),
        transforms.Normalize(uavimgs_info['mean'], uavimgs_info['std'])
    ]

    return transforms.Compose(transform_uav_list)


def mk_sat_transform_train(config, satimgs_info):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    transform_sat_list = []

    transform_sat_list += [transforms.Resize(da_setting.imgsize2net, interpolation=3)]  # 缩放
    if 'rr' in da_setting.sat_da:
        transform_sat_list += [transforms.RandomRotation(180, interpolation=3)]  # 旋转+缩放+中心剪裁到512正方形图像
    # transform_sat_list += [
    #     transforms.RandomHorizontalFlip(),
    # ]
    transform_sat_list += mk_da_transform_list(da_setting, mode='satellite')

    transform_sat_list += [
        transforms.ToTensor(),
        transforms.Normalize(satimgs_info['mean'], satimgs_info['std'])
    ]

    return transforms.Compose(transform_sat_list)


def mk_satensor_transform_train(config):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    transform_sat_list = []
    transform_sat_list += [transforms.Resize(da_setting.imgsize2net, interpolation=3, antialias=False)]
    if 'rr' in da_setting.sat_da:
        transform_sat_list += [transforms.RandomRotation(180, interpolation=3)]  # 旋转+缩放+中心剪裁到512正方形图像
    # transform_sat_list += [
    #     transforms.RandomHorizontalFlip(),
    # ]
    if 'ra' in da_setting.sat_da:
        transform_sat_list += [transforms.RandomAffine(180)]
    if 're' in da_setting.sat_da:
        # transform_sat_list = transform_sat_list +  [RandomErasing(probability=opt.erasing_p)]
        transform_sat_list += [transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random"  # 随机填充（在归一化范围内生成随机值）
        )]
    if 'cj' in da_setting.sat_da:
        transform_sat_list = transform_sat_list + [
            transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)]

    return transforms.Compose(transform_sat_list)


def mk_uav_transform_test(config, uavimgs_info=None):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    transform_uav = transforms.Compose([
        transforms.Resize(da_setting.imgsize2net, interpolation=3),
        transforms.CenterCrop(da_setting.imgsize2net),
        transforms.ToTensor(),
        transforms.Normalize(uavimgs_info['mean'], uavimgs_info['std'])
    ])
    return transform_uav


def mk_sat_transform_test(config,satimgs_info=None):
    da_setting = config.data_setting if isinstance(config,omegaconf.dictconfig.DictConfig) else config
    transform_sat = transforms.Compose([
        transforms.Resize(da_setting.imgsize2net, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(satimgs_info['mean'], satimgs_info['std'])
    ])
    return transform_sat