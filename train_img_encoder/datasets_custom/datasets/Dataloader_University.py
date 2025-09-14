import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import glob

class Dataloader_University(Dataset):
    def __init__(self, root, transforms, names=['satellite', 'drone']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        # 获取所有图片的相对路径分别放到对应的类别中,一个类别对应一个文件夹，文件夹下可以有多个图像/实例
        # dict_cls2paths={'satelite':{0839:[0839_01.jpg,0839_02.jpg,...],...},'drone':{0839:[0839_01.jpg,0839_02.jpg,...],...}}
        # dict_cls2paths['drone']
        # |-dict_['0839']
        # ...
        # dict_cls2paths['satellite']
        # |-dict_['0839']
        # ...
        dict_cls2paths = {}
        for name in names: #遍历'satellite'和'drone'两个文件夹
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root, name, cls_name))
                img_path_list = [os.path.join(
                    root, name, cls_name, img) for img in img_list]
                dict_[cls_name] = img_path_list #一个key对应一个图像路径列表
            dict_cls2paths[name] = dict_   # dict_cls2paths[name+"/"+cls_name] = img_path_list

        # 获取设置名字与索引之间的镜像,dict:index->'cls',cls即类名/文件夹名称
        cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.nclasses = len(self.cls_names)
        self.map_dict = map_dict
        self.dict_cls2paths = dict_cls2paths
        self.index_cls_nums = 2

    # 从对应的类别中抽一张出来
    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_cls2paths[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)
        # exp3
        # img_s = transforms.Resize((224, 224), interpolation=3)(img)
        # img_s = transforms.RandomAffine(180)(img)
        # from datasets.queryDataset import RotateAndCrop, RandomCrop, RandomErasing
        # img_s = RandomErasing(probability=0.3)(img_s)
        # to_tensor = transforms.Compose(  [
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # img_s = to_tensor(img_s)

        img = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_d, index

    def __len__(self):
        return len(self.cls_names)


class DataLoader_Inference(Dataset):
    def __init__(self, root, transforms):
        super(DataLoader_Inference, self).__init__()
        self.root = root
        self.imgs = glob.glob(root+"/*.tif")
        self.tranforms = transforms
        sorted(self.imgs)
        self.labels = [os.path.basename(img).split(".tif")[
            0] for img in self.imgs]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        return self.tranforms(img), self.labels[index]

    def __len__(self):
        return len(self.imgs)


class Sampler_University(object):
    r"""Base ctlass for all Samplers.
    Every Sampler subclass has to provide an :method:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :method:`__len__` method
    that returns the length of the returned iteraors.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8, sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0, self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]


if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list = {"satellite": transforms.Compose(transform_train_list),
                            "train": transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",
                                     transforms=transform_train_list, names=['satellite', 'drone'])
    samper = Sampler_University(datasets, 8)
    dataloader = DataLoader(datasets, batch_size=8, num_workers=0,
                            sampler=samper, collate_fn=train_collate_fn)
    for data_s, data_d in dataloader:
        print()
