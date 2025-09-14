import torch
import torch.nn as nn
from .utils import ClassBlock

class FSRA(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_classes = opt.nclasses
        droprate = opt.droprate
        in_planes = opt.in_planes #decided by the backbone
        num_bottleneck = opt.num_bottleneck #decided in the trian.py manually
        self.class_name = "classifier_heat"
        self.block = opt.block

        if opt.w_classify:
            # global classifier
            self.classifier1 = ClassBlock(in_planes, num_classes, droprate, num_bottleneck = num_bottleneck)
            # local classifier
            for i in range(self.block):
                name = self.class_name + str(i + 1)
                setattr(self, name, ClassBlock(in_planes, num_classes, droprate,num_bottleneck = num_bottleneck))
        else:
            # global classifier
            self.classifier1 = ClassBlock(in_planes, droprate=droprate,num_bottleneck = num_bottleneck, w_cls=False)
            # local classifier
            for i in range(self.block):
                name = self.class_name + str(i + 1)
                setattr(self, name, ClassBlock(in_planes, droprate=droprate, num_bottleneck = num_bottleneck, w_cls=False))

    def forward(self, features):
        global_cls, global_feature = self.classifier1(features[:, 0])
        # tranformer_feature = torch.mean(features,dim=1)
        # tranformer_feature = self.classifier1(tranformer_feature)
        if self.block == 1:
            return global_cls, global_feature

        part_features = features[:, 1:]

        heat_result = self.get_heartmap_pool(part_features)
        cls_list, features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)

        total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls, total_features]

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            cls_list.append(res[0])
            features_list.append(res[1])
        return cls_list, features_list

class FSRA_wo_CLS(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        droprate = opt.droprate
        in_planes = opt.in_planes
        num_bottleneck = opt.num_bottleneck  # decided in the trian.py manually
        self.class_name = "classifier_heat"
        self.block = opt.block

        if opt.w_classify:
            num_classes = opt.nclasses
            # global classifier
            self.classifier1 = ClassBlock(in_planes, num_classes, droprate)
            # local classifier
            for i in range(self.block):
                name = self.class_name + str(i + 1)
                setattr(self, name, ClassBlock(in_planes, num_classes, droprate))
        else:
            # global classifier
            self.classifier1 = ClassBlock(in_planes, droprate=droprate, w_cls=False, num_bottleneck=num_bottleneck)
            # local classifier
            for i in range(self.block):
                name = self.class_name + str(i + 1)
                setattr(self, name, ClassBlock(in_planes, droprate=droprate, w_cls=False,num_bottleneck=num_bottleneck))

    def forward(self, features):
        global_feature = self.classifier1(features[:, 0])
        if self.block == 1:
            return  global_feature

        part_features = features[:, 1:]

        heat_result = self.get_heartmap_pool(part_features) #到此，已经完成根据热力图按区域tokens进行平均池化 #语义的分区是通过热力图数值的中位数来划分的
        # cls_list, features_list = self.part_classifier(
        #     self.block, heat_result, cls_name=self.class_name)
        # 开始对特征进行降维
        features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)  #每个block执行的功能都是对特征进行降维，只不过若有两个block，就是假设有两个区域/两种语义，每个block单独处理一种语义的降维

        # total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        # return [total_cls, total_features]
        return total_features

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            # cls_list.append(res[0])
            # features_list.append(res[1])
            features_list.append(res)
        # return cls_list, features_list
        return features_list

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        #debug:
        # import numpy as np
        # heatmap2vis = heatmap[0].reshape(14,14).detach().cpu().numpy()
        # median = np.median(heatmap2vis)
        # status_above_median = heatmap2vis > median
        # overlay_data = status_above_median.astype(np.int)
        # vmin = heatmap2vis.min()
        # vmax = heatmap2vis.max()
        # from matplotlib import pyplot as plt
        # fig,ax = plt.subplots()
        # # base_cmap = 'viridis'  # 或者你选择的其他 colormap
        # # im_base = ax.imshow(heatmap2vis, cmap=base_cmap, aspect='auto')
        # im_base = ax.imshow(heatmap2vis, cmap = 'jet', interpolation = 'bilinear', vmin = vmin, vmax = vmax)
        # # 添加基础热力图的颜色条
        # fig.colorbar(im_base, ax=ax)
        # # 方法一：使用简单的标准 colormap (如 'Reds')，并设置全局 alpha
        # # 'Reds' 颜色图: 0 接近白色/透明，1 是红色
        # overlay_cmap = 'viridis' #viridis,Reds
        # alpha_level = 0.2  # 设置叠加层的透明度 (0=完全透明, 1=完全不透明)
        # ax.imshow(overlay_data,
        #                        cmap=overlay_cmap,  # 使用 'Reds' cmap
        #                        alpha=alpha_level,  # 应用全局透明度
        #                        interpolation='nearest',  # 使用 'nearest' 避免模糊状态边界
        #                        aspect='auto',
        #                        vmin=vmin, vmax=vmax)  # 确保 0 映射到 cmap 的开始，1 映射到结束
        # fig.tight_layout()
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad/heatmap.jpg')

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_




class FSRA_CNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_classes = opt.nclasses
        droprate = opt.droprate
        in_planes = opt.in_planes
        self.class_name = "classifier_heat"
        self.block = opt.block
        # global classifier
        self.classifier1 = ClassBlock(in_planes, num_classes, droprate)
        # local classifier
        for i in range(self.block):
            name = self.class_name + str(i+1)
            setattr(self, name, ClassBlock(in_planes, num_classes, droprate))

    def forward(self, features):
        # global_cls, global_feature = self.classifier1(features[:, 0])
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1,2)
        global_feature = torch.mean(features,dim=1)
        global_cls, global_feature = self.classifier1(global_feature)
        if self.block == 1:
            return global_cls, global_feature

        part_features = features
        # print(part_features.shape)


        heat_result = self.get_heartmap_pool(part_features)
        cls_list, features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)

        total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls, total_features]

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            cls_list.append(res[0])
            features_list.append(res[1])
        return cls_list, features_list