import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftFocalLoss(nn.Module):
    """
    适用于软标签（soft labels）或概率回归的Focal Loss。
    它将Focal Loss的思想推广到目标是[0, 1]之间的连续值的场景。

    该损失函数旨在解决类别不平衡问题，它通过一个调节因子来降低
    大量易分样本（通常是负样本）在损失计算中的权重，从而让模型
    更专注于学习难分的样本。

    参数:
        alpha (float): 平衡正负样本权重的因子，取值范围[0, 1]。
                       对于正样本，权重为alpha；对于负样本，权重为1-alpha。
                       通常设置为0.25。
        gamma (float): 聚焦参数，用于调节难易样本的权重。gamma > 0。
                       当gamma=0时，Focal Loss退化为标准的加权交叉熵损失。
                       通常设置为2.0。
        reduction (str): 指定应用于输出的规约方法: 'mean', 'sum', 'none'。
                         'mean': 输出的损失为所有样本损失的平均值。
                         'sum': 输出的损失为所有样本损失的总和。
                         'none': 不进行规约，返回每个样本的损失。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, logits, targets):
        """
        前向传播。

        参数:
            logits (torch.Tensor): 模型的原始输出，未经Sigmoid激活。
            targets (torch.Tensor): 真实的概率标签，值在[0, 1]之间。
        """
        # 基础的BCE损失，它完美支持软标签
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        loss = bce_loss[targets > 0].mean()+ bce_loss[targets == 0].mean()
        return loss


    # def forward(self, logits, targets):
    #     """
    #     前向传播。
    #
    #     参数:
    #         logits (torch.Tensor): 模型的原始输出，未经Sigmoid激活。
    #         targets (torch.Tensor): 真实的概率标签，值在[0, 1]之间。
    #     """
    #     # 基础的BCE损失，它完美支持软标签
    #     bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    #
    #     # 将logits通过sigmoid函数得到预测概率
    #     probs = torch.sigmoid(logits)
    #
    #     # --- 核心修改点 ---
    #     # 对于软标签，我们定义"正确度"pt为 1 - |预测概率 - 真实概率|
    #     # 这衡量了预测值与真值的接近程度
    #     pt = 1 - torch.abs(probs - targets)
    #
    #     # alpha项仍然可以像原来一样使用，它现在作为一个平滑的权重因子
    #     # 当targets接近1时，权重偏向alpha；当targets接近0时，权重偏向1-alpha
    #     # alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
    #
    #     # 调节因子 (1 - pt)^gamma
    #     # 当预测很准时(pt -> 1), 调节因子 -> 0, 损失被抑制
    #     # 当预测很差时(pt -> 0), 调节因子 -> 1, 损失被保留
    #     modulating_factor = (1.0 - pt).pow(self.gamma)
    #
    #     # 最终的Focal Loss
    #     # focal_loss = alpha_factor * modulating_factor * bce_loss
    #     focal_loss =  modulating_factor * bce_loss
    #
    #     if self.reduction == 'mean':
    #         return focal_loss.mean()
    #     elif self.reduction == 'sum':
    #         return focal_loss.sum()
    #     else:
    #         return focal_loss

        # id = torch.argsort(targets)
        # from matplotlib import pyplot as plt
        # plt.plot(modulating_factor[id].detach().cpu().numpy())
        # plt.savefig('/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/vis_occloc/modulating_factor.png')