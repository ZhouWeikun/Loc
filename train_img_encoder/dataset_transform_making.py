import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random


# 改进后的自定义 Transform 类，能正确处理 Batch
class RandomRotationWithAngle(torch.nn.Module):
    """
    对图像或图像Batch进行随机旋转。
    - 如果输入是单个图像，返回旋转后的图像，self.angle为单个角度。
    - 如果输入是一个Batch，对每张图应用不同随机角度，返回旋转后的Batch，self.angle为一个角度列表。
    Args:
        degrees (float or sequence): 旋转的角度范围。
            - 如果是单个数字 d，范围是 [-d, d]。
            - 如果是元组 (min, max)，范围是 [min, max]。
        same_on_batch (bool):
            - 如果为 False (默认), 对批次中的每个样本应用不同的随机旋转。
            - 如果为 True, 对批次中的所有样本应用同一个随机旋转，行为与 torchvision 标准库对齐。
    """

    def __init__(self, degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=0, same_on_batch = False):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.fill = fill
        self.same_on_batch = same_on_batch  # <--- 新增的控制参数

        # 用来存储角度，可以是单个浮点数或一个列表
        self.angle = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 4:  # --- 处理一个 Batch ---

            if self.same_on_batch:
                # --- 行为B: 对整个Batch应用同一个旋转 ---
                # 1. 只生成一个随机角度
                random_angle = random.uniform(self.degrees[0], self.degrees[1])
                self.angle = random_angle  # 保存这个单值

                # 2. F.rotate 可以直接处理Batch，当angle是标量时，它会对所有图应用相同旋转
                return F.rotate(img, self.angle, self.interpolation, fill=self.fill)

            else:
                # --- 行为A: 对Batch中每个样本应用不同旋转 (原始逻辑) ---
                batch_size = img.shape[0]

                # 1. 为每个图片生成一个随机角度列表
                angles = [random.uniform(self.degrees[0], self.degrees[1]) for _ in range(batch_size)]
                self.angle = angles  # 保存角度列表

                # 2. 逐个旋转Batch中的图片
                rotated_images = [
                    F.rotate(img[i], angle, self.interpolation, fill=self.fill)
                    for i, angle in enumerate(angles)
                ]

                # 3. 将旋转后的图片列表重新堆叠成一个Batch Tensor
                return torch.stack(rotated_images)

        elif img.ndim == 3:  # --- 处理单个图像 (逻辑不变) ---
            random_angle = random.uniform(self.degrees[0], self.degrees[1])
            self.angle = random_angle
            return F.rotate(img, self.angle, self.interpolation, fill=self.fill)

        else:
            raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")


def mk_sat_tensor_transform(
        imgsize2net=224,
        rand_rot = False,
        rand_affine = False,
        affine_para = None,
        rand_erase = False,
        ):
    transform_list = [transforms.Resize(imgsize2net,antialias=True)]

    rotator = None
    if rand_rot:
        # 创建我们自定义的类的实例
        rotator = RandomRotationWithAngle(degrees=180,same_on_batch=True)
        # 将这个实例加入到变换列表中
        transform_list.append(rotator)

    if rand_affine:
        # 1. 为 RandomAffine 定义一套默认参数
        default_affine_params = {
            'degrees': 0,
            'translate': (0, 0),
            'scale': (1.0, 1.0),
            'shear': 5,
        }
        # 2. 如果提供了affine_para字典，用它来更新默认值
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)
            # 3. 使用字典解包(**)将参数传递给函数
        transform_list.append(
            transforms.RandomAffine(
                **default_affine_params,  # <--- 核心改动
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
        )

    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0))

    transform2ret = transforms.Compose(transform_list)
    return transform2ret,rotator


def mk_transform(
        mean,
        std,
        imgsize2net=224,
        rand_affine = False,
        affine_para = None,
        rand_rot = False,
        rand_erase = False,
        color_jitter = False,
        center_crop = False,
        rand_crop = False,
        ):
    transform_list = [transforms.Resize(imgsize2net)]
    if center_crop:
        transform_list += [transforms.CenterCrop(imgsize2net)]
    if rand_crop:
        transform_list += [transforms.RandomCrop(imgsize2net)]
    if rand_rot:
        transform_list.append(transforms.RandomRotation(180, interpolation=3))
    if rand_affine:
        # 1. 为 RandomAffine 定义一套默认参数
        default_affine_params = {
            'degrees': 180,
            'translate': (0, 0),
            'scale': (1.0, 1.0),
            'shear': 5,
        }
        # 2. 如果提供了affine_para字典，用它来更新默认值
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)
            # 3. 使用字典解包(**)将参数传递给函数
        transform_list.append(
            transforms.RandomAffine(
                **default_affine_params,  # <--- 核心改动
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            )
        )
    if color_jitter:
        transform_list.append(
            transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                # transforms.RandomAutocontrast(p=0.3),
                # transforms.RandomGrayscale(p=0.3)
            ])
        )
    transform_list += [transforms.ToTensor()]
    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=1))
    transform_list +=[transforms.Normalize(mean=mean, std=std)]

    transform2ret = transforms.Compose(transform_list)
    return transform2ret