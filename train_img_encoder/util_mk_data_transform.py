import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import List, Union


def _resolve_fill_value(img, fill):
    if fill != "mean":
        return fill
    if torch.is_tensor(img):
        if img.ndim == 3:
            return img.mean(dim=(1, 2)).tolist()
        if img.ndim == 4:
            return img.mean(dim=(2, 3))
    img_np = np.asarray(img)
    if img_np.ndim == 0:
        return float(img_np)
    if img_np.ndim == 2:
        return float(img_np.mean())
    mean_vals = img_np.mean(axis=(0, 1))
    return tuple(int(round(x)) for x in mean_vals.tolist())


def _get_image_size(img):
    if torch.is_tensor(img):
        return [img.shape[-1], img.shape[-2]]
    return F.get_image_size(img)

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

    def __init__(self, degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill="mean", same_on_batch = False):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.fill = fill
        self.same_on_batch = same_on_batch  # <--- 新增的控制参数

        # 用来存储角度，可以是单个浮点数或一个列表
        self.angle = None

    def forward(self, img: torch.Tensor, angle: Union[float, List[float], None] = None) -> torch.Tensor:
        """
        Args:
            img (Tensor): 输入图像 (3D) 或图像批次 (4D)。
            angle (float, List[float], optional):
                - 如果提供，则使用此角度/角度列表进行旋转。
                - 如果为 None，则进行随机旋转。
                - 对于批处理，可以是一个浮点数为整个批次应用相同旋转，
                  也可以是一个列表为批次中每个图像应用不同旋转。
                Defaults to None.
        Returns:
            Tensor: 旋转后的图像或图像批次。
        """
        # --- 1. 处理一个 Batch ---
        if img.ndim == 4:
            batch_size = img.shape[0]

            # 步骤 A: 决定要使用的角度
            angles_to_apply = None
            if angle is not None:
                # 优先使用外部传入的、确定的角度
                angles_to_apply = angle
            else:
                # Fallback: 生成随机角度
                if self.same_on_batch:
                    angles_to_apply = random.uniform(self.degrees[0], self.degrees[1])
                else:
                    angles_to_apply = [random.uniform(self.degrees[0], self.degrees[1]) for _ in range(batch_size)]

            self.angle = angles_to_apply  # 记录最终使用的角度

            # 步骤 B: 根据角度类型执行旋转
            if isinstance(angles_to_apply, (int, float)):
                # 单个角度 -> 应用于整个批次
                fill_val = _resolve_fill_value(img, self.fill)
                return F.rotate(img, angles_to_apply, self.interpolation, fill=fill_val)

            elif isinstance(angles_to_apply, (list, tuple)):
                # 角度列表 -> 为每个图像应用不同旋转
                if len(angles_to_apply) != batch_size:
                    raise ValueError(f"提供的角度列表长度 ({len(angles_to_apply)}) 与批次大小 ({batch_size}) 不匹配。")

                rotated_images = []
                for i, ang in enumerate(angles_to_apply):
                    fill_val = _resolve_fill_value(img[i], self.fill)
                    if torch.is_tensor(fill_val):
                        fill_val = fill_val[i].tolist()
                    rotated_images.append(
                        F.rotate(img[i], ang, self.interpolation, fill=fill_val)
                    )
                return torch.stack(rotated_images)
            else:
                raise TypeError(f"要应用的旋转角度必须是浮点数或浮点数列表，但得到的是 {type(angles_to_apply)}。")

        # --- 2. 处理单个图像 ---
        elif img.ndim == 3:
            angle_to_apply = None
            if angle is not None:
                # 优先使用外部传入的、确定的角度
                angle_to_apply = angle
            else:
                # Fallback: 生成随机角度
                angle_to_apply = random.uniform(self.degrees[0], self.degrees[1])

            self.angle = angle_to_apply  # 记录最终使用的角度
            fill_val = _resolve_fill_value(img, self.fill)
            return F.rotate(img, angle_to_apply, self.interpolation, fill=fill_val)

        # --- 3. 错误处理 ---
        else:
            raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")


class RandomAffineWithMean(torch.nn.Module):
    """
    RandomAffine with per-image mean fill support for tensor/PIL inputs.
    """
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill="mean",
        center=None,
        same_on_batch=False,
    ):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center
        self.same_on_batch = same_on_batch

    def _sample_params(self, img):
        img_size = _get_image_size(img)
        return transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, img_size
        )

    def forward(self, img):
        if img.ndim == 4:
            batch_size = img.shape[0]
            rotated_images = []
            if self.same_on_batch:
                angle, translations, scale, shear = self._sample_params(img[0])
            for i in range(batch_size):
                if not self.same_on_batch:
                    angle, translations, scale, shear = self._sample_params(img[i])
                fill_val = _resolve_fill_value(img[i], self.fill)
                if torch.is_tensor(fill_val):
                    fill_val = fill_val[i].tolist()
                rotated_images.append(
                    F.affine(
                        img[i],
                        angle=angle,
                        translate=translations,
                        scale=scale,
                        shear=shear,
                        interpolation=self.interpolation,
                        fill=fill_val,
                        center=self.center,
                    )
                )
            return torch.stack(rotated_images)
        if img.ndim == 3:
            angle, translations, scale, shear = self._sample_params(img)
            fill_val = _resolve_fill_value(img, self.fill)
            return F.affine(
                img,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=self.interpolation,
                fill=fill_val,
                center=self.center,
            )
        raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")

def mk_tensor_transform(
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
            RandomAffineWithMean(
                **default_affine_params,  # <--- 核心改动
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill="mean"
            )
        )

    if rand_erase:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0))

    transform2ret = transforms.Compose(transform_list)
    return transform2ret,rotator


def mk_pil_transform(
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
    """
    不接受 已经是 Tensor 的输入
    """
    transform_list = [transforms.Resize(imgsize2net)]
    if center_crop:
        transform_list += [transforms.CenterCrop(imgsize2net)]
    if rand_crop:
        transform_list += [transforms.RandomCrop(imgsize2net)]
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

    fill_val = _mean_fill_from_stats(mean)
    if rand_rot:
        transform_list.append(transforms.RandomRotation(180, interpolation=3, fill=fill_val))
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
                fill=fill_val
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
