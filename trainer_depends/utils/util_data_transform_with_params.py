"""
可追踪参数的Transform类
用于数据增强时记录旋转、缩放等参数，并将其应用到4D坐标上
"""
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import Tuple, Optional


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


class RandomRotationWithParams(torch.nn.Module):
    """
    随机旋转Transform，能够返回实际使用的旋转角度

    注意：torchvision.transforms.functional.rotate 的旋转约定与我们的不一致
    我们约定：逆时针旋转为正（与 batch_rotate_images_per_sample 一致）
    但 F.rotate 在图像坐标系下的行为相反
    因此需要对角度取反
    """
    def __init__(self, degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill="mean"):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation
        self.fill = fill
        self.last_angle = None  # 记录最后一次使用的角度（按我们的约定：逆时针为正）

    def forward(self, img):
        """
        Args:
            img: PIL Image or Tensor
        Returns:
            rotated_img: 旋转后的图像
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        self.last_angle = angle  # 记录角度（按我们的约定）

        # 关键修正：F.rotate 的约定与我们相反，需要取反
        fill_val = _resolve_fill_value(img, self.fill)
        return F.rotate(img, -angle, self.interpolation, fill=fill_val)

    def get_last_angle(self):
        """返回最后一次旋转的角度（度数，逆时针为正）"""
        return self.last_angle


class RandomScaleWithParams(torch.nn.Module):
    """
    随机缩放Transform，模拟UAV高度变化

    语义：
    - scale > 1: UAV飞得更低，地面覆盖范围变小，物体变大
      实现：先从原图中心裁剪较小区域，再放大到base_size
    - scale < 1: UAV飞得更高，地面覆盖范围变大，物体变小
      实现：先缩小整个图像，再padding到base_size
    - scale = 1: 保持原样

    注意：这个transform应该在旋转之前应用，以保留更多有效图像内容
    """
    def __init__(self, scale_range=(0.8, 1.2), base_size=224, interpolation=transforms.InterpolationMode.BILINEAR, pad_mode=None):
        super().__init__()
        self.scale_range = scale_range
        self.base_size = base_size
        self.interpolation = interpolation
        self.pad_mode = str(pad_mode).lower() if pad_mode is not None else None
        self.last_scale = None  # 记录最后一次使用的缩放因子

    def forward(self, img):
        """
        Args:
            img: PIL Image or Tensor (C, H, W)
        Returns:
            scaled_img: 缩放后的图像，尺寸为 (C, base_size, base_size)
        """
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        self.last_scale = scale

        if scale > 1.0:
            # UAV飞低了，地面覆盖范围变小
            # 策略：从原图中心裁剪 (base_size / scale) 的区域，然后放大到 base_size
            crop_size = int(self.base_size / scale)
            img_cropped = F.center_crop(img, crop_size)
            img_scaled = F.resize(img_cropped, self.base_size, self.interpolation)

        elif scale < 1.0:
            # UAV飞高了，地面覆盖范围变大
            # 策略：将图像缩小到 (base_size * scale)，然后padding到 base_size
            resize_size = int(self.base_size * scale)
            img_resized = F.resize(img, resize_size, self.interpolation)

            # 计算padding
            pad_total = self.base_size - resize_size
            pad_left = pad_total // 2
            pad_top = pad_total // 2
            pad_right = pad_total - pad_left
            pad_bottom = pad_total - pad_top

            if self.pad_mode in {"zero", "zeros", "0"}:
                fill_val = 0
            else:
                fill_val = _resolve_fill_value(img_resized, "mean")
            if torch.is_tensor(img_resized):
                if img_resized.ndim == 2:
                    if isinstance(fill_val, (list, tuple, np.ndarray)):
                        fill_scalar = float(np.mean(fill_val))
                    else:
                        fill_scalar = float(fill_val)
                    img_scaled = img_resized.new_full(
                        (self.base_size, self.base_size), fill_scalar
                    )
                    img_scaled[
                        pad_top:pad_top + resize_size,
                        pad_left:pad_left + resize_size
                    ] = img_resized
                else:
                    if isinstance(fill_val, (list, tuple, np.ndarray)):
                        fill_channels = torch.tensor(
                            fill_val, dtype=img_resized.dtype, device=img_resized.device
                        )
                    elif torch.is_tensor(fill_val):
                        fill_channels = fill_val.to(img_resized.device, img_resized.dtype)
                    else:
                        fill_channels = torch.full(
                            (img_resized.shape[0],),
                            float(fill_val),
                            dtype=img_resized.dtype,
                            device=img_resized.device,
                        )
                    img_scaled = img_resized.new_empty(
                        (img_resized.shape[0], self.base_size, self.base_size)
                    )
                    img_scaled[:] = fill_channels.view(-1, 1, 1)
                    img_scaled[
                        :,
                        pad_top:pad_top + resize_size,
                        pad_left:pad_left + resize_size
                    ] = img_resized
            else:
                img_scaled = F.pad(
                    img_resized,
                    [pad_left, pad_top, pad_right, pad_bottom],
                    fill=fill_val
                )

        else:
            # scale == 1.0, 保持原样（仍然需要确保尺寸正确）
            img_scaled = F.resize(img, self.base_size, self.interpolation)

        return img_scaled

    def get_last_scale(self):
        """返回最后一次缩放的因子"""
        return self.last_scale


class TransformWithParams:
    """
    包装transform pipeline，能够提取旋转和缩放参数
    """
    def __init__(self, transform_list):
        """
        Args:
            transform_list: list of transforms
        """
        self.transforms = transform_list
        self.rotation_transform = None
        self.scale_transform = None

        # 找到旋转和缩放的transform
        for t in transform_list:
            if isinstance(t, RandomRotationWithParams):
                self.rotation_transform = t
            elif isinstance(t, RandomScaleWithParams):
                self.scale_transform = t

    def __call__(self, img):
        """应用所有transforms"""
        for t in self.transforms:
            img = t(img)
        return img

    def get_params(self):
        """
        获取最后一次变换的参数
        Returns:
            dict: {'rotation_deg': float, 'scale': float}
        """
        params = {}
        if self.rotation_transform is not None:
            params['rotation_deg'] = self.rotation_transform.get_last_angle()
        else:
            params['rotation_deg'] = 0.0

        if self.scale_transform is not None:
            params['scale'] = self.scale_transform.get_last_scale()
        else:
            params['scale'] = 1.0

        return params


def mk_pil_transform_with_params(
        mean,
        std,
        imgsize2net=224,
        rand_rot=False,
        rotation_range_deg=180,
        rand_scale=False,
        scale_range=(0.8, 1.2),
        rand_affine=False,
        affine_para=None,
        rand_erase=False,
        color_jitter=False,
        center_crop=False,
        rand_crop=False,
        pad_mode=None,
):
    """
    创建带参数追踪的transform pipeline

    Transform顺序（重要）：
    1. Resize/Crop: 初始尺寸调整
    2. Scale: 模拟UAV高度变化（先处理，避免旋转导致内容丢失）
    3. Rotation: 模拟UAV朝向变化（后处理）
    4. 其他增强: Affine, ColorJitter等
    5. ToTensor + Normalize

    Args:
        rand_rot: 是否启用旋转增强
        rotation_range_deg: 旋转范围（度）
        rand_scale: 是否启用尺度增强
        scale_range: 尺度范围 (min, max)
        其他参数同 mk_pil_transform

    Returns:
        TransformWithParams: 可以获取变换参数的transform对象
    """
    transform_list = [transforms.Resize(imgsize2net)]

    if center_crop:
        transform_list.append(transforms.CenterCrop(imgsize2net))
    if rand_crop:
        transform_list.append(transforms.RandomCrop(imgsize2net))

    pad_mode_norm = str(pad_mode).lower() if pad_mode is not None else None
    use_zero = pad_mode_norm in {"zero", "zeros", "0"}

    # ===== 关键顺序：先缩放，再旋转 =====
    # 1. 先添加可追踪的缩放（避免旋转导致的内容丢失）
    if rand_scale:
        transform_list.append(RandomScaleWithParams(
            scale_range=scale_range,
            base_size=imgsize2net,
            pad_mode=pad_mode_norm,
        ))

    # 2. 再添加可追踪的旋转
    if rand_rot:
        transform_list.append(RandomRotationWithParams(
            degrees=rotation_range_deg,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0 if use_zero else "mean"
        ))

    # 其他标准transforms
    if rand_affine:
        default_affine_params = {
            'degrees': 0,  # 旋转已经单独处理
            'translate': (0, 0),
            'scale': (1.0, 1.0),  # 缩放已经单独处理
            'shear': 5,
        }
        if affine_para:
            if not isinstance(affine_para, dict):
                raise TypeError("affine_para必须是一个字典。")
            default_affine_params.update(affine_para)

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

        fill_val = 0 if use_zero else _mean_fill_from_stats(mean)
        transform_list.append(
            transforms.RandomAffine(
                **default_affine_params,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=fill_val
            )
        )

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        )

    transform_list.append(transforms.ToTensor())

    if rand_erase:
        transform_list.append(
            transforms.RandomErasing(p=0.1, scale=(0.05, 0.15), ratio=(0.3, 3.3), value=0.)
        )

    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return TransformWithParams(transform_list)


def apply_augment_to_coords(coords_4d, rotation_deg, scale_factor):
    """
    将数据增强参数应用到4D坐标上

    Args:
        coords_4d: torch.Tensor [N, 4] or [4], format: [nr, nc, rot_rad, scale_ratio]
        rotation_deg: float, 旋转角度（度数），顺时针为正
        scale_factor: float, 缩放因子

    Returns:
        coords_4d_augmented: torch.Tensor, 增强后的坐标

    注意：
        - 位置(nr, nc)不变，因为UAV的地理位置没有改变
        - 旋转角度需要加上增强的旋转：rot_new = rot_old + rot_augment
        - 尺度需要乘以增强的尺度：scale_new = scale_old * scale_augment
    """
    coords_augmented = coords_4d.clone()

    # 旋转增强：将度数转换为弧度并加到原始旋转上
    # 注意：图像旋转是逆时针为正，但航向角通常是顺时针为正
    # 需要根据你的坐标系定义来调整符号
    rot_augment_rad = torch.deg2rad(torch.tensor(rotation_deg, device=coords_4d.device, dtype=coords_4d.dtype))

    if coords_4d.ndim == 1:
        coords_augmented[2] = coords_augmented[2] + rot_augment_rad
        coords_augmented[2] = torch.atan2(torch.sin(coords_augmented[2]), torch.cos(coords_augmented[2]))
    else:  # ndim == 2
        coords_augmented[:, 2] = coords_augmented[:, 2] + rot_augment_rad
        coords_augmented[:, 2] = torch.atan2(torch.sin(coords_augmented[:, 2]), torch.cos(coords_augmented[:, 2]))
    # 尺度增强：乘以缩放因子
    # scale_factor > 1 表示图像放大，相当于UAV飞得更低，视野变小
    # scale_factor < 1 表示图像缩小，相当于UAV飞得更高，视野变大
    if coords_4d.ndim == 1:
        coords_augmented[3] = coords_augmented[3] / scale_factor
    else:
        coords_augmented[:, 3] = coords_augmented[:, 3] / scale_factor

    return coords_augmented


# 测试代码
if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    # 创建测试图像
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # 创建带参数追踪的transform
    transform = mk_pil_transform_with_params(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        imgsize2net=224,
        rand_rot=True,
        rotation_range_deg=180,
        rand_scale=True,
        scale_range=(0.8, 1.2),
        center_crop=True,
    )

    # 应用transform
    img_transformed = transform(img)
    print(f"Transformed image shape: {img_transformed.shape}")

    # 获取参数
    params = transform.get_params()
    print(f"Transform parameters: {params}")

    # 测试坐标变换
    coords_original = torch.tensor([0.5, 0.5, 0.0, 1.0])  # [nr, nc, rot, scale]
    coords_augmented = apply_augment_to_coords(
        coords_original,
        rotation_deg=params['rotation_deg'],
        scale_factor=params['scale']
    )
    print(f"Original coords: {coords_original}")
    print(f"Augmented coords: {coords_augmented}")
