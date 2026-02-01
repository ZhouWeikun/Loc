"""
批量旋转工具
高效地对图像应用多个旋转角度
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import numpy as np


def _mean_fill_from_tensor(img):
    if img.ndim == 3:
        return img.mean(dim=(1, 2)).tolist()
    if img.ndim == 4:
        return img.mean(dim=(2, 3))
    raise TypeError(f"输入必须是3D或4D的Tensor，但收到了维度为{img.ndim}的输入。")

def batch_rotate_images_per_sample(images, angles_deg):
    """
    对每个图像应用不同的旋转角度（一对一）

    Args:
        images: [N, C, H, W] 输入图像
        angles_deg: [N] 或 list of N floats，每个图像对应的旋转角度（度数）

    Returns:
        rotated_images: [N, C, H, W] 旋转后的图像
    """
    N, C, H, W = images.shape
    device = images.device

    # 转换为tensor
    if isinstance(angles_deg, list):
        angles_deg = torch.tensor(angles_deg, dtype=torch.float32, device=device)
    elif isinstance(angles_deg, np.ndarray):
        angles_deg = torch.from_numpy(angles_deg).float().to(device)
    else:
        angles_deg = angles_deg.to(device)

    assert angles_deg.shape[0] == N, f"角度数量({angles_deg.shape[0]})必须等于图像数量({N})"

    # 转为弧度
    angles_rad = angles_deg * (torch.pi / 180.0)

    # 构造每个图像的旋转矩阵
    cos_vals = torch.cos(angles_rad)  # [N]
    sin_vals = torch.sin(angles_rad)  # [N]

    # 仿射变换矩阵 [N, 2, 3]
    theta = torch.zeros(N, 2, 3, device=device)
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = sin_vals
    theta[:, 1, 0] = -sin_vals
    theta[:, 1, 1] = cos_vals

    # 生成采样网格
    grid = F.affine_grid(theta, images.size(), align_corners=False)

    # 应用旋转
    rotated_images = F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return rotated_images


def batch_rotate_images(images, angles_deg):
    """
    批量旋转图像到多个角度（高效实现）

    Args:
        images: [N, C, H, W] 输入图像
        angles_deg: list or array of float, 旋转角度列表（度数）

    Returns:
        rotated_images: [N, R, C, H, W] 旋转后的图像
                        其中R是角度数量
    """
    N, C, H, W = images.shape
    n_angles = len(angles_deg)

    # 方案1：使用torch.nn.functional.affine_grid（推荐，最快）
    # 预计算所有旋转的仿射矩阵
    device = images.device
    angles_rad = torch.tensor(angles_deg, dtype=torch.float32, device=device) * (torch.pi / 180.0)

    # 构造旋转矩阵
    cos_vals = torch.cos(angles_rad)  # [R]
    sin_vals = torch.sin(angles_rad)  # [R]

    # 仿射变换矩阵 [R, 2, 3]
    theta = torch.zeros(n_angles, 2, 3, device=device)
    theta[:, 0, 0] = cos_vals   # cos
    theta[:, 0, 1] = sin_vals   # sin
    theta[:, 1, 0] = -sin_vals  # -sin
    theta[:, 1, 1] = cos_vals   # cos

    # 扩展到所有图像: [N*R, 2, 3]
    theta_expanded = theta.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N * n_angles, 2, 3)

    # 扩展图像: [N*R, C, H, W]
    images_expanded = images.unsqueeze(1).repeat(1, n_angles, 1, 1, 1).reshape(N * n_angles, C, H, W)

    # 生成采样网格
    grid = F.affine_grid(theta_expanded, images_expanded.size(), align_corners=False)

    # 应用旋转
    rotated = F.grid_sample(images_expanded, grid, mode='bilinear', padding_mode='border', align_corners=False)

    # Reshape回 [N, R, C, H, W]
    rotated_images = rotated.reshape(N, n_angles, C, H, W)

    return rotated_images


def batch_rotate_images_sequential(images, angles_deg, batch_size=8):
    """
    批量旋转图像（分批处理版本，节省内存）

    Args:
        images: [N, C, H, W] 输入图像
        angles_deg: list of float, 旋转角度列表（度数）
        batch_size: 每次处理的角度数量

    Returns:
        rotated_images: [N, R, C, H, W] 旋转后的图像
    """
    N, C, H, W = images.shape
    n_angles = len(angles_deg)
    device = images.device

    result_list = []

    # 分批处理角度
    for i in range(0, n_angles, batch_size):
        angles_batch = angles_deg[i:i + batch_size]
        rotated_batch = batch_rotate_images(images, angles_batch)
        result_list.append(rotated_batch)

    # 拼接所有批次
    rotated_images = torch.cat(result_list, dim=1)  # [N, R, C, H, W]

    return rotated_images


def rotate_with_torchvision(images, angles_deg):
    """
    使用torchvision的rotate函数（较慢，但兼容性好）

    Args:
        images: [N, C, H, W] 输入图像
        angles_deg: list of float, 旋转角度列表（度数）

    Returns:
        rotated_images: [N, R, C, H, W] 旋转后的图像
    """
    N, C, H, W = images.shape
    n_angles = len(angles_deg)

    rotated_list = []

    fill_vals = _mean_fill_from_tensor(images)
    if torch.is_tensor(fill_vals):
        fill_vals = fill_vals.detach().cpu().tolist()
    for angle in angles_deg:
        # 对所有图像应用当前角度
        per_angle = []
        for i in range(N):
            fill_val = fill_vals[i] if isinstance(fill_vals, list) else fill_vals
            per_angle.append(
                TF.rotate(images[i], angle, interpolation=TF.InterpolationMode.BILINEAR, fill=fill_val)
            )
        rotated_list.append(torch.stack(per_angle))

    # Stack: [R, N, C, H, W] -> [N, R, C, H, W]
    rotated_images = torch.stack(rotated_list, dim=0).permute(1, 0, 2, 3, 4)

    return rotated_images


# ==================== 性能测试 ====================

if __name__ == '__main__':
    import time

    # 创建测试数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 100  # 图像数量
    C, H, W = 3, 224, 224
    n_angles = 36  # 旋转角度数

    images = torch.randn(N, C, H, W, device=device)
    angles_deg = [-180 + 10 * i for i in range(n_angles)]

    print(f"测试配置:")
    print(f"  设备: {device}")
    print(f"  图像数量: {N}")
    print(f"  图像尺寸: {C}x{H}x{W}")
    print(f"  旋转角度数: {n_angles}")
    print()

    # ========== 方法1：批量旋转（推荐） ==========
    print("方法1: batch_rotate_images (全批量)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated1 = batch_rotate_images(images, angles_deg)

    torch.cuda.synchronize() if device == 'cuda' else None
    time1 = time.time() - start
    print(f"  时间: {time1:.3f}s")
    print(f"  输出shape: {rotated1.shape}")
    print()

    # ========== 方法2：分批旋转（节省内存） ==========
    print("方法2: batch_rotate_images_sequential (分批)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated2 = batch_rotate_images_sequential(images, angles_deg, batch_size=8)

    torch.cuda.synchronize() if device == 'cuda' else None
    time2 = time.time() - start
    print(f"  时间: {time2:.3f}s")
    print(f"  输出shape: {rotated2.shape}")
    print()

    # ========== 方法3：torchvision逐个旋转（慢） ==========
    print("方法3: rotate_with_torchvision (逐个角度)")
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated3 = rotate_with_torchvision(images, angles_deg)

    torch.cuda.synchronize() if device == 'cuda' else None
    time3 = time.time() - start
    print(f"  时间: {time3:.3f}s")
    print(f"  输出shape: {rotated3.shape}")
    print()

    # ========== 方法4：原始方法（最慢，作为baseline） ==========
    print("方法4: 原始循环方法 (baseline)")
    from util_mk_data_transform import RandomRotationWithAngle
    rotater = RandomRotationWithAngle(degrees=180, same_on_batch=True)

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()

    rotated4_list = []
    for angle in angles_deg:
        rotated = rotater(images, angle)
        rotated4_list.append(rotated)
    rotated4 = torch.stack(rotated4_list, dim=1)

    torch.cuda.synchronize() if device == 'cuda' else None
    time4 = time.time() - start
    print(f"  时间: {time4:.3f}s")
    print(f"  输出shape: {rotated4.shape}")
    print()

    # ========== 性能对比 ==========
    print("=" * 50)
    print("性能对比:")
    print(f"  方法1 (全批量):    {time1:.3f}s  (1.00x)")
    print(f"  方法2 (分批):      {time2:.3f}s  ({time2/time1:.2f}x)")
    print(f"  方法3 (torchvision): {time3:.3f}s  ({time3/time1:.2f}x)")
    print(f"  方法4 (原始循环):  {time4:.3f}s  ({time4/time1:.2f}x)")
    print()
    print(f"加速比 (相对于原始方法):")
    print(f"  方法1: {time4/time1:.1f}x")
    print(f"  方法2: {time4/time2:.1f}x")
    print(f"  方法3: {time4/time3:.1f}x")
