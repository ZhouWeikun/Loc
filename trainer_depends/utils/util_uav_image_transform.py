import torch
import torch.nn.functional as TF


def warp_uav_imgs(imgs, rot_rad=None, scale_f=None):
    if rot_rad is None and scale_f is None:
        return imgs

    batch_size = imgs.shape[0]
    device = imgs.device
    dtype = imgs.dtype

    if rot_rad is None:
        rot_rad = torch.zeros(batch_size, device=device, dtype=dtype)
    else:
        rot_rad = rot_rad.to(device=device, dtype=dtype)

    if scale_f is None:
        scale_f = torch.ones(batch_size, device=device, dtype=dtype)
    else:
        scale_f = scale_f.to(device=device, dtype=dtype)

    cos_v = torch.cos(rot_rad)
    sin_v = torch.sin(rot_rad)

    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_v * scale_f
    theta[:, 0, 1] = sin_v * scale_f
    theta[:, 1, 0] = -sin_v * scale_f
    theta[:, 1, 1] = cos_v * scale_f

    grid = TF.affine_grid(theta, imgs.size(), align_corners=False)
    return TF.grid_sample(imgs, grid, mode='bilinear', padding_mode='border', align_corners=False)
