import torch
# 我们的函数保持不变
def unravel_index(flat_indices, shape):
    shape_tensor = torch.as_tensor(shape, device=flat_indices.device)
    strides = torch.flip(torch.cumprod(torch.flip(shape_tensor, dims=[0]), dim=0), dims=[0])
    strides = torch.cat((strides[1:], torch.tensor([1], device=flat_indices.device)))

    multi_dim_coords = []
    remainder = flat_indices
    for i in range(len(shape)):
        coord = remainder // strides[i]
        multi_dim_coords.append(coord)
        remainder = remainder % strides[i]

    return tuple(multi_dim_coords)