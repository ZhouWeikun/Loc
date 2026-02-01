import math
import torch
import torch.nn.functional as F


def _get_gaussian_kernel_3d(kernel_size=3, sigma=1.0, channels=1, device='cpu'):
    """
    生成 3D 高斯核
    """
    # 创建网格坐标
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # 计算 2D 高斯 (XY)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # 扩展到 3D (Z/Rot) - 这里简化为各向同性，也可以分别设置
    # 为了简单，我们生成三个维度的 grid
    range_vec = torch.arange(kernel_size, dtype=torch.float32, device=device) - mean
    xx, yy, zz = torch.meshgrid(range_vec, range_vec, range_vec, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * variance))

    # 归一化，保证能量守恒
    kernel = kernel / torch.sum(kernel)

    # Reshape 为 conv3d 权重格式: [out_channels, in_channels, k_d, k_h, k_w]
    # 这里 in_channels=1, out_channels=1
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    # 如果需要处理多通道输入（虽然这里通常是单通道概率），可以 repeat
    if channels > 1:
        kernel = kernel.repeat(channels, 1, 1, 1, 1)

    return kernel.to(device)


def _apply_gaussian_smoothing_3d(preds, kernel_size=3, sigma=1.0):
    """
    对概率体进行 3D 高斯平滑，专门处理 Rot 维度的循环填充。
    Args:
        preds: [B, H, W, Rot]
    Returns:
        smoothed_preds: [B, H, W, Rot]
    """
    B, H, W, Rot = preds.shape
    device = preds.device

    # 1. 调整维度以适配 conv3d: [B, C, D, H, W]
    # 我们把 H, W, Rot 对应到 Depth, Height, Width
    # Input needs to be [B, 1, H, W, Rot]
    x = preds.unsqueeze(1)

    # 2. 填充 (Padding)
    # H 和 W 维度使用 'replicate' (边缘复制) 或 'constant' (补0)
    # Rot 维度必须使用 'circular' (循环填充) ! 这一点对无人机朝向至关重要
    pad = kernel_size // 2

    # F.pad 的顺序是倒序的: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # 对应最后三个维度 (Rot, W, H)

    # 先对 Rot (最后一维) 做 Circular Padding
    # 注意：F.pad 对 5D tensor 的 circular padding 支持可能有限，
    # 我们先手动拼接实现 Rot 的 circular padding，或者分开 pad

    # 方案：先 pad W, H (replicate), 再 pad Rot (circular)
    # 这里的维度对应 x: [B, 1, H, W, Rot]

    # Step 2.1: Pad W and H (spatial) using replicate
    # pad_W_left, pad_W_right, pad_H_top, pad_H_bottom
    # 此时只处理最后两维? 不，conv3d 的 pad 比较复杂。
    # 让我们用更通用的方式：手动处理 Circular

    # 取出 Rot 维度的首尾进行拼接
    left_slice = x[..., -pad:]
    right_slice = x[..., :pad]
    x_padded_rot = torch.cat([left_slice, x, right_slice], dim=-1)  # [B, 1, H, W, Rot + 2*pad]

    # Step 2.2: Pad H and W (spatial)
    # 对倒数第2、3维进行 padding
    # F.pad for 5D input pads the last 3 dimensions (D, H, W).
    # Our mapping is (H->D, W->H, Rot->W) for conv3d names, but let's just stick to indices.
    # 我们已经手动处理了最后一维(Rot)，现在只需要 pad H 和 W
    # 使用 replicate 模式
    # F.pad 对 5D tensor: (pad_last_dim, pad_last_dim, pad_2nd_last, pad_2nd_last, pad_3rd_last, pad_3rd_last)
    # 我们刚才手动 pad 了 last dim，现在需要 pad 2nd_last(W) 和 3rd_last(H)
    # 但由于手动 pad 改变了 tensor 形状，混合使用 F.pad 容易出错。
    # 建议：直接构建一个全 padded 的 tensor

    # 简便方法：使用 F.pad 一次性处理 (如果 PyTorch 版本支持 5D circular，否则报错)
    # 许多版本不支持 5D circular。最稳妥的方法是：
    # 1. Rot 维度手动 circular
    # 2. H, W 维度使用 F.pad replicate

    # 手动处理 Rot 已经在上面做了 (x_padded_rot)
    # 现在对 x_padded_rot 的 H(dim 2), W(dim 3) 进行 replicate padding
    # pad 格式: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # 对应维度: (Rot_padded, W, H)
    # 我们需要 pad H 和 W，Rot 已经 pad 过了，所以 Rot 的 pad 为 0
    x_fully_padded = F.pad(x_padded_rot, (0, 0, pad, pad, pad, pad), mode='replicate')

    # 3. 生成核
    kernel = _get_gaussian_kernel_3d(kernel_size, sigma, channels=1, device=device)

    # 4. 卷积
    # x_fully_padded: [B, 1, H+2p, W+2p, Rot+2p]
    out = F.conv3d(x_fully_padded, kernel, padding=0)

    return out.squeeze(1)


def get_batch_adaptive_indices(scores_flat, top_p=0.95, prob_threshold=0.0, min_k=1, normalize=True):
    """
    自适应获取 Batch 中每个样本的 Top-K 索引 (增强版)

    Args:
        scores_flat: [B, N] 原始分数 (Logits 或 Probabilities)
        top_p:       (float) 累积概率阈值 (e.g., 0.95)
        prob_threshold: (float) 绝对概率阈值 (e.g., 0.001)，低于此值的点强制丢弃
        min_k:       (int) 每个样本最少保留多少个点
        normalize:   (bool) 是否对输入进行归一化。建议为 True，除非你确定输入已经是 Sum=1 的概率。

    Returns:
        indices: [B, Max_K]  排序后的索引
        mask:    [B, Max_K]  布尔掩码 (True=有效, False=Padding)
        probs:   [B, Max_K]  对应的概率值
    """
    B, N = scores_flat.shape

    # --- 1. 安全归一化 (关键修复点) ---
    if normalize:
        # 如果包含负数，通常是 Logits -> 使用 Softmax
        if scores_flat.min() < 0 or scores_flat.max() > 1.0:
            probs_flat = F.softmax(scores_flat, dim=-1)
        else:
            # 如果全是正数但和不为1 -> 使用 L1 归一化
            probs_flat = scores_flat / (scores_flat.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        probs_flat = scores_flat

    # --- 2. 全局排序 ---
    sorted_probs, sorted_indices = torch.sort(probs_flat, descending=True, dim=-1)

    # --- 3. 计算 Mask (Top-P) ---
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask_p: 累积概率 < top_p 的部分
    # Shift right: 也就是保留导致累积概率刚刚超过 top_p 的那个临界点
    mask_p = cumulative_probs < top_p
    mask_p = torch.cat([torch.ones(B, 1, device=scores_flat.device, dtype=torch.bool),
                        mask_p[:, :-1]], dim=1)

    # --- 4. 计算 Mask (绝对阈值 - 双重保险) ---
    # 就算 Top-P 囊括了所有点，如果点的值太小，也应该丢弃
    if prob_threshold > 0:
        mask_t = sorted_probs >= prob_threshold
        final_mask = mask_p & mask_t
    else:
        final_mask = mask_p

    # --- 5. 强制 Min-K ---
    if min_k > 0:
        final_mask[:, :min_k] = True

    # --- 6. 动态截断 ---
    lengths = final_mask.sum(dim=1)  # [B]
    max_k = lengths.max().item()

    # 限制 max_k 不超过 N (防止 min_k 设置过大越界)
    max_k = min(max_k, N)

    selected_indices = sorted_indices[:, :max_k]
    selected_probs = sorted_probs[:, :max_k]
    selected_mask = final_mask[:, :max_k]

    return selected_indices, selected_mask, selected_probs, lengths


def filter_batch_by_dynamic_threshold(scores_flat, thresholds, min_topN=64):
    """
    基于每个样本独立的动态阈值进行过滤，并保证最少保留 min_topN 个点。

    Args:
        scores_flat: [B, N] 输入的分数或概率 (需要与 thresholds 的量纲一致，如都是logits或都是probs)
        thresholds:  [B]    每个样本对应的阈值 Tensor
        min_topN:    (int)  兜底参数，每个样本最少保留前 N 个点 (即使它们低于阈值)

    Returns:
        indices: [B, Max_K]  排序后的索引 (已 Padding)
        mask:    [B, Max_K]  布尔掩码 (True=有效选中, False=Padding或低于阈值)
        values:  [B, Max_K]  对应的分数
    """
    B, N = scores_flat.shape

    # 1. 全局排序 (Descending)
    # 无论是 Top-K 还是 Threshold，先排序都是处理 Ragged Batch 的最高效手段
    sorted_scores, sorted_indices = torch.sort(scores_flat, descending=True, dim=-1)

    # 2. 构造动态掩码 (Vectorized)
    # 扩展 thresholds 维度以便广播: [B] -> [B, 1]
    thresholds_exp = thresholds.view(B, 1)

    # 逻辑 A: 阈值过滤 (Scores >= Threshold)
    # 利用广播机制，每行会对比该行对应的 threshold
    mask_threshold = sorted_scores >= thresholds_exp

    # 逻辑 B: 强制保留前 min_topN (Safety Net)
    # 无论阈值多高，前 min_topN 个点必须保留
    if min_topN > 0:
        # 创建一个只在前 min_topN 位置为 True 的掩码
        # 注意：这里我们利用索引位置，因为数据已经排好序了，前 min_topN 就是最大的 min_topN
        # mask_min 形状 [1, N] 或 [B, N] 都可以，这里用广播更省内存
        mask_min = torch.arange(N, device=scores_flat.device).expand(B, N) < min_topN

        # 合并掩码 (逻辑或 OR)
        # 只要满足 "大于阈值" OR "在前 min_topN 里"，就是 True
        final_mask = mask_threshold | mask_min
    else:
        final_mask = mask_threshold

    # 3. 动态截断 (Dynamic Slicing)
    # 计算当前 Batch 里最大的保留长度 (Max K)
    lengths = final_mask.sum(dim=1)  # [B]
    max_k = lengths.max().item()

    # 安全限制 (防止 max_k 意外越界，虽然理论上不会)
    max_k = min(max_k, N)

    # 4. 统一截取
    # 截取到 max_k 长度，PyTorch 会自动处理好数据搬运
    selected_indices = sorted_indices[:, :max_k]
    selected_scores = sorted_scores[:, :max_k]
    selected_mask = final_mask[:, :max_k]

    return selected_indices, selected_mask, selected_scores

