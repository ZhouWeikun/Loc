import torch
import torch.nn.functional as F


def compute_l2_dist(X, Y, normalize=False):
    """
    使用 PyTorch 矩阵运算快速计算两组向量之间的欧式距离。

    参数:
    X: torch.Tensor, 形状为 (m, d)，代表第一组的 ma 个 d 维向量。
    Y: torch.Tensor, 形状为 (n, d)，代表第二组的 n 个 d 维向量。
    normalize: bool, 如果为True，则在计算距离前对所有向量进行L2归一化。

    返回:
    D: torch.Tensor, 形状为 (m, n)，D[i, j] 是 X[i] 和 Y[j] 之间的距离。
    """
    # 确保张量在同一设备上
    assert X.device == Y.device, "Tensors must be on the same device"

    # --- 新增：向量归一化步骤 ---
    if normalize:
        # 增加一个极小值以保证数值稳定性 (防止除以0)
        eps = 1e-8

        # 对 X 中的每个向量进行L2归一化
        # torch.norm(X, p=2, dim=1, keepdim=True) 计算每个向量的L2范数，形状为 (m, 1)
        X_norms = torch.norm(X, p=2, dim=1, keepdim=True)
        X = X / (X_norms + eps)

        # 对 Y 中的每个向量进行L2归一化
        Y_norms = torch.norm(Y, p=2, dim=1, keepdim=True)
        Y = Y / (Y_norms + eps)
    # --------------------------

    # 计算 X 中每个向量的模长平方
    sum_X_sq = X.pow(2).sum(dim=1, keepdim=True)

    # 计算 Y 中每个向量的模长平方
    sum_Y_sq = Y.pow(2).sum(dim=1)

    # 计算点积项: -2 * X @ Y.T
    dot_product = -2 * (X @ Y.T)

    # 利用广播机制计算距离的平方
    D_sq = sum_X_sq + sum_Y_sq + dot_product

    # 处理浮点数精度问题并开方
    D = torch.sqrt(torch.clamp(D_sq, min=0.0))

    return D
##################################################################################################


def find_4neighbors_topleft(pred_pdf_agged):
    """
    pred_pdf_agged:  a 3d map with shape = [C, H, W]
    """
    kernel = torch.tensor([
        [1, 1],
        [1, 1]
    ], dtype=torch.float32)
    input_4d = pred_pdf_agged.unsqueeze(1)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    sum_map = F.conv2d(input_4d, kernel_4d.to(pred_pdf_agged.device), padding=0)
    sum_map_squeezed = sum_map.squeeze()
    out_channels, out_height, out_width = sum_map_squeezed.shape
    sum_map_flat = sum_map_squeezed.view(out_channels, -1)
    max_sums, top_left_flat_indices = torch.max(sum_map_flat, dim=1)
    top_left_rows = top_left_flat_indices // out_width
    top_left_cols = top_left_flat_indices % out_width
    top_left_coords = torch.stack((top_left_rows, top_left_cols), dim=1)
    return top_left_coords


def find_nneighbors_topleft(pred_pdf_agged, n_len):
    """
    寻找具有最大值的 n_len x n_len 区域的左上角坐标。

    Args:
        pred_pdf_agged: 一个形状为 [H, W] 的二维图或形状为 [C, H, W] 的三维图。
        n_len (int): 正方形卷积核的边长。

    Returns:
        一个形状为 [C, 2] 的张量，包含每个通道中最大和区域的左上角 (行, 列) 坐标。
    """
    # 动态创建一个 n_len x n_len 的全1卷积核
    kernel = torch.ones((n_len, n_len), dtype=torch.float32)

    # 准备用于卷积的四维张量
    # 输入需要是 4D: [批次数, 输入通道数, 高, 宽]
    # pred_pdf_agged 是 [C, H, W]，我们把 C 当作批次数
    input_4d = pred_pdf_agged.unsqueeze(1)  # 形状变为 [C, 1, H, W]

    # 卷积核需要是 4D: [输出通道数, 输入通道数, 核高, 核宽]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, n, n]

    # 使用卷积计算每个 n x n 窗口的总和
    # 结果图中的每个像素值，等于原图中以该像素为左上角的 n x n 区域的和
    sum_map = F.conv2d(input_4d, kernel_4d.to(pred_pdf_agged.device), padding=0)

    # 压缩张量，移除大小为1的维度，形状变为 [C, out_height, out_width]
    sum_map_squeezed = sum_map.squeeze(1)

    # 获取输出图的尺寸
    out_channels, out_height, out_width = sum_map_squeezed.shape

    # 将空间维度展平，以便于寻找最大值
    sum_map_flat = sum_map_squeezed.view(out_channels, -1)

    # 找到最大和及其对应的展平后的一维索引
    max_sums, top_left_flat_indices = torch.max(sum_map_flat, dim=1)

    # 将一维索引转换回二维坐标
    top_left_rows = top_left_flat_indices // out_width
    top_left_cols = top_left_flat_indices % out_width

    # 将行和列坐标堆叠起来
    top_left_coords = torch.stack((top_left_rows, top_left_cols), dim=1)

    return top_left_coords


def compute_agged_pred_4neighbors_id(pred_seq_agged):
    """
    pred_seq_agged: with shape = [C, H, W]
    """
    h,w = pred_seq_agged.shape[-2:]
    id_toplefts = find_4neighbors_topleft(pred_seq_agged)
    id_toprights = id_toplefts + torch.tensor([0, 1],device=pred_seq_agged.device)
    id_buttonlefts = id_toplefts + torch.tensor([1, 0],device=pred_seq_agged.device)
    id_buttonrights = id_toplefts + torch.tensor([1, 1],device=pred_seq_agged.device)
    id_4neighbors = torch.stack([id_toplefts, id_toprights, id_buttonlefts, id_buttonrights]).permute(1, 0, 2)
    id_4neighbors_flat = id_4neighbors[..., 0] * w + id_4neighbors[..., 1]
    return id_4neighbors_flat


def compute_agged_pred_nneighbors_id(pred_seq_agged,n_len,ret_2d=False):
    """
    pred_seq_agged: with shape = [C, H, W]
    """
    h,w = pred_seq_agged.shape[-2:]
    # 1. 找到和最大 n x n 窗口的左上角坐标
    id_toplefts = find_nneighbors_topleft(pred_seq_agged,n_len)
    # 2. 生成一个 n x n 正方形内的所有相对偏移量
    #    这将创建一个从 (0,0) 到 (n-1, n-1) 的坐标网格
    row_offsets, col_offsets = torch.meshgrid(
        torch.arange(n_len, device=pred_seq_agged.device),
        torch.arange(n_len, device=pred_seq_agged.device),
        indexing='ij'  # 保证行和列的顺序正确
    )
    # 将偏移量网格展平成 [n*n, 2] 的坐标列表
    offsets = torch.stack([row_offsets.flatten(), col_offsets.flatten()], dim=1)

    # 3. 将偏移量加到左上角坐标上，得到所有 n*n 个点的坐标
    #    这里使用了 PyTorch 的广播机制，非常高效:
    #    id_toplefts.unsqueeze(1)  -> [C, 1, 2]
    #    offsets.unsqueeze(0)      -> [1, n*n, 2]
    #    id_n_neighbors (结果)      -> [C, n*n, 2]
    id_n_neighbors = id_toplefts.unsqueeze(1) + offsets.unsqueeze(0)

    # 4. 将 n*n 个点的二维坐标转换为一维索引
    #    公式: 索引 = 行坐标 * 宽度 + 列坐标
    id_n_neighbors_flat = id_n_neighbors[..., 0] * w + id_n_neighbors[..., 1]

    if not ret_2d:
        return id_n_neighbors_flat
    else:
        return id_n_neighbors_flat,id_n_neighbors


def agg_seq_pdf(mlp_pred_pdf, window_len = 10,padding=False):
    """
    mlp_pred_pdf:  with shape = [C, H*W]
    """
    agg_gs_dustb = torch.distributions.Normal(loc=window_len,scale=window_len)  # 2sigma that contains 99% pdf = grid_cell_radius,scale=sigma
    agg_weight_gs = torch.exp(agg_gs_dustb.log_prob(torch.arange(window_len, dtype=torch.float32, device=mlp_pred_pdf.device)))
    agg_weight_gs = (agg_weight_gs / agg_weight_gs.sum(dim=-1, keepdim=True)).to(mlp_pred_pdf.device)
    agg_weight = agg_weight_gs
    if padding:
        pred_pdf_agged = torch.concatenate([torch.zeros(window_len - 1, mlp_pred_pdf.shape[1], device=mlp_pred_pdf.device), mlp_pred_pdf], dim=0)
        pred_pdf_agged = F.conv1d(pred_pdf_agged.T.unsqueeze(1), agg_weight[None, None, :])
        pred_pdf_agged = pred_pdf_agged.squeeze().T
    else:
        pred_pdf_agged = F.conv1d(mlp_pred_pdf.T.unsqueeze(1), agg_weight[None, None, :])
        pred_pdf_agged = pred_pdf_agged.squeeze().T

    return pred_pdf_agged


def compute_recall(self, pred_pdf, gt_pdf, gt_nrcs):
    id_gt = torch.argmax(gt_pdf, dim=-1, keepdim=False)
    id_pred = torch.argmax(pred_pdf, dim=-1, keepdim=False)
    recall_1 = (id_pred == id_gt).sum() / id_pred.shape[0]
    # print(f"recall={recall_1:.4f}")
    pred_rcs = self.grid_centers[id_pred]
    dist_rel2radius = torch.norm(pred_rcs - gt_nrcs, dim=-1) / self.sat_dataloader.dataset.grid_cell_radius
    dist_rel2radius_recall_1 = dist_rel2radius.mean()
    # print(f"dist_rel2radius_recall_1={dist_rel2radius_recall_1:.4f}")
    return recall_1,dist_rel2radius_recall_1

