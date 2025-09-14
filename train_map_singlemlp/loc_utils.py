import torch
import torch.nn.functional as F

def find_4neighbors_topleft(pred_pdf_agged):
    """
    pred_pdf_agged: a 2d map with shape = [H, W] or a 3d map with shape = [C, H, W]
    """
    kernel = torch.tensor([
        [1, 1],
        [1, 1]
    ], dtype=torch.float32)
    if len(pred_pdf_agged.shape)==2:
        # 找到最大的四个相邻值位置，单个响应图：
        # input_4d = pred_pdf_agged.reshape(8, 10).unsqueeze(0).unsqueeze(0)
        input_4d = pred_pdf_agged.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        # 这会导致输出尺寸比输入尺寸在长和宽上各小1
        sum_map = F.conv2d(input_4d, kernel_4d.to(pred_pdf_agged.device), padding=0)
        sum_map_2d = sum_map.squeeze()
        # 找出最大和及其左上角位置 ---
        top_left_flat_index = torch.argmax(sum_map_2d)
        top_left_coords = (top_left_flat_index // sum_map_2d.shape[1], top_left_flat_index % sum_map_2d.shape[1])
        top_left_row, top_left_col = top_left_coords
        return torch.tensor([top_left_row, top_left_col])
    # 找到最大的四个相邻值位置，batched：
    else:
        # input_4d = pred_pdf_agged.reshape(-1, 1, 8, 10)
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


def compute_agged_pred_4neighbors_id(pred_seq_agged,h,w):
    """
    pred_seq_agged: with shape = [C, H, W] or shape = [C, H*W]
    """
    id_toplefts = find_4neighbors_topleft(pred_seq_agged.reshape(-1, h, w))
    id_toprights = id_toplefts + torch.tensor([0, 1],device=pred_seq_agged.device)
    id_buttonlefts = id_toplefts + torch.tensor([1, 0],device=pred_seq_agged.device)
    id_buttonrights = id_toplefts + torch.tensor([1, 1],device=pred_seq_agged.device)
    id_4neighbors = torch.stack([id_toplefts, id_toprights, id_buttonlefts, id_buttonrights]).permute(1, 0, 2)
    id_4neighbors_flat = id_4neighbors[..., 0] * w + id_4neighbors[..., 1]
    return id_4neighbors_flat


def agg_seq_pdf(mlp_pred_pdf,window_len = 10,padding=False):
    """
    mlp_pred_pdf:  with shape = [C, H*W]
    """
    agg_gs_dustb = torch.distributions.Normal(loc=window_len,scale=window_len)  # 2sigma that contains 99% pdf = grid_cell_radius,scale=sigma
    agg_weight_gs = torch.exp(agg_gs_dustb.log_prob(torch.arange(window_len, dtype=torch.float32, device=mlp_pred_pdf.device)))
    agg_weight_gs = (agg_weight_gs / agg_weight_gs.sum(dim=-1, keepdim=True)).to(mlp_pred_pdf.device)
    # agg_weight_uniform = torch.ones(window_len).to(mlp_pred_pdf.device)
    agg_weight = agg_weight_gs
    if padding:
        pred_pdf_agged = torch.concatenate([torch.zeros(window_len - 1, mlp_pred_pdf.shape[1], device=mlp_pred_pdf.device), mlp_pred_pdf], dim=0)
        pred_pdf_agged = F.conv1d(pred_pdf_agged.T.unsqueeze(1), agg_weight[None, None, :])
        pred_pdf_agged = pred_pdf_agged.squeeze().T
        # pred_pdf_agged_np = pred_pdf_agged.detach().cpu().numpy()
        # q_labels = torch.argmax(gt_pdf, dim=-1)
        # pred_vals_per_query, pred_labels_per_query = torch.sort(pred_pdf_agged, dim=-1, descending=True)
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