import numpy as np
import torch


def compute_recall_by_label(q_labels, pred_labels_per_query, k_values=(1, 5), normalize=True):
    """
    Compute recall@k for label-based retrieval.

    Args:
        q_labels: sequence where each item is a set/list/array of valid labels for one query
        pred_labels_per_query: [N, K] predicted labels
        k_values: recall cutoffs
        normalize: if False, return hit counts instead of percentages
    """
    pred_labels = np.asarray(pred_labels_per_query)
    if pred_labels.ndim != 2:
        raise ValueError(f"pred_labels_per_query must be 2D, got shape {pred_labels.shape}")

    correct_at_k = np.zeros(len(k_values), dtype=np.float64)
    for q_idx, pred in enumerate(pred_labels):
        gt_labels = np.asarray(q_labels[q_idx]).reshape(-1)
        for i, k in enumerate(k_values):
            if np.any(np.in1d(pred[:k], gt_labels)):
                correct_at_k[i:] += 1.0
                break

    if normalize:
        correct_at_k /= max(len(pred_labels), 1)
        correct_at_k *= 100.0
    return {f"top{k}_acc": float(v) for k, v in zip(k_values, correct_at_k)}


def compute_topk_acc_from_coords(
        coords_pred,
        coords_gt,
        dist_th,
        rot_th_deg=None,
        scale_ratio_th=None,
        k_values=(1, 5, 10, 20),
):
    """
    Compute coordinate retrieval accuracy and top-1 error stats.

    Args:
        coords_pred: [B, K, 4]
        coords_gt: [B, 4]
        dist_th: threshold on nr/nc distance
        rot_th_deg: threshold on rotation error in degrees; None to ignore
        scale_ratio_th: threshold on multiplicative scale error; None to ignore
        k_values: top-k cutoffs
    """
    if not torch.is_tensor(coords_pred):
        coords_pred = torch.as_tensor(coords_pred, dtype=torch.float32)
    if not torch.is_tensor(coords_gt):
        coords_gt = torch.as_tensor(coords_gt, dtype=torch.float32)

    coords_pred = coords_pred.to(torch.float32)
    coords_gt = coords_gt.to(coords_pred.device, dtype=torch.float32)

    if coords_pred.ndim != 3 or coords_pred.shape[-1] != 4:
        raise ValueError(f"coords_pred must be [B, K, 4], got {tuple(coords_pred.shape)}")
    if coords_gt.ndim != 2 or coords_gt.shape[-1] != 4:
        raise ValueError(f"coords_gt must be [B, 4], got {tuple(coords_gt.shape)}")

    coords_gt_expanded = coords_gt.unsqueeze(1)

    dist_errors = torch.norm(coords_pred[..., :2] - coords_gt_expanded[..., :2], p=2, dim=-1)

    rot_diff_rad = torch.abs(coords_pred[..., 2] - coords_gt_expanded[..., 2])
    rot_errors_rad = torch.min(rot_diff_rad, 2 * torch.pi - rot_diff_rad)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)

    pred_scale = coords_pred[..., 3].clamp(min=1e-6)
    gt_scale = coords_gt_expanded[..., 3].clamp(min=1e-6)
    scale_ratio = torch.maximum(pred_scale / gt_scale, gt_scale / pred_scale)

    is_hit = dist_errors <= float(dist_th)
    if rot_th_deg is not None:
        is_hit = is_hit & (rot_errors_deg <= float(rot_th_deg))
    if scale_ratio_th is not None:
        is_hit = is_hit & (scale_ratio <= float(scale_ratio_th))

    metrics = {}
    max_pred = coords_pred.shape[1]
    for k in k_values:
        k = int(k)
        if k <= 0:
            raise ValueError("k_values must be positive.")
        if k > max_pred:
            metrics[f"top{k}_acc"] = 0.0
            continue
        hit_in_k = is_hit[:, :k].any(dim=1)
        metrics[f"top{k}_acc"] = float(hit_in_k.float().mean().item() * 100.0)

    top1_dist = dist_errors[:, 0]
    top1_rot = rot_errors_deg[:, 0]
    top1_scale = scale_ratio[:, 0]
    errors = {
        "mean_dist_err_top1": float(top1_dist.mean().item()),
        "median_dist_err_top1": float(torch.median(top1_dist).item()),
        "mean_rot_err_top1": float(top1_rot.mean().item()),
        "median_rot_err_top1": float(torch.median(top1_rot).item()),
        "mean_scale_ratio_top1": float(top1_scale.mean().item()),
        "median_scale_ratio_top1": float(torch.median(top1_scale).item()),
    }
    return metrics, errors


def print_topk_eval_results(metrics, errors, thresholds, report_title="Retrieval Accuracy Report", report_meta=None, log_lines=None):
    """
    Pretty-print retrieval metrics with the same threshold semantics as fine localization.
    """
    def _emit(line=""):
        print(line)
        if log_lines is not None:
            log_lines.append(line)

    dist_th = thresholds.get("norm_dist", "N/A")
    rot_th = thresholds.get("rot", None)
    scale_ratio_th = thresholds.get("scale_ratio", None)
    report_meta = report_meta or {}

    rot_msg = "None (Ignored)" if rot_th is None else f"{rot_th}°"
    scale_msg = "None (Ignored)" if scale_ratio_th is None else f"{scale_ratio_th:.3f}x"
    integrate_scale = bool(report_meta.get("integrate_scale", scale_ratio_th is not None))
    scale_select_mode = report_meta.get("scale_select_mode", None)
    dist_msg = f"{dist_th:.4f}" if isinstance(dist_th, (int, float)) else str(dist_th)

    _emit(
        f"\n{'=' * 20} {report_title} | "
        f"Integerate Scale:{integrate_scale} | "
        f"scale_select_mode:{scale_select_mode} {'=' * 20}"
    )
    _emit(f"Thresholds -> Dist: {dist_msg}, Rot: {rot_msg}, Scale: {scale_msg}")
    _emit("-" * 75)
    _emit(f"{'Metric':<15} | {'Accuracy (%)':<15}")
    _emit("-" * 35)

    keys = sorted(
        [k for k in metrics.keys() if k.startswith("top") and k.endswith("_acc")],
        key=lambda x: int(x.replace("top", "").replace("_acc", "")),
    )
    for key in keys:
        _emit(f"{key:<15} | {metrics[key]:<15.2f}")

    _emit("-" * 75)
    _emit("Top-1 Error Stats (Global Average & Median):")
    _emit(
        f"  Dist  Error: Mean={errors['mean_dist_err_top1']:.4f},   "
        f"Median={errors['median_dist_err_top1']:.4f}"
    )
    _emit(
        f"  Rot   Error: Mean={errors['mean_rot_err_top1']:.2f}°,    "
        f"Median={errors['median_rot_err_top1']:.2f}°"
    )
    _emit(
        f"  Scale Error: Mean={errors['mean_scale_ratio_top1']:.3f}x,    "
        f"Median={errors['median_scale_ratio_top1']:.3f}x (Ratio)"
    )
    _emit(f"{'=' * 75}\n")
