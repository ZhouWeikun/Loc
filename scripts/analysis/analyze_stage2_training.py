#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze Stage-2 training logs and plot metric curves.

Example:
  /root/miniconda3/envs/neuloc_wisp/bin/python scripts/analysis/util_analyze_stage2_training.py \
    --log-files \
      gen_fm_exps/logs/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1.log \
      gen_fm_exps/logs/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B2/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B2.log \
    --metrics recall1 recall5 loss \
    --dir2save gen_fm_exps/analysis/stage2_plots \
    --name2save width_compare
"""

import argparse
import os
import re
from collections import defaultdict


SUPPORTED_METRICS = (
    "loss",
    "recall1",
    "recall5",
    "recall10",
    "recall20",
    "recall50",
    "recall256",
    "recall512",
    "recall1024",
    "error_rc_norm",
    "error_rc_meter",
    "mean_l2",
    "mean_cos",
    "hashgrid_var",
    "vis_var",
    "recall_gap_1_5",
)

METRIC_LABELS = {
    "loss": "Loss",
    "recall1": "Recall@1 (%)",
    "recall5": "Recall@5 (%)",
    "recall10": "Recall@10 (%)",
    "recall20": "Recall@20 (%)",
    "recall50": "Recall@50 (%)",
    "recall256": "Recall@256 (%)",
    "recall512": "Recall@512 (%)",
    "recall1024": "Recall@1024 (%)",
    "error_rc_norm": "RC Error Norm",
    "error_rc_meter": "RC Error (m)",
    "mean_l2": "Mean L2 Distance",
    "mean_cos": "Mean Cosine Similarity",
    "hashgrid_var": "HashGrid Variance Mean",
    "vis_var": "Vis Variance Mean",
    "recall_gap_1_5": "Recall@5 - Recall@1 Gap (%)",
}

EPOCH_RE = re.compile(r"Epoch (\d+)/\d+")
LOSS_RE = re.compile(r"loss=([-+0-9.eE]+)")
MEAN_L2_RE = re.compile(r"mean值- L2距离: ([-+0-9.eE]+)")
MEAN_COS_RE = re.compile(r"mean值- 余弦相似度: ([-+0-9.eE]+)")
HASHGRID_VAR_RE = re.compile(r"HashGrid方差均值: ([-+0-9.eE]+)")
VIS_VAR_RE = re.compile(r"Vis方差均值: ([-+0-9.eE]+)")
RC_RE = re.compile(
    r"\[Train Eval\]\[RC\] N=\d+ "
    r"R@1=([-+0-9.eE]+)% \| "
    r"R@5=([-+0-9.eE]+)% \| "
    r"R@10=([-+0-9.eE]+)% \| "
    r"R@20=([-+0-9.eE]+)% \| "
    r"R@50=([-+0-9.eE]+)% \| "
    r"R@256=([-+0-9.eE]+)% \| "
    r"R@512=([-+0-9.eE]+)% \| "
    r"R@1024=([-+0-9.eE]+)%"
)
RC_ERR_RE = re.compile(
    r"\[Train Eval\]\[RC\] error_rc_norm=([-+0-9.eE]+), "
    r"error_rc_meter=([-+0-9.eE]+)m"
)


def _ensure_metric_supported(metrics):
    invalid = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if invalid:
        raise ValueError(
            f"Unsupported metrics: {invalid}. Supported metrics: {list(SUPPORTED_METRICS)}"
        )


def _normalize_metrics(metrics):
    normalized = []
    for metric in metrics:
        if metric is None:
            continue
        for item in str(metric).split(","):
            item = item.strip()
            if item:
                normalized.append(item)
    return normalized


def _resolve_log_path(log_path_or_dir):
    path = os.path.abspath(log_path_or_dir)
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        dirname = os.path.basename(path.rstrip(os.sep))
        candidates = [
            os.path.join(path, f"{dirname}.log"),
            os.path.join(path, "train.log"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No log file found under directory: {path}. "
            f"Tried: {candidates}"
        )

    raise FileNotFoundError(f"Log file or directory not found: {path}")


def _infer_series_label(log_path):
    basename = os.path.basename(log_path)
    stem, _ = os.path.splitext(basename)
    if stem and stem != "train":
        return stem
    return os.path.basename(os.path.dirname(log_path))


def _build_gap_series(left_points, right_points):
    left_map = {epoch: value for epoch, value in left_points}
    right_map = {epoch: value for epoch, value in right_points}
    epochs = sorted(set(left_map) & set(right_map))
    return [(epoch, right_map[epoch] - left_map[epoch]) for epoch in epochs]


def parse_stage2_training_log(log_path):
    """
    Parse a Stage-2 training log into per-metric epoch-value series.

    Args:
        log_path: Path to one Stage-2 log file.

    Returns:
        dict[str, list[tuple[int, float]]]: metric -> [(epoch, value), ...]
    """
    log_path = _resolve_log_path(log_path)

    series = defaultdict(list)
    current_epoch = None

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            match = EPOCH_RE.search(line)
            if match:
                current_epoch = int(match.group(1))

            if current_epoch is None:
                continue

            match = LOSS_RE.search(line)
            if match:
                series["loss"].append((current_epoch, float(match.group(1))))
                continue

            match = MEAN_L2_RE.search(line)
            if match:
                series["mean_l2"].append((current_epoch, float(match.group(1))))
                continue

            match = MEAN_COS_RE.search(line)
            if match:
                series["mean_cos"].append((current_epoch, float(match.group(1))))
                continue

            match = HASHGRID_VAR_RE.search(line)
            if match:
                series["hashgrid_var"].append((current_epoch, float(match.group(1))))
                continue

            match = VIS_VAR_RE.search(line)
            if match:
                series["vis_var"].append((current_epoch, float(match.group(1))))
                continue

            match = RC_RE.search(line)
            if match:
                series["recall1"].append((current_epoch, float(match.group(1))))
                series["recall5"].append((current_epoch, float(match.group(2))))
                series["recall10"].append((current_epoch, float(match.group(3))))
                series["recall20"].append((current_epoch, float(match.group(4))))
                series["recall50"].append((current_epoch, float(match.group(5))))
                series["recall256"].append((current_epoch, float(match.group(6))))
                series["recall512"].append((current_epoch, float(match.group(7))))
                series["recall1024"].append((current_epoch, float(match.group(8))))
                continue

            match = RC_ERR_RE.search(line)
            if match:
                series["error_rc_norm"].append((current_epoch, float(match.group(1))))
                series["error_rc_meter"].append((current_epoch, float(match.group(2))))

    if "recall1" in series and "recall5" in series:
        series["recall_gap_1_5"] = _build_gap_series(series["recall1"], series["recall5"])

    return dict(series)


def plot_stage2_training_curves(log_files, metrics, dir2save, name2save, labels=None, dpi=160):
    """
    Plot one figure per metric for one or more Stage-2 log files.

    Args:
        log_files: List of Stage-2 log file paths.
        metrics: Metrics to plot. One metric -> one figure.
        dir2save: Output directory.
        name2save: Filename prefix for all figures.
        labels: Optional legend labels, same length as log_files.
        dpi: Figure dpi.

    Returns:
        list[str]: Saved figure paths.
    """
    if not log_files:
        raise ValueError("log_files must not be empty.")
    if not name2save:
        raise ValueError("name2save must not be empty.")

    metrics = _normalize_metrics(metrics)
    _ensure_metric_supported(metrics)

    resolved_logs = [_resolve_log_path(path) for path in log_files]
    if labels is not None and len(labels) != len(resolved_logs):
        raise ValueError("labels length must match log_files length.")

    parsed = {path: parse_stage2_training_log(path) for path in resolved_logs}
    if labels is None:
        labels = [_infer_series_label(path) for path in resolved_logs]

    dir2save = os.path.abspath(dir2save)
    os.makedirs(dir2save, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        has_curve = False

        for log_path, label in zip(resolved_logs, labels):
            points = parsed[log_path].get(metric, [])
            if not points:
                continue
            has_curve = True
            epochs = [item[0] for item in points]
            values = [item[1] for item in points]
            ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=label)

        if not has_curve:
            plt.close(fig)
            raise ValueError(f"Metric {metric!r} was not found in any provided log file.")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Epoch")
        ax.grid(True, alpha=0.3)
        if len(resolved_logs) > 1:
            ax.legend()

        save_path = os.path.join(dir2save, f"{name2save}_{metric}.png")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Plot Stage-2 training curves from one or more log files."
    )
    parser.add_argument(
        "--log-files",
        nargs="+",
        required=True,
        help="One or more Stage-2 log files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help=f"Metrics to plot. Supported: {list(SUPPORTED_METRICS)}",
    )
    parser.add_argument(
        "--dir2save",
        required=True,
        help="Directory used to save the generated figures.",
    )
    parser.add_argument(
        "--name2save",
        required=True,
        help="Filename prefix for the generated figures.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional legend labels. Length must match --log-files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure dpi. Default: 160.",
    )
    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    save_paths = plot_stage2_training_curves(
        log_files=args.log_files,
        metrics=args.metrics,
        dir2save=args.dir2save,
        name2save=args.name2save,
        labels=args.labels,
        dpi=args.dpi,
    )

    print("Saved figures:")
    for path in save_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
