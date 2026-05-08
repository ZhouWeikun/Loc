"""
Stage2 ablation analysis: compare two training runs, or plot a single run.
Plots R@k metrics, L2 distance curves, and loss curves.

Compare two runs:
    python scripts/analysis/analyse_stage2_ablation.py \
        --log1 <path1> --label1 <name1> \
        --log2 <path2> --label2 <name2> \
        --metrics r@1 l2_mean loss \
        --output_dir gen_fm_exps/analysis/stage2_analysis

Single run (output_dir defaults to the log file's parent directory):
    python scripts/analysis/analyse_stage2_ablation.py --log1 <path1>
"""

import argparse
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_RE_EPOCH   = re.compile(r"Epoch (\d+)/\d+")
_RE_LOSS    = re.compile(r"loss=([\d.]+)")
_RE_RECALL  = re.compile(
    r"\[Train Eval\]\[RC\].*?"
    r"R@1=([\d.]+)%.*?R@5=([\d.]+)%.*?R@10=([\d.]+)%.*?"
    r"R@20=([\d.]+)%.*?R@50=([\d.]+)%.*?"
    r"R@256=([\d.]+)%.*?R@512=([\d.]+)%.*?R@1024=([\d.]+)%"
)
_RE_L2      = re.compile(
    r"\[Train Eval\]\[RC\].*?error_rc_meter=([\d.]+)m.*?error_rc_meter_median=([\d.]+)m"
)


def parse_log(log_path: str) -> dict:
    """Return dict with keys: epochs_loss, loss, epochs_eval, r@1..r@1024, l2_mean, l2_median."""
    data = {
        "epochs_loss": [],
        "loss": [],
        "epochs_eval": [],
        "r@1": [], "r@5": [], "r@10": [], "r@20": [],
        "r@50": [], "r@256": [], "r@512": [], "r@1024": [],
        "l2_mean": [],
        "l2_median": [],
    }

    current_epoch = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _RE_EPOCH.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue

            m = _RE_LOSS.search(line)
            if m and current_epoch is not None:
                data["epochs_loss"].append(current_epoch)
                data["loss"].append(float(m.group(1)))
                continue

            m = _RE_RECALL.search(line)
            if m and current_epoch is not None:
                data["epochs_eval"].append(current_epoch)
                for i, key in enumerate(["r@1", "r@5", "r@10", "r@20", "r@50", "r@256", "r@512", "r@1024"]):
                    data[key].append(float(m.group(i + 1)))
                continue

            m = _RE_L2.search(line)
            if m:
                data["l2_mean"].append(float(m.group(1)))
                data["l2_median"].append(float(m.group(2)))

    return data


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

METRIC_CONFIG = {
    "r@1":    ("R@1 (%)",              "epochs_eval",  "r@1"),
    "r@5":    ("R@5 (%)",              "epochs_eval",  "r@5"),
    "r@10":   ("R@10 (%)",             "epochs_eval",  "r@10"),
    "r@20":   ("R@20 (%)",             "epochs_eval",  "r@20"),
    "r@50":   ("R@50 (%)",             "epochs_eval",  "r@50"),
    "r@256":  ("R@256 (%)",            "epochs_eval",  "r@256"),
    "r@512":  ("R@512 (%)",            "epochs_eval",  "r@512"),
    "r@1024": ("R@1024 (%)",           "epochs_eval",  "r@1024"),
    "l2_mean":   ("L2 Error Mean (m)", "epochs_eval",  "l2_mean"),
    "l2_median": ("L2 Error Median (m)", "epochs_eval","l2_median"),
    "loss":   ("Training Loss",        "epochs_loss",  "loss"),
}

ALL_METRICS = list(METRIC_CONFIG.keys())


def plot_metric(metric_key, d1, label1, output_dir, d2=None, label2=None):
    ylabel, x_key, y_key = METRIC_CONFIG[metric_key]

    x1, y1 = d1[x_key], d1[y_key]
    x2 = d2[x_key] if d2 else []
    y2 = d2[y_key] if d2 else []

    if not x1 and not x2:
        print(f"  [skip] No data for metric '{metric_key}'")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if x1:
        ax.plot(x1, y1, label=label1, linewidth=1.5)
    if x2:
        ax.plot(x2, y2, label=label2, linewidth=1.5, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(output_dir) / f"{metric_key.replace('@', 'at')}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

LOG_DIR = Path(__file__).parent.parent.parent / "gen_fm_exps" / "logs"

DEFAULT_LOG1 = str(
    LOG_DIR
    / "stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_noPoseCond"
    / "stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_noPoseCond.log"
)
DEFAULT_LOG2 = str(
    LOG_DIR
    / "stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie"
    / "stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie.log"
)
DEFAULT_OUTPUT = str(
    Path(__file__).parent.parent.parent / "gen_fm_exps" / "analysis" / "stage2_analysis"
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot stage2 training metrics. "
                    "Omit --log2 for single-run mode (output saved next to the log file)."
    )
    parser.add_argument("--log1",   default=DEFAULT_LOG1,  help="Path to first .log file")
    parser.add_argument("--log2",   default=None,          help="Path to second .log file (optional)")
    parser.add_argument("--label1", default=None,          help="Legend label for run 1 (default: log filename stem)")
    parser.add_argument("--label2", default=None,          help="Legend label for run 2 (default: log filename stem)")
    parser.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS,
        choices=ALL_METRICS,
        help="Metrics to plot (default: all)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for output figures. "
             "Defaults to the log file's parent directory in single-run mode, "
             f"or {DEFAULT_OUTPUT} in compare mode.",
    )
    args = parser.parse_args()

    single_mode = args.log2 is None

    label1 = args.label1 or Path(args.log1).stem
    label2 = args.label2 or (Path(args.log2).stem if args.log2 else None)

    if args.output_dir:
        output_dir = args.output_dir
    elif single_mode:
        output_dir = str(Path(args.log1).parent)
    else:
        output_dir = DEFAULT_OUTPUT

    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing {args.log1} ...")
    d1 = parse_log(args.log1)
    print(f"  epochs_loss={len(d1['epochs_loss'])}, epochs_eval={len(d1['epochs_eval'])}")

    d2 = None
    if not single_mode:
        print(f"Parsing {args.log2} ...")
        d2 = parse_log(args.log2)
        print(f"  epochs_loss={len(d2['epochs_loss'])}, epochs_eval={len(d2['epochs_eval'])}")

    print(f"\nGenerating plots -> {output_dir}")
    for metric in args.metrics:
        plot_metric(metric, d1, label1, output_dir, d2=d2, label2=label2)

    print("\nDone.")


if __name__ == "__main__":
    main()
