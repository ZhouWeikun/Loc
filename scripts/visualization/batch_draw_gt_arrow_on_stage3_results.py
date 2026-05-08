#!/usr/bin/env python3
"""Batch draw pose arrows on selected Stage-3 exported result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.visualization.draw_gt_arrow_on_stage3_preds import (
    DEFAULT_GT_COLOR,
    DEFAULT_PRED_COLOR,
    DEFAULT_STAGES,
    _parse_color,
    draw_for_run,
)


DEFAULT_RESULT_ROOTS = (
    Path(
        "/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B1_PN1cubie_1/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_visloc03_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_visloc04_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_zuchwil_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/res_epoch999"),
    Path(
        "/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie_1/res_epoch990"),
)


def _is_stage3_run_dir(path: Path) -> bool:
    return (path / "manifest.json").is_file() and (path / "samples").is_dir()


def _find_stage3_run_dirs(root: Path) -> list[Path]:
    if _is_stage3_run_dir(root):
        return [root]
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.glob("stage3_triplets_test_*")
        if path.is_dir() and _is_stage3_run_dir(path)
    )


def _select_run_dirs(roots: list[Path], latest_only: bool) -> tuple[list[Path], list[Path]]:
    selected: list[Path] = []
    missing: list[Path] = []
    for root in roots:
        run_dirs = _find_stage3_run_dirs(root)
        if not run_dirs:
            missing.append(root)
            continue
        if latest_only:
            selected.append(run_dirs[-1])
        else:
            selected.extend(run_dirs)
    return selected, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RESULT_ROOTS),
        help="Result roots or direct Stage-3 run dirs. Defaults to the hardcoded experiment list.",
    )
    parser.add_argument("--latest-only", action="store_true", help="Only draw the latest Stage-3 run under each root.")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), help="Stages to process, e.g. grid mode evo")
    parser.add_argument("--suffix", default="_gtarrow", help="Suffix appended before .png")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images")
    parser.add_argument("--dry-run", action="store_true", help="Print matched run dirs without writing images")
    parser.add_argument("--no-badge", action="store_true", help="Do not draw the scale-ratio text badge")
    parser.add_argument("--no-gt-arrow", action="store_true", help="Do not draw the GT arrow on prediction crops")
    parser.add_argument("--draw-pred-arrow", action="store_true", help="Draw the predicted pose arrow at the prediction crop center")
    parser.add_argument("--draw-gt-crops", action="store_true", help="Also draw the GT pose arrow on gt_satmap*.png crops")
    parser.add_argument("--gt-color", type=_parse_color, default=DEFAULT_GT_COLOR, help="GT arrow color")
    parser.add_argument("--pred-color", type=_parse_color, default=DEFAULT_PRED_COLOR, help="Prediction arrow color")
    parser.add_argument("--base-arrow-len", type=float, default=34.0)
    parser.add_argument("--min-arrow-len", type=float, default=20.0)
    parser.add_argument("--max-arrow-len", type=float, default=58.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs, missing_roots = _select_run_dirs(list(args.roots), latest_only=bool(args.latest_only))

    print(f"Matched {len(run_dirs)} Stage-3 run dirs.")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    if missing_roots:
        print(f"Skipped {len(missing_roots)} roots without Stage-3 export inputs:")
        for root in missing_roots:
            print(f"  {root}")

    if args.dry_run:
        return

    summaries = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] Drawing arrows: {run_dir}")
        summary = draw_for_run(
            run_dir=run_dir,
            stages=tuple(args.stages),
            suffix=args.suffix,
            overwrite=bool(args.overwrite),
            draw_badge=not bool(args.no_badge),
            draw_gt_arrow=not bool(args.no_gt_arrow),
            draw_pred_arrow=bool(args.draw_pred_arrow),
            draw_gt_crops=bool(args.draw_gt_crops),
            gt_color=args.gt_color,
            pred_color=args.pred_color,
            base_arrow_len=float(args.base_arrow_len),
            min_arrow_len=float(args.min_arrow_len),
            max_arrow_len=float(args.max_arrow_len),
            angle_offset_deg=float(args.angle_offset_deg),
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True))

    total_written = sum(int(item.get("written", 0)) for item in summaries)
    total_gt_crop_written = sum(int(item.get("gt_crop_written", 0)) for item in summaries)
    total_missing = sum(int(item.get("missing", 0)) for item in summaries)
    total_outside = sum(int(item.get("gt_center_outside_image", 0)) for item in summaries)
    print(
        json.dumps(
            {
                "run_dirs": len(run_dirs),
                "written": total_written,
                "gt_crop_written": total_gt_crop_written,
                "missing": total_missing,
                "gt_center_outside_image": total_outside,
                "skipped_roots": [str(root) for root in missing_roots],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
