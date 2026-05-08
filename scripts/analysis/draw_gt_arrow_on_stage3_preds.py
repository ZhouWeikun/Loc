#!/usr/bin/env python3
"""Draw GT pose arrows on exported Stage-3 prediction crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


DEFAULT_STAGES = ("grid", "mode", "evo")
DEFAULT_GT_COLOR = (255, 30, 30)
DEFAULT_PRED_COLOR = (30, 144, 255)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _parse_color(value: str) -> tuple[int, int, int]:
    value = str(value).strip()
    named = {
        "red": (255, 30, 30),
        "blue": (30, 144, 255),
        "green": (0, 180, 80),
        "yellow": (255, 210, 0),
        "cyan": (0, 210, 255),
        "magenta": (255, 0, 180),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    if value.lower() in named:
        return named[value.lower()]
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6:
        try:
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid hex color: {value}") from exc
    parts = value.split(",")
    if len(parts) == 3:
        try:
            rgb = tuple(int(part.strip()) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid RGB color: {value}") from exc
        if all(0 <= channel <= 255 for channel in rgb):
            return rgb  # type: ignore[return-value]
    raise argparse.ArgumentTypeError(
        f"invalid color '{value}', expected name, #RRGGBB, RRGGBB, or R,G,B"
    )


def _nrc_to_crop_xy(
    *,
    gt_nr: float,
    gt_nc: float,
    pred_nr: float,
    pred_nc: float,
    pred_rot: float,
    pred_scale: float,
    halfimg_radius_nrc: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Project GT NRC into the crop frame generated from pred coord.

    This inverts the local-to-satmap rotation used by
    crop_satimg_by_4d_coords_fast:
        dcol = cos(rot) * x + sin(rot) * y
        drow = -sin(rot) * x + cos(rot) * y
    """
    drow = gt_nr - pred_nr
    dcol = gt_nc - pred_nc
    cos_v = math.cos(pred_rot)
    sin_v = math.sin(pred_rot)

    x_local = cos_v * dcol - sin_v * drow
    y_local = sin_v * dcol + cos_v * drow

    half_nrc = max(float(halfimg_radius_nrc) * max(float(pred_scale), 1e-6), 1e-12)
    x_px = (x_local / half_nrc + 1.0) * 0.5 * (width - 1)
    y_px = (y_local / half_nrc + 1.0) * 0.5 * (height - 1)
    return x_px, y_px


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    angle_rad: float,
    length_px: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    cx, cy = center
    dx = math.sin(angle_rad) * length_px
    dy = -math.cos(angle_rad) * length_px
    end = (cx + dx, cy + dy)
    draw.line((cx, cy, end[0], end[1]), fill=color, width=width)

    head_len = max(7.0, length_px * 0.28)
    head_half = math.radians(28.0)
    for sign in (-1.0, 1.0):
        a = angle_rad + math.pi + sign * head_half
        hx = end[0] + math.sin(a) * head_len
        hy = end[1] - math.cos(a) * head_len
        draw.line((end[0], end[1], hx, hy), fill=color, width=width)


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=width)
    draw.line((cx - radius * 1.6, cy, cx + radius * 1.6, cy), fill=color, width=width)
    draw.line((cx, cy - radius * 1.6, cx, cy + radius * 1.6), fill=color, width=width)


def _draw_text_badge(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: tuple[int, int],
    fill: tuple[int, int, int],
    background: tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=background,
    )
    draw.text((x, y), text, fill=fill)


def _draw_pose_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    angle_rad: float,
    length_px: float,
    color: tuple[int, int, int],
) -> None:
    _draw_crosshair(draw, center=center, radius=7.0, color=color, width=1)
    _draw_arrow(draw, center=center, angle_rad=angle_rad, length_px=length_px, color=color, width=2)


def _iter_sample_meta(run_dir: Path) -> Iterable[Path]:
    samples_dir = run_dir / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples directory not found: {samples_dir}")
    yield from sorted(samples_dir.glob("*/meta.json"))


def draw_for_run(
    *,
    run_dir: Path,
    stages: tuple[str, ...],
    suffix: str,
    overwrite: bool,
    draw_badge: bool,
    draw_gt_arrow: bool,
    draw_pred_arrow: bool,
    draw_gt_crops: bool,
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
    base_arrow_len: float,
    min_arrow_len: float,
    max_arrow_len: float,
    angle_offset_deg: float,
) -> dict:
    manifest = _load_json(run_dir / "manifest.json")
    halfimg_radius_nrc = float(manifest["dataset_metrics"]["halfimg_radius_nrc"])

    n_written = 0
    n_missing = 0
    n_outside = 0
    n_gt_crop_written = 0
    stage_counts = {stage: 0 for stage in stages}
    angle_offset = math.radians(float(angle_offset_deg))

    for meta_path in _iter_sample_meta(run_dir):
        meta = _load_json(meta_path)
        sample_dir = meta_path.parent
        gt = meta["gt_coord"]
        gt_nr, gt_nc, gt_rot, gt_scale = map(float, gt)
        stage_predictions = meta.get("stage_predictions", {})

        if draw_gt_crops:
            for gt_filename in (name for name in meta.get("saved_files", []) if str(name).startswith("gt_satmap")):
                gt_path = sample_dir / gt_filename
                if not gt_path.exists():
                    n_missing += 1
                    continue
                gt_output_path = gt_path.with_name(f"{gt_path.stem}{suffix}{gt_path.suffix}")
                if gt_output_path.exists() and not overwrite:
                    continue
                with Image.open(gt_path) as im:
                    out_gt = im.convert("RGB")
                gt_width, gt_height = out_gt.size
                draw_gt = ImageDraw.Draw(out_gt)
                _draw_pose_marker(
                    draw_gt,
                    center=((gt_width - 1) * 0.5, (gt_height - 1) * 0.5),
                    angle_rad=angle_offset,
                    length_px=base_arrow_len,
                    color=gt_color,
                )
                out_gt.save(gt_output_path)
                n_gt_crop_written += 1

        for stage in stages:
            stage_meta = stage_predictions.get(stage)
            if not stage_meta:
                n_missing += 1
                continue

            pred_filename = stage_meta.get("pred_filename", f"pred_{stage}_top1.png")
            image_path = sample_dir / pred_filename
            if not image_path.exists():
                n_missing += 1
                continue

            output_path = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
            if output_path.exists() and not overwrite:
                continue

            pred_nr, pred_nc, pred_rot, pred_scale = map(float, stage_meta["coord_top1"])
            with Image.open(image_path) as im:
                out = im.convert("RGB")
            width, height = out.size

            draw = ImageDraw.Draw(out)
            scale_ratio_signed = gt_scale / max(pred_scale, 1e-6)
            if draw_pred_arrow:
                _draw_pose_marker(
                    draw,
                    center=((width - 1) * 0.5, (height - 1) * 0.5),
                    angle_rad=angle_offset,
                    length_px=base_arrow_len,
                    color=pred_color,
                )
            if draw_gt_arrow:
                gt_xy = _nrc_to_crop_xy(
                    gt_nr=gt_nr,
                    gt_nc=gt_nc,
                    pred_nr=pred_nr,
                    pred_nc=pred_nc,
                    pred_rot=pred_rot,
                    pred_scale=pred_scale,
                    halfimg_radius_nrc=halfimg_radius_nrc,
                    width=width,
                    height=height,
                )
                if not (0.0 <= gt_xy[0] < width and 0.0 <= gt_xy[1] < height):
                    n_outside += 1

                arrow_len = base_arrow_len * scale_ratio_signed
                arrow_len = min(max_arrow_len, max(min_arrow_len, arrow_len))
                angle_local = _wrap_pi(gt_rot - pred_rot + angle_offset)
                _draw_pose_marker(
                    draw,
                    center=gt_xy,
                    angle_rad=angle_local,
                    length_px=arrow_len,
                    color=gt_color,
                )

            if draw_badge:
                _draw_text_badge(
                    draw,
                    text=f"Pred/GT Scale = {1.0 / max(scale_ratio_signed, 1e-6):.2f}x",
                    xy=(6, 6),
                    fill=(255, 255, 255),
                    background=(0, 0, 0),
                )

            out.save(output_path)
            n_written += 1
            stage_counts[stage] += 1

    return {
        "run_dir": str(run_dir),
        "suffix": suffix,
        "written": n_written,
        "gt_crop_written": n_gt_crop_written,
        "missing": n_missing,
        "gt_center_outside_image": n_outside,
        "stage_counts": stage_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Stage-3 export directory containing manifest.json and samples/")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), help="Stages to process, e.g. grid mode evo")
    parser.add_argument("--suffix", default="_gtarrow", help="Suffix appended before .png")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images")
    parser.add_argument("--no-badge", action="store_true", help="Do not draw the scale-ratio text badge")
    parser.add_argument("--no-gt-arrow", action="store_true", help="Do not draw the GT arrow on prediction crops")
    parser.add_argument("--draw-pred-arrow", action="store_true", help="Draw the predicted pose arrow at the prediction crop center")
    parser.add_argument("--draw-gt-crops", action="store_true", help="Also draw the GT pose arrow on gt_satmap*.png crops")
    parser.add_argument("--gt-color", type=_parse_color, default=DEFAULT_GT_COLOR, help="GT arrow color: name, #RRGGBB, RRGGBB, or R,G,B")
    parser.add_argument("--pred-color", type=_parse_color, default=DEFAULT_PRED_COLOR, help="Prediction arrow color: name, #RRGGBB, RRGGBB, or R,G,B")
    parser.add_argument("--base-arrow-len", type=float, default=34.0, help="Arrow length in pixels when gt_scale == pred_scale")
    parser.add_argument("--min-arrow-len", type=float, default=20.0)
    parser.add_argument("--max-arrow-len", type=float, default=58.0)
    parser.add_argument(
        "--angle-offset-deg",
        type=float,
        default=0.0,
        help="Manual angle offset for convention calibration. Default keeps gt_rot == pred_rot pointing upward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = draw_for_run(
        run_dir=args.run_dir,
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
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
