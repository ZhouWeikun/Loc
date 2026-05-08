#!/usr/bin/env python3
"""Export Stage-3 error ranking and arrow-annotated contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.draw_gt_arrow_on_stage3_preds import (
    DEFAULT_GT_COLOR,
    DEFAULT_PRED_COLOR,
    _draw_pose_marker,
    _nrc_to_crop_xy,
    _parse_color,
    _wrap_pi,
)


DEFAULT_STAGES = ("grid", "mode", "evo")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def _load_summary_rows(run_dir: Path) -> list[dict]:
    summary_csv = run_dir / "stage3_triplets_summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary csv not found: {summary_csv}")
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rank_rows(rows: list[dict], nrc2meter: float) -> list[dict]:
    ranked = []
    for row in rows:
        out = dict(row)
        dist_nrc = _to_float(row, "dist_2d_nrc")
        out["dist_2d_meter"] = f"{dist_nrc * nrc2meter:.6f}"
        out["rot_error_deg_abs"] = f"{abs(_to_float(row, 'rot_error_deg')):.6f}"
        out["scale_error_abs"] = f"{abs(_to_float(row, 'scale_ratio') - 1.0):.6f}"
        ranked.append(out)

    by_dist = sorted(ranked, key=lambda r: float(r["dist_2d_meter"]), reverse=True)
    by_rot = sorted(ranked, key=lambda r: float(r["rot_error_deg_abs"]), reverse=True)
    by_scale = sorted(ranked, key=lambda r: float(r["scale_error_abs"]), reverse=True)

    for rank, row in enumerate(by_dist, start=1):
        row["rank_dist"] = rank
    for rank, row in enumerate(by_rot, start=1):
        row["rank_rot"] = rank
    for rank, row in enumerate(by_scale, start=1):
        row["rank_scale"] = rank
    return by_dist


def _write_rank_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "rank_dist",
        "rank_rot",
        "rank_scale",
        "query_id",
        "source_index",
        "query_filename",
        "sample_dir",
        "dist_2d_nrc",
        "dist_2d_meter",
        "rot_error_deg",
        "scale_ratio",
        "grid_dist_2d_nrc",
        "grid_rot_error_deg",
        "grid_scale_ratio",
        "mode_dist_2d_nrc",
        "mode_rot_error_deg",
        "mode_scale_ratio",
        "evo_dist_2d_nrc",
        "evo_rot_error_deg",
        "evo_scale_ratio",
    ]
    fieldnames = []
    for key in preferred + list(rows[0].keys()):
        if key not in fieldnames and key in rows[0]:
            fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    fill=(255, 255, 255),
    background=(0, 0, 0),
    font=None,
) -> None:
    x, y = xy
    lines = str(text).splitlines()
    pad = 5
    line_h = 14
    widths = []
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_h = max(line_h, bbox[3] - bbox[1] + 4)
    w = max(widths) if widths else 0
    h = line_h * len(lines)
    draw.rectangle((x - pad, y - pad, x + w + pad, y + h + pad), fill=background)
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * line_h), line, fill=fill, font=font)


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as im:
        out = im.convert("RGB")
    if size is not None and out.size != size:
        out = out.resize(size, Image.Resampling.BILINEAR)
    return out


def _annotate_prediction_crop(
    image: Image.Image,
    *,
    meta: dict,
    stage: str,
    halfimg_radius_nrc: float,
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
    base_arrow_len: float,
    angle_offset_deg: float,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    gt_nr, gt_nc, gt_rot, gt_scale = map(float, meta["gt_coord"])
    stage_meta = meta["stage_predictions"][stage]
    pred_nr, pred_nc, pred_rot, pred_scale = map(float, stage_meta["coord_top1"])
    angle_offset = math.radians(float(angle_offset_deg))

    # Prediction crop is centered at the predicted pose.
    pred_center = ((width - 1) * 0.5, (height - 1) * 0.5)
    _draw_pose_marker(
        draw,
        center=pred_center,
        angle_rad=angle_offset,
        length_px=base_arrow_len,
        color=pred_color,
    )

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
    arrow_len = base_arrow_len * gt_scale / max(pred_scale, 1e-6)
    arrow_len = min(58.0, max(20.0, arrow_len))
    _draw_pose_marker(
        draw,
        center=gt_xy,
        angle_rad=_wrap_pi(gt_rot - pred_rot + angle_offset),
        length_px=arrow_len,
        color=gt_color,
    )
    return out


def _make_contact_sheet(
    *,
    run_dir: Path,
    row: dict,
    output_path: Path,
    halfimg_radius_nrc: float,
    nrc2meter: float,
    stages: Iterable[str],
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
    angle_offset_deg: float,
) -> None:
    sample_dir = run_dir / row["sample_dir"]
    meta = _read_json(sample_dir / "meta.json")
    tile_size = (224, 224)
    header_h = 56
    footer_h = 82
    gap = 10
    margin = 14
    labels = ["Query", "GT"] + [stage.upper() for stage in stages]

    images = [
        _load_rgb(sample_dir / "query.png", tile_size),
        _load_rgb(sample_dir / "gt_satmap00.png", tile_size),
    ]
    for stage in stages:
        filename = meta["stage_predictions"][stage]["pred_filename"]
        img = _load_rgb(sample_dir / filename, tile_size)
        images.append(
            _annotate_prediction_crop(
                img,
                meta=meta,
                stage=stage,
                halfimg_radius_nrc=halfimg_radius_nrc,
                gt_color=gt_color,
                pred_color=pred_color,
                base_arrow_len=34.0,
                angle_offset_deg=angle_offset_deg,
            )
        )

    w = margin * 2 + len(images) * tile_size[0] + (len(images) - 1) * gap
    h = margin * 2 + header_h + tile_size[1] + footer_h
    canvas = Image.new("RGB", (w, h), (245, 245, 238))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    dist_m = float(row["dist_2d_nrc"]) * nrc2meter
    title = (
        f"Rank #{row['rank_dist']} | query {row['query_id']} | "
        f"2D err {dist_m:.1f} m | rot {float(row['rot_error_deg']):.1f} deg | "
        f"scale {float(row['scale_ratio']):.3f}x"
    )
    _draw_text_box(draw, title, (margin, margin), background=(30, 30, 30), font=font)

    y0 = margin + header_h
    x = margin
    for label, img in zip(labels, images):
        canvas.paste(img, (x, y0))
        _draw_text_box(draw, label, (x + 6, y0 + 6), background=(0, 0, 0), font=font)
        x += tile_size[0] + gap

    legend = (
        f"Blue arrow/cross: predicted crop center pose. Red arrow/cross: GT pose projected into each prediction crop.\n"
        f"grid err {float(row['grid_dist_2d_nrc']) * nrc2meter:.1f}m, "
        f"mode err {float(row['mode_dist_2d_nrc']) * nrc2meter:.1f}m, "
        f"evo err {float(row['evo_dist_2d_nrc']) * nrc2meter:.1f}m"
    )
    _draw_text_box(draw, legend, (margin, y0 + tile_size[1] + 18), background=(45, 45, 45), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def export_error_rank_visuals(
    *,
    run_dir: Path,
    output_dir: Path | None,
    top_k: int,
    stages: tuple[str, ...],
    gt_color: tuple[int, int, int],
    pred_color: tuple[int, int, int],
    angle_offset_deg: float,
) -> dict:
    run_dir = run_dir.expanduser().resolve()
    manifest = _read_json(run_dir / "manifest.json")
    nrc2meter = float(manifest.get("dataset_metrics", {}).get("nrc2meter_factor", 1.0))
    halfimg_radius_nrc = float(manifest["dataset_metrics"]["halfimg_radius_nrc"])
    rows = _rank_rows(_load_summary_rows(run_dir), nrc2meter=nrc2meter)

    output_dir = output_dir or (run_dir / "error_rank_visuals")
    output_dir = output_dir.expanduser().resolve()
    ranking_csv = output_dir / "error_ranking_by_evo_2d.csv"
    _write_rank_csv(rows, ranking_csv)

    sheet_dir = output_dir / "annotated_top_errors"
    written = []
    for row in rows[: int(top_k)]:
        out_path = sheet_dir / f"rank{int(row['rank_dist']):03d}_qid{int(row['query_id']):05d}_err{float(row['dist_2d_meter']):.1f}m.png"
        _make_contact_sheet(
            run_dir=run_dir,
            row=row,
            output_path=out_path,
            halfimg_radius_nrc=halfimg_radius_nrc,
            nrc2meter=nrc2meter,
            stages=stages,
            gt_color=gt_color,
            pred_color=pred_color,
            angle_offset_deg=angle_offset_deg,
        )
        written.append(str(out_path))

    summary_path = output_dir / "error_rank_visuals_summary.json"
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "ranking_csv": str(ranking_csv),
        "top_k": int(top_k),
        "n_rows": len(rows),
        "annotated_images": written,
        "sort_key": "dist_2d_nrc desc (primary/evo stage)",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--gt-color", type=_parse_color, default=DEFAULT_GT_COLOR)
    parser.add_argument("--pred-color", type=_parse_color, default=DEFAULT_PRED_COLOR)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_error_rank_visuals(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        top_k=int(args.top_k),
        stages=tuple(args.stages),
        gt_color=args.gt_color,
        pred_color=args.pred_color,
        angle_offset_deg=float(args.angle_offset_deg),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
