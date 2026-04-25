#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export a maintainable summary for current Stage-1 control ckpts that match:
  stage1_wingtra_xxxx_xxRandSatNeg_msloss_dinov2B2_xxx
  stage1_visloc_xxxx_xxRandSatNeg_msloss_dinov2B2_xxx

Outputs are written to:
  gen_fm_exps/analysis/stage1_crtl_ckpts_info/
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path("/home/data/zwk/pyproj_neuloc_v0")
CKPTS_ROOT = PROJECT_ROOT / "gen_fm_exps" / "ckpts"
LOGS_ROOT = PROJECT_ROOT / "gen_fm_exps" / "logs"
ANALYSIS_ROOT = PROJECT_ROOT / "gen_fm_exps" / "analysis"
OUTPUT_ROOT = ANALYSIS_ROOT / "stage1_crtl_ckpts_info"
OLD_CTRL_TXT = ANALYSIS_ROOT / "ctrl_best_epoch.txt"
GALLERY_SUMMARY_CSV = ANALYSIS_ROOT / "stage1_interval82_gallery_summary_progressive_multi_cfg.csv"
GALLERY_CURRENT_ROWS_CSV = OUTPUT_ROOT / "gallery_current" / "gallery_rows_full.csv"
MATCH_PATTERN = "stage1_*_*_*RandSatNeg_msloss_dinov2B2_*"

EXP_RE = re.compile(
    r"^(stage1_(?P<dataset>wingtra|visloc)_(?P<split_tag>(?:interval|segment)\d+)_(?:w|wo)RandSatNeg_msloss_dinov2B2_(?P<method>.+))$"
)
SPLIT_TAG_RE = re.compile(r"^(?P<split_mode>interval|segment)(?P<train_ratio_tag>\d+)$")
EPOCH_CKPT_RE = re.compile(r"^epoch(?P<epoch>\d+)(?:[^/]*)\.pth$")
SCENE_R1_RE = re.compile(r"\[Scene:\s*(?P<scene>[^\]]+)\]\s+nrc:R@1=(?P<r1>[0-9.]+)%")
EPOCH_DONE_RE = re.compile(r"epoch\s+(?P<epoch>\d+)\s+完成")
EXPERIMENT_DIR_RE = re.compile(r"^\s*-\s+experiment_dir:\s+(?P<exp>stage1_(?:wingtra|visloc)_.+_(?:wo|w)RandSatNeg_msloss_dinov2B2_.+)\s*$")
GALLERY_EXP_BASE_RE = re.compile(r"^(?P<base>stage1_(?:wingtra|visloc)_.+_(?:w|wo)RandSatNeg_msloss_dinov2B2_.+)_epoch\d+$")


@dataclass
class ParsedLog:
    path: Optional[Path]
    parsed_eval_epochs: int
    best_epoch: Optional[int]
    best_sum_top1: Optional[float]
    best_scene_scores: Dict[str, float]
    tied_best_epochs: List[int]


def _safe_rel(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _json_compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _list_matching_experiments() -> List[Tuple[str, str, str, str, str]]:
    items = []
    for path in sorted(CKPTS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        m = EXP_RE.match(path.name)
        if not m:
            continue
        split_m = SPLIT_TAG_RE.match(m.group("split_tag"))
        if not split_m:
            continue
        items.append(
            (
                path.name,
                m.group("dataset"),
                split_m.group("split_mode"),
                split_m.group("train_ratio_tag"),
                m.group("method"),
            )
        )
    return items


def _extract_method_base_and_variant(method: str) -> Tuple[str, str]:
    m = re.match(r"^(?P<base>.+?)(?:_(?P<variant>\d+))?$", method)
    assert m is not None
    return m.group("base"), m.group("variant") or ""


def _list_epoch_ckpts(ckpt_dir: Path) -> List[Tuple[int, str]]:
    rows = []
    for path in sorted(ckpt_dir.iterdir()):
        if not path.is_file():
            continue
        m = EPOCH_CKPT_RE.match(path.name)
        if not m:
            continue
        epoch = int(m.group("epoch") or m.group("epoch2"))
        rows.append((epoch, path.name))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def _parse_stage1_log(log_path: Path) -> ParsedLog:
    epoch_to_scene_scores: Dict[int, Dict[str, float]] = {}
    current_scene_scores: Dict[str, float] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = SCENE_R1_RE.search(line)
            if m:
                current_scene_scores[m.group("scene")] = float(m.group("r1"))
                continue
            m = EPOCH_DONE_RE.search(line)
            if m:
                epoch = int(m.group("epoch"))
                if current_scene_scores:
                    epoch_to_scene_scores[epoch] = dict(current_scene_scores)
                    current_scene_scores = {}

    if not epoch_to_scene_scores:
        return ParsedLog(
            path=log_path,
            parsed_eval_epochs=0,
            best_epoch=None,
            best_sum_top1=None,
            best_scene_scores={},
            tied_best_epochs=[],
        )

    scored = []
    for epoch, scene_scores in epoch_to_scene_scores.items():
        scored.append((epoch, sum(scene_scores.values()), scene_scores))
    scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)

    max_sum = max(item[1] for item in scored)
    tied = sorted(epoch for epoch, total, _ in scored if abs(total - max_sum) < 1e-9)
    best_epoch = tied[0]
    best_scene_scores = epoch_to_scene_scores[best_epoch]

    return ParsedLog(
        path=log_path,
        parsed_eval_epochs=len(epoch_to_scene_scores),
        best_epoch=best_epoch,
        best_sum_top1=max_sum,
        best_scene_scores=best_scene_scores,
        tied_best_epochs=tied,
    )


def _select_best_log(log_dir: Path) -> ParsedLog:
    log_files = sorted(p for p in log_dir.glob("*.log") if p.is_file())
    if not log_files:
        return ParsedLog(None, 0, None, None, {}, [])

    candidates: List[Tuple[int, float, int, ParsedLog]] = []
    for p in log_files:
        parsed = _parse_stage1_log(p)
        stat = p.stat()
        candidates.append((parsed.parsed_eval_epochs, stat.st_mtime, stat.st_size, parsed))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _parse_old_ctrl_experiments() -> set[str]:
    found = set()
    if not OLD_CTRL_TXT.is_file():
        return found
    with open(OLD_CTRL_TXT, "r", encoding="utf-8") as f:
        for line in f:
            m = EXPERIMENT_DIR_RE.match(line)
            if m:
                found.add(m.group("exp"))
    return found


def _load_gallery_rows(current_experiments: set[str]) -> Tuple[List[dict], Dict[str, List[dict]]]:
    rows: List[dict] = []
    by_exp: Dict[str, List[dict]] = defaultdict(list)
    if not GALLERY_SUMMARY_CSV.is_file():
        return rows, by_exp

    with open(GALLERY_SUMMARY_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_dir = row.get("experiment_dir", "")
            m = GALLERY_EXP_BASE_RE.match(exp_dir)
            if not m:
                continue
            base = m.group("base")
            if base not in current_experiments:
                continue
            out_row = dict(row)
            out_row["experiment_base_dir"] = base
            rows.append(out_row)
            by_exp[base].append(out_row)
    return rows, by_exp


def _build_master_rows() -> Tuple[List[dict], List[dict], Dict[str, int]]:
    matching = _list_matching_experiments()
    current_set = {item[0] for item in matching}
    old_ctrl_set = _parse_old_ctrl_experiments()
    gallery_rows, gallery_by_exp = _load_gallery_rows(current_set)

    master_rows: List[dict] = []
    counts = Counter()

    for exp_name, dataset, split_mode, train_ratio_tag, method in matching:
        counts[f"{dataset}_{split_mode}{train_ratio_tag}"] += 1
        method_base, method_variant = _extract_method_base_and_variant(method)

        ckpt_dir = CKPTS_ROOT / exp_name
        log_dir = LOGS_ROOT / exp_name
        ckpt_rows = _list_epoch_ckpts(ckpt_dir)
        saved_epochs = [epoch for epoch, _ in ckpt_rows]
        latest_ckpt_epoch = saved_epochs[-1] if saved_epochs else None
        latest_ckpt_file = ckpt_rows[-1][1] if ckpt_rows else ""

        parsed_log = _select_best_log(log_dir) if log_dir.is_dir() else ParsedLog(None, 0, None, None, {}, [])
        exact_best_ckpt_file = ""
        if parsed_log.best_epoch is not None:
            exact_matches = [name for epoch, name in ckpt_rows if epoch == parsed_log.best_epoch]
            if exact_matches:
                exact_best_ckpt_file = sorted(exact_matches)[0]

        gallery_exp_rows = gallery_by_exp.get(exp_name, [])
        gallery_scenes = sorted({row.get("scene", "") for row in gallery_exp_rows if row.get("scene")})
        gallery_source_epochs = sorted({str(row.get("epoch", "")).strip() for row in gallery_exp_rows if row.get("epoch", "") != ""})

        master_rows.append(
            {
                "experiment_dir": exp_name,
                "dataset": dataset,
                "split_mode": split_mode,
                "train_ratio_tag": train_ratio_tag,
                "method": method,
                "method_base": method_base,
                "method_variant": method_variant,
                "ckpt_dir": _safe_rel(ckpt_dir),
                "log_dir": _safe_rel(log_dir if log_dir.is_dir() else None),
                "log_file_selected": _safe_rel(parsed_log.path),
                "selected_log_name": parsed_log.path.name if parsed_log.path else "",
                "listed_in_old_ctrl_best_epoch": exp_name in old_ctrl_set,
                "saved_epoch_count": len(saved_epochs),
                "saved_epochs_json": _json_compact(saved_epochs),
                "latest_saved_epoch": latest_ckpt_epoch if latest_ckpt_epoch is not None else "",
                "latest_ckpt_file": latest_ckpt_file,
                "parsed_eval_epochs": parsed_log.parsed_eval_epochs,
                "best_epoch": parsed_log.best_epoch if parsed_log.best_epoch is not None else "",
                "best_sum_top1": f"{parsed_log.best_sum_top1:.3f}" if parsed_log.best_sum_top1 is not None else "",
                "best_scene_top1_json": _json_compact(parsed_log.best_scene_scores) if parsed_log.best_scene_scores else "",
                "tied_best_epochs_json": _json_compact(parsed_log.tied_best_epochs) if parsed_log.tied_best_epochs else "",
                "exact_best_epoch_ckpt_exists": bool(exact_best_ckpt_file),
                "exact_best_epoch_ckpt_file": exact_best_ckpt_file,
                "gallery_rows_count": len(gallery_exp_rows),
                "gallery_scenes": ",".join(gallery_scenes),
                "gallery_source_epochs": ",".join(gallery_source_epochs),
            }
        )

    master_rows.sort(key=lambda row: (row["dataset"], row["split_mode"], row["train_ratio_tag"], row["method"], row["experiment_dir"]))
    gallery_rows.sort(key=lambda row: (row["experiment_base_dir"], row.get("scene", ""), row.get("epoch", "")))
    return master_rows, gallery_rows, dict(counts)


def _write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = fieldnames or list(rows[0].keys())
    elif fieldnames is None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gallery_fieldnames() -> List[str]:
    if not GALLERY_SUMMARY_CSV.is_file():
        return []
    with open(GALLERY_SUMMARY_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header and "experiment_base_dir" not in header:
        header.append("experiment_base_dir")
    return header


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_gallery_current_by_exp() -> Dict[str, List[dict]]:
    by_exp: Dict[str, List[dict]] = defaultdict(list)
    if not GALLERY_CURRENT_ROWS_CSV.is_file():
        return by_exp
    with open(GALLERY_CURRENT_ROWS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row.get("experiment_dir", "")
            if exp_name:
                by_exp[exp_name].append(dict(row))
    for exp_name in list(by_exp):
        by_exp[exp_name].sort(key=lambda row: (row.get("scene", ""), row.get("epoch", "")))
    return by_exp


def _path_from_rel_or_abs(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_current_experiments_txt(master_rows: List[dict]) -> str:
    gallery_by_exp = _load_gallery_current_by_exp()
    lines = [
        "# Stage1 Ctrl Current Experiments",
        f"# Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"# Match pattern: {MATCH_PATTERN}",
        f"# Master CSV: {OUTPUT_ROOT / 'stage1_ctrl_current_experiments_master.csv'}",
        f"# Gallery source CSV: {GALLERY_CURRENT_ROWS_CSV}",
        "",
    ]

    for row in master_rows:
        exp_name = row["experiment_dir"]
        best_epoch = row.get("best_epoch", "")
        exact_file = row.get("exact_best_epoch_ckpt_file", "")
        best_ckpt_path = "MISSING"
        if exact_file:
            best_ckpt_path = str(_path_from_rel_or_abs(row["ckpt_dir"]) / exact_file)

        gallery_rows = gallery_by_exp.get(exp_name, [])
        gallery_epochs = sorted({r.get("epoch", "") for r in gallery_rows if r.get("epoch", "")})
        gallery_scenes = sorted({r.get("scene", "") for r in gallery_rows if r.get("scene", "")})
        epoch_matches = bool(gallery_rows and best_epoch and set(gallery_epochs) == {str(best_epoch)})

        lines.append(f"- experiment_dir: {exp_name}")
        lines.append(f"  dataset: {row.get('dataset', '')}")
        lines.append(f"  split: {row.get('split_mode', '')}{row.get('train_ratio_tag', '')}")
        lines.append(f"  method: {row.get('method', '')}")
        lines.append(f"  best_epoch: {best_epoch}")
        lines.append(f"  best_sum_top1: {row.get('best_sum_top1', '')}")
        lines.append(f"  best_ckpt_exists: {row.get('exact_best_epoch_ckpt_exists', '')}")
        lines.append(f"  best_ckpt_path: {best_ckpt_path}")
        lines.append(f"  gallery_available: {bool(gallery_rows)}")
        if gallery_rows:
            lines.append(f"  gallery_epoch_matches_best_epoch: {epoch_matches}")
            lines.append(f"  gallery_epochs: {','.join(gallery_epochs)}")
            lines.append(f"  gallery_scenes: {','.join(gallery_scenes)}")
            lines.append("  gallery_paths:")
            for gallery_row in gallery_rows:
                lines.append(f"    - scene: {gallery_row.get('scene', '')}")
                lines.append(f"      epoch: {gallery_row.get('epoch', '')}")
                lines.append(f"      gallery_save_dir: {gallery_row.get('gallery_save_dir', '')}")
                lines.append(f"      report_path: {gallery_row.get('report_path', '')}")
                lines.append(f"      bundle_path: {gallery_row.get('bundle_path', '')}")
        else:
            lines.append("  gallery_paths: []")
        lines.append("")

    return "\n".join(lines)


def _build_summary_md(master_rows: List[dict], counts: Dict[str, int]) -> str:
    total = len(master_rows)
    with_logs = sum(1 for row in master_rows if int(row["parsed_eval_epochs"] or 0) > 0)
    with_gallery = sum(1 for row in master_rows if int(row["gallery_rows_count"] or 0) > 0)
    new_vs_old = [row["experiment_dir"] for row in master_rows if not row["listed_in_old_ctrl_best_epoch"]]
    missing_best_ckpt = [row["experiment_dir"] for row in master_rows if row["best_epoch"] and not row["exact_best_epoch_ckpt_exists"]]

    ranked = []
    for row in master_rows:
        if not row["best_sum_top1"]:
            continue
        ranked.append((float(row["best_sum_top1"]), row))
    ranked.sort(key=lambda x: x[0], reverse=True)

    lines = []
    lines.append("# Stage1 Ctrl Ckpts Info")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"- Source ckpts root: `{CKPTS_ROOT}`")
    lines.append(f"- Source logs root: `{LOGS_ROOT}`")
    lines.append(f"- Match pattern: `{MATCH_PATTERN}`")
    lines.append(f"- Source gallery summary: `{GALLERY_SUMMARY_CSV}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Current matched ckpt directories: `{total}`")
    for key in sorted(counts):
        lines.append(f"- `{key}`: `{counts[key]}`")
    lines.append(f"- Experiments with parseable logs: `{with_logs}`")
    lines.append(f"- Experiments covered by interval82 gallery summary csv: `{with_gallery}`")
    lines.append("")
    lines.append("## Top Current Experiments By best_sum_top1")
    lines.append("")
    if ranked:
        for idx, (_, row) in enumerate(ranked[:10], start=1):
            scene_text = row["best_scene_top1_json"] or "{}"
            lines.append(
                f"{idx}. `{row['experiment_dir']}` | best_epoch=`{row['best_epoch']}` | "
                f"best_sum_top1=`{row['best_sum_top1']}%` | scene_top1=`{scene_text}`"
            )
    else:
        lines.append("- No parseable log metrics.")
    lines.append("")
    lines.append("## New Current Experiments Not Covered By Old ctrl_best_epoch.txt")
    lines.append("")
    if new_vs_old:
        for exp in new_vs_old:
            lines.append(f"- `{exp}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Experiments Without Exact best_epoch Checkpoint")
    lines.append("")
    if missing_best_ckpt:
        for exp in missing_best_ckpt:
            row = next(r for r in master_rows if r["experiment_dir"] == exp)
            lines.append(
                f"- `{exp}` | best_epoch=`{row['best_epoch']}` | saved_epochs=`{row['saved_epochs_json']}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `stage1_ctrl_current_experiments_master.csv`: one row per current matching ckpt dir")
    lines.append("- `stage1_ctrl_current_gallery_multi_cfg_rows.csv`: filtered rows from the existing interval82 gallery multi-cfg summary")
    lines.append("- `stage1_ctrl_current_experiments.txt`: current matching experiments with best ckpt path and gallery paths when available")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    master_rows, gallery_rows, counts = _build_master_rows()

    master_csv = OUTPUT_ROOT / "stage1_ctrl_current_experiments_master.csv"
    gallery_csv = OUTPUT_ROOT / "stage1_ctrl_current_gallery_multi_cfg_rows.csv"
    experiments_txt = OUTPUT_ROOT / "stage1_ctrl_current_experiments.txt"
    summary_md = OUTPUT_ROOT / "README.md"

    _write_csv(master_csv, master_rows)
    _write_csv(gallery_csv, gallery_rows, _gallery_fieldnames() or None)
    _write_text(experiments_txt, _build_current_experiments_txt(master_rows))
    _write_text(summary_md, _build_summary_md(master_rows, counts))

    print(f"[ok] wrote: {master_csv}")
    print(f"[ok] wrote: {gallery_csv}")
    print(f"[ok] wrote: {experiments_txt}")
    print(f"[ok] wrote: {summary_md}")


if __name__ == "__main__":
    main()
