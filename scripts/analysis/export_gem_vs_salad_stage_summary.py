#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_stage3_test_best_stage2_wreject.sh"
DEFAULT_RECALL_CFG_YAML = REPO_ROOT / "trainer_depends" / "configs" / "stage3_recall_thresholds.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_fm_exps" / "analysis" / "ours_ckpt_best" / "gem_vs_salad"


def _expand_vars(text: str, mapping: dict) -> str:
    for key, value in mapping.items():
        text = text.replace(f"${key}", value)
    return text


def _load_run_specs(run_script_path: Path):
    text = run_script_path.read_text(encoding="utf-8")
    marker = "declare -a RUN_SPECS=("
    start = text.rfind(marker)
    if start < 0:
        raise RuntimeError(f"Could not find active RUN_SPECS block in {run_script_path}")

    block = text[start + len(marker) :]
    end = block.find("\n)")
    if end < 0:
        raise RuntimeError(f"Could not find closing ')' for RUN_SPECS block in {run_script_path}")
    block = block[:end]

    var_map = {
        "REPO_ROOT": str(REPO_ROOT),
        "VISLOC_YAML": str(REPO_ROOT / "trainer_depends" / "configs" / "stage3_visloc.yaml"),
        "WINGTRA_YAML": str(REPO_ROOT / "trainer_depends" / "configs" / "stage3_wingtra.yaml"),
    }

    specs = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^"([^"]+)"$', line)
        if not match:
            continue
        item = _expand_vars(match.group(1), var_map)
        parts = item.split("|")
        if len(parts) == 4:
            base_yaml, scene, ckpt_path, opts_path = parts
            recall_cfg = "per_scene"
        elif len(parts) == 5:
            base_yaml, scene, ckpt_path, opts_path, recall_cfg = parts
        else:
            raise RuntimeError(f"Unexpected RUN_SPECS entry: {item}")
        specs.append(
            {
                "base_yaml": base_yaml,
                "scene": scene,
                "ckpt_path": ckpt_path,
                "opts_path": opts_path,
                "recall_cfg": recall_cfg,
            }
        )
    return specs


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scene_aliases(scene: str):
    mapping = {
        "zurich": ["zurich"],
        "zuchwil": ["zuchwil"],
        "visloc_03": ["visloc03", "visloc_03", "03"],
        "visloc_04": ["visloc04", "visloc_04", "04"],
    }
    return mapping.get(scene, [scene.lower()])


def _parse_stage_scene_metrics_from_name(ckpt_name: str, scene: str):
    stem = Path(ckpt_name).stem
    stem_lower = stem.lower()
    for alias in _scene_aliases(scene):
        alias = alias.lower()
        pattern = re.compile(
            rf"{re.escape(alias)}.*?r1=?(\d+).*?(?:mederr|miderr)=?(\d+)m",
            re.IGNORECASE,
        )
        match = pattern.search(stem_lower)
        if match:
            return {
                "r1": int(match.group(1)),
                "med_err_m": int(match.group(2)),
            }

    match = re.search(r"r1=(\d+).*?(?:mederr|miderr)=?(\d+)m", stem_lower, re.IGNORECASE)
    if match:
        return {
            "r1": int(match.group(1)),
            "med_err_m": int(match.group(2)),
        }
    raise RuntimeError(f"Could not parse scene metrics from checkpoint name: {ckpt_name}")


def _find_best_metric_ckpt(stage2_dir: Path):
    matches = sorted(stage2_dir.glob("epoch*_R1=*.pth"))
    if not matches:
        raise FileNotFoundError(f"No metric-named stage2 checkpoint found in {stage2_dir}")
    return matches[-1]


def _latest_stage3_run_dir(stage2_loaded_ckpt: Path):
    output_root = stage2_loaded_ckpt.parent / f"res_{stage2_loaded_ckpt.stem}"
    candidates = sorted(
        p
        for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("stage3_triplets_test_") and (p / "manifest.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No stage3 run dir found under {output_root}")
    return candidates[-1]


def _resolve_recall_cfg(spec_recall_cfg: str, scene: str, recall_cfg_doc: dict):
    if spec_recall_cfg == "per_scene":
        resolved = recall_cfg_doc["per_scene"][scene]
        source = "per_scene"
    else:
        resolved = spec_recall_cfg
        source = "explicit"
    cfg = dict(recall_cfg_doc["configs"][resolved])
    return resolved, source, cfg


def _threshold_summary(stage3_eval_cfg: dict):
    return (
        f"dist={stage3_eval_cfg['dist_th_m']:.1f}m, "
        f"rot={stage3_eval_cfg['rot_th']:.1f}deg, "
        f"scale={stage3_eval_cfg['scale_ratio_th']:.2f}"
    )


def _progressive_top1(stage_section: dict, key: str):
    return float(stage_section["progressive_acc_metrics"][key]["top1_acc"])


def _progressive_top5(stage_section: dict, key: str):
    values = stage_section["progressive_acc_metrics"][key]
    if "top5_acc" in values:
        return float(values["top5_acc"])
    return None


def _build_row(spec: dict, recall_cfg_doc: dict):
    opts_path = Path(spec["opts_path"])
    stage2_loaded_ckpt = Path(spec["ckpt_path"])
    stage2_dir = opts_path.parent
    opts = _load_yaml(opts_path)

    scene = spec["scene"]
    aggregator = str(opts["network_setting"]["aggregator_type"]).lower()
    stage1_ckpt = Path(opts["exp_setting"]["load_stage1_ckpt"])
    stage1_metrics = _parse_stage_scene_metrics_from_name(stage1_ckpt.name, scene)

    stage2_best_metric_ckpt = _find_best_metric_ckpt(stage2_dir)
    stage2_best_metrics = _parse_stage_scene_metrics_from_name(stage2_best_metric_ckpt.name, scene)

    run_dir = _latest_stage3_run_dir(stage2_loaded_ckpt)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    seed_reports = json.loads((run_dir / "seed_mode_reports.json").read_text(encoding="utf-8"))
    seed_eval_cfg = json.loads((run_dir / "seed_mode_eval_config.json").read_text(encoding="utf-8"))
    stage3_errors = json.loads((run_dir / "stage3_top1_error_stats.json").read_text(encoding="utf-8"))

    stage3_cfg_name, stage3_cfg_source, _ = _resolve_recall_cfg(spec["recall_cfg"], scene, recall_cfg_doc)
    stage3_eval = dict(seed_eval_cfg["eval_thresh_cfg_resolved"])
    nrc2meter = float(manifest["dataset_metrics"]["nrc2meter_factor"])

    stage3_final_dist_median_nrc = float(stage3_errors["metrics"]["dist_2d_nrc"]["median"])
    stage3_final_dist_median_m = stage3_final_dist_median_nrc * nrc2meter
    stage3_final_rot_median = float(stage3_errors["metrics"]["rot_error_deg"]["median"])
    stage3_final_scale_median = float(stage3_errors["metrics"]["scale_ratio"]["median"])

    dataset = "visloc" if scene.startswith("visloc") else "wingtra"
    return {
        "dataset": dataset,
        "scene": scene,
        "aggregator": aggregator,
        "stage2_experiment": stage2_dir.name,
        "stage1_ckpt": stage1_ckpt.name,
        "stage1_r1": stage1_metrics["r1"],
        "stage1_med_err_m": stage1_metrics["med_err_m"],
        "stage2_loaded_ckpt": stage2_loaded_ckpt.name,
        "stage2_best_metric_ckpt": stage2_best_metric_ckpt.name,
        "stage2_r1": stage2_best_metrics["r1"],
        "stage2_med_err_m": stage2_best_metrics["med_err_m"],
        "stage3_run_dir": str(run_dir),
        "stage3_recall_cfg": spec["recall_cfg"],
        "stage3_recall_cfg_resolved": stage3_cfg_name,
        "stage3_recall_cfg_source": stage3_cfg_source,
        "stage3_threshold_summary": _threshold_summary(stage3_eval),
        "stage3_dist_th_m": float(stage3_eval["dist_th_m"]),
        "stage3_rot_th_deg": float(stage3_eval["rot_th"]),
        "stage3_scale_ratio_th": float(stage3_eval["scale_ratio_th"]),
        "coarse_dist_top1": _progressive_top1(seed_reports["coarse_retrieval"], "dist_recall"),
        "coarse_dist_rot_top1": _progressive_top1(seed_reports["coarse_retrieval"], "dist_rot_recall"),
        "coarse_dist_rot_scale_top1": _progressive_top1(seed_reports["coarse_retrieval"], "dist_rot_scale_recall"),
        "init_dist_top1": _progressive_top1(seed_reports["seed_mode_init"], "dist_recall"),
        "init_dist_rot_top1": _progressive_top1(seed_reports["seed_mode_init"], "dist_rot_recall"),
        "init_dist_rot_scale_top1": _progressive_top1(seed_reports["seed_mode_init"], "dist_rot_scale_recall"),
        "final_dist_top1": _progressive_top1(seed_reports["seed_mode_final"], "dist_recall"),
        "final_dist_rot_top1": _progressive_top1(seed_reports["seed_mode_final"], "dist_rot_recall"),
        "final_dist_rot_scale_top1": _progressive_top1(seed_reports["seed_mode_final"], "dist_rot_scale_recall"),
        "final_dist_rot_scale_top5": _progressive_top5(seed_reports["seed_mode_final"], "dist_rot_scale_recall"),
        "final_median_dist_m": stage3_final_dist_median_m,
        "final_median_rot_deg": stage3_final_rot_median,
        "final_median_scale_ratio": stage3_final_scale_median,
    }


def _write_csv(rows, output_path: Path):
    fieldnames = [
        "dataset",
        "scene",
        "aggregator",
        "stage2_experiment",
        "stage1_ckpt",
        "stage1_r1",
        "stage1_med_err_m",
        "stage2_loaded_ckpt",
        "stage2_best_metric_ckpt",
        "stage2_r1",
        "stage2_med_err_m",
        "stage3_recall_cfg",
        "stage3_recall_cfg_resolved",
        "stage3_recall_cfg_source",
        "stage3_threshold_summary",
        "stage3_dist_th_m",
        "stage3_rot_th_deg",
        "stage3_scale_ratio_th",
        "coarse_dist_top1",
        "coarse_dist_rot_top1",
        "coarse_dist_rot_scale_top1",
        "init_dist_top1",
        "init_dist_rot_top1",
        "init_dist_rot_scale_top1",
        "final_dist_top1",
        "final_dist_rot_top1",
        "final_dist_rot_scale_top1",
        "final_dist_rot_scale_top5",
        "final_median_dist_m",
        "final_median_rot_deg",
        "final_median_scale_ratio",
        "stage3_run_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows, output_path: Path):
    payload = {
        "source_run_script": str(DEFAULT_RUN_SCRIPT),
        "n_experiments": len(rows),
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_pct(value):
    return f"{float(value):.2f}"


def _fmt_num(value):
    return f"{float(value):.2f}"


def _write_markdown(rows, output_path: Path):
    lines = []
    lines.append("# Gem vs Salad Stage Summary")
    lines.append("")
    lines.append("注：`stage2_loaded_ckpt` 来自当前 `run_stage3_test_best_stage2_wreject.sh` 的实际加载 checkpoint。")
    lines.append("注：`stage2_r1/stage2_med_err_m` 来自同目录中带指标命名的 stage2 best checkpoint，用于补全 stage2 阶段结果。")
    lines.append("")

    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = sorted((row for row in rows if row["scene"] == scene), key=lambda x: x["aggregator"])
        lines.append(f"## {scene}")
        lines.append("")
        lines.append("| aggregator | stage1 R1 | stage1 MedErr(m) | stage2 R1 | stage2 MedErr(m) | coarse top1 (D / D+R / D+R+S) | init top1 (D / D+R / D+R+S) | final top1 (D / D+R / D+R+S) | final D+R+S top5 | final median dist(m) | final median rot(deg) | final median scale | recall cfg |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in scene_rows:
            lines.append(
                "| {aggregator} | {stage1_r1} | {stage1_med_err_m} | {stage2_r1} | {stage2_med_err_m} | "
                "{coarse} | {init} | {final} | {final_top5} | {dist} | {rot} | {scale} | {cfg} |".format(
                    aggregator=row["aggregator"],
                    stage1_r1=row["stage1_r1"],
                    stage1_med_err_m=row["stage1_med_err_m"],
                    stage2_r1=row["stage2_r1"],
                    stage2_med_err_m=row["stage2_med_err_m"],
                    coarse=(
                        f"{_fmt_pct(row['coarse_dist_top1'])} / "
                        f"{_fmt_pct(row['coarse_dist_rot_top1'])} / "
                        f"{_fmt_pct(row['coarse_dist_rot_scale_top1'])}"
                    ),
                    init=(
                        f"{_fmt_pct(row['init_dist_top1'])} / "
                        f"{_fmt_pct(row['init_dist_rot_top1'])} / "
                        f"{_fmt_pct(row['init_dist_rot_scale_top1'])}"
                    ),
                    final=(
                        f"{_fmt_pct(row['final_dist_top1'])} / "
                        f"{_fmt_pct(row['final_dist_rot_top1'])} / "
                        f"{_fmt_pct(row['final_dist_rot_scale_top1'])}"
                    ),
                    final_top5=_fmt_pct(row["final_dist_rot_scale_top5"]),
                    dist=_fmt_num(row["final_median_dist_m"]),
                    rot=_fmt_num(row["final_median_rot_deg"]),
                    scale=_fmt_num(row["final_median_scale_ratio"]),
                    cfg=f"{row['stage3_recall_cfg_resolved']} ({row['stage3_threshold_summary']})",
                )
            )
        if len(scene_rows) == 2:
            by_agg = {row["aggregator"]: row for row in scene_rows}
            if "gem" in by_agg and "salad" in by_agg:
                gem = by_agg["gem"]
                salad = by_agg["salad"]
                lines.append("")
                lines.append(
                    "对比: `salad - gem`, final `dist+rot+scale top1` = "
                    f"{salad['final_dist_rot_scale_top1'] - gem['final_dist_rot_scale_top1']:+.2f}, "
                    f"median dist = {salad['final_median_dist_m'] - gem['final_median_dist_m']:+.2f} m."
                )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--recall-cfg-yaml", type=Path, default=DEFAULT_RECALL_CFG_YAML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    specs = _load_run_specs(args.run_script)
    recall_cfg_doc = _load_yaml(args.recall_cfg_yaml)
    rows = [_build_row(spec, recall_cfg_doc) for spec in specs]
    rows.sort(key=lambda x: (x["dataset"], x["scene"], x["aggregator"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "gem_vs_salad_stage_summary.csv")
    _write_json(rows, args.output_dir / "gem_vs_salad_stage_summary.json")
    _write_markdown(rows, args.output_dir / "gem_vs_salad_stage_summary.md")

    print(f"[OK] wrote {len(rows)} experiment rows to {args.output_dir}")


if __name__ == "__main__":
    main()
