#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import select
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CTRL_BEST_EPOCH_PATH = PROJECT_ROOT / "gen_fm_exps" / "analysis" / "ctrl_best_epoch.txt"
TRAINER_SCRIPT_PATH = PROJECT_ROOT / "trainers" / "stage1_visual_encoder_w_ANCE.py"
ANALYSIS_DIR = PROJECT_ROOT / "gen_fm_exps" / "analysis"
SUMMARY_CSV_PATH = ANALYSIS_DIR / "stage1_eval_ctrl1.csv"
RUN_LOG_PATH = ANALYSIS_DIR / "stage1_eval_ctrl1_run.log"

DATASET_CFG = {
    "visloc": {
        "scenes": ("visloc_03", "visloc_04"),
        "dist_th_meter": 100.0,
    },
    "wingtra": {
        "scenes": ("zurich", "zuchwil"),
        "dist_th_meter": 25.0,
    },
}

METRIC_KEYS = (
    "top1_acc",
    "top5_acc",
    "top10_acc",
    "top20_acc",
    "top50_acc",
    "top128_acc",
    "top256_acc",
    "top512_acc",
    "top1024_acc",
)


def parse_ctrl_best_epoch(ctrl_path: Path):
    rows = []
    current_pattern = None
    current_row = None
    for raw_line in ctrl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("[") and line.endswith("]"):
            current_pattern = line[1:-1]
            current_row = None
            continue
        if line.startswith("- experiment_dir: "):
            current_row = {
                "pattern": current_pattern,
                "experiment_dir": line.split(": ", 1)[1].strip(),
            }
            rows.append(current_row)
            continue
        if current_row is None:
            continue
        stripped = line.strip()
        if stripped.startswith("best_epoch: "):
            current_row["best_epoch"] = int(stripped.split(": ", 1)[1])
        elif stripped.startswith("best_sum_top1: "):
            value = stripped.split(": ", 1)[1].rstrip("%")
            current_row["best_sum_top1"] = float(value)
        elif stripped.startswith("scene_top1_at_best_epoch: "):
            current_row["scene_top1_at_best_epoch"] = stripped.split(": ", 1)[1]
    return [row for row in rows if "best_epoch" in row]


def infer_dataset_name(experiment_dir: str):
    if experiment_dir.startswith("stage1_visloc_"):
        return "visloc"
    if experiment_dir.startswith("stage1_wingtra_"):
        return "wingtra"
    raise ValueError(f"Cannot infer dataset name from experiment_dir={experiment_dir}")


def parse_ckpt_epoch(path: Path):
    match = re.search(r"epoch(\d{3})", path.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint epoch from {path}")
    return int(match.group(1))


def resolve_checkpoint(ckpt_dir: Path, best_epoch: int):
    exact_matches = sorted(ckpt_dir.glob(f"epoch{best_epoch:03d}*.pth"))
    if exact_matches:
        ckpt_path = exact_matches[0]
        return {
            "ckpt_path": ckpt_path,
            "eval_ckpt_epoch": best_epoch,
            "ckpt_resolution": "exact",
            "ckpt_epoch_delta": 0,
        }

    candidates = []
    for ckpt_path in sorted(ckpt_dir.glob("epoch*.pth")):
        ckpt_epoch = parse_ckpt_epoch(ckpt_path)
        candidates.append((ckpt_epoch, ckpt_path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    ckpt_epoch, ckpt_path = min(
        candidates,
        key=lambda item: (abs(item[0] - best_epoch), 0 if item[0] <= best_epoch else 1, item[0]),
    )
    return {
        "ckpt_path": ckpt_path,
        "eval_ckpt_epoch": ckpt_epoch,
        "ckpt_resolution": "nearest",
        "ckpt_epoch_delta": ckpt_epoch - best_epoch,
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_save_dir(line: str):
    marker = "save_dir="
    if marker not in line:
        return None
    return line.split(marker, 1)[1].strip()


def find_report_path(scene_name: str, experiment_dir: str, eval_ckpt_epoch: int):
    gallery_root = PROJECT_ROOT / "gen_fm_exps" / "gallery_bank_stage1"
    pattern = f"{scene_name}_*/{experiment_dir}_epoch{eval_ckpt_epoch:03d}*/stage1_retrieval_eval_report.json"
    matches = sorted(gallery_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_single_eval(task, scene_name: str, trainer_script: Path, run_log_path: Path, dry_run: bool):
    cmd = [
        sys.executable,
        str(trainer_script),
        "--test_only",
        "true",
        "--test_mode",
        "gallery_bank",
        "--scene_name",
        scene_name,
        "--p_yaml",
        str(task["opts_yaml"]),
        "--load2test",
        str(task["ckpt_path"]),
        "--dist_th_meter",
        str(task["dist_th_meter"]),
    ]
    result = {
        "scene_name": scene_name,
        "status": "pending",
        "save_dir": "",
        "report_path": "",
        "threshold_dist_meter": task["dist_th_meter"],
    }
    if dry_run:
        result["status"] = "dry_run"
        result["command"] = " ".join(cmd)
        return result

    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(
        f"  RUN scene={scene_name} dist_th_meter={task['dist_th_meter']:.1f} "
        f"ckpt={task['ckpt_path'].name}"
    )
    recent_lines = deque(maxlen=20)
    save_dir = None
    saved_report_dir = None
    start_time = time.time()
    last_heartbeat = start_time

    with run_log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n" + "=" * 80 + "\n")
        log_handle.write(
            f"[START] experiment={task['experiment_dir']} scene={scene_name} cmd={' '.join(cmd)}\n"
        )
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None

        while True:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if line:
                    log_handle.write(line)
                    log_handle.flush()
                    recent_lines.append(line.rstrip("\n"))
                    maybe_save_dir = extract_save_dir(line)
                    if maybe_save_dir is not None:
                        save_dir = maybe_save_dir
                    if "[Gallery Eval] saved structured report to " in line:
                        saved_report_dir = line.split("to ", 1)[1].strip()
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break

            now = time.time()
            if now - last_heartbeat >= 30.0:
                elapsed_min = (now - start_time) / 60.0
                print(f"    still running scene={scene_name} elapsed={elapsed_min:.1f}m")
                last_heartbeat = now

        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            recent_lines.append(line.rstrip("\n"))
            maybe_save_dir = extract_save_dir(line)
            if maybe_save_dir is not None:
                save_dir = maybe_save_dir
            if "[Gallery Eval] saved structured report to " in line:
                saved_report_dir = line.split("to ", 1)[1].strip()

        return_code = process.wait()
        elapsed_sec = time.time() - start_time
        result["elapsed_sec"] = round(elapsed_sec, 2)

        if save_dir is None and saved_report_dir is not None:
            save_dir = saved_report_dir
        result["save_dir"] = "" if save_dir is None else save_dir

        if return_code != 0:
            result["status"] = f"failed({return_code})"
            result["tail"] = " | ".join(recent_lines)
            print(f"  FAIL scene={scene_name} rc={return_code}")
            return result

        report_path = None
        if save_dir:
            candidate = Path(save_dir) / "stage1_retrieval_eval_report.json"
            if candidate.exists():
                report_path = candidate
        if report_path is None:
            report_path = find_report_path(
                scene_name=scene_name,
                experiment_dir=task["experiment_dir"],
                eval_ckpt_epoch=task["eval_ckpt_epoch"],
            )
        if report_path is None or not report_path.exists():
            result["status"] = "missing_report"
            result["tail"] = " | ".join(recent_lines)
            print(f"  FAIL scene={scene_name} missing report")
            return result

        report = load_json(report_path)
        thresholds = report.get("thresholds", {})
        acc_metrics = report.get("acc_metrics", {})
        err_stats = report.get("err_stats", {})
        nrc2meter = thresholds.get("nrc2meter", None)

        result.update(
            {
                "status": "ok",
                "report_path": str(report_path),
                "save_dir": str(report_path.parent),
                "n_queries": report.get("n_queries", ""),
                "threshold_dist_meter_report": thresholds.get("dist_meter", ""),
                "threshold_norm_dist_report": thresholds.get("norm_dist", ""),
                "top1_dist_nrc_mean": err_stats.get("mean_dist_err_top1", ""),
                "top1_dist_nrc_median": err_stats.get("median_dist_err_top1", ""),
            }
        )
        if nrc2meter is not None and err_stats.get("mean_dist_err_top1", None) is not None:
            result["top1_dist_meter_mean"] = err_stats["mean_dist_err_top1"] * nrc2meter
        else:
            result["top1_dist_meter_mean"] = ""
        if nrc2meter is not None and err_stats.get("median_dist_err_top1", None) is not None:
            result["top1_dist_meter_median"] = err_stats["median_dist_err_top1"] * nrc2meter
        else:
            result["top1_dist_meter_median"] = ""
        for key in METRIC_KEYS:
            result[key] = acc_metrics.get(key, "")

        print(
            f"  DONE scene={scene_name} top1={result.get('top1_acc', '')} "
            f"top5={result.get('top5_acc', '')} save_dir={result['save_dir']}"
        )
        return result


def build_tasks(ctrl_rows, ckpt_root: Path):
    tasks = []
    for row in ctrl_rows:
        experiment_dir = row["experiment_dir"]
        dataset_name = infer_dataset_name(experiment_dir)
        cfg = DATASET_CFG[dataset_name]
        ckpt_dir = ckpt_root / experiment_dir
        opts_yaml = ckpt_dir / "opts.yaml"
        task = {
            "pattern": row.get("pattern", ""),
            "experiment_dir": experiment_dir,
            "dataset_name": dataset_name,
            "best_epoch": int(row["best_epoch"]),
            "best_sum_top1_log": row.get("best_sum_top1", ""),
            "scene_top1_at_best_epoch": row.get("scene_top1_at_best_epoch", ""),
            "ckpt_dir": ckpt_dir,
            "opts_yaml": opts_yaml,
            "dist_th_meter": float(cfg["dist_th_meter"]),
            "scenes": tuple(cfg["scenes"]),
        }
        task.update(resolve_checkpoint(ckpt_dir=ckpt_dir, best_epoch=task["best_epoch"]))
        tasks.append(task)
    return tasks


def flatten_task_to_row(task, scene_results):
    row = {
        "pattern": task["pattern"],
        "experiment_dir": task["experiment_dir"],
        "dataset_name": task["dataset_name"],
        "best_epoch": task["best_epoch"],
        "best_sum_top1_log": task["best_sum_top1_log"],
        "scene_top1_at_best_epoch_log": task["scene_top1_at_best_epoch"],
        "dist_th_meter": task["dist_th_meter"],
        "opts_yaml": str(task["opts_yaml"]),
        "ckpt_path": str(task["ckpt_path"]),
        "eval_ckpt_epoch": task["eval_ckpt_epoch"],
        "ckpt_resolution": task["ckpt_resolution"],
        "ckpt_epoch_delta": task["ckpt_epoch_delta"],
    }

    overall_status = "ok"
    top1_values = []
    top5_values = []
    top10_values = []
    for idx, scene_name in enumerate(task["scenes"], start=1):
        scene_key = f"scene{idx}"
        scene_result = scene_results.get(scene_name, {"scene_name": scene_name, "status": "missing"})
        row[f"{scene_key}_name"] = scene_name
        row[f"{scene_key}_status"] = scene_result.get("status", "")
        row[f"{scene_key}_gallery_save_dir"] = scene_result.get("save_dir", "")
        row[f"{scene_key}_report_path"] = scene_result.get("report_path", "")
        row[f"{scene_key}_n_queries"] = scene_result.get("n_queries", "")
        row[f"{scene_key}_threshold_dist_meter_report"] = scene_result.get("threshold_dist_meter_report", "")
        row[f"{scene_key}_threshold_norm_dist_report"] = scene_result.get("threshold_norm_dist_report", "")
        row[f"{scene_key}_top1_dist_meter_mean"] = scene_result.get("top1_dist_meter_mean", "")
        row[f"{scene_key}_top1_dist_meter_median"] = scene_result.get("top1_dist_meter_median", "")
        row[f"{scene_key}_elapsed_sec"] = scene_result.get("elapsed_sec", "")
        for metric_key in METRIC_KEYS:
            value = scene_result.get(metric_key, "")
            row[f"{scene_key}_{metric_key}"] = value
        if scene_result.get("status") != "ok":
            overall_status = "partial" if overall_status == "ok" else overall_status
        else:
            if scene_result.get("top1_acc", "") != "":
                top1_values.append(float(scene_result["top1_acc"]))
            if scene_result.get("top5_acc", "") != "":
                top5_values.append(float(scene_result["top5_acc"]))
            if scene_result.get("top10_acc", "") != "":
                top10_values.append(float(scene_result["top10_acc"]))

    row["sum_top1_acc"] = sum(top1_values) if top1_values else ""
    row["mean_top1_acc"] = (sum(top1_values) / len(top1_values)) if top1_values else ""
    row["sum_top5_acc"] = sum(top5_values) if top5_values else ""
    row["mean_top5_acc"] = (sum(top5_values) / len(top5_values)) if top5_values else ""
    row["sum_top10_acc"] = sum(top10_values) if top10_values else ""
    row["mean_top10_acc"] = (sum(top10_values) / len(top10_values)) if top10_values else ""
    row["status"] = overall_status
    row["notes"] = ""
    if task["ckpt_resolution"] != "exact":
        row["notes"] = (
            f"best_epoch has no exact checkpoint; used epoch {task['eval_ckpt_epoch']} "
            f"(delta={task['ckpt_epoch_delta']:+d})"
        )
    return row


def write_summary_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run stage1 gallery eval for control experiments.")
    parser.add_argument("--ctrl-file", type=Path, default=CTRL_BEST_EPOCH_PATH)
    parser.add_argument("--trainer-script", type=Path, default=TRAINER_SCRIPT_PATH)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV_PATH)
    parser.add_argument("--run-log", type=Path, default=RUN_LOG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ctrl_rows = parse_ctrl_best_epoch(args.ctrl_file)
    tasks = build_tasks(ctrl_rows=ctrl_rows, ckpt_root=PROJECT_ROOT / "gen_fm_exps" / "ckpts")

    print(f"Loaded {len(tasks)} experiments from {args.ctrl_file}")
    total_scene_jobs = sum(len(task["scenes"]) for task in tasks)
    print(f"Total scene evaluations: {total_scene_jobs}")

    summary_rows = []
    job_index = 0
    for exp_index, task in enumerate(tasks, start=1):
        print(
            f"[{exp_index}/{len(tasks)}] experiment={task['experiment_dir']} "
            f"best_epoch={task['best_epoch']} eval_ckpt_epoch={task['eval_ckpt_epoch']} "
            f"ckpt_resolution={task['ckpt_resolution']}"
        )
        scene_results = {}
        for scene_name in task["scenes"]:
            job_index += 1
            print(f"  [{job_index}/{total_scene_jobs}] scene={scene_name}")
            scene_results[scene_name] = run_single_eval(
                task=task,
                scene_name=scene_name,
                trainer_script=args.trainer_script,
                run_log_path=args.run_log,
                dry_run=args.dry_run,
            )
        summary_rows.append(flatten_task_to_row(task, scene_results))
        write_summary_csv(summary_rows, args.summary_csv)
        print(f"  wrote partial summary to {args.summary_csv}")

    print(f"Completed. Summary CSV: {args.summary_csv}")
    print(f"Run log: {args.run_log}")


if __name__ == "__main__":
    main()
