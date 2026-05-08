#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SCRIPT = REPO_ROOT / "trainers" / "stage1_visual_encoder_wANCE.py"
EXPORTER_SCRIPT = REPO_ROOT / "scripts" / "res_export" / "export_stage1_gallery_summary_from_plan.py"


def _iter_jobs(plan_payload):
    shared = dict(plan_payload.get("shared", {}))
    gallery = dict(shared.get("gallery", {}))
    output_gallery_root_dir = str(shared.get("output_gallery_root_dir", "")).strip()
    query_split_policies = dict(shared.get("query_split_policies", {}))
    for exp in plan_payload.get("experiments", []):
        exp = dict(exp)
        policy_name = str(exp.get("query_split_policy_ref", "")).strip()
        policy = dict(query_split_policies.get(policy_name, {}) or {})
        subset_policy = dict(policy.get("subset_on_selected_split", {}) or {})
        for job in exp.get("jobs", []):
            yield {
                "experiment_dir": str(exp["experiment_dir"]),
                "dataset": str(exp["dataset"]),
                "scene_name": str(job["scene_name"]),
                "opts_yaml": str(exp["opts_yaml"]),
                "selected_ckpt": str(exp["selected_ckpt"]),
                "gallery_root_dir": output_gallery_root_dir,
                "gallery_overlap": float(gallery["overlap"]),
                "gallery_n_rot": int(gallery["n_rot"]),
                "gallery_n_scale": int(gallery["n_scale"]),
                "gallery_scale_mode": str(gallery["scale_mode"]),
                "planned_gallery_save_dir": str(job.get("planned_gallery_save_dir", "") or "").strip(),
                "query_subset_mode": str(subset_policy.get("mode", "")).strip(),
                "query_subset_ratio": subset_policy.get("train_ratio", None),
                "query_subset_take": str(subset_policy.get("take", "test")).strip(),
                "query_subset_seed": int(subset_policy.get("random_seed", 2026)),
            }


def _bundle_path(job):
    planned = str(job.get("planned_gallery_save_dir", "")).strip()
    if not planned:
        return None
    return Path(planned) / "stage1_retrieval_eval_bundle.pt"


def _bundle_glob_suffix(job):
    ckpt_stem = Path(job["selected_ckpt"]).stem
    return f"{job['experiment_dir']}_{ckpt_stem}/stage1_retrieval_eval_bundle.pt"


def _resolve_bundle_path(job):
    expected = _bundle_path(job)
    if expected is not None and expected.is_file():
        return expected

    gallery_root = Path(job["gallery_root_dir"])
    suffix = _bundle_glob_suffix(job)
    matches = sorted(gallery_root.glob(f"**/{suffix}"))
    scene_token = f"/{job['scene_name']}_"
    scene_matches = [m for m in matches if scene_token in str(m)]
    if len(scene_matches) == 1:
        return scene_matches[0]
    if len(scene_matches) > 1:
        raise RuntimeError(
            f"Multiple bundle matches found for {job['experiment_dir']} / {job['scene_name']}: "
            + ", ".join(str(m) for m in scene_matches)
        )
    if expected is not None:
        return expected
    return gallery_root / "__missing__" / suffix


def _run_one(job, env, log_handle):
    cmd = [
        sys.executable,
        str(TRAINER_SCRIPT),
        "--test_only",
        "true",
        "--test_mode",
        "gallery_bank",
        "--scene_name",
        job["scene_name"],
        "--p_yaml",
        job["opts_yaml"],
        "--load2test",
        job["selected_ckpt"],
        "--gallery_root_dir",
        job["gallery_root_dir"],
        "--gallery_overlap",
        str(job["gallery_overlap"]),
        "--gallery_n_rot",
        str(job["gallery_n_rot"]),
        "--gallery_n_scale",
        str(job["gallery_n_scale"]),
        "--gallery_scale_mode",
        job["gallery_scale_mode"],
    ]
    if job["query_subset_mode"]:
        cmd.extend([
            "--eval_query_subset_mode",
            job["query_subset_mode"],
            "--eval_query_subset_ratio",
            str(job["query_subset_ratio"]),
            "--eval_query_subset_take",
            job["query_subset_take"],
            "--eval_query_subset_seed",
            str(job["query_subset_seed"]),
        ])

    header = (
        f"[Stage1PlanRun] scene={job['scene_name']} exp={job['experiment_dir']} "
        f"ckpt={Path(job['selected_ckpt']).name}"
    )
    print(header)
    log_handle.write(header + "\n")
    log_handle.write("[CMD] " + " ".join(cmd) + "\n")
    log_handle.flush()

    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        log_handle.write(line)
    return_code = process.wait()
    log_handle.write(f"[RET] {return_code}\n\n")
    log_handle.flush()
    if return_code != 0:
        raise RuntimeError(f"Job failed with exit code {return_code}: {header}")

    bundle_path = _resolve_bundle_path(job)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Expected bundle file not found after run: {bundle_path}")
    expected = _bundle_path(job)
    if expected is not None and bundle_path != expected:
        msg = f"[Stage1PlanRun] resolved bundle_path={bundle_path}"
        print(msg)
        log_handle.write(msg + "\n")
        log_handle.flush()


def main():
    parser = argparse.ArgumentParser(description="Run Stage1 gallery eval jobs from a plan YAML.")
    parser.add_argument("--plan-yaml", type=Path, required=True, help="Plan YAML file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Summary CSV output path.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional run log path. Defaults to <output-csv>.run.log",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose expected bundle file already exists.",
    )
    parser.add_argument(
        "--scale-ratio-th",
        type=float,
        default=1.15,
        help="Scale ratio threshold passed to the export script for all recall configs (default: 1.15).",
    )
    args = parser.parse_args()

    plan_payload = yaml.safe_load(args.plan_yaml.read_text(encoding="utf-8"))
    jobs = list(_iter_jobs(plan_payload))
    log_path = args.log_path or args.output_csv.with_suffix(args.output_csv.suffix + ".run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    completed = 0
    skipped = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"[PLAN] {args.plan_yaml}\n")
        log_handle.write(f"[OUTPUT_CSV] {args.output_csv}\n")
        log_handle.write(f"[TOTAL_JOBS] {len(jobs)}\n\n")
        for idx, job in enumerate(jobs, start=1):
            bundle_path = _resolve_bundle_path(job)
            if args.skip_existing and bundle_path.is_file():
                skipped += 1
                msg = f"[Stage1PlanRun] skip-existing {idx}/{len(jobs)} -> {bundle_path}"
                print(msg)
                log_handle.write(msg + "\n")
                continue
            print(f"[Stage1PlanRun] start {idx}/{len(jobs)}")
            _run_one(job, env=env, log_handle=log_handle)
            completed += 1
            print(f"[Stage1PlanRun] done {idx}/{len(jobs)}")

    export_cmd = [
        sys.executable,
        str(EXPORTER_SCRIPT),
        "--plan-yaml",
        str(args.plan_yaml),
        "--output-csv",
        str(args.output_csv),
        "--scale-ratio-th",
        str(args.scale_ratio_th),
    ]
    print("[Stage1PlanRun] export summary csv")
    subprocess.run(export_cmd, cwd=str(REPO_ROOT), env=env, check=True)
    print(
        f"[Stage1PlanRun] finished: completed={completed}, skipped={skipped}, "
        f"csv={args.output_csv}, log={log_path}"
    )


if __name__ == "__main__":
    main()
