#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


def _maybe_reexec_with_env_lib():
    if os.environ.get("_APR_PROGRESSIVE_BOOTSTRAPPED") == "1":
        return

    python_bin = os.path.abspath(sys.executable)
    if not python_bin.endswith("/bin/python"):
        return

    env_root = os.path.dirname(os.path.dirname(python_bin))
    env_lib = os.path.join(env_root, "lib")
    if not os.path.isdir(env_lib):
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    if ld_parts and ld_parts[0] == env_lib:
        return

    env = os.environ.copy()
    env["_APR_PROGRESSIVE_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else env_lib
    os.execve(python_bin, [python_bin] + sys.argv, env)


_maybe_reexec_with_env_lib()

import torch


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.util_core_eval import compute_progressive_topk_acc_from_coords


DEFAULT_EXP_DIRS = [
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_visloc04_segment82",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_visloc03_segment82",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_zuchwil_segment91",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_zurich_segment91",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_visloc04_interval82",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_visloc03_interval82",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_zuchwil_interval91",
    REPO_ROOT / "gen_fm_exps" / "ckpts" / "stage2_apr_mlp_zurich_interval91_4",
]


def _json_cell(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _to_jsonable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_epoch(path):
    match = re.search(r"epoch(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _parse_rcmed(path):
    match = re.search(r"rcmed=([0-9.]+)m", path.name)
    return float(match.group(1)) if match else None


def _select_best_ckpt(exp_dir):
    marked = sorted(exp_dir.glob("*rcmed=*.pth"))
    if marked:
        return min(marked, key=lambda p: (_parse_rcmed(p), -_parse_epoch(p)))
    ckpts = sorted(exp_dir.glob("epoch*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No epoch*.pth checkpoint found in {exp_dir}")
    return max(ckpts, key=_parse_epoch)


def _resolve_exp_dirs(raw_dirs):
    if not raw_dirs:
        return [Path(p).resolve() for p in DEFAULT_EXP_DIRS]
    return [Path(p).expanduser().resolve() for p in raw_dirs]


def _scene_from_opt(opt):
    scenes_setting = getattr(opt, "scenes_setting", None)
    if isinstance(scenes_setting, dict):
        selected = scenes_setting.get("selected_scene_name", "")
        if selected:
            return str(selected)
        scenes = scenes_setting.get("scenes", [])
        if scenes and isinstance(scenes[0], dict):
            return str(scenes[0].get("name", ""))
    return str(getattr(opt, "selected_scene_name", ""))


def _dataset_from_opt(opt):
    scenes_setting = getattr(opt, "scenes_setting", None)
    if isinstance(scenes_setting, dict) and scenes_setting.get("dataset_name"):
        return str(scenes_setting["dataset_name"])
    return str(getattr(opt, "dataset_name", ""))


def _make_recall_specs(dataset_metrics, preset, scale_ratio_th):
    nrc2meter = float(dataset_metrics["nrc2meter_factor"])
    halfimg_radius_nrc = float(dataset_metrics["halfimg_radius_nrc"])

    def _nrc_from_meter(meter):
        return float(meter) / max(nrc2meter, 1e-8)

    def _spec(spec_id, label, dist_th_nrc, rot_th_deg, scale_th):
        return {
            "id": spec_id,
            "label": label,
            "dist_th_nrc": float(dist_th_nrc),
            "dist_th_meter": float(dist_th_nrc) * nrc2meter,
            "rot_th_deg": None if rot_th_deg is None else float(rot_th_deg),
            "scale_ratio_th": None if scale_th is None else float(scale_th),
        }

    if preset == "stage3_default":
        scale = 1.2 if scale_ratio_th is None else float(scale_ratio_th)
        return [
            _spec("cfg01", f"distLambda0p55_rot5p5_scale{scale}", halfimg_radius_nrc * 0.55, 5.5, scale),
            _spec("cfg02", f"100m_rot10_scale{scale}", _nrc_from_meter(100.0), 10.0, scale),
            _spec("cfg03", f"50m_rot10_scale{scale}", _nrc_from_meter(50.0), 10.0, scale),
            _spec("cfg04", f"25m_rot10_scale{scale}", _nrc_from_meter(25.0), 10.0, scale),
        ]

    scale = 1.15 if scale_ratio_th is None else float(scale_ratio_th)
    return [
        _spec("cfg01", f"distLambda0p55_rot5p5_scale{scale}", halfimg_radius_nrc * 0.55, 5.5, scale),
        _spec("cfg02", f"100m_rot10_scale{scale}", _nrc_from_meter(100.0), 10.0, scale),
        _spec("cfg03", f"100m_rot5_scale{scale}", _nrc_from_meter(100.0), 5.0, scale),
        _spec("cfg04", f"50m_rot10_scale{scale}", _nrc_from_meter(50.0), 10.0, scale),
        _spec("cfg05", f"50m_rot5_scale{scale}", _nrc_from_meter(50.0), 5.0, scale),
        _spec("cfg06", f"25m_rot10_scale{scale}", _nrc_from_meter(25.0), 10.0, scale),
        _spec("cfg07", f"25m_rot5_scale{scale}", _nrc_from_meter(25.0), 5.0, scale),
    ]


def _parse_opt_for_apr(opts_path, ckpt_path, args):
    from trainer_depends.config.parser import get_parse

    argv = [str(Path(__file__)), "--p_yaml", str(opts_path), "--load2test", str(ckpt_path)]
    if args.gpu_ids is not None:
        argv.extend(["--gpu_ids", str(args.gpu_ids)])

    old_argv = sys.argv
    try:
        sys.argv = argv
        opt = get_parse(print_summary=False)
    finally:
        sys.argv = old_argv

    opt.load2test = str(ckpt_path)
    if args.batch_size is not None:
        opt.batchsize_uav_test = int(args.batch_size)
    if args.num_worker is not None:
        opt.num_worker = int(args.num_worker)
    return opt


def _run_apr_prediction(exp_dir, ckpt_path, opts_path, bundle_path, args):
    from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
    from trainers.stage2_arp_mlp import APRMLPTrainer

    opt = _parse_opt_for_apr(opts_path, ckpt_path, args)
    trainer = APRMLPTrainer(opt=opt)
    trainer._init_datasets(create_train_loader=False)
    trainer.coord_normer = CoordsNormProcessor(trainer.sat_dataset)
    trainer.uav_dataloader_test = trainer._make_uav_loader(
        trainer.uav_dataset_test,
        shuffle=False,
        drop_last=False,
        batch_size=int(getattr(opt, "batchsize_uav_test", opt.batchsize_uav)),
    )
    trainer._load_checkpoints_for_test()

    predictions = trainer.collect_predictions(
        trainer.uav_dataloader_test,
        max_batches=None if args.max_batches is None else int(args.max_batches),
    )
    nrc2meter = float(trainer.sat_dataset.satmap_hw_max) * float(trainer.sat_dataset.geo_res_m)
    dataset_metrics = {
        "halfimg_radius_nrc": float(trainer.sat_dataset.halfimg_radius_nrc),
        "halfimg_radius_meter": float(trainer.sat_dataset.halfimg_radius_nrc) * nrc2meter,
        "nrc2meter_factor": nrc2meter,
    }
    meta = {
        "experiment_dir": str(exp_dir),
        "experiment_name": exp_dir.name,
        "ckpt_path": str(ckpt_path),
        "ckpt_name": ckpt_path.name,
        "ckpt_epoch": _parse_epoch(ckpt_path),
        "opts_path": str(opts_path),
        "scene_name": _scene_from_opt(opt),
        "dataset_name": _dataset_from_opt(opt),
        "split_mode": str(getattr(opt, "split_mode", "")),
        "backbone": str(getattr(opt, "backbone", "")),
        "aggregator_type": str(getattr(opt, "aggregator_type", "")),
        "apr_target_mode": str(getattr(opt, "apr_target_mode", "")),
        "n_queries": int(predictions["coords_gt"].shape[0]),
        "dataset_metrics": dataset_metrics,
    }
    bundle = {
        "coords_pred": predictions["coords_pred"].to(torch.float32),
        "coords_gt": predictions["coords_gt"].to(torch.float32),
        "loss": predictions["loss"].to(torch.float32),
        "meta": meta,
        "dataset_metrics": dataset_metrics,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, bundle_path)
    with (bundle_path.parent / "apr_prediction_meta.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(meta), f, indent=2, ensure_ascii=False)
    return bundle


def _load_or_create_bundle(exp_dir, ckpt_path, opts_path, output_root, args):
    save_dir = output_root / f"{exp_dir.name}_{ckpt_path.stem}"
    bundle_path = save_dir / "apr_prediction_bundle.pt"
    if bundle_path.is_file() and not args.force:
        return torch.load(bundle_path, map_location="cpu"), bundle_path, True
    bundle = _run_apr_prediction(exp_dir, ckpt_path, opts_path, bundle_path, args)
    return bundle, bundle_path, False


def _base_row(meta, bundle_path):
    dataset_metrics = meta["dataset_metrics"]
    return {
        "experiment_name": meta["experiment_name"],
        "scene_name": meta["scene_name"],
        "dataset_name": meta["dataset_name"],
        "split_mode": meta["split_mode"],
        "backbone": meta["backbone"],
        "aggregator_type": meta["aggregator_type"],
        "apr_target_mode": meta["apr_target_mode"],
        "ckpt_epoch": meta["ckpt_epoch"],
        "ckpt_name": meta["ckpt_name"],
        "ckpt_path": meta["ckpt_path"],
        "opts_path": meta["opts_path"],
        "bundle_path": str(bundle_path),
        "n_queries": meta["n_queries"],
        "halfimg_radius_nrc": dataset_metrics["halfimg_radius_nrc"],
        "halfimg_radius_meter": dataset_metrics["halfimg_radius_meter"],
        "nrc2meter": dataset_metrics["nrc2meter_factor"],
    }


def _threshold_cell(spec):
    return (
        f"dist={spec['dist_th_nrc']:.6f} nrc / {spec['dist_th_meter']:.3f} m; "
        f"rot={spec['rot_th_deg']} deg; scale={spec['scale_ratio_th']}x"
    )


def _add_all_query_errors(row, err_stats, nrc2meter):
    row.update({
        "dist_mean_nrc": float(err_stats["mean_dist_err_top1"]),
        "dist_median_nrc": float(err_stats["median_dist_err_top1"]),
        "dist_mean_meter": float(err_stats["mean_dist_err_top1"]) * nrc2meter,
        "dist_median_meter": float(err_stats["median_dist_err_top1"]) * nrc2meter,
        "rot_mean_deg": float(err_stats["mean_rot_err_top1"]),
        "rot_median_deg": float(err_stats["median_rot_err_top1"]),
        "scale_ratio_mean": float(err_stats["mean_scale_ratio_top1"]),
        "scale_ratio_median": float(err_stats["median_scale_ratio_top1"]),
    })


def _success_error_columns(errors, nrc2meter):
    out = {
        "n_success_top1": errors.get("n_success_top1"),
        "top1_success_rate": errors.get("top1_success_rate"),
        "success_dist_mean_nrc": errors.get("mean_dist_err_top1_given_success"),
        "success_dist_median_nrc": errors.get("median_dist_err_top1_given_success"),
        "success_rot_mean_deg": errors.get("mean_rot_err_top1_given_success"),
        "success_rot_median_deg": errors.get("median_rot_err_top1_given_success"),
        "success_scale_ratio_mean": errors.get("mean_scale_ratio_top1_given_success"),
        "success_scale_ratio_median": errors.get("median_scale_ratio_top1_given_success"),
    }
    if out["success_dist_mean_nrc"] is None:
        out["success_dist_mean_meter"] = None
    else:
        out["success_dist_mean_meter"] = float(out["success_dist_mean_nrc"]) * nrc2meter
    if out["success_dist_median_nrc"] is None:
        out["success_dist_median_meter"] = None
    else:
        out["success_dist_median_meter"] = float(out["success_dist_median_nrc"]) * nrc2meter
    return out


def _compute_rows_for_bundle(bundle, bundle_path, args):
    meta = dict(bundle["meta"])
    dataset_metrics = dict(bundle["dataset_metrics"])
    nrc2meter = float(dataset_metrics["nrc2meter_factor"])
    coords_pred = bundle["coords_pred"].to(torch.float32)
    coords_gt = bundle["coords_gt"].to(torch.float32)
    coords_pred_top1 = coords_pred.unsqueeze(1)

    specs = _make_recall_specs(
        dataset_metrics=dataset_metrics,
        preset=args.preset,
        scale_ratio_th=args.scale_ratio_th,
    )
    base = _base_row(meta, bundle_path)
    wide_row = dict(base)
    long_rows = []
    error_row = None
    full_report = {
        "meta": meta,
        "recall_preset": args.preset,
        "recall_specs": specs,
        "configs": {},
    }

    for spec in specs:
        metrics, err_stats = compute_progressive_topk_acc_from_coords(
            coords_pred=coords_pred_top1,
            coords_gt=coords_gt,
            dist_th=spec["dist_th_nrc"],
            rot_th_deg=spec["rot_th_deg"],
            scale_ratio_th=spec["scale_ratio_th"],
            k_values=(1,),
        )
        progressive = dict(metrics["progressive_acc_metrics"])
        progressive_errors = dict(metrics["progressive_error_metrics"])

        if error_row is None:
            error_row = dict(base)
            _add_all_query_errors(error_row, err_stats, nrc2meter)

        cfg_id = spec["id"]
        wide_row[f"{cfg_id}_label"] = spec["label"]
        wide_row[f"{cfg_id}_threshold"] = _threshold_cell(spec)
        for criterion in ("dist_recall", "dist_rot_recall", "dist_rot_scale_recall"):
            acc = progressive[criterion]
            success_errors = _success_error_columns(progressive_errors[criterion], nrc2meter)
            wide_row[f"{cfg_id}_{criterion}"] = _json_cell(acc)
            wide_row[f"{cfg_id}_{criterion}_top1"] = float(acc.get("top1_acc", 0.0))
            wide_row[f"{cfg_id}_{criterion}_success_dist_median_meter"] = success_errors[
                "success_dist_median_meter"
            ]
            wide_row[f"{cfg_id}_{criterion}_success_rot_median_deg"] = success_errors[
                "success_rot_median_deg"
            ]
            wide_row[f"{cfg_id}_{criterion}_success_scale_ratio_median"] = success_errors[
                "success_scale_ratio_median"
            ]

            long_row = dict(base)
            long_row.update({
                "config_id": cfg_id,
                "config_label": spec["label"],
                "threshold": _threshold_cell(spec),
                "dist_th_nrc": spec["dist_th_nrc"],
                "dist_th_meter": spec["dist_th_meter"],
                "rot_th_deg": spec["rot_th_deg"],
                "scale_ratio_th": spec["scale_ratio_th"],
                "criterion": criterion,
                "top1_acc": float(acc.get("top1_acc", 0.0)),
            })
            long_row.update(success_errors)
            long_rows.append(long_row)

        full_report["configs"][cfg_id] = {
            "spec": spec,
            "progressive_acc_metrics": progressive,
            "progressive_error_metrics": progressive_errors,
            "top1_error_stats": err_stats,
            "top1_error_stats_meter": {
                "mean_dist_err_top1_meter": float(err_stats["mean_dist_err_top1"]) * nrc2meter,
                "median_dist_err_top1_meter": float(err_stats["median_dist_err_top1"]) * nrc2meter,
            },
        }

    return wide_row, long_rows, error_row, full_report


def main():
    parser = argparse.ArgumentParser(description="Export APR top1 predictions and progressive recall metrics.")
    parser.add_argument("--exp-dirs", nargs="*", default=None, help="APR checkpoint directories. Defaults to the 8 known APR dirs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "gen_fm_exps" / "analysis" / "apr_progressive_metrics",
    )
    parser.add_argument("--preset", choices=("stage1_gallery", "stage3_default"), default="stage1_gallery")
    parser.add_argument("--scale-ratio-th", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-worker", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute prediction bundles even if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Only list resolved inputs and outputs.")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    exp_dirs = _resolve_exp_dirs(args.exp_dirs)
    jobs = []
    for exp_dir in exp_dirs:
        opts_path = exp_dir / "opts.yaml"
        if not exp_dir.is_dir():
            raise NotADirectoryError(exp_dir)
        if not opts_path.is_file():
            raise FileNotFoundError(f"Missing opts.yaml: {opts_path}")
        ckpt_path = _select_best_ckpt(exp_dir)
        save_dir = output_root / f"{exp_dir.name}_{ckpt_path.stem}"
        jobs.append({
            "exp_dir": exp_dir,
            "opts_path": opts_path,
            "ckpt_path": ckpt_path,
            "bundle_path": save_dir / "apr_prediction_bundle.pt",
        })

    print(f"[Output root] {output_root}")
    print(f"[Preset] {args.preset}")
    print(f"[Jobs] {len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        print(f"  {idx:02d}. {job['exp_dir'].name}")
        print(f"      ckpt: {job['ckpt_path'].name}")
        print(f"      bundle: {job['bundle_path']}")

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    wide_rows = []
    long_rows = []
    error_rows = []
    reports = []
    for idx, job in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] {job['exp_dir'].name}")
        bundle, bundle_path, reused = _load_or_create_bundle(
            exp_dir=job["exp_dir"],
            ckpt_path=job["ckpt_path"],
            opts_path=job["opts_path"],
            output_root=output_root,
            args=args,
        )
        print(f"  bundle: {bundle_path} ({'reused' if reused else 'created'})")
        wide_row, per_job_long_rows, error_row, report = _compute_rows_for_bundle(bundle, bundle_path, args)
        wide_rows.append(wide_row)
        long_rows.extend(per_job_long_rows)
        error_rows.append(error_row)
        reports.append(report)

    wide_csv = output_root / "apr_progressive_summary_wide.csv"
    long_csv = output_root / "apr_progressive_summary_long.csv"
    error_csv = output_root / "apr_top1_error_summary.csv"
    report_json = output_root / "apr_progressive_summary.json"
    _write_csv(wide_csv, wide_rows)
    _write_csv(long_csv, long_rows)
    _write_csv(error_csv, error_rows)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(reports), f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(f"  {wide_csv}")
    print(f"  {long_csv}")
    print(f"  {error_csv}")
    print(f"  {report_json}")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
