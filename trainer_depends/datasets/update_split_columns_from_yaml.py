#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import tempfile
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from trainer_depends.utils.compat_runtime import preload_conda_libstdcpp

preload_conda_libstdcpp()

import pandas as pd

from trainer_depends.config.parser import _expand_selected_scene_config
from trainer_depends.datasets.dataset_neuloc_4d import (
    SatDataset,
    UAVDataset,
    _build_split_column_name,
)


def _load_yaml_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    scenes_setting = config.get("scenes_setting", {})
    if scenes_setting:
        scenes_setting = _expand_selected_scene_config(scenes_setting)

    data_setting = config.get("data_setting", {})
    split_train_ratio = float(data_setting.get("split_train_ratio", 0.9))
    split_mode = str(data_setting.get("split_mode", "segment")).strip().lower()
    imgsize2net = int(data_setting.get("imgsize2net", 224))
    dataset_name = scenes_setting.get("dataset_name", config.get("dataset_name", "default"))
    scenes = scenes_setting.get("scenes", [])
    if not scenes:
        raise ValueError(f"No scenes found in YAML: {yaml_path}")

    return {
        "yaml_path": yaml_path,
        "split_train_ratio": split_train_ratio,
        "split_mode": split_mode,
        "imgsize2net": imgsize2net,
        "dataset_name": dataset_name,
        "scenes": scenes,
    }


def _build_labels(total_rows, train_indices, test_indices):
    labels = np.full(total_rows, "", dtype=object)
    labels[np.asarray(train_indices, dtype=np.int64)] = "train"
    labels[np.asarray(test_indices, dtype=np.int64)] = "test"
    return labels


def _write_split_column(csv_path, column_name, labels):
    df = pd.read_csv(csv_path)
    if len(df) != len(labels):
        raise ValueError(
            f"Row count mismatch when writing split column '{column_name}' to {csv_path}: "
            f"csv rows={len(df)}, labels={len(labels)}"
        )
    df[column_name] = labels
    df.to_csv(csv_path, index=False)


def _make_legacy_temp_csv(csv_path, drop_column):
    df = pd.read_csv(csv_path)
    legacy_df = df.drop(columns=[drop_column], errors="ignore")
    tmp = tempfile.NamedTemporaryFile(prefix="legacy_split_", suffix=".csv", delete=False)
    tmp.close()
    legacy_df.to_csv(tmp.name, index=False)
    return tmp.name, len(df)


def _process_scene(config_info, scene):
    scene_name = scene["name"]
    split_mode = config_info["split_mode"]
    split_train_ratio = config_info["split_train_ratio"]
    imgsize2net = config_info["imgsize2net"]
    dataset_name = config_info["dataset_name"]
    csv_path = scene["p_uav_geocsv"]
    column_name = _build_split_column_name(split_mode, split_train_ratio)

    temp_csv_path, total_rows = _make_legacy_temp_csv(csv_path, drop_column=column_name)
    try:
        sat_dataset = SatDataset(
            p_satinfo_json=scene["p_satinfo_json"],
            p_uav_geocsv=temp_csv_path,
            imgsize2net=imgsize2net,
            split_train_ratio=split_train_ratio,
            split_mode=split_mode,
            name=scene_name,
            device="cpu",
        )
        uav_dataset = UAVDataset(
            p_uavinfo_json=scene["p_uavinfo_json"],
            p_uav_geocsv=temp_csv_path,
            sat_dataset=sat_dataset,
            imgsize2net=imgsize2net,
            stage="train",
            use_augmentation=False,
            name=scene_name,
            device="cpu",
            dataset_name=dataset_name,
            split_train_ratio=split_train_ratio,
            split_mode=split_mode,
        )

        labels = _build_labels(
            total_rows=total_rows,
            train_indices=uav_dataset.uav_row_indices_train,
            test_indices=uav_dataset.uav_row_indices_test,
        )
        _write_split_column(csv_path, column_name, labels)

        print(
            f"[updated] {scene_name}: {csv_path}\n"
            f"  column={column_name} train={int((labels == 'train').sum())} "
            f"test={int((labels == 'test').sum())} dropped={int((labels == '').sum())}"
        )
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Populate split_<mode><xx> columns in p_uav_geocsv files from stage YAML configs."
    )
    parser.add_argument(
        "yaml_paths",
        nargs="+",
        help="One or more YAML config paths, e.g. trainer_depends/configs/stage1_visual_encoder_wingtra.yaml",
    )
    args = parser.parse_args()

    for yaml_path in args.yaml_paths:
        config_info = _load_yaml_config(yaml_path)
        print(
            f"[yaml] {yaml_path} -> mode={config_info['split_mode']} "
            f"ratio={config_info['split_train_ratio']} dataset={config_info['dataset_name']}"
        )
        for scene in config_info["scenes"]:
            _process_scene(config_info, scene)


if __name__ == "__main__":
    main()
