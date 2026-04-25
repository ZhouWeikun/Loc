import json
from pathlib import Path

import torch

import trainer_depends.datasets.dataset_neuloc_4d as dataset_mod


OUTPUT_DIR = Path("/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/dataset_setting")


SCENES = [
    {
        "scene_key": "visloc03",
        "name": "visloc_03",
        "init_name": "visloc_03",
        "dataset_name": "visloc",
        "p_satinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/03/satellite03_epsg32650_res03m_multTifs.json",
        "p_uavinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/03/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/dataset_UAV-VisLoc/03/uavimgs_geo_corrected.csv",
        "configs": [
            {"config_key": "segment82", "split_train_ratio": 0.8, "split_mode": "segment"},
            {"config_key": "interval82", "split_train_ratio": 0.8, "split_mode": "interval"},
        ],
    },
    {
        "scene_key": "visloc04",
        "name": "visloc_04",
        "init_name": "visloc_04",
        "dataset_name": "visloc",
        "p_satinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/04/satellite04_epsg32650_res03m_multTifs.json",
        "p_uavinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_geo_corrected.csv",
        "configs": [
            {"config_key": "segment82", "split_train_ratio": 0.8, "split_mode": "segment"},
            {"config_key": "interval82", "split_train_ratio": 0.8, "split_mode": "interval"},
        ],
    },
    {
        "scene_key": "zurich",
        "name": "zurich",
        "init_name": "wingtra",
        "dataset_name": "wingtra",
        "p_satinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json",
        "p_uavinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv",
        "configs": [
            {"config_key": "segment91", "split_train_ratio": 0.9, "split_mode": "segment"},
            {"config_key": "interval91", "split_train_ratio": 0.9, "split_mode": "interval"},
        ],
    },
    {
        "scene_key": "zuchwil",
        "name": "zuchwil",
        "init_name": "wingtra",
        "dataset_name": "wingtra",
        "p_satinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/zuchwil_blocks12_proj2056_res03m.json",
        "p_uavinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_info/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_info/uavimgs_geo_corrected_v1.csv",
        "configs": [
            {"config_key": "segment91", "split_train_ratio": 0.9, "split_mode": "segment"},
            {"config_key": "interval91", "split_train_ratio": 0.9, "split_mode": "interval"},
        ],
    },
]


class _FastToTensor:
    """Keep SatDataset init logic but skip caching full satellite rasters as tensors."""

    def __call__(self, img):
        return torch.zeros((3, 1, 1), dtype=torch.float32)


def _augment_export_df(df, row_indices, img_paths, scene, config, split_name):
    export_df = df.copy()
    export_df.insert(0, "source_csv_row_index", row_indices.astype(int))
    export_df.insert(1, "uavimg_path", list(img_paths))
    export_df.insert(2, "scene_key", scene["scene_key"])
    export_df.insert(3, "scene_name", scene["name"])
    export_df.insert(4, "dataset_name", scene["dataset_name"])
    export_df.insert(5, "split_config", config["config_key"])
    export_df.insert(6, "split_name", split_name)
    return export_df


def _build_datasets(scene, config):
    sat_dataset = dataset_mod.SatDataset(
        p_satinfo_json=scene["p_satinfo_json"],
        p_uav_geocsv=scene["p_uav_geocsv"],
        imgsize2net=224,
        name=scene["init_name"],
        device="cpu",
    )
    uav_dataset = dataset_mod.UAVDataset(
        p_uavinfo_json=scene["p_uavinfo_json"],
        p_uav_geocsv=scene["p_uav_geocsv"],
        imgsize2net=224,
        sat_dataset=sat_dataset,
        stage="train",
        use_augmentation=False,
        name=scene["init_name"],
        device="cpu",
        dataset_name=scene["dataset_name"],
        split_train_ratio=config["split_train_ratio"],
        split_mode=config["split_mode"],
    )
    return sat_dataset, uav_dataset


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    orig_to_tensor = dataset_mod.transforms.ToTensor
    try:
        dataset_mod.transforms.ToTensor = _FastToTensor
        for scene in SCENES:
            for config in scene["configs"]:
                sat_dataset, uav_dataset = _build_datasets(scene, config)

                train_df = _augment_export_df(
                    uav_dataset.uav_df_train,
                    uav_dataset.uav_row_indices_train,
                    uav_dataset.uavimg_paths_train,
                    scene,
                    config,
                    "train",
                )
                test_df = _augment_export_df(
                    uav_dataset.uav_df_test,
                    uav_dataset.uav_row_indices_test,
                    uav_dataset.uavimg_paths_test,
                    scene,
                    config,
                    "test",
                )

                train_path = OUTPUT_DIR / f"{scene['scene_key']}_{config['config_key']}_train.csv"
                test_path = OUTPUT_DIR / f"{scene['scene_key']}_{config['config_key']}_test.csv"
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                manifest.append(
                    {
                        "scene_key": scene["scene_key"],
                        "scene_name": scene["name"],
                        "dataset_name": scene["dataset_name"],
                        "config_key": config["config_key"],
                        "split_mode": config["split_mode"],
                        "split_train_ratio": config["split_train_ratio"],
                        "p_satinfo_json": scene["p_satinfo_json"],
                        "p_uavinfo_json": scene["p_uavinfo_json"],
                        "p_uav_geocsv": scene["p_uav_geocsv"],
                        "scale_ref_m": float(uav_dataset.scale_ref_m),
                        "geo_res_m": float(sat_dataset.geo_res_m),
                        "n_filtered_total": int(len(uav_dataset.uav_row_indices)),
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "train_csv": str(train_path),
                        "test_csv": str(test_path),
                    }
                )
                print(
                    f"{scene['scene_key']} {config['config_key']}: "
                    f"filtered={len(uav_dataset.uav_row_indices)}, "
                    f"train={len(train_df)}, test={len(test_df)}"
                )
    finally:
        dataset_mod.transforms.ToTensor = orig_to_tensor

    manifest_path = OUTPUT_DIR / "dataset_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
