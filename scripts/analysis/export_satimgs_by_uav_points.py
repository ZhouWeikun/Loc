import json
from pathlib import Path

import pandas as pd
import torch

import trainer_depends.datasets.dataset_neuloc_4d as dataset_mod


SCENES = [
    {
        "scene_key": "zurich",
        "name": "zurich",
        "init_name": "wingtra",
        "dataset_name": "wingtra",
        "p_satinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json",
        "p_uavinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv",
        "output_root": "/home/data/zwk/data_uavimgs_wingtra/Zurich",
    },
    {
        "scene_key": "zuchwil",
        "name": "zuchwil",
        "init_name": "wingtra",
        "dataset_name": "wingtra",
        "p_satinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/zuchwil_blocks12_proj2056_res03m.json",
        "p_uavinfo_json": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_info/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil/uavimgs_info/uavimgs_geo_corrected_v1.csv",
        "output_root": "/home/data/zwk/data_uavimgs_wingtra/Zuchwil",
    },
    {
        "scene_key": "visloc03",
        "name": "visloc_03",
        "init_name": "visloc_03",
        "dataset_name": "visloc",
        "p_satinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/03/satellite03_epsg32650_res03m_multTifs.json",
        "p_uavinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/03/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/dataset_UAV-VisLoc/03/uavimgs_geo_corrected.csv",
        "output_root": "/home/data/zwk/dataset_UAV-VisLoc/03",
    },
    {
        "scene_key": "visloc04",
        "name": "visloc_04",
        "init_name": "visloc_04",
        "dataset_name": "visloc",
        "p_satinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/04/satellite04_epsg32650_res03m_multTifs.json",
        "p_uavinfo_json": "/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_metainfo.json",
        "p_uav_geocsv": "/home/data/zwk/dataset_UAV-VisLoc/04/uavimgs_geo_corrected.csv",
        "output_root": "/home/data/zwk/dataset_UAV-VisLoc/04",
    },
]


class _FastToTensor:
    """Keep SatDataset init logic but skip caching full satellite rasters as tensors."""

    def __call__(self, img):
        return torch.zeros((3, 1, 1), dtype=torch.float32)


def _build_datasets(scene):
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
    )
    return sat_dataset, uav_dataset


def _build_output_path(output_dir, uavimg_path):
    uav_name = Path(uavimg_path).name
    return output_dir / f"{Path(uav_name).stem}.png"


def _export_scene(scene):
    sat_dataset, uav_dataset = _build_datasets(scene)
    crop_size = int(sat_dataset.satimgsize2crop_mean_int)
    output_dir = Path(scene["output_root"]) / f"satimgs_h{crop_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_df = uav_dataset.uav_df_filtered.copy()
    export_df.insert(0, "source_csv_row_index", uav_dataset.uav_row_indices.astype(int))
    export_df.insert(1, "uavimg_path", list(uav_dataset.uavimg_paths))
    export_df.insert(2, "scene_key", scene["scene_key"])
    export_df.insert(3, "scene_name", scene["name"])
    export_df.insert(4, "dataset_name", scene["dataset_name"])
    export_df["satimg_path"] = ""
    export_df["satimg_crop_size"] = crop_size
    export_df["sat_nrc_row"] = uav_dataset.uav_nrcs[:, 0]
    export_df["sat_nrc_col"] = uav_dataset.uav_nrcs[:, 1]

    n_samples = len(uav_dataset.uavimg_paths)
    for idx, (uavimg_path, nrc) in enumerate(zip(uav_dataset.uavimg_paths, uav_dataset.uav_nrcs)):
        satimg = sat_dataset.crop_satimg_by_nrc(
            nrc=nrc,
            satimgsize2crop=crop_size,
            type="pil",
            random_satmap=False,
        )
        if satimg.size != (crop_size, crop_size):
            raise ValueError(
                f"Unexpected crop size for {scene['scene_key']} sample {idx}: "
                f"got {satimg.size}, expected {(crop_size, crop_size)}"
            )
        output_path = _build_output_path(output_dir, uavimg_path)
        satimg.save(output_path)
        export_df.at[idx, "satimg_path"] = str(output_path)

        if (idx + 1) % 500 == 0 or idx + 1 == n_samples:
            print(
                f"{scene['scene_key']}: exported {idx + 1}/{n_samples} "
                f"to {output_dir}"
            )

    manifest_path = output_dir / "manifest.csv"
    export_df.to_csv(manifest_path, index=False)

    for satmap in sat_dataset.satmaps:
        try:
            satmap.close()
        except Exception:
            pass

    return {
        "scene_key": scene["scene_key"],
        "scene_name": scene["name"],
        "dataset_name": scene["dataset_name"],
        "crop_size": crop_size,
        "n_exported": int(n_samples),
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_path),
    }


def main():
    summary = []
    orig_to_tensor = dataset_mod.transforms.ToTensor
    try:
        dataset_mod.transforms.ToTensor = _FastToTensor
        for scene in SCENES:
            summary.append(_export_scene(scene))
    finally:
        dataset_mod.transforms.ToTensor = orig_to_tensor

    summary_path = Path("/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/dataset_setting/satimg_export_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
