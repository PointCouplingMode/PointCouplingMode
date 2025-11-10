"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import glob
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants
from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1


# def read_plymesh(filepath):
#     """Read ply file and return it as numpy array. Returns None if emtpy."""
#     with open(filepath, "rb") as f:
#         plydata = plyfile.PlyData.read(f)
#     if plydata.elements:
#         vertices = pd.DataFrame(plydata["vertex"].data).values
#         faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
#         return vertices, faces
#


def read_plymesh(filepath):
    """Read a .ply file and return (vertices, faces) as numpy arrays. Returns None if empty or corrupted."""
    try:
        with open(filepath, "rb") as f:
            plydata = plyfile.PlyData.read(f)
        if plydata.elements:
            vertices = pd.DataFrame(plydata["vertex"].data).values
            faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
            return vertices, faces
    except Exception as e:
        print(f"[WARNING] Failed to read ply file: {filepath}, error: {e}")
    return None


# Map the raw category id to the point cloud
def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0

    # Only store for the valid categories
    if label_id20 in CLASS_IDS20:
        label_id20 = CLASS_IDS20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX

    if label_id200 in CLASS_IDS200:
        label_id200 = CLASS_IDS200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return point_idx, label_id20, label_id200


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def handle_process(
    scene_path, output_path, labels_pd, train_scenes, val_scenes, parse_normals=True
):
    try:
        scene_id = os.path.basename(scene_path)
        mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
        segments_file = os.path.join(
            scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}"
        )
        aggregations_file = os.path.join(
            scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}"
        )
        info_file = os.path.join(scene_path, f"{scene_id}.txt")

        # ---------- 缺失文件检查 ----------
        required_files = [mesh_path, segments_file, aggregations_file, info_file]
        missing = [f for f in required_files if not os.path.isfile(f)]
        if missing:
            print(f"[SKIP] Scene {scene_id}: missing files {missing}")
            return

        # ---------- 输出目录 ----------
        if scene_id in train_scenes:
            output_path = os.path.join(output_path, "train", f"{scene_id}")
            split_name = "train"
        elif scene_id in val_scenes:
            output_path = os.path.join(output_path, "val", f"{scene_id}")
            split_name = "val"
        else:
            output_path = os.path.join(output_path, "test", f"{scene_id}")
            split_name = "test"

        print(f"Processing: {scene_id} in {split_name}")

        # ---------- 读取 mesh ----------
        mesh = read_plymesh(mesh_path)
        if mesh is None:
            print(f"[SKIP] Scene {scene_id}: invalid or unreadable ply")
            return
        vertices, faces = mesh

        coords = vertices[:, :3]
        colors = vertices[:, 3:6]
        save_dict = dict(
            coord=coords.astype(np.float32),
            color=colors.astype(np.uint8),
        )

        if parse_normals:
            save_dict["normal"] = vertex_normal(coords, faces).astype(np.float32)

        # ---------- test 集无标签 ----------
        if split_name == "test":
            os.makedirs(output_path, exist_ok=True)
            for key, value in save_dict.items():
                np.save(os.path.join(output_path, f"{key}.npy"), value)
            return

        # ---------- 读取 segments ----------
        try:
            with open(segments_file) as f:
                segments = json.load(f)
                seg_indices = np.array(segments["segIndices"])
        except Exception as e:
            print(f"[SKIP] Scene {scene_id}: broken segments file - {e}")
            return

        # ---------- 读取 aggregations ----------
        try:
            with open(aggregations_file) as f:
                aggregation = json.load(f)
                seg_groups = np.array(aggregation["segGroups"])
        except Exception as e:
            print(f"[SKIP] Scene {scene_id}: broken aggregations file - {e}")
            return

        # ---------- 生成标签 ----------
        semantic_gt20 = np.full(vertices.shape[0], IGNORE_INDEX, dtype=np.int16)
        semantic_gt200 = np.full(vertices.shape[0], IGNORE_INDEX, dtype=np.int16)
        instance_ids = np.full(vertices.shape[0], IGNORE_INDEX, dtype=np.int16)

        for group in seg_groups:
            point_idx, label_id20, label_id200 = point_indices_from_group(
                seg_indices, group, labels_pd
            )
            semantic_gt20[point_idx] = label_id20
            semantic_gt200[point_idx] = label_id200
            instance_ids[point_idx] = group["id"]

        save_dict["segment20"] = semantic_gt20
        save_dict["segment200"] = semantic_gt200
        save_dict["instance"] = instance_ids

        # ---------- 保存 ----------
        os.makedirs(output_path, exist_ok=True)
        for key, value in save_dict.items():
            np.save(os.path.join(output_path, f"{key}.npy"), value)

    except Exception as e:
        print(f"[ERROR] Scene {os.path.basename(scene_path)} unexpected error - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--parse_normals", default=True, type=bool, help="Whether parse point normals"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()
    print("dataset_root:", config.dataset_root)
    # Load label map
    labels_pd = pd.read_csv(
        "/home/oem/Pycharm_Pytorch_Projects/pointcept/datasets/preprocessing/scannet/meta_data/scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(
        "/home/oem/Pycharm_Pytorch_Projects/pointcept/datasets/preprocessing/scannet/meta_data/scannetv2_train.txt"
    ) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(
        "/home/oem/Pycharm_Pytorch_Projects/pointcept/datasets/preprocessing/scannet/meta_data/scannetv2_val.txt"
    ) as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans*/scene*"))
    # scene_paths = sorted(glob.glob(config.dataset_root + "scans/scene*"))
    print(f"Found {len(scene_paths)} scene path: {scene_paths[:3]}...")

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(labels_pd),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(config.parse_normals),
        )
    )
