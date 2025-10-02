# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

# HML3D 数据集的标注帧率为 20 FPS
HML3D_FPS = 20


def amass_to_amassx(file_path):
    """将 AMASS 数据集的文件路径转换为 AMASS-X 数据集的格式。"""
    file_path = file_path.replace("_poses", "_stageii")
    file_path = file_path.replace("SSM_synced", "SSM")
    file_path = file_path.replace("MPI_HDM05", "HMD05")
    file_path = file_path.replace("MPI_mosh", "MoSh")
    file_path = file_path.replace("MPI_Limits", "PosePrior")
    file_path = file_path.replace("TCD_handMocap", "TCDHands")
    file_path = file_path.replace("Transitions_mocap", "Transitions")
    file_path = file_path.replace("DFaust_67", "DFaust")
    file_path = file_path.replace("BioMotionLab_NTroje", "BMLrub")
    return file_path


@dataclass
class ProcessingOptions:
    """存储处理选项的数据类。"""

    ignore_occlusions: bool  # 是否忽略遮挡
    occlusion_bound: int = 0  # 因遮挡边界太小而被忽略的动作数量
    occlusion: int = 0  # 因无法恢复的遮挡问题而被忽略的动作数量


def fix_motion_fps(motion, dur):
    """
    根据给定的持续时间修正动作的 FPS。
    注意：此函数在当前脚本中并未被调用。
    """
    true_fps = motion.local_rotation.shape[0] / dur

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        motion.local_rotation,
        motion.root_translation,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=true_fps)

    return new_motion


def is_valid_motion(
    occlusion_data: dict,
    motion_name: str,
    options: ProcessingOptions,
):
    """
    根据遮挡数据判断一个动作是否有效。
    如果 'issue' 是 'sitting' 或 'airborne', 并且遮挡边界大于等于10帧(按30FPS计算), 
    则认为动作有效，并返回遮挡边界。
    否则，认为动作无效。
    """
    if not options.ignore_occlusions and len(occlusion_data) > 0:
        issue = occlusion_data["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in occlusion_data:
            bound = occlusion_data["idxes"][
                0
            ]  # This bounded is calculated assuming 30 FPS.....
            if bound < 10:
                options.occlusion_bound += 1
                print("bound too small", motion_name, bound)
                return False, 0
            else:
                return True, bound
        else:
            options.occlusion += 1
            print("issue irrecoverable", motion_name, issue)
            return False, 0

    return True, None


def main(
    outfile: Path,
    amass_data_path: Path,
    text_dir: Path = Path("data/hml3d/texts"),
    csv_file: Path = Path("data/hml3d/index.csv"),
    hml3d_file: Path = Path("data/hml3d/train_val.txt"),
    motion_fps_path: Path = Path("data/yaml_files/motion_fps_smpl.yaml"),
    occlusion_data_path: Path = Path("data/amass/amass_copycat_occlusion_v3.pkl"),
    humanoid_type: str = "smpl",
    dataset: str = "",
    max_length_seconds: Optional[int] = None,  # 90
    min_length_seconds: Optional[float] = 0.5,
    ignore_occlusions: bool = False,
):
    """
    该脚本用于处理 HML3D 数据集，将其转换为适用于模型训练的 YAML 格式。
    主要步骤包括：
    1. 从 CSV 和文本文件中读取 HML3D 数据集的索引和元数据。
    2. 加载 AMASS 动作捕捉数据。
    3. 根据文本描述、动作时长、遮挡等信息对动作进行筛选和切分。
    4. 将处理后的动作数据 (包括文件路径、起止时间、FPS、文本标签等) 整理并保存为 YAML 文件。

    注意：需要 babel 文件来获取剪辑的持续时间，以便调整 fps, 但此脚本似乎使用预先计算好的 FPS 文件。
    """
    num_too_long = 0
    num_too_short = 0
    total_time = 0
    total_motions = 0
    total_sub_motions = 0

    # 加载 hml3d/index.csv 文件
    df = pd.read_csv(csv_file)
    # 加载 hml3d/train_val.txt 文件，并逐行迭代
    hml3d_indices = []
    with open(hml3d_file) as f:
        # 逐行读取文件，并将行中的整数存入列表
        for line in f:
            entry = line
            # 忽略以 "M" 开头的镜像文件
            if entry.startswith("M"):
                continue
            # 去除行尾的换行符
            entry = entry.strip()
            hml3d_indices.append(int(entry))

    # 加载预处理的遮挡数据
    occlusion_data = joblib.load(occlusion_data_path)

    # 加载每个动作对应的 FPS（帧率）数据
    motion_fps_dict = yaml.load(open(motion_fps_path, "r"), Loader=yaml.FullLoader)

    output_motions = {}

    options = ProcessingOptions(
        ignore_occlusions=ignore_occlusions,
    )
    
    for k, hml3d_idx in enumerate(tqdm(hml3d_indices)):
        # 从 HML3D 的 CSV 中获取原始 AMASS 文件路径
        path = (
            df["source_path"][hml3d_idx][12:]
            .replace(".npz", ".npy")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
        )

        # 如果指定了数据集，则只处理该数据集内的动作
        if dataset not in path and dataset != "":
            continue

        # 构建用于访问 AMASS 数据的 key
        path_parts = path.split(os.path.sep)
        path_parts[0] = path_parts[0] + "-" + humanoid_type
        key = os.path.join(*(path_parts))

        # 根据 humanoid 类型（smpl/smplx）调整路径和 key
        if humanoid_type == "smplx":
            occlusion_key = ("_".join(path.split("/")))[:-4]
            key = amass_to_amassx(key)
            path = key.replace("-smplx", "")

            occlusion_key = amass_to_amassx(occlusion_key)
        else:
            occlusion_key = "-".join(["0"] + ["_".join(path.split("/"))])[:-4]

        # 检查处理后的 AMASS 动作文件是否存在
        if not os.path.exists(f"{amass_data_path}/{key}"):
            continue

        # 获取当前动作的遮挡信息
        if occlusion_key in occlusion_data:
            this_motion_occlusion = occlusion_data[occlusion_key]
        else:
            this_motion_occlusion = []

        # 检查动作的 FPS 是否在字典中
        if path not in motion_fps_dict:
            raise Exception(f"{path} not in motion_fps_dict.")
        else:
            motion_fps = motion_fps_dict[path]

        # 验证动作是否有效（基于遮挡数据）
        is_valid, fps_30_bound_frame = is_valid_motion(
            this_motion_occlusion, occlusion_key, options
        )
        if not is_valid:
            continue

        rid = hml3d_idx

        # 获取 HML3D 索引对应的行数据
        row = df.iloc[rid].to_dict()

        new_name = row["new_name"]
        label_path = (text_dir / new_name).with_suffix(".txt")
        # 读取原始文本标签
        raw_labels = label_path.read_text().strip().split("\n")

        processed_labels = []
        # 处理文本标签，去除 '#' 后面的注释和句末的 '.'
        for raw_label in raw_labels:
            label = raw_label.split("#")[0].strip()
            if label.endswith("."):
                label = label[:-1]
            processed_labels.append(label)

        # 提取动作片段的起止帧
        raw_start_frame = row["start_frame"]
        if fps_30_bound_frame is not None:
            # 如果存在遮挡边界，则截断结束帧
            # 遮挡边界是按 30 FPS 计算的，需要转换为 HML3D 的 20 FPS
            raw_end_frame = min(
                row["end_frame"], int(np.floor(fps_30_bound_frame * 1.0 / 30 * 20))
            )
        else:
            raw_end_frame = row["end_frame"]

        # 将帧数转换为秒
        start_time = raw_start_frame / HML3D_FPS
        end_time = raw_end_frame / HML3D_FPS
        length_seconds = end_time - start_time
        # 根据最大和最小长度过滤动作
        if max_length_seconds is not None and length_seconds > max_length_seconds:
            num_too_long += 1
            continue

        if length_seconds < min_length_seconds:
            num_too_short += 1
            continue

        # 将处理好的动作信息存入 output_motions 字典，按文件路径 key 组织
        if key not in output_motions:
            output_motions[key] = []
            total_motions += 1

        output_motions[key].append(
            {
                "start": start_time,
                "end": end_time,
                "fps": motion_fps,
                "hml3d_id": rid,
                "labels": processed_labels,
            }
        )

        total_time += end_time - start_time
        total_sub_motions += 1

    # 将 output_motions 字典转换为最终的 YAML 格式
    yaml_dict_format = {"motions": []}
    num_motions = 0
    num_sub_motions = 0
    for key, value in output_motions.items():
        if humanoid_type == "smplx":
            # 更改文件名以匹配 AMASS-X 的命名约定
            key = key.replace("_poses.npy", "_stageii.npy").replace(
                "-smpl/", "-smplx/"
            )
        item_dict = {
            "file": key,
            "fps": value[0]["fps"],
            "sub_motions": [],
            "idx": num_sub_motions,
        }
        num_motions += 1
        for sub_motion in value:
            item_dict["sub_motions"].append(
                {
                    "timings": {"start": sub_motion["start"], "end": sub_motion["end"]},
                    "weight": 1.0,
                    "idx": num_sub_motions,
                    "hml3d_id": sub_motion["hml3d_id"],
                    "labels": sub_motion["labels"],
                }
            )
            num_sub_motions += 1

        yaml_dict_format["motions"].append(item_dict)

    # 打印处理结果的统计信息
    print(f"Saving {len(output_motions)} motions to {outfile}")
    print(
        f"Total of {num_motions} motions, and {num_sub_motions} sub-motions, equaling to {total_time / 60} minutes of motion."
    )
    print(f"Num too long: {num_too_long}")
    print(f"Num too short: {num_too_short}")
    print(
        f"Num occluded: {options.occlusion}, occluded_bound: {options.occlusion_bound}"
    )

    # 将结果字典写入 YAML 文件
    with open(outfile, "w") as file:
        yaml.dump(yaml_dict_format, file)


if __name__ == "__main__":
    typer.run(main)
