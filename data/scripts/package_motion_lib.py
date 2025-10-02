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
from pathlib import Path

import torch
import typer
import yaml
import tempfile
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize


def main(
    motion_file: Path,
    amass_data_path: Path,
    outpath: Path,
    humanoid_type: str = "smpl",
    create_text_embeddings: bool = False,
    num_data_splits: int = None,
):
    """
    主函数，用于处理运动文件，创建并保存MotionLib的状态。

    Args:
        motion_file (Path): 包含运动数据列表的yaml文件路径。
        amass_data_path (Path): AMASS数据集的根目录路径，用于解析运动文件中的相对路径。
        outpath (Path): 保存处理后的MotionLib状态文件的输出路径。
        humanoid_type (str, optional): 使用的人形模型类型，默认为 "smpl"。
        create_text_embeddings (bool, optional): 是否为运动数据创建文本嵌入，默认为 False。
        num_data_splits (int, optional): 将运动文件分割成的小文件数量。如果为None，则不分割。默认为 None。
    """
    # Hydra配置文件的相对路径
    config_path = "../../phys_anim/config/robot"

    # 使用hydra初始化并加载配置
    # version_base=None 允许使用旧版本的hydra API
    # job_name 用于标识本次运行，这里设为 "test_app"
    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        # 根据humanoid_type组合配置文件，例如 smpl.yaml
        cfg = compose(config_name=humanoid_type)

    # 从配置中获取关键身体部位的索引
    key_body_ids = torch.tensor(
        [
            # 在dfs_body_names（深度优先搜索顺序的身体部位名称列表）中查找每个关键身体部位的索引
            cfg.robot.dfs_body_names.index(key_body_name)
            for key_body_name in cfg.robot.key_bodies
        ],
        dtype=torch.long,
    )
    
    # 计算每个关节自由度（DOF）在总自由度列表中的起始偏移量
    dof_offsets = []
    previous_dof_name = "null"
    for dof_offset, dof_name in enumerate(cfg.robot.dfs_dof_names):
        # 自由度名称通常带有 "_x", "_y", "_z" 后缀，我们通过去掉最后两个字符来获取关节名称
        if dof_name[:-2] != previous_dof_name:
            previous_dof_name = dof_name[:-2]
            dof_offsets.append(dof_offset)
    # 添加最后一个偏移量，即总自由度数量
    dof_offsets.append(len(cfg.robot.dfs_dof_names))

    print("Creating motion state")
    motion_files = []
    # 如果指定了 num_data_splits，则将原始运动文件分割成多个小文件
    if num_data_splits is not None:
        # 运动文件是一个yaml文件
        # 加载yaml文件并将其分成 num_data_splits 份
        # 将每个分割后的部分保存为单独的文件
        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            motions = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        num_motions = len(motions)
        split_size = num_motions // num_data_splits
        for i in range(num_data_splits):
            # 最后一个分片包含所有剩余的动作，以防除不尽
            if i == num_data_splits - 1:
                split_motions = motions[i * split_size :]
            else:
                split_motions = motions[i * split_size : (i + 1) * split_size]

            # 为每个分片中的动作重新编制索引
            motion_idx = 0
            for motion in split_motions:
                motion["idx"] = motion_idx
                if "sub_motions" in motion:
                    for sub_motion in motion["sub_motions"]:
                        sub_motion["idx"] = motion_idx
                        motion_idx += 1
                else:
                    motion_idx += 1

            # 为分片文件生成新的文件名，例如 "original_name_0.yaml"
            split_name = motion_file.with_name(
                motion_file.stem + f"_{i}" + motion_file.suffix
            )
            # 将分片数据写入新的yaml文件
            with open(split_name, "w") as f:
                yaml.dump({"motions": split_motions}, f)

            # 将分片文件名和对应的输出文件名添加到待处理列表中
            motion_files.append(
                (
                    str(split_name),
                    outpath.with_name(outpath.stem + f"_{i}" + outpath.suffix),
                )
            )
    else:
        # 如果不分割，则直接处理原始文件
        motion_files.append((motion_file, motion_file))

    # 遍历所有待处理的运动文件（可能是分割后的文件，也可能是原始文件）
    for motion_file, outpath in motion_files:
        # 打开并读取yaml运动文件
        with open(motion_file, "r") as f:
            motion_data = yaml.safe_load(f)

        # 修改文件路径，将相对路径转换为基于amass_data_path的绝对路径
        for motion in motion_data["motions"]:
            motion["file"] = str(amass_data_path.resolve() / motion["file"])

        # 将修改后的运动数据保存到一个临时文件中
        # MotionLib需要一个文件路径作为输入，所以我们不能直接传递数据
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(motion_data, temp_file)
            temp_file_path = temp_file.name

        # 使用临时文件路径来实例化MotionLib
        cfg.motion_lib.motion_file = temp_file_path
        mlib = instantiate(
            cfg.motion_lib,
            dof_body_ids=cfg.robot.dfs_dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device="cpu", # 在CPU上处理以节省GPU内存
            create_text_embeddings=create_text_embeddings,
            skeleton_tree=None,
        )

        print("Saving motion state")

        # 将实例化的MotionLib的状态保存到二进制文件中
        with open(outpath, "wb") as file:
            torch.save(mlib.state, file)

        # 删除临时文件
        os.unlink(temp_file_path)


if __name__ == "__main__":
    # 使用typer库来运行主函数，这样可以方便地通过命令行参数调用
    typer.run(main)