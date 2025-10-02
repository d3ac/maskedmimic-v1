import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml


def main(
    main_motion_dir: Path,
    humanoid_type: str = "smpl",
    amass_fps_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    """
    递归扫描指定的运动数据目录，提取每个动作的帧率，并将结果保存为 YAML 文件。

    Args:
        main_motion_dir (Path): 包含运动数据 (.npz) 文件的根目录。
        humanoid_type (str, optional): 人体模型类型，'smpl' 或 'smplx'。默认为 'smpl'。
                                    这会影响帧率的提取方式。
        amass_fps_file (Optional[Path], optional): 包含 AMASS 数据集预计算帧率的 YAML 文件路径。
                                                   当 humanoid_type 为 'smplx' 时是必需的。默认为 None。
        output_path (Optional[Path], optional): 输出 YAML 文件的保存目录。默认为当前工作目录。
    """
    if humanoid_type == "smplx":
        assert (
            amass_fps_file is not None
        ), "Please provide the amass_fps_file since amass-x fps is wrong."
        amass_fps = yaml.load(open(amass_fps_file, "r"), Loader=yaml.SafeLoader)

    # 初始化一个字典来存储文件名和对应的帧率
    motion_fps_dict = {}
    # 递归遍历目录及所有子目录
    for root, dirs, files in os.walk(main_motion_dir):
        # 忽略包含 "-retarget"、"-smpl" 或 "-smplx" 的文件夹
        if "-retarget" in root or "-smpl" in root or "-smplx" in root:
            continue
        for file in files:
            # 只处理 .npz 文件，并排除特定的文件名（如 shape.npz）
            if (
                file.endswith(".npz")
                and file != "shape.npz"
                and "stagei.npz" not in file
            ):
                # --- 文件路径和名称的规范化处理 ---
                # 从根路径中移除主运动目录部分，得到相对路径
                save_root = root.replace(str(main_motion_dir), "")
                # 移除路径开头的任何斜杠
                save_root = save_root.lstrip("/")

                # 重命名文件：将 .npz 替换为 .npy，并将特殊字符替换为下划线
                # 这是为了创建一个统一的、将用作字典键的标识符
                file_rename = (
                    save_root
                    + "/"
                    + file.replace(".npz", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                # --- 根据人体模型类型提取帧率 ---
                if humanoid_type == "smplx":
                    # 对于 smplx 类型，需要特殊处理文件名以匹配 AMASS 数据集的帧率文件
                    amass_filename = file_rename.replace("_stageii", "_poses")
                    amass_filename = amass_filename.replace("SSM/", "SSM_synced/")
                    amass_filename = amass_filename.replace("HMD05/", "MPI_HDM05/")
                    amass_filename = amass_filename.replace("MoSh/", "MPI_mosh/")
                    amass_filename = amass_filename.replace("PosePrior/", "MPI_Limits/")
                    amass_filename = amass_filename.replace(
                        "TCDHands/", "TCD_handMocap/"
                    )
                    amass_filename = amass_filename.replace(
                        "Transitions/", "Transitions_mocap/"
                    )
                    amass_filename = amass_filename.replace("DFaust/", "DFaust_67/")
                    amass_filename = amass_filename.replace(
                        "BMLrub/", "BioMotionLab_NTroje/"
                    )

                    # 尝试从预加载的 amass_fps 文件中获取帧率
                    if amass_filename in amass_fps:
                        framerate = amass_fps[amass_filename]
                    else:
                        # 如果在 amass_fps 文件中找不到，则从 .npz 文件中加载或使用启发式规则
                        motion_data = dict(
                            np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                        )
                        # 根据文件名中的关键字或文件内的数据来确定帧率
                        if "TotalCapture" in file_rename or "SSM" in file_rename:
                            framerate = 60
                        elif "KIT" in file_rename:
                            framerate = 100
                        elif "mocap_frame_rate" in motion_data:
                            framerate = motion_data["mocap_frame_rate"]
                        else:
                            raise Exception(f"{file_rename} has no framerate")
                else:
                    # 对于 smpl 类型，直接从 .npz 文件中加载帧率
                    motion_data = dict(
                        np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                    )
                    if "mocap_framerate" in motion_data:
                        framerate = motion_data["mocap_framerate"]
                    else:
                        raise Exception(f"{file_rename} has no framerate")

                motion_fps_dict[file_rename] = int(framerate)

    # --- 保存结果 ---
    if output_path is None:
        output_path = Path.cwd()
    # 将最终的字典保存到 YAML 文件
    with open(output_path / f"motion_fps_{humanoid_type}.yaml", "w") as f:
        yaml.dump(motion_fps_dict, f)


if __name__ == "__main__":
    typer.run(main)
