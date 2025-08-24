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

import random
import yaml
import os
from easydict import EasyDict
import torch
import numpy as np
from isaac_utils import torch_utils
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Dict

if TYPE_CHECKING:
    from phys_anim.envs.env_utils.terrains import Terrain
else:
    Terrain = None


@dataclass
class ObjectState:
    """
    Represents the state of a scene object.

    Attributes:
        translations (torch.Tensor): The translation of the object.
        rotations (torch.Tensor): The rotation of the object.
        is_static (torch.Tensor): Boolean tensor indicating if the scene can be interacted by one or many robots.
    """

    translations: torch.Tensor
    rotations: torch.Tensor
    is_static: torch.Tensor


class SceneLib:
    """
    A class for managing scene libraries and object motions.

    This class handles loading scenes from YAML files, creating spawn lists,
    and managing object motions within scenes.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize the SceneLib.

        Args:
            scene_file (str): Path to the YAML file containing scene definitions.
            config (dict): Configuration dictionary.
            device (str, optional): The device to use for tensor operations. Defaults to "cpu".
        """
        super().__init__()
        self.config = config
        self._device = device
        self.scenes = []
        self.object_spawn_list = []
        self.object_id_to_path = []
        self.object_path_to_id = {}

        self.load_scenes_from_yaml(self.config.scene_yaml_path)

    def call_when_terrain_init_scene_spacing(self, terrain: Terrain):
        """
        在地形和场景间距初始化后调用此函数。

        该函数负责初始化与场景和对象相关的各种数据结构，为模拟环境的准备工作。
        它设置了场景到对象的映射，加载对象数据，并为后续的场景和对象生成做准备。

        Args:
            terrain (Terrain): 初始化完成的地形对象。
        """
        self.terrain = terrain
        self.create_scene_spawn_list() # 创建场景生成列表，放在了self.scenes里面

        # 确保配置的最大对象数不小于任何场景中的实际对象数
        assert self.config.max_objects_per_scene >= max(
            [len(scene["objects"]) for scene in self.scenes]
        )
        # 初始化场景到对象ID的映射张量，-1表示没有对象
        self.scene_to_object_ids = torch.full(
            (len(self.scenes), self.config.max_objects_per_scene),
            -1,
            dtype=torch.long,
            device=self._device,
        )
        # 初始化场景偏移量张量
        self.scene_offsets = torch.zeros(
            (len(self.scenes), 2), dtype=torch.float, device=self._device
        )

        # 遍历所有场景，填充场景到对象ID的映射，并建立对象ID和路径的双向映射
        for scene_idx, scene in enumerate(self.scenes):
            for obj_idx, obj in enumerate(scene["objects"]):
                self.scene_to_object_ids[scene_idx, obj_idx] = obj["id"]
                self.object_id_to_path.append(obj["path"])
                # 确认对象ID是连续的
                assert len(self.object_id_to_path) == obj["id"] + 1
                self.object_path_to_id[obj["path"]] = obj["id"]

        self.load_object_motions()         # 加载场景中物体的运动数据 （一帧的就是静态物体）
        self.create_object_spawn_list()

        # 创建张量以跟踪场景是否正在使用
        self.scene_in_use = torch.zeros(
            len(self.scenes), dtype=torch.bool, device=self._device
        )
        # 创建张量以标记哪些是单机器人场景
        self.single_robot_in_scene = torch.tensor(
            [scene["single_robot_in_scene"] for scene in self.scenes],
            dtype=torch.bool,
            device=self._device,
        )

    def call_at_terrain_done_init(self, y_offset):
        self.add_scene_y_offsets(y_offset)

    def add_scene_y_offsets(self, y_offset):
        for scene_idx, scene in enumerate(self.scenes):
            scene.y_offset += y_offset

            # Convert to terrain map coordinates
            scene_x = int(scene.x_offset / self.terrain.horizontal_scale)
            scene_y = int(scene.y_offset / self.terrain.horizontal_scale)

            locations = torch.tensor([[scene_x, scene_y]], device=self.terrain.device)

            # Check if the scene location is valid
            assert (
                self.terrain.is_valid_spawn_location(locations).cpu().item()
            ), f"Scene {scene_idx}: Scene overlaps with another scene."

            assert (
                self.terrain.tot_cols
                - self.terrain.border
                - self.terrain.object_playground_cols
                <= scene_y
                < self.terrain.tot_cols - self.terrain.border
            ), f"Scene {scene_idx}: Scene spawn location (y={scene_y}) is not in the designated spawn area. Should be between {self.terrain.tot_cols - self.terrain.border - self.terrain.object_playground_cols} and {self.terrain.tot_cols - self.terrain.border}"

            # Mark the scene location as occupied
            self.terrain.mark_scene_location(scene_x, scene_y)

        # Store scene offsets in a tensor
        self.scene_offsets = torch.zeros(
            (len(self.scenes), 2), dtype=torch.float, device=self._device
        )
        for scene_idx, scene in enumerate(self.scenes):
            self.scene_offsets[scene_idx, 0] = scene["x_offset"]
            self.scene_offsets[scene_idx, 1] = scene["y_offset"]

    def load_scenes_from_yaml(self, yaml_path: str):
        """
        Load scenes from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing scene definitions.
        """
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        self.raw_scenes = data["scenes"]

    def total_num_objects(self):
        total_objects = sum(
            len(scene.get("objects", []))
            * scene.get("replications", 1)
            * self.config.scene_replications
            for scene in self.raw_scenes
        )

        if self.config.max_num_objects is not None:
            total_objects = min(total_objects, self.config.max_num_objects)

        return total_objects

    @property
    def total_spawned_scenes(self):
        return len(self.scenes)

    def create_scene_spawn_list(self):
        """
        为所有场景创建一个对象生成位置列表。

        此方法处理从YAML文件加载的原始场景，根据它们的复制权重随机选择场景，
        并填充场景列表。如果设置了 max_num_objects，则会遵守该配置。

        该方法还会使用选定的场景初始化 self.scenes 列表，并跟踪每个场景被复制的次数。

        Side effects:
            - 使用选定的场景配置填充 self.scenes。
            - 更新 self.raw_scenes 中每个场景的 'replications' 计数。

        打印:
            - 将要生成的对象总数。
            - 将要生成的场景数量。

        +---------------------------------------------------------------------------+
        |                            BORDER                                         |
        +-------------------------------+-------------------------------------------+
        |                               |                                           |
        |           BORDER              |           Object Playground               |
        |                               |       (where your scenes placed)          |
        |          40 len               |                  200 len                  |
        +-------------------------------+-------------------------------------------+
        |                            BORDER                                         |
        +---------------------------------------------------------------------------+


        """
        total_objects = self.total_num_objects()

        print(f"will spawn {total_objects} objects.")
        total_spawned_objects = 0

        weighted_scenes = [
            [scene, scene.get("replications", 1) * self.config.scene_replications]
            for scene in self.raw_scenes
        ]
        for scene in self.raw_scenes:
            scene["replications"] = 0

        object_id = 0
        scene_count = 0
        while total_spawned_objects < total_objects and weighted_scenes:
            scene_index, (scene, _) = random.choices(
                list(enumerate(weighted_scenes)),
                weights=[w for _, w in weighted_scenes],
                k=1,
            )[0]
            weighted_scenes[scene_index][1] -= 1 # 在这里标记了后面又弹出去了

            # Calculate scene position in terrain coordinates
            x_offset = (                                                     # 计算放在哪一排
                (scene_count % self.terrain.num_scenes_per_column + 1)
                * self.terrain.spacing_between_scenes
                + self.terrain.border * self.terrain.horizontal_scale
            )
            y_offset = (                                                     # 计算放在哪一列
                scene_count // self.terrain.num_scenes_per_column + 1
            ) * self.terrain.spacing_between_scenes

            scene_objects = scene["objects"]

            scene_copy = scene.copy()
            scene_copy["x_offset"] = x_offset
            scene_copy["y_offset"] = y_offset
            scene_copy["objects"] = []
            # 如果 force_single_robot_per_scene 为 True，则每个场景只允许一个机器人（强制所有场景为“动态”）
            scene_copy["single_robot_in_scene"] = (
                self.config.force_single_robot_per_scene
            )

            for obj in scene_objects:
                obj_copy = obj.copy()
                obj_copy["id"] = object_id
                scene_copy["objects"].append(obj_copy)
                if not obj_copy["is_static"]:
                    scene_copy["single_robot_in_scene"] = (
                        True  # 如果任何对象是动态的，则该场景不能被多个机器人交互
                    )
                object_id += 1
                total_spawned_objects += 1
                if total_spawned_objects >= total_objects:
                    break

            self.scenes.append(EasyDict(scene_copy))

            if total_spawned_objects >= total_objects:
                break

            if weighted_scenes[scene_index][1] <= 0:
                weighted_scenes.pop(scene_index)

            scene_count += 1

        print(f"Will spawn {len(self.scenes)} scenes.")

    def get_scene_ids(self):
        """
        Retrieve the IDs of all scenes in the scene library.

        Returns:
            list: A list containing the ID of each scene in self.scenes.
        """
        return [scene.id for scene in self.scenes]

    def load_object_motions(self):
        """
        为场景中的所有对象加载运动数据。

        该方法处理所有场景和对象，以实现以下功能：
        1. 为每个具有指定运动路径的对象加载运动数据。
        2. 为没有运动路径的对象创建默认的静态运动数据。
        3. 使用加载的运动数据填充 object_translations 和 object_rotations 张量。
        4. 为每个对象设置 motion_lengths、motion_starts 和 motion_dts, 以跟踪其运动序列。

        执行后，将填充以下属性：
        - object_translations: 包含所有对象平移的张量。
        - object_rotations: 包含所有对象旋转的张量。
        - motion_lengths: 包含每个对象运动长度（以秒为单位）的张量。
        - motion_starts: 包含每个对象运动在平移和旋转张量中起始索引的张量。
        - motion_dts: 包含每个对象运动的时间步长 (1/fps) 的张量。
        - motion_num_frames: 包含每个对象运动的帧数的张量。
        """
        total_motion_length = 0
        motion_data = []
        motion_lengths_list = []
        motion_dts_list = []
        motion_num_frames_list = []

        # Process scenes and objects
        for scene in self.scenes:
            for obj in scene["objects"]:
                object_id = obj["id"]
                motion_path = obj.get("motion")

                if motion_path: # 对于每一个物体来说，他的运动文件里面都只包含了一帧的数据
                    object_motion_data = self._load_motion(motion_path)
                    motion_length = object_motion_data["translation"].shape[0]
                    total_motion_length += motion_length
                    motion_data.append((object_id, object_motion_data))
                    fps = object_motion_data.get("fps", 30.0)
                    dt = 1.0 / fps
                    motion_lengths_list.append(motion_length * dt)
                    motion_dts_list.append(dt)
                    motion_num_frames_list.append(motion_length)
                else:                                              # 没进来过
                    motion_data.append(
                        (
                            object_id,
                            {
                                "translation": torch.zeros(1, 3),
                                "rotation": torch.tensor([[0, 0, 0, 1]]),
                                "translation_offset": torch.zeros(3),
                                "rotation_offset": torch.tensor([0, 0, 0, 1]),
                            },
                        )
                    )
                    total_motion_length += 1
                    motion_lengths_list.append(1.0 / 30)  # -1 表示静态物体
                    motion_dts_list.append(1.0 / 30)
                    motion_num_frames_list.append(1)

        total_objects = len(motion_lengths_list)

        # 物体的平移和旋转是相对于场景坐标系的局部坐标。
        self.object_translations = torch.zeros(
            (total_motion_length, 3), device=self._device
        )
        self.object_rotations = torch.zeros(
            (total_motion_length, 4), device=self._device
        )

        # 每个物体运动的时长（单位：秒）。
        self.motion_lengths = torch.tensor(
            motion_lengths_list, dtype=torch.float, device=self._device
        )

        # 类似于 MotionLib，平移和旋转张量是一个长张量。
        # 我们使用 motion_starts 将运动标识符映射到统一张量中运动序列的起始位置。
        self.motion_starts = torch.zeros(
            total_objects, dtype=torch.long, device=self._device
        )
        self.motion_dts = torch.tensor(
            motion_dts_list, dtype=torch.float, device=self._device
        )
        self.motion_num_frames = torch.tensor(                      # 每个物体运动的帧数 (都是1)
            motion_num_frames_list, dtype=torch.long, device=self._device
        )

        # CT hack：平移和旋转的 offset 是为了解决 obj 文件中的问题。
        # 例如，在 SAMP 数据集中，物体的“前进”方向实际上指向侧面。
        # 控制器观测到的是物体的朝向。
        # 我们使用旋转 offset 来修正控制器感知到的朝向。
        self.object_translation_offsets = torch.zeros(
            (total_objects, 3), device=self._device
        )
        self.object_rotation_offsets = torch.zeros(
            (total_objects, 4), device=self._device
        )

        current_start = 0
        for object_id, object_motion_data in motion_data:
            motion_length = object_motion_data["translation"].shape[0]
            self.motion_starts[object_id] = current_start

            self.object_translations[current_start : current_start + motion_length] = (
                object_motion_data["translation"]
            )
            self.object_rotations[current_start : current_start + motion_length] = (
                object_motion_data["rotation"]
            )
            self.object_translation_offsets[object_id] = object_motion_data[
                "translation_offset"
            ]
            self.object_rotation_offsets[object_id] = object_motion_data[
                "rotation_offset"
            ]

            current_start += motion_length

    def _load_motion(self, motion_path: str) -> Dict[str, torch.Tensor]:
        """
        Load motion data from a file.

        Args:
            motion_path (str): Path to the motion file.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'translation', 'rotation', 'translation_offset', 'rotation_offset', and 'fps' tensors.

        Raises:
            FileNotFoundError: If the motion file is not found.
        """
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")

        motion_data = np.load(motion_path, allow_pickle=True).item()

        # Ensure the required keys are present in the loaded data
        required_keys = [
            "translation",
            "rotation",
            "translation_offset",
            "rotation_offset",
        ]
        for key in required_keys:
            if key not in motion_data:
                raise KeyError(f"Required key '{key}' not found in motion data")

        for key in motion_data:
            motion_data[key] = torch.tensor(motion_data[key], device=self._device)

        # Add fps to motion_data, default to 30 if not provided
        motion_data["fps"] = motion_data.get("fps", 30.0)

        return motion_data

    def get_object_pose(
        self, object_id: torch.Tensor, time: torch.Tensor
    ) -> ObjectState:
        """
        Get the pose of an object at a specific time.

        Args:
            object_id (torch.Tensor): The ID of the object. Shape: [B]
            time (torch.Tensor): The time at which to get the object's pose. Shape: [B]

        Returns:
            ObjectState: A ObjectState object containing the object's translation, rotation, and whether it's static.

        Note:
            If the object is static (motion_length == -1), it will return the initial pose.
            For dynamic objects, it interpolates between two frames based on the given time.
        """
        motion_length = self.motion_lengths[object_id]
        motion_start = self.motion_starts[object_id]
        motion_dt = self.motion_dts[object_id]
        motion_num_frames = self.motion_num_frames[object_id]

        is_static = motion_num_frames == 1

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            time, motion_length, motion_num_frames, motion_dt
        )

        translation0 = self.object_translations[motion_start + frame_idx0]
        rotation0 = self.object_rotations[motion_start + frame_idx0]

        translation1 = self.object_translations[motion_start + frame_idx1]
        rotation1 = self.object_rotations[motion_start + frame_idx1]

        translation = (1 - blend).unsqueeze(-1) * translation0 + blend.unsqueeze(
            -1
        ) * translation1
        rotation = torch_utils.slerp(rotation0, rotation1, blend.unsqueeze(-1))

        scene_state = ObjectState(
            translations=translation,
            rotations=rotation,
            is_static=is_static,
        )

        return scene_state

    def _calc_frame_blend(
        self,
        time: torch.Tensor,
        length: torch.Tensor,
        num_frames: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate frame indices and blend factor for interpolation.

        Args:
            time (torch.Tensor): Current time.
            length (torch.Tensor): Length of the motion sequence in seconds.
            num_frames (torch.Tensor): Number of frames in the motion sequence.
            dt (torch.Tensor): Time step between frames.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Frame index 0, frame index 1, and blend factor.
        """
        phase = time / length
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def create_object_spawn_list(self):
        """
        为所有场景创建物体生成位置列表，不包含具体的位姿信息。

        此方法会填充 self.object_spawn_list 类变量。
        """
        for scene_idx, scene in enumerate(self.scenes):
            for obj_idx, obj in enumerate(scene["objects"]):
                object_id = obj["id"]

                self.object_spawn_list.append(
                    EasyDict(
                        {
                            "object_path": obj["path"],
                            "scene_idx": scene_idx,
                            "object_id": object_id,
                            "is_static": obj["is_static"],
                            "object_options": obj.get("object_options", {}), # 一些物体的物理参数，不太用管
                        }
                    )
                )

    def mark_scene_in_use(self, scene_idx):
        """
        Mark one or more scenes as in use, while respecting static scene constraints.

        This method updates the `scene_in_use` attribute to indicate which scenes are currently being used.
        It ensures that static scenes are not marked as in use, preventing potential conflicts.

        Args:
            scene_idx (int or torch.Tensor): Index or indices of the scene(s) to mark as in use.
                Can be a single integer or a tensor of indices.

        Note:
            - Single-robot scenes (as indicated by `not self.single_robot_in_scene`) will never be marked as in use.
            - This method is thread-safe for concurrent calls with different scene_idx values.
        """
        valid_scenes = scene_idx[scene_idx >= 0]
        single_robot_scenes = self.single_robot_in_scene[valid_scenes]
        self.scene_in_use[valid_scenes[single_robot_scenes]] = True

    def mark_scene_not_in_use(self, scene_idx):
        """
        Mark a scene as not in use.

        Args:
            scene_idx (int or torch.Tensor): Index or indices of the scene(s) to mark as not in use.
        """
        valid_scenes = scene_idx[scene_idx >= 0]
        single_robot_scenes = self.single_robot_in_scene[valid_scenes]
        self.scene_in_use[valid_scenes[single_robot_scenes]] = False

    def get_available_scenes(self):
        """
        Get a list of indices of available scenes (not in use).

        Returns:
            torch.Tensor: Indices of available scenes.
        """
        return torch.where(~self.scene_in_use)[0]

    def get_available_scenes_mask(self):
        """
        Get a boolean mask of available scenes (not in use).

        Returns:
            torch.Tensor: Boolean mask of available scenes, where True indicates
                          an available scene and False indicates a scene in use.
                          The size of the mask is equal to the total number of scenes.
        """
        return ~self.scene_in_use

    def sample_scenes(self, valid_scenes, valid_count, get_first_matching_scene=False):
        """
        Sample scenes from a matrix of valid scenes, maximizing selection and respecting static/dynamic constraints.

        Args:
            valid_scenes (torch.Tensor): Matrix of valid scene indices [batch, N].
            valid_count (torch.Tensor): Number of valid entries in each row of valid_scenes [batch].
            get_first_matching_scene (bool): Whether to get the first matching scene or a random one.

        Returns:
            torch.Tensor: Sampled scene indices [batch].
            torch.Tensor: Mask indicating environments without a valid scene [batch].
        """
        batch_size, max_scenes = valid_scenes.shape
        device = valid_scenes.device

        sampled_scenes = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Track which non-static scenes have been sampled
        available_scenes = ~self.scene_in_use.clone()

        for i in range(batch_size):
            # Handle special cases. -1 means no scene needed, 0 means no scene spawned.
            if valid_count[i] == -1:
                continue
            elif valid_count[i] == 0:
                valid_mask[i] = False
                continue

            valid_indices = valid_scenes[i, : valid_count[i]]

            # Filter out unavailable scenes and already sampled non-static scenes
            available_mask = available_scenes[valid_indices]
            available_indices = valid_indices[available_mask]

            if len(available_indices) > 0:
                if get_first_matching_scene:
                    # Get the first matching scene
                    sampled_idx = available_indices[0]
                else:
                    # Randomly sample from available scenes
                    sampled_idx = available_indices[
                        torch.randint(len(available_indices), (1,), device=device)
                    ]

                sampled_scenes[i] = sampled_idx

                # Mark single-robot scenes as sampled
                if self.single_robot_in_scene[sampled_idx]:
                    available_scenes[sampled_idx] = False
            else:
                valid_mask[i] = False

        return sampled_scenes, valid_mask


if __name__ == "__main__":
    import tempfile
    from phys_anim.envs.env_utils.terrains import FlatTerrain

    # Flag to choose between dummy scene and SAMP scene
    use_samp_scene = True  # Set to False for dummy scene

    # Load terrain configuration
    config = EasyDict(
        {
            "max_num_objects": 1024,
            "spacing_between_scenes": 5,
            "horizontal_scale": 0.1,
            "vertical_scale": 0.005,
            "border_size": 20.0,
            "map_length": 20.0,
            "map_width": 20.0,
            "num_levels": 10,
            "num_terrains": 10,
            "terrain_proportions": [0.2, 0.1, 0.15, 0.15, 0.05, 0, 0, 0.35],
            "slope_threshold": 0.9,
            "load_terrain": False,
            "save_terrain": False,
        }
    )

    # Create FlatTerrain instance
    terrain = FlatTerrain(config, device="cpu")

    if use_samp_scene:
        # Load SAMP scenes from the specified YAML file
        with open("data/yaml_files/samp_scenes_train.yaml", "r") as file:
            samp_scenes = yaml.safe_load(file)
        dummy_scenes = samp_scenes
    else:
        # Create dummy scenes
        dummy_scenes = {
            "scenes": [
                {
                    "replications": 2,
                    "objects": [
                        {
                            "path": "object1.urdf",
                            "motion": "motion1.pkl",
                            "is_static": False,
                        },
                        {
                            "path": "object2.urdf",
                            "motion": "motion2.pkl",
                            "is_static": True,
                        },
                    ],
                },
                {
                    "replications": 1,
                    "objects": [
                        {
                            "path": "object3.urdf",
                            "motion": "motion3.pkl",
                            "is_static": False,
                        },
                        {
                            "path": "object4.urdf",
                            "motion": "motion4.pkl",
                            "is_static": True,
                        },
                        {
                            "path": "object5.urdf",
                            "motion": "motion5.pkl",
                            "is_static": False,
                        },
                    ],
                },
            ]
        }

    if not use_samp_scene:
        # Create dummy motion files (only for dummy scenes)
        motion_files = [
            "motion1.pkl",
            "motion2.pkl",
            "motion3.pkl",
            "motion4.pkl",
            "motion5.pkl",
        ]
        for motion_file in motion_files:
            motion_data = {
                "translation": np.random.rand(10, 3).astype(np.float32),
                "rotation": np.random.rand(10, 4).astype(np.float32),
                "translation_offset": np.random.rand(3).astype(np.float32),
                "rotation_offset": np.random.rand(4).astype(np.float32),
            }
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pkl", delete=False
            ) as temp_motion_file:
                np.save(temp_motion_file, motion_data)
                if motion_file == "motion1.pkl":
                    dummy_scenes["scenes"][0]["objects"][0][
                        "motion"
                    ] = temp_motion_file.name
                elif motion_file == "motion2.pkl":
                    dummy_scenes["scenes"][0]["objects"][1][
                        "motion"
                    ] = temp_motion_file.name
                elif motion_file == "motion3.pkl":
                    dummy_scenes["scenes"][1]["objects"][0][
                        "motion"
                    ] = temp_motion_file.name
                elif motion_file == "motion4.pkl":
                    dummy_scenes["scenes"][1]["objects"][1][
                        "motion"
                    ] = temp_motion_file.name
                elif motion_file == "motion5.pkl":
                    dummy_scenes["scenes"][1]["objects"][2][
                        "motion"
                    ] = temp_motion_file.name

    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(dummy_scenes, temp_file)
        temp_file_path = temp_file.name

    _config = EasyDict(
        {
            "max_num_objects": config.max_num_objects,
            "scene_yaml_path": temp_file_path,
            "num_object_types": 7,
            "force_single_robot_per_scene": True,
            "scene_replications": 1024,
        }
    )

    scene_lib = SceneLib(config=_config, terrain=terrain, device="cpu")

    print("Object ID to Path:", scene_lib.object_id_to_path)
    print("Object Path to ID:", scene_lib.object_path_to_id)
    print("Spawn list:", scene_lib.object_spawn_list)
    print("Total number of objects:", scene_lib.total_num_objects)
    print("Number of scenes:", len(scene_lib.scenes))
