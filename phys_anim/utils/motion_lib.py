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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Tuple
from easydict import EasyDict

import numpy as np
import torch
import yaml
from lightning_fabric.utilities.rank_zero import _get_rank
from torch import Tensor, nn

from isaac_utils import rotations, torch_utils
from phys_anim.utils.device_dtype_mixin import DeviceDtypeModuleMixin
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


@dataclass
class MotionState:
    root_pos: Tensor
    root_rot: Tensor
    dof_pos: Tensor
    root_vel: Tensor
    root_ang_vel: Tensor
    dof_vel: Tensor
    key_body_pos: Tensor
    rb_pos: Tensor
    rb_rot: Tensor
    local_rot: Tensor
    rb_vel: Tensor
    rb_ang_vel: Tensor


class LoadedMotions(nn.Module):
    def __init__(
        self,
        motions: Tuple[SkeletonMotion],
        motion_lengths: Tensor,
        motion_weights: Tensor,
        motion_timings: Tensor,
        motion_fps: Tensor,
        motion_dt: Tensor,
        motion_num_frames: Tensor,
        motion_files: Tuple[str],
        sub_motion_to_motion: Tensor,
        ref_respawn_offsets: Tensor,
        text_embeddings: Tensor = None,
        has_text_embeddings: Tensor = None,
        supported_scene_ids: List[List[str]] = None,
        motion_labels = None,
        motion_labels_raw = None,
        **kwargs,  # Catch some nn.Module arguments that aren't needed
    ):
        super().__init__()
        self.motions = motions
        self.motion_files = motion_files
        self.register_buffer("motion_lengths", motion_lengths, persistent=False)
        self.register_buffer("motion_weights", motion_weights, persistent=False)
        self.register_buffer("motion_timings", motion_timings, persistent=False) # 每一个动作的有效开始和结束的时间
        self.register_buffer("motion_fps", motion_fps, persistent=False)
        self.register_buffer("motion_dt", motion_dt, persistent=False)
        self.register_buffer("motion_num_frames", motion_num_frames, persistent=False)
        self.register_buffer(
            "sub_motion_to_motion", sub_motion_to_motion, persistent=False
        )
        self.register_buffer(
            "ref_respawn_offsets", ref_respawn_offsets, persistent=False
        )
        if text_embeddings is None:
            text_embeddings = torch.zeros(len(motions), 3, 512, dtype=torch.float32)
            has_text_embeddings = torch.zeros(len(motions), dtype=torch.bool)
        self.register_buffer("text_embeddings", text_embeddings, persistent=False)
        self.register_buffer(
            "has_text_embeddings", has_text_embeddings, persistent=False
        )
        # motion_labels 是字符串列表，不能用 register_buffer，直接作为属性存储
        self.motion_labels = motion_labels
        self.motion_labels_raw = motion_labels_raw

        if supported_scene_ids is None:
            supported_scene_ids = [None for _ in range(len(motions))]
        self.supported_scene_ids = supported_scene_ids


class MotionLib(DeviceDtypeModuleMixin):
    gts: Tensor
    grs: Tensor
    lrs: Tensor
    gvs: Tensor
    gavs: Tensor
    grvs: Tensor
    gravs: Tensor
    dvs: Tensor
    length_starts: Tensor
    motion_ids: Tensor
    key_body_ids: Tensor

    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        device="cpu",
        ref_height_adjust: float = 0,
        target_frame_rate: int = 30,
        create_text_embeddings: bool = False,
        spawned_scene_ids: List[str] = None,
        fix_motion_heights: bool = True,
        skeleton_tree: Any = None,
        rb_conversion: Tensor = None,
        dof_conversion: Tensor = None,
        local_rot_conversion: Tensor = None,
        w_last: bool = True,
    ):
        super().__init__()
        self.w_last = w_last # 是否四元数最后一位为w分量 （实部）
        self.fix_heights = fix_motion_heights # 是否修正运动高度
        self.skeleton_tree = skeleton_tree # 骨骼树结构
        self.create_text_embeddings = create_text_embeddings # 是否创建文本嵌入
        self.dof_body_ids = dof_body_ids # 每个自由度对应的刚体id
        self.dof_offsets = dof_offsets # 每个关节在自由度列表中的起始偏移
        self.num_dof = dof_offsets[-1] # 总自由度数量
        self.ref_height_adjust = ref_height_adjust # 参考高度修正值
        self.rb_conversion = rb_conversion # 刚体转换矩阵
        self.dof_conversion = dof_conversion # 自由度转换矩阵
        self.local_rot_conversion = local_rot_conversion # 局部旋转转换矩阵

        self.register_buffer(
            "key_body_ids",
            torch.tensor(key_body_ids, dtype=torch.long, device=device),
            persistent=False,
        )

        if str(motion_file).split(".")[-1] in ["yaml", "npy", "npz", "np"]:
            print("Loading motions from yaml/npy file")
            self._load_motions(motion_file, target_frame_rate)
        else:
            rank = _get_rank()
            if rank is None:
                rank = 0
            # This is used for large motion files that are split across multiple GPUs
            motion_file = motion_file.replace("_slurmrank", f"_{rank}")
            print(f"Loading motions from state file: {motion_file}")

            with open(motion_file, "rb") as file:
                state: LoadedMotions = torch.load(file, map_location="cpu")

            # Create LoadedMotions instance with loaded state dict
            # We re-create to enable backwards compatibility. This allows LoadedMotions class to accept "None" values and set defaults if needed.
            state_dict = {
                **vars(state),
                **{k: v for k, v in state._buffers.items() if v is not None},
            }
            self.state = LoadedMotions(**state_dict)

        motions = self.state.motions
        self.register_buffer(
            "gts",
            torch.cat([m.global_translation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grs",
            torch.cat([m.global_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "lrs",
            torch.cat([m.local_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grvs",
            torch.cat([m.global_root_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gravs",
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gavs",
            torch.cat([m.global_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gvs",
            torch.cat([m.global_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "dvs",
            torch.cat([m.dof_vels for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

        lengths = self.state.motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0                # 这个变量意思就是说，每个动作是从哪里开始的，所以全部右移一个然后第一个清零
        self.register_buffer(
            "length_starts", lengths_shifted.cumsum(0), persistent=False
        )

        self.register_buffer(
            "motion_ids",
            torch.arange(
                len(self.state.motions), dtype=torch.long, device=self._device
            ),
            persistent=False,
        )

        scenes_per_motion, motion_to_scene_ids = self.parse_scenes(spawned_scene_ids)

        self.register_buffer(
            "scenes_per_motion",
            torch.tensor(scenes_per_motion, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "motion_to_scene_ids",
            torch.tensor(motion_to_scene_ids, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.to(device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.state.motions)

    def num_sub_motions(self):
        """Returns the number of sub-motions in the state.

        A sub-motion is a segment or a part of a larger motion sequence.
        In the context of this code, a motion can be divided into multiple sub-motions,
        each representing a smaller portion of the overall motion.
        These sub-motions are used to manage and manipulate parts of the motion sequence
        independently, allowing for more granular control and analysis of the motion data.

        Returns:
            int: The number of sub-motions.
        """
        return self.state.motion_weights.shape[0]

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.state.motion_lengths)

    def get_total_trainable_length(self):
        """Returns the total trainable length of all motions.

        The total trainable length is calculated by summing the differences
        between the end and start times of each motion timing.

        Returns:
            int: The total trainable length of all motions.
        """
        return sum(self.state.motion_timings[:, 1] - self.state.motion_timings[:, 0])

    def get_motion(self, motion_id):
        return self.state.motions[motion_id]

    def sample_motions(self, n, valid_mask=None):
        if valid_mask is not None:
            weights = self.state.motion_weights.clone()
            weights[~valid_mask] = 0
        else:
            weights = self.state.motion_weights

        sub_motion_ids = torch.multinomial(weights, num_samples=n, replacement=True)

        return sub_motion_ids

    def sample_other_motions(self, already_chosen_ids: Tensor) -> Tensor:
        """Samples other motions that are not in the already chosen IDs.

        Args:
            already_chosen_ids (Tensor): A tensor containing the IDs of motions that have already been chosen.

        Returns:
            Tensor: A tensor containing the IDs of the sampled motions that are not in the already chosen IDs.
        """
        n = already_chosen_ids.shape[0]
        motion_weights = self.state.motion_weights.unsqueeze(0).tile([n, 1])
        motion_weights = motion_weights.scatter(
            1, already_chosen_ids.unsqueeze(-1), torch.zeros_like(motion_weights)
        )
        sub_motion_ids = torch.multinomial(motion_weights, num_samples=1).squeeze(-1)
        return sub_motion_ids

    def sample_text_embeddings(self, sub_motion_ids: Tensor) -> Tensor:
        """Samples text embeddings for the given sub-motion IDs.

        Args:
            sub_motion_ids (Tensor): A tensor containing the IDs of the sub-motions.

        Returns:
            Tensor: A tensor containing the sampled text embeddings for the given sub-motion IDs.
        """
        if hasattr(self.state, "text_embeddings"):
            indices = torch.randint(
                0, 3, (sub_motion_ids.shape[0],), device=self.device
            )
            return self.state.text_embeddings[sub_motion_ids, indices]
        return 0

    def sample_time(self, sub_motion_ids, max_time=None, truncate_time=None):
        phase = torch.rand(sub_motion_ids.shape, device=self.device)

        motion_len = (
            self.state.motion_timings[sub_motion_ids, 1]
            - self.state.motion_timings[sub_motion_ids, 0]
        )
        if max_time is not None:
            motion_len = torch.clamp(
                motion_len,
                max=max_time,
            )

        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
            assert torch.all(motion_len >= 0)

        motion_time = phase * motion_len
        return motion_time + self.state.motion_timings[sub_motion_ids, 0]

    def get_sub_motion_length(self, sub_motion_ids):
        return (
            self.state.motion_timings[sub_motion_ids, 1]
            - self.state.motion_timings[sub_motion_ids, 0]
        )

    def get_motion_length(self, sub_motion_ids):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        return self.state.motion_lengths[motion_ids]

    def get_mimic_motion_state(
        self, sub_motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> MotionState:
        """
        获取模仿学习所需的参考动作状态。
        
        该函数是训练中最核心的函数之一，它根据给定的动作ID和时间，通过帧间插值
        从预加载的动作库中提取参考状态，用于与仿真环境中的实际状态进行对比计算奖励。
        
        Args:
            sub_motion_ids: 子动作ID，shape (num_envs,)
            motion_times: 当前动作进行到的时间点，shape (num_envs,)
            joint_3d_format: 关节3D表示格式，默认为 "exp_map"
            
        Returns:
            MotionState: 包含参考动作状态的对象，包括位置、旋转、速度等信息
        """
        # 1. 将子动作ID映射到主动作ID
        # 因为一个主动作可能包含多个子动作片段
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        # 2. 获取动作长度并确保时间在有效范围内
        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # 将时间裁剪到 [0, motion_len] 范围内

        # 3. 获取动作的帧数和帧间时间间隔
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]  # 每帧之间的时间间隔

        # 4. 计算帧混合参数
        # 返回：frame_idx0 (前一帧索引), frame_idx1 (后一帧索引), blend (混合权重)
        # blend=0 表示完全使用 frame_idx0，blend=1 表示完全使用 frame_idx1
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        # 5. 计算在全局张量中的实际索引位置
        # length_starts 记录了每个动作在连续张量中的起始位置
        f0l = frame_idx0 + self.length_starts[motion_ids]  # 前一帧的全局索引
        f1l = frame_idx1 + self.length_starts[motion_ids]  # 后一帧的全局索引

        # 6. 从预加载的张量中提取前后两帧的数据
        # gts: global translations (全局位置)
        global_translation0 = self.gts[f0l]
        global_translation1 = self.gts[f1l]

        # grs: global rotations (全局旋转，四元数)
        global_rotation0 = self.grs[f0l]
        global_rotation1 = self.grs[f1l]

        # lrs: local rotations (局部旋转，相对于父关节)
        local_rotation0 = self.lrs[f0l]
        local_rotation1 = self.lrs[f1l]

        # gvs: global velocities (全局线速度)
        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        # gavs: global angular velocities (全局角速度)
        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        # dvs: dof velocities (关节速度)
        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        # 7. 扩展 blend 的维度以便进行广播操作
        blend = blend.unsqueeze(-1)          # shape: (num_envs, 1)
        blend_exp = blend.unsqueeze(-1)      # shape: (num_envs, 1, 1)

        # 8. 对前后两帧进行插值，得到指定时间点的平滑状态
        
        # 位置插值：线性插值
        global_translation: Tensor = (
            1.0 - blend_exp
        ) * global_translation0 + blend_exp * global_translation1
        
        # 旋转插值：球面线性插值 (SLERP)，确保旋转的平滑性
        global_rotation: Tensor = torch_utils.slerp(
            global_rotation0, global_rotation1, blend_exp
        )

        local_rotation: Tensor = torch_utils.slerp(
            local_rotation0, local_rotation1, blend_exp
        )

        # 9. 计算或插值关节位置 (dof_pos)
        if hasattr(self, "dof_pos"):  # 针对H1等特殊机器人，直接存储了dof_pos
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:  # 对于SMPL等模型，从局部旋转转换为关节位置
            dof_pos: Tensor = self._local_rotation_to_dof(
                local_rotation, joint_3d_format
            )

        # 速度插值：线性插值
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        # 10. 应用高度调整（如果需要）
        # 对z轴（高度）进行修正，用于适配不同的地形或机器人高度
        global_translation[:, :, 2] += self.ref_height_adjust

        # 11. 四元数格式转换（如果需要）
        # 根据配置决定四元数的w分量是在最后还是最前
        if not self.w_last:
            global_rotation = rotations.xyzw_to_wxyz(global_rotation)
            local_rotation = rotations.xyzw_to_wxyz(local_rotation)

        # 12. 应用刚体和关节的索引转换（如果需要）
        # 这些转换用于适配不同机器人模型的刚体和关节顺序
        if self.rb_conversion is not None:
            global_translation = global_translation[:, self.rb_conversion]
            global_rotation = global_rotation[:, self.rb_conversion]
            global_vel = global_vel[:, self.rb_conversion]
            global_ang_vel = global_ang_vel[:, self.rb_conversion]
        if self.dof_conversion is not None:
            dof_pos = dof_pos[:, self.dof_conversion]
            dof_vel = dof_vel[:, self.dof_conversion]
        if self.local_rot_conversion is not None:
            local_rotation = local_rotation[:, self.local_rot_conversion]

        # 13. 构建并返回 MotionState 对象
        # 这个对象包含了训练时用于对比的所有参考状态信息
        motion_state = MotionState(
            root_pos=None,              # 根部位置（这里未使用，会从rb_pos中提取）
            root_rot=None,              # 根部旋转（这里未使用，会从rb_rot中提取）
            root_vel=None,              # 根部速度（这里未使用，会从rb_vel中提取）
            root_ang_vel=None,          # 根部角速度（这里未使用，会从rb_ang_vel中提取）
            key_body_pos=None,          # 关键身体部位（会在后续处理中提取）
            dof_pos=dof_pos,            # 关节位置（局部坐标系，相对于父关节的角度/姿态）
            dof_vel=dof_vel,            # 关节速度（局部坐标系，关节角速度）
            rb_pos=global_translation,  # 所有刚体的全局位置
            rb_rot=global_rotation,     # 所有刚体的全局旋转
            local_rot=local_rotation,   # 局部旋转
            rb_vel=global_vel,          # 所有刚体的线速度
            rb_ang_vel=global_ang_vel,  # 所有刚体的角速度
        )

        return motion_state

    def get_motion_state(
        self, sub_motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> MotionState:
        # ------------------------------------ 1. 输入处理与帧计算 ---------------------------------
        # 将子动作ID映射到全局动作ID
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        # 获取动作长度，并将输入时间裁剪到有效范围内 [0, motion_len]
        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        # 获取动作的总帧数和每帧的时间间隔 (dt)
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        # 根据时间点计算用于插值的前后两个关键帧索引(frame_idx0, frame_idx1)和混合权重(blend)
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        # ------------------------------------ 2. 从关键帧提取数据 ---------------------------------
        # 计算在全局数据张量中的绝对帧索引
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        # 从两个关键帧(f0l, f1l)中提取所有相关的运动数据
        # 包括根关节、局部关节、速度、角速度等信息
        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel0 = self.grvs[f0l]
        root_vel1 = self.grvs[f1l]

        root_ang_vel0 = self.gravs[f0l]
        root_ang_vel1 = self.gravs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        key_body_pos0 = self.gts[f0l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]
        key_body_pos1 = self.gts[f1l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rb_pos0 = self.gts[f0l]
        rb_pos1 = self.gts[f1l]

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]

        # (调试用) 检查所有提取出的张量数据类型是否正确
        vals = [
            root_pos0,
            root_pos1,
            local_rot0,
            local_rot1,
            root_vel0,
            root_vel1,
            root_ang_vel0,
            root_ang_vel1,
            global_vel0,
            global_vel1,
            global_ang_vel0,
            global_ang_vel1,
            dof_vel0,
            dof_vel1,
            key_body_pos0,
            key_body_pos1,
            rb_pos0,
            rb_pos1,
            rb_rot0,
            rb_rot1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        # ------------------------------------ 3. 插值计算目标状态 ---------------------------------
        # 准备混合权重张量(blend)以便进行广播运算
        blend = blend.unsqueeze(-1)

        # 对位置和速度等数据进行线性插值
        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        # 应用参考高度调整
        root_pos[:, 2] += self.ref_height_adjust

        # 对旋转数据(四元数)进行球面线性插值 (slerp)
        root_rot: Tensor = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_body_pos = (1.0 - blend_exp) * key_body_pos0 + blend_exp * key_body_pos1
        key_body_pos[:, :, 2] += self.ref_height_adjust

        local_rot = torch_utils.slerp(
            local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
        )

        # 计算关节自由度(DOF)的位置
        if hasattr(self, "dof_pos"):  # H1 joints
            # 针对H1模型，直接插值
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            # 其他模型，通过局部旋转计算得到
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        # 对其他速度和位置数据进行插值
        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rb_pos = (1.0 - blend_exp) * rb_pos0 + blend_exp * rb_pos1
        rb_pos[:, :, 2] += self.ref_height_adjust
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1
        
        # ------------------------------------ 4. 数据格式转换与重排 ---------------------------------
        # 如果四元数的格式不是 w 在最后 (w_last=False)，则转换为 wxyz 格式
        if not self.w_last:
            root_rot = rotations.xyzw_to_wxyz(root_rot)
            rb_rot = rotations.xyzw_to_wxyz(rb_rot)
            local_rot = rotations.xyzw_to_wxyz(local_rot)

        # 根据机器人模型的具体定义，对刚体(rb)、自由度(dof)和局部旋转(local_rot)数据进行重排，以匹配仿真环境
        if self.rb_conversion is not None:
            rb_pos = rb_pos[:, self.rb_conversion]
            rb_rot = rb_rot[:, self.rb_conversion]
            global_vel = global_vel[:, self.rb_conversion]
            global_ang_vel = global_ang_vel[:, self.rb_conversion]
        if self.dof_conversion is not None:
            dof_pos = dof_pos[:, self.dof_conversion]
            dof_vel = dof_vel[:, self.dof_conversion]
        if self.local_rot_conversion is not None:
            local_rot = local_rot[:, self.local_rot_conversion]

        # ------------------------------------ 5. 封装并返回运动状态 ---------------------------------
        # 将所有计算出的数据封装到 MotionState 对象中
        motion_state = MotionState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            key_body_pos=key_body_pos,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            local_rot=local_rot,
            rb_pos=rb_pos,
            rb_rot=rb_rot,
            rb_vel=global_vel,
            rb_ang_vel=global_ang_vel,
        )

        return motion_state

    @staticmethod
    def _load_motion_file(motion_file):
        return SkeletonMotion.from_file(motion_file)

    def _load_motions(self, motion_file, target_frame_rate):
        if self.create_text_embeddings:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []
        text_embeddings = []
        has_text_embeddings = []
        (
            motion_files,            # 一共有哪些动作文件
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,    # 这个submotion属于哪一个motion
            ref_respawn_offsets,
            motion_labels,
            supported_scene_ids,
            motion_labels_raw,       # 原始个数的label
        ) = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        for f in range(num_motion_files):  # 遍历所有动作文件
            curr_file = motion_files[f]  # 获取当前动作文件路径

            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )
            curr_motion = self._load_motion_file(curr_file)  # 加载动作文件
            curr_motion = fix_motion_fps(  # 修正动作帧率以匹配目标帧率
                curr_motion, motion_fpses[f], target_frame_rate, self.skeleton_tree
            )
            motion_fpses[f] = float(curr_motion.fps)  # 更新动作的帧率记录

            if self.fix_heights:  # 如果启用了高度修正
                curr_motion = fix_heights(curr_motion, self.skeleton_tree)  # 修正动作的高度，让最低的高度挨着地面，不要让他飘起来

            curr_dt = 1.0 / motion_fpses[f]  # 计算当前动作的帧间隔时间

            num_frames = curr_motion.global_translation.shape[0]  # 获取动作的总帧数
            curr_len = 1.0 / motion_fpses[f] * (num_frames - 1)  # 计算当前动作的总时长

            motion_dt.append(curr_dt)  # 记录帧间隔时间
            motion_num_frames.append(num_frames)  # 记录总帧数

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)  # 计算动作的自由度速度
            curr_motion.dof_vels = curr_dof_vels  # 将计算出的速度保存到动作数据中

            motions.append(curr_motion)  # 将处理后的动作添加到列表
            motion_lengths.append(curr_len)  # 将动作时长添加到列表

        num_sub_motions = len(sub_motion_to_motion)  # 获取子动作的总数

        for f in range(num_sub_motions):
            # Incase start/end weren't provided, set to (0, motion_length)
            motion_f = sub_motion_to_motion[f]
            if motion_timings[f][1] == -1:
                motion_timings[f][1] = motion_lengths[motion_f]

            motion_timings[f][1] = min(
                motion_timings[f][1], motion_lengths[motion_f]
            )  # CT hack: fix small timing differences

            assert (
                motion_timings[f][0] < motion_timings[f][1]
            ), f"Motion start {motion_timings[f][0]} >= motion end {motion_timings[f][1]} in motion {motion_f}"

            if self.create_text_embeddings and motion_labels[f][0] != "":
                with torch.inference_mode():
                    inputs = tokenizer(
                        motion_labels[f],
                        padding=True,            # padding那些比较短的句子，把所有的句子长度都弄成一样的
                        truncation=True,         # 如果 input 超出了最长的长度就截断
                        return_tensors="pt",
                    )
                    outputs = model(**inputs)  # 输出包括了pooler_output（表示整句话的池化，一个token代表整个句子, [3 ,512]）, last_hidden_state（表示每个token的隐藏状态, [3, 12, 512]）
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    text_embeddings.append(pooled_output)  # should be [3, 512]
                    has_text_embeddings.append(True)
            else:
                text_embeddings.append(
                    torch.zeros((3, 512), dtype=torch.float32)
                )  # just hold something temporary
                has_text_embeddings.append(False)

        motion_lengths = torch.tensor(
            motion_lengths, device=self._device, dtype=torch.float32
        )

        motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self._device
        )
        motion_weights /= motion_weights.sum()

        motion_timings = torch.tensor(
            motion_timings, dtype=torch.float32, device=self._device
        )

        sub_motion_to_motion = torch.tensor(
            sub_motion_to_motion, dtype=torch.long, device=self._device
        )

        ref_respawn_offsets = torch.tensor(
            ref_respawn_offsets, dtype=torch.float32, device=self._device
        )

        motion_fpses = torch.tensor(
            motion_fpses, device=self._device, dtype=torch.float32
        )
        motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        motion_num_frames = torch.tensor(motion_num_frames, device=self._device)

        text_embeddings = torch.stack(text_embeddings).detach().to(device=self._device)
        has_text_embeddings = torch.tensor(
            has_text_embeddings, dtype=torch.bool, device=self._device
        )

        self.state = LoadedMotions(
            motions=tuple(motions),
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_timings=motion_timings,
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            sub_motion_to_motion=sub_motion_to_motion,
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
            supported_scene_ids=supported_scene_ids,
            motion_labels=motion_labels,
            motion_labels_raw=motion_labels_raw,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        num_sub_motions = self.num_sub_motions()
        total_trainable_len = self.get_total_trainable_length()

        print(
            "Loaded {:d} sub motions with a total trainable length of {:.3f}s.".format(
                num_sub_motions, total_trainable_len
            )
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            sub_motion_to_motion = []
            ref_respawn_offsets = []
            motion_weights = []
            motion_timings = []
            motion_fpses = []
            motion_labels = []
            supported_scene_ids = []
            motion_labels_raw = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            motion_list = sorted(
                motion_config.motions,
                key=lambda x: 1e6 if "idx" not in x else int(x.idx),
            )

            motion_index = 0

            for motion_id, motion_entry in enumerate(motion_list):
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)

                motion_fpses.append(motion_entry.get("fps", None))

                if "sub_motions" not in motion_entry:
                    motion_entry.sub_motions = [deepcopy(motion_entry)]
                    motion_entry.sub_motions[0].idx = motion_index # 其实没暖用，都是对的

                for sub_motion in sorted(
                    motion_entry.sub_motions, key=lambda x: int(x.idx)
                ):
                    curr_weight = sub_motion.weight
                    assert curr_weight >= 0

                    assert motion_index == sub_motion.idx

                    motion_weights.append(curr_weight)

                    sub_motion_to_motion.append(motion_id) 

                    ref_respawn_offset = sub_motion.get("ref_respawn_offset", 0)
                    ref_respawn_offsets.append(ref_respawn_offset)

                    if "timings" in sub_motion:
                        curr_timing = sub_motion.timings
                        start = curr_timing.start
                        end = curr_timing.end
                    else:
                        start = 0
                        end = -1

                    motion_timings.append([start, end])

                    sub_motion_labels = []
                    sub_motion_labels_raw = []
                    if "labels" in sub_motion:                         # 这个地方的 motion_lable不用非得在 json文件里面读取，也可以直接去读txt文件，有 hml3d_id
                        # 我们假设每个动作有3个标签。
                        # 如果标签少于3个，则重复最后一个标签以补足数量。
                        # 如果没有标签，则使用空字符串作为标签。
                        for label in sub_motion.labels:
                            sub_motion_labels.append(label)
                            if len(sub_motion_labels) == 3:
                                break
                        if len(sub_motion_labels) == 0:
                            sub_motion_labels.append("")
                        while len(sub_motion_labels) < 3:
                            sub_motion_labels.append(sub_motion_labels[-1])
                        
                        for label in sub_motion.labels:
                            sub_motion_labels_raw.append(label)

                    else:
                        print(f"!!! No labels found for sub motion !!!")
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")

                    motion_labels.append(sub_motion_labels)
                    motion_labels_raw.append(sub_motion_labels_raw)

                    if "supported_scenes" in sub_motion:
                        supported_scene_ids.append(sub_motion.supported_scenes)
                    else:
                        supported_scene_ids.append(None)

                    motion_index += 1
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_timings = [[0, -1]]
            motion_fpses = [None]
            sub_motion_to_motion = [0]
            ref_respawn_offsets = [0]
            motion_labels = [["", "", ""]]
            supported_scene_ids = [None]

        return (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
            supported_scene_ids,
            motion_labels_raw
        )

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion: SkeletonMotion):
        num_frames = motion.global_translation.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    # jp hack
    # get rid of this ASAP, need a proper way of projecting from max coords to reduced coords
    def _local_rotation_to_dof(self, local_rot, joint_3d_format):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self.num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                if joint_3d_format == "exp_map":
                    formatted_joint = torch_utils.quat_to_exp_map(joint_q, w_last=True)
                elif joint_3d_format == "xyz":
                    x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
                    formatted_joint = torch.stack([x, y, z], dim=-1)
                else:
                    raise ValueError(f"Unknown 3d format '{joint_3d_format}'")

                dof_pos[:, joint_offset : (joint_offset + joint_size)] = formatted_joint
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(
                    joint_q, w_last=True
                )
                joint_theta = (
                    joint_theta * joint_axis[..., 1]
                )  # assume joint is always along y axis

                joint_theta = rotations.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        dof_vel = torch.zeros([self.num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[
                    1
                ]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

    def parse_scenes(self, spawned_scene_ids):
        # If motions may have supported scenes, create the mapping to allow sampling scenes for motions.
        motion_to_scene_ids = []
        scenes_per_motion = []
        if hasattr(self.state, "supported_scene_ids") and spawned_scene_ids is not None:

            def indices(lst, element):
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(element, offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            max_num_scenes = max(
                max(
                    [
                        len(scene_ids) if scene_ids is not None else 0
                        for scene_ids in self.state.supported_scene_ids
                    ]
                ),
                len(spawned_scene_ids),
            )

            for i in range(len(self.state.supported_scene_ids)):
                if self.state.supported_scene_ids[i] is None:
                    motion_to_scene_ids.append([-1] * max_num_scenes)
                    scenes_per_motion.append(-1)
                else:
                    all_scene_ids = []
                    for scene_id in self.state.supported_scene_ids[i]:
                        if scene_id in spawned_scene_ids:
                            # store all indices that match, multiple options may exist
                            scene_indices = indices(spawned_scene_ids, scene_id)
                            for scene_index in scene_indices:
                                all_scene_ids.append(scene_index)

                    scenes_per_motion.append(len(all_scene_ids))

                    if len(all_scene_ids) == 0:
                        all_scene_ids = [-1]
                    while len(all_scene_ids) < max_num_scenes:
                        all_scene_ids.append(-1)
                    motion_to_scene_ids.append(all_scene_ids)

        return scenes_per_motion, motion_to_scene_ids

    def sample_motions_scene_aware(
        self,
        num_motions,
        available_scenes,
        single_robot_in_scene,
        with_replacement=True,
        available_motion_mask=None,
    ):
        sampled_motions = []
        occupied_scenes = []

        if available_motion_mask is None:
            available_motion_mask = torch.ones(
                len(self.scenes_per_motion), dtype=torch.bool, device=self.device
            )

        motion_weights = self.state.motion_weights.clone()

        while len(sampled_motions) < num_motions:
            # Create a view of available motions
            for i, num_scenes in enumerate(self.scenes_per_motion):
                if num_scenes != -1 and not torch.any(
                    available_scenes[self.motion_to_scene_ids[i, :num_scenes]]
                ):
                    available_motion_mask[i] = False

            # Sample a motion based on weights
            motion_weights[~available_motion_mask] = 0
            if motion_weights.sum() == 0:
                raise ValueError("No more valid motions available")
            sampled_motion = torch.multinomial(motion_weights, num_samples=1).item()
            sampled_motions.append(sampled_motion)

            if not with_replacement:
                available_motion_mask[sampled_motion] = False

            # Sample a scene for the motion if needed
            if self.scenes_per_motion[sampled_motion] != -1:
                num_scenes = self.scenes_per_motion[sampled_motion]
                available_scene_mask = available_scenes[
                    self.motion_to_scene_ids[sampled_motion, :num_scenes]
                ]
                valid_scenes = self.motion_to_scene_ids[sampled_motion, :num_scenes][
                    available_scene_mask
                ]
                if valid_scenes.numel() > 0:
                    scene = valid_scenes[
                        torch.randint(0, valid_scenes.numel(), (1,)).item()
                    ]
                    occupied_scenes.append(scene)
                    if single_robot_in_scene[scene]:
                        available_scenes[scene] = False
                else:
                    raise ValueError("No more valid scenes available")
            else:
                occupied_scenes.append(-1)

        return torch.tensor(
            sampled_motions, device=self.device, dtype=torch.long
        ), torch.tensor(occupied_scenes, device=self.device, dtype=torch.long)


def fix_motion_fps(motion, orig_fps, target_frame_rate, skeleton_tree):
    if skeleton_tree is None:
        if hasattr(motion, "skeleton_tree"):
            skeleton_tree = motion.skeleton_tree
        else:
            return motion

    if orig_fps is None:
        orig_fps = motion.fps

    skip = int(np.round(orig_fps / target_frame_rate))

    lr = motion.local_rotation[::skip] # 每一个关节相对于其父关节的局部旋转
    rt = motion.root_translation[::skip]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        lr,
        rt,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_frame_rate)

    return new_motion


def fix_heights(motion, skeleton_tree):
    if skeleton_tree is None:
        if hasattr(motion, "skeleton_tree"):
            skeleton_tree = motion.skeleton_tree
    body_heights = motion.global_translation[..., 2]
    min_height = body_heights.min()

    if skeleton_tree is None:
        motion.global_translation[..., 2] -= min_height
        return motion

    root_translation = motion.root_translation
    root_translation[:, 2] -= min_height

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        motion.global_rotation,
        root_translation,
        is_local=False,
    )

    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion
