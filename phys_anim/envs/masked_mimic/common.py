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

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from torch import Tensor

from isaac_utils import torch_utils
from phys_anim.envs.masked_mimic.masked_mimic_utils import (
    build_historical_body_poses,
    build_sparse_target_poses,
    get_object_bounding_box_obs,
)
from phys_anim.envs.mimic.mimic_utils import dof_to_local, exp_tracking_reward
from phys_anim.envs.humanoid.humanoid_utils import quat_diff_norm
from phys_anim.envs.env_utils.general import HistoryBuffer, StepTracker

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.isaacgym import MaskedMimicHumanoid
else:
    MaskedMimicHumanoid = object


class BaseMaskedMimic(MaskedMimicHumanoid):
    def __init__(self, config, device: torch.device):
        self.masked_mimic_tensors_init = False

        # +1 for heading and velocity. One is the translation entry and the other as the rotation.
        self.num_conditionable_bodies = (
            len(config.masked_mimic_conditionable_bodies) + 1
        )

        self.motion_text_embeddings = torch.zeros(
            config.num_envs,
            config.masked_mimic_obs.text_embedding_dim,
            dtype=torch.float,
            device=device,
        )
        self.motion_text_embeddings_mask = torch.zeros(
            config.num_envs, 1, dtype=torch.bool, device=device
        )

        self.masked_mimic_target_poses = torch.zeros(  # + 1 for "far-away future pose"
            config.num_envs,
            (config.masked_mimic_obs.num_future_steps + 1)
            * config.masked_mimic_obs.num_obs_per_sparse_target_pose,
            dtype=torch.float,
            device=device,
        )

        self.time_gap_mask_steps = StepTracker(
            config.num_envs,
            min_steps=config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps,
            max_steps=config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps,
            device=device,
        )

        self.masked_mimic_target_bodies_masks = torch.zeros(
            config.num_envs,
            self.num_conditionable_bodies
            * 2
            * config.masked_mimic_obs.num_future_steps,
            dtype=torch.bool,
            device=device,
        )
        self.masked_mimic_target_poses_masks = torch.zeros(
            config.num_envs,
            config.masked_mimic_obs.num_future_steps + 1,
            dtype=torch.bool,
            device=device,  # + 1 for the inbetweening pose
        )

        self.object_bounding_box_obs = torch.zeros(
            # 8 * 3 for the bounding box, 1 * 6 for the tanh rot, len(self.object_types) for the one-hot object category
            config.num_envs,
            8 * 3 + 6 + len(config.object_types),
            dtype=torch.float,
            device=device,
        )
        self.object_bounding_box_obs_mask = torch.zeros(
            config.num_envs, 1, dtype=torch.bool, device=device
        )

        self.target_pose_time = torch.zeros(
            config.num_envs, dtype=torch.float, device=device
        )
        # which joints are visible
        self.target_pose_joints = torch.zeros(
            config.num_envs,
            self.num_conditionable_bodies * 2,
            dtype=torch.bool,
            device=device,
        )
        self.target_pose_obs_mask = torch.zeros(
            config.num_envs, 1, dtype=torch.bool, device=device
        )

        # Historical pose information
        self.body_pos_hist_buf = HistoryBuffer(
            config.masked_mimic_obs.num_historical_stored_steps + 1,
            config.num_envs,
            shape=(config.robot.num_bodies * 3,),
            device=device,
        )
        self.body_rot_hist_buf = HistoryBuffer(
            config.masked_mimic_obs.num_historical_stored_steps + 1,
            config.num_envs,
            shape=(config.robot.num_bodies * 4,),
            device=device,
        )
        self.valid_hist_buf = HistoryBuffer(
            config.masked_mimic_obs.num_historical_stored_steps + 1,
            config.num_envs,
            shape=(1,),
            device=device,
            dtype=torch.bool,
        )

        self.historical_pose_obs = torch.zeros(
            config.num_envs,
            config.masked_mimic_obs.num_historical_conditioned_steps
            * (config.robot.num_bodies * (3 + 6) + 1),
            device=device,
            dtype=torch.float,
        )
        self.historical_pose_obs_mask = torch.zeros(
            config.num_envs,
            config.masked_mimic_obs.num_historical_conditioned_steps,
            device=device,
            dtype=torch.bool,
        )

        # Sampling vectors
        self.long_term_gap_probs = (
            torch.ones(config.num_envs, dtype=torch.float, device=device)
            * config.masked_mimic_masking.joint_masking.with_conditioning_max_gap_probability
        )

        self.visible_object_bounding_box_probs = (
            torch.ones(config.num_envs, dtype=torch.float, device=device)
            * config.masked_mimic_masking.object_bounding_box_visible_prob
        )

        self.visible_text_embeddings_probs = (
            torch.ones(config.num_envs, dtype=torch.float, device=device)
            * config.masked_mimic_masking.motion_text_embeddings_visible_prob
        )

        self.visible_target_pose_probs = (
            torch.ones(config.num_envs, dtype=torch.float, device=device)
            * config.masked_mimic_masking.target_pose_visible_prob
        )

        self.visible_target_pose_joint_probs = (
            torch.ones(
                config.num_envs,
                self.num_conditionable_bodies * 2,
                dtype=torch.float,
                device=device,
            )
            * config.masked_mimic_masking.target_pose_joint_probs
        )

        self.start_without_history_probs = (
            torch.ones(config.num_envs, dtype=torch.float, device=device)
            * config.masked_mimic_masking.start_without_history_prob
        )

        super().__init__(config, device)

    def init_masked_mimic_tensors(self):
        self.masked_mimic_conditionable_bodies_ids = self.build_body_ids_tensor(
            self.config.masked_mimic_conditionable_bodies
        )
        self.masked_mimic_tensors_init = True

    def sample_body_masks(self, num_envs, env_ids, reset_track):
        """
        This function samples body masks for the given environments. The masks are used to condition the behavior of the
        humanoid model in the simulation. The function takes into account various factors such as time gaps, probabilities
        for repeating masks, and the number of active bodies to generate the masks.

        Args:
            num_envs (int): The number of environments.
            env_ids (torch.Tensor): The IDs of the environments.
            reset_track (bool): A flag indicating whether the track is being reset.

        Returns:
            None
        """
        if not self.masked_mimic_tensors_init:
            self.init_masked_mimic_tensors()

        # +1 for speed and heading
        new_mask = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        no_time_gap_or_repeated = torch.ones(
            num_envs, dtype=torch.bool, device=self.device
        )

        if not reset_track:
            # Reset the time gap mask if the time has expired
            remaining_time = (
                self.time_gap_mask_steps.cur_max_steps[env_ids]
                - self.time_gap_mask_steps.steps[env_ids]
            )

            # Mark envs with active time-gap
            active_time_gap = remaining_time > 0
            no_time_gap_or_repeated[active_time_gap] = False

            # For those without, check if it should start one
            restart_timegap = (remaining_time <= 0) & (
                torch.rand(num_envs, device=self.device)
                < self.config.masked_mimic_masking.joint_masking.masked_mimic_time_gap_probability
            )
            self.time_gap_mask_steps.reset_steps(env_ids[restart_timegap])

            # If we have text or object conditioning or target pose, then allow longer time gaps
            text_mask = restart_timegap & self.motion_text_embeddings_mask[
                env_ids
            ].view(-1)
            object_mask = restart_timegap & self.object_bounding_box_obs_mask[
                env_ids
            ].view(-1)
            target_pose_obs_mask = restart_timegap & self.target_pose_obs_mask[
                env_ids
            ].view(-1)
            allow_longer_time_gap = text_mask | object_mask | target_pose_obs_mask
            self.time_gap_mask_steps.cur_max_steps[env_ids[allow_longer_time_gap]] *= (
                self.config.masked_mimic_masking.joint_masking.with_conditioning_time_gap_mask_max_steps
                // self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps
            )

            # Where there's no time-gap, we can repeat the last mask
            repeat_mask = (remaining_time < 0) & (
                torch.rand(num_envs, device=self.device)
                < self.config.masked_mimic_masking.joint_masking.masked_mimic_repeat_mask_probability
            )
            single_step_mask_size = self.num_conditionable_bodies * 2
            new_mask[repeat_mask] = self.masked_mimic_target_bodies_masks[
                env_ids[repeat_mask], -single_step_mask_size:
            ].view(-1, self.num_conditionable_bodies, 2)

            no_time_gap_or_repeated[repeat_mask] = False

        # Compute number of active bodies for each env
        num_bodies_mask = (
            torch.rand(num_envs, device=self.device)
            >= self.config.masked_mimic_masking.joint_masking.force_small_num_conditioned_bodies_prob
        )
        # With low probability we ensure the number of active bodies is 1-3. Otherwise we allow all.
        max_num_bodies = torch.where(num_bodies_mask, self.num_conditionable_bodies, 3)
        random_floats = torch.rand(num_envs, device=self.device)
        scaled_floats = random_floats * (max_num_bodies - 1) + 1
        num_active_bodies = torch.round(scaled_floats).long()
        # With low probability force all bodies to be active
        max_bodies_mask = (
            torch.rand(num_envs, device=self.device)
            <= self.config.masked_mimic_masking.joint_masking.force_max_conditioned_bodies_prob
        )
        num_active_bodies[max_bodies_mask] = self.num_conditionable_bodies

        # Create tensor of [num_envs, num_conditionable_bodies] with the max_bodies dim being arange
        active_body_ids = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            device=self.device,
            dtype=torch.bool,
        )
        constraint_states = torch.randint(
            0, 3, (num_envs, self.num_conditionable_bodies), device=self.device
        )

        if (
            self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
            is None
        ):
            for idx in range(num_envs):
                # Sample the active body ids for each env
                active_body_ids[
                    idx,
                    np.random.choice(
                        self.num_conditionable_bodies,
                        size=num_active_bodies[idx].item(),
                        replace=False,
                    ),
                ] = True
        else:
            # If we have fixed conditioning, then we need to construct the active body ids based on the fixed conditioning
            body_names = [
                entry.body_name
                for entry in self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
            ]
            conditionable_body_ids = self.masked_mimic_conditionable_bodies_ids.tolist()
            conditioned_body_ids = self.build_body_ids_tensor(body_names).tolist()
            fixed_body_ids = [
                conditionable_body_ids.index(body_id)
                for body_id in conditioned_body_ids
            ]

            for body_id in fixed_body_ids:
                conditionable_body_names = self.config.masked_mimic_conditionable_bodies
                conditioned_body_names = [
                    entry.body_name
                    for entry in self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
                ]
                body_name = conditionable_body_names[body_id]
                id_in_constraint_list = conditioned_body_names.index(body_name)

                active_body_ids[:, body_id] = True
                constraint_states[:, body_id] = (
                    self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning[
                        id_in_constraint_list
                    ].constraint_state
                )

        translation_mask = (constraint_states <= 1) & active_body_ids
        rotation_mask = (constraint_states >= 1) & active_body_ids

        new_mask[no_time_gap_or_repeated, :, 0] = translation_mask[
            no_time_gap_or_repeated
        ]  # also velocity, at last idx
        new_mask[no_time_gap_or_repeated, :, 1] = rotation_mask[
            no_time_gap_or_repeated
        ]  # also heading, at last idx

        return new_mask.view(num_envs, -1)

    def sample_visible_target_pose(self, num_envs):
        """
        This function samples the visible target pose joints for the given environments. The function takes into account
        various factors such as fixed conditioning and the number of active bodies to generate the masks.
        """
        visible_target_pose_joints = (
            torch.bernoulli(self.visible_target_pose_joint_probs[:num_envs]) > 0
        )
        if (
            self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
            is not None
        ):
            visible_target_pose_joints[:] = False

            # If we have fixed conditioning, then we need to construct the active body ids based on the fixed conditioning
            body_names = [
                entry.body_name
                for entry in self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
            ]
            conditionable_body_ids = self.masked_mimic_conditionable_bodies_ids.tolist()
            conditioned_body_ids = self.build_body_ids_tensor(body_names).tolist()
            fixed_body_ids = [
                conditionable_body_ids.index(body_id)
                for body_id in conditioned_body_ids
            ]

            for body_id in fixed_body_ids:
                conditionable_body_names = self.config.masked_mimic_conditionable_bodies
                conditioned_body_names = [
                    entry.body_name
                    for entry in self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning
                ]
                body_name = conditionable_body_names[body_id]
                id_in_constraint_list = conditioned_body_names.index(body_name)
                if (
                    self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning[
                        id_in_constraint_list
                    ].constraint_state
                    <= 1
                ):
                    visible_target_pose_joints[:, body_id * 2] = True
                if (
                    self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning[
                        id_in_constraint_list
                    ].constraint_state
                    >= 1
                ):
                    visible_target_pose_joints[:, body_id * 2 + 1] = True

        return visible_target_pose_joints

    def reset_track(self, env_ids, new_motion_ids=None):
        if self.disable_reset_track:
            return

        if not self.masked_mimic_tensors_init:
            self.init_masked_mimic_tensors()

        new_motion_ids, new_times = super().reset_track(env_ids, new_motion_ids)

        # First sample how the motion will be constrained, then sample the masks (which may rely on this information)
        if hasattr(self.motion_lib.state, "has_text_embeddings"):
            visible_text_embeddings = (
                torch.bernoulli(self.visible_text_embeddings_probs[: len(env_ids)]) > 0
            )

            self.motion_text_embeddings[env_ids] = (
                self.motion_lib.sample_text_embeddings(new_motion_ids)
            )
            has_text = self.motion_lib.state.has_text_embeddings[new_motion_ids]
            self.motion_text_embeddings_mask[env_ids] = visible_text_embeddings.view(
                -1, 1
            ) & has_text.view(-1, 1)
        else:
            self.motion_text_embeddings_mask[env_ids] = False

        visible_object_bounding_box = (
            torch.bernoulli(self.visible_object_bounding_box_probs[: len(env_ids)]) > 0
        )
        has_scene = self.scene_ids[env_ids] >= 0
        self.object_bounding_box_obs_mask[env_ids] = visible_object_bounding_box.view(
            -1, 1
        ) & has_scene.view(-1, 1)

        visible_target_pose = (
            torch.bernoulli(self.visible_target_pose_probs[: len(env_ids)]) > 0
        )
        self.target_pose_obs_mask[env_ids] = visible_target_pose.view(-1, 1)

        max_time = self.motion_lib.state.motion_timings[new_motion_ids, 1]
        target_pose_time = (
            torch.rand(len(env_ids), device=self.device) * (max_time - new_times)
            + new_times
        )
        # With a small probability, set the target time to the max time. Since we re-sample the target-pose once it is
        # reached, don't need a high probability here.
        sample_max_time = torch.rand(len(env_ids), device=self.device) < 0.1
        target_pose_time[sample_max_time] = max_time[sample_max_time]
        self.target_pose_time[env_ids] = target_pose_time

        visible_target_pose_joints = self.sample_visible_target_pose(env_ids.shape[0])
        self.target_pose_joints[env_ids] = visible_target_pose_joints

        new_body_masks = self.sample_body_masks(len(env_ids), env_ids, reset_track=True)

        single_step_mask_size = self.num_conditionable_bodies * 2

        new_body_masks = (
            new_body_masks.view(len(env_ids), 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(len(env_ids), -1)
        )
        self.time_gap_mask_steps.reset_steps(env_ids)

        if (
            self.config.masked_mimic_masking.joint_masking.masked_mimic_time_gap_probability
            == 1
        ):
            new_body_masks[:, :] = 0

        has_long_term_conditioning = (
            self.motion_text_embeddings_mask[env_ids].view(-1)
            | self.object_bounding_box_obs_mask[env_ids].view(-1)
            | self.target_pose_obs_mask[env_ids].view(-1)
        )
        if has_long_term_conditioning.any():
            long_term_gap = (
                torch.bernoulli(self.long_term_gap_probs[: len(env_ids)])[
                    has_long_term_conditioning
                ]
                > 0
            )

            if long_term_gap.any():
                long_term_gap_env_ids = env_ids[has_long_term_conditioning][
                    long_term_gap
                ]

                self.time_gap_mask_steps.cur_max_steps[long_term_gap_env_ids] = (
                    self.config.max_episode_length * 2
                )  # Set beyond the max episode length
                new_body_masks[has_long_term_conditioning][long_term_gap, :] = 0

        self.masked_mimic_target_bodies_masks[env_ids] = new_body_masks

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)

        if env_ids.shape[0] > 0:
            self.reset_hist_buf(env_ids)

    def reset_hist_buf(self, env_ids):
        """
        This function resets the history buffer for the given environments. It handles the initialization of various
        tensors and sampling of masks that are used to condition the behavior of the humanoid model in the simulation.
        """
        num_envs = len(env_ids)

        time_offsets = (
            torch.arange(
                0,
                self.config.masked_mimic_obs.num_historical_stored_steps + 1,
                device=self.device,
                dtype=torch.long,
            )
            * self.dt
        )

        raw_historical_times = self.motion_times[env_ids].unsqueeze(
            -1
        ) - time_offsets.unsqueeze(0)
        motion_ids = (
            self.motion_ids[env_ids]
            .unsqueeze(-1)
            .tile([1, self.config.masked_mimic_obs.num_historical_stored_steps + 1])
        )
        flat_ids = motion_ids.view(-1)
        flat_times = torch.clamp(raw_historical_times.view(-1), min=0)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_gt = ref_state.rb_pos
        flat_gr = ref_state.rb_rot

        gt = flat_gt.view(
            num_envs, self.config.masked_mimic_obs.num_historical_stored_steps + 1, -1
        )
        gr = flat_gr.view(
            num_envs, self.config.masked_mimic_obs.num_historical_stored_steps + 1, -1
        )

        self.body_pos_hist_buf.set_hist(gt[:, 1:].permute(1, 0, 2), env_ids)
        self.body_rot_hist_buf.set_hist(gr[:, 1:].permute(1, 0, 2), env_ids)

        self.body_pos_hist_buf.set_curr(gt[:, 0], env_ids)
        self.body_rot_hist_buf.set_curr(gr[:, 0], env_ids)

        valid_pose_mask = (raw_historical_times >= 0).unsqueeze(-1).permute(1, 0, 2)

        sampled_with_history = (
            torch.bernoulli(self.start_without_history_probs[:num_envs]) == 0
        ).view(1, num_envs, 1)
        envs_with_history = torch.logical_and(sampled_with_history, valid_pose_mask)

        self.valid_hist_buf.set_all(envs_with_history, env_ids)

    def update_hist_buf(self):
        """
        This function updates the history buffer for the given environments. It handles the initialization of various
        tensors and sampling of masks that are used to condition the behavior of the humanoid model in the simulation.
        """
        current_state = self.get_bodies_state()
        cur_gt, cur_gr = (
            current_state.body_pos,
            current_state.body_rot,
        )
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )

        self.body_pos_hist_buf.set_curr(cur_gt.view(self.num_envs, -1))
        self.body_rot_hist_buf.set_curr(cur_gr.view(self.num_envs, -1))
        self.valid_hist_buf.set_curr(True)

    def post_physics_step(self):
        """
        This function updates the masked mimic tensors after the physics step. It handles the initialization of various
        tensors and sampling of masks that are used to condition the behavior of the humanoid model in the simulation.
        """
        super().post_physics_step()

        self.time_gap_mask_steps.advance()

        # Long-term conditioning ("inbetweening") can be re-sampled.
        # We want to make sure that there's always some conditioning.
        # If "suddenly" all conditioning is gone, reset the time-gap-mask-counter.
        had_long_term_conditioning = (
            self.motion_text_embeddings_mask.view(-1)
            | self.object_bounding_box_obs_mask.view(-1)
            | self.target_pose_obs_mask.view(-1)
        )

        passed_target_pose_time = self.target_pose_time < self.motion_times

        new_target_pose = (
            torch.bernoulli(
                self.visible_target_pose_probs[
                    : self.target_pose_obs_mask[passed_target_pose_time].shape[0]
                ]
            )
            > 0
        )
        self.target_pose_obs_mask[passed_target_pose_time] = new_target_pose.view(-1, 1)

        max_time = self.motion_lib.state.motion_timings[
            self.motion_ids[passed_target_pose_time], 1
        ]
        target_pose_time = (
            torch.rand(
                self.target_pose_obs_mask[passed_target_pose_time].shape[0],
                device=self.device,
            )
            * (max_time - self.motion_times[passed_target_pose_time])
            + self.motion_times[passed_target_pose_time]
        )
        sample_max_time = (
            torch.rand(
                self.target_pose_obs_mask[passed_target_pose_time].shape[0],
                device=self.device,
            )
            > 0.5
        )
        # CT hack: "- self.dt * 2" probably not needed, but just making sure there's no overflow.
        target_pose_time[sample_max_time] = max_time[sample_max_time] - self.dt * 2
        self.target_pose_time[passed_target_pose_time] = target_pose_time

        has_no_long_term_conditioning = torch.logical_not(
            self.motion_text_embeddings_mask.view(-1)
            | self.object_bounding_box_obs_mask.view(-1)
            | self.target_pose_obs_mask.view(-1)
        )

        lost_long_term_conditioning = (
            had_long_term_conditioning & has_no_long_term_conditioning
        )

        # If lost long-term conditioning, set the "max" time for the time-gap to -1 to ensure a new mask is sampled
        if lost_long_term_conditioning.any():
            self.time_gap_mask_steps.cur_max_steps[lost_long_term_conditioning] = -1

        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        new_body_masks = self.sample_body_masks(
            self.num_envs, all_env_ids, reset_track=False
        )

        single_step_mask_size = self.num_conditionable_bodies * 2

        self.masked_mimic_target_bodies_masks[:, :-single_step_mask_size] = (
            self.masked_mimic_target_bodies_masks[:, single_step_mask_size:].clone()
        )
        self.masked_mimic_target_bodies_masks[:, -single_step_mask_size:] = (
            new_body_masks
        )

        self.body_pos_hist_buf.rotate()
        self.body_rot_hist_buf.rotate()
        self.valid_hist_buf.rotate()
        self.update_hist_buf()

    def store_motion_data(self, skip=False):
        # Set skip=True in the super call.
        # This ensures we don't store the standard mimic data, only masked-mimic format.
        super().store_motion_data(skip=True)
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )

        target_pos[:, :, -1] += self.get_ground_heights(target_pos[:, 0, :2]).view(
            self.num_envs, 1
        )

        self.motion_recording["target_poses"].append(
            target_pos[:, self.masked_mimic_conditionable_bodies_ids, :].cpu().numpy()
        )

    def compute_reward(self, actions):
        """
        计算训练奖励函数 - 这是训练中最核心的函数之一
        
        该函数通过对比仿真环境中机器人的实际状态和从动作库加载的参考动作状态，
        计算多个奖励组件，引导智能体学习模仿参考动作。
        
        Args:
            actions (Tensor): 智能体采取的动作，shape (num_envs, num_actions)

        Returns:
            无直接返回值，但会更新 self.rew_buf (每个环境的总奖励)
            
        奖励计算包含以下组件：
            - gt_rew: 全局刚体位置奖励
            - rt_rew: 根部位置奖励
            - rh_rew: 根部高度奖励
            - rv_rew: 根部线速度奖励
            - rav_rew: 根部角速度奖励
            - gv_rew: 全局线速度奖励
            - gav_rew: 全局角速度奖励
            - kb_rew: 关键身体部位奖励
            - gr_rew: 全局旋转奖励
            - lr_rew: 局部旋转奖励
            - dv_rew: 关节速度奖励
            - kbf_rew: 接触力奖励（鼓励平滑的接触）
            - pow_rew: 功率惩罚（鼓励节能运动）
        """
        
        # ================================ 第一部分：获取参考动作状态 ================================
        # 从动作库中根据当前的动作ID和时间，提取参考状态（ground truth）
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        # 提取参考状态的各个组件（这些都是从pt文件中加载的预处理数据）
        ref_gt = ref_state.rb_pos           # 参考：所有刚体的全局位置 (世界坐标系)
        ref_gr = ref_state.rb_rot           # 参考：所有刚体的全局旋转 (世界坐标系，四元数)
        ref_lr = ref_state.local_rot        # 参考：所有关节的局部旋转 (相对父关节，四元数)
        ref_gv = ref_state.rb_vel           # 参考：所有刚体的全局线速度 (世界坐标系)
        ref_gav = ref_state.rb_ang_vel      # 参考：所有刚体的全局角速度 (世界坐标系)
        ref_dv = ref_state.dof_vel          # 参考：所有关节的角速度 (局部坐标系)

        # 只保留有DOF的刚体对应的局部旋转
        ref_lr = ref_lr[:, self.dof_body_ids]
        # 提取参考状态的关键身体部位（如手、脚）
        ref_kb = self.process_kb(ref_gt, ref_gr)

        # ================================ 第二部分：获取当前仿真状态 ================================
        # 从仿真环境中获取机器人的当前实际状态
        current_state = self.get_bodies_state()
        gt, gr, gv, gav = (
            current_state.body_pos,         # 当前：所有刚体的全局位置
            current_state.body_rot,         # 当前：所有刚体的全局旋转
            current_state.body_vel,         # 当前：所有刚体的线速度
            current_state.body_ang_vel,     # 当前：所有刚体的角速度
        )
        
        # 对当前位置进行坐标修正，使其与参考数据对齐
        # 第一步：根据当前根部位置获取地面高度，并从所有刚体的Z坐标中减去
        # 这样可以消除地形高度的影响，让机器人"站在"同一水平面上比较
        gt[:, :, -1:] -= self.get_ground_heights(gt[:, 0, :2]).view(self.num_envs, 1, 1)
        
        # 第二步：减去重生时的XY偏移量，回到参考数据的坐标系
        # 这是因为机器人可能在地图的不同位置重生，需要对齐坐标系
        gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )

        # 提取当前状态的关键身体部位
        kb = self.process_kb(gt, gr)

        # ================================ 第三部分：提取根部（Root）状态 ================================
        # 根部通常是Pelvis（骨盆），是机器人的中心刚体，索引为0
        rt = gt[:, 0]           # 当前根部位置
        ref_rt = ref_gt[:, 0]   # 参考根部位置

        # 如果配置为忽略高度（只关心XY平面运动），则只保留XY坐标
        if self.config.mimic_reward_config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]           # 当前根部旋转
        ref_rr = ref_gr[:, 0]   # 参考根部旋转

        # 计算当前和参考状态的朝向的逆四元数
        # 这用于将全局坐标转换到以机器人朝向为基准的局部坐标系
        inv_heading = torch_utils.calc_heading_quat_inv(rr, self.w_last)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, self.w_last)

        rv = gv[:, 0]           # 当前根部线速度
        ref_rv = ref_gv[:, 0]   # 参考根部线速度

        rav = gav[:, 0]         # 当前根部角速度
        ref_rav = ref_gav[:, 0] # 参考根部角速度

        # ================================ 第四部分：获取关节状态 ================================
        # 从仿真环境获取关节位置和关节速度
        dp, dv = self.get_dof_state()  # dp: dof position, dv: dof velocity

        # 将关节位置（dof_pos）转换为局部旋转四元数
        # 这是因为奖励计算需要用四元数形式的局部旋转
        lr = dof_to_local(dp, self.get_dof_offsets(), self.w_last)

        # 如果配置要求，将根部旋转也加入局部旋转列表
        # 这样可以同时对根部朝向和关节姿态进行奖励计算
        if self.config.mimic_reward_config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]
            # 在第一个维度拼接根部旋转
            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        # ================================ 第五部分：计算跟踪奖励 ================================
        # 调用核心奖励函数，计算当前状态与参考状态之间的匹配程度
        # 该函数返回一个字典，包含多个奖励组件（gt_rew, rt_rew, kb_rew等）
        rew_dict = exp_tracking_reward(
            # 当前状态
            gt=gt,      # 全局刚体位置
            rt=rt,      # 根部位置
            kb=kb,      # 关键身体部位
            gr=gr,      # 全局刚体旋转
            lr=lr,      # 局部关节旋转
            rv=rv,      # 根部线速度
            rav=rav,    # 根部角速度
            gv=gv,      # 全局线速度
            gav=gav,    # 全局角速度
            dv=dv,      # 关节速度
            # 参考状态
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
            # 配置参数
            joint_reward_weights=self.reward_joint_weights,  # 不同关节的权重
            config=self.config.mimic_reward_config,
            w_last=self.w_last,
        )

        # ================================ 第六部分：计算额外奖励组件 ================================
        
        # 6.1 接触力奖励（Key Body Force Reward）
        # 鼓励机器人平滑地与地面接触，惩罚突然的接触力变化
        current_contact_forces = self.get_bodies_contact_buf()
        # 计算接触力的减少量（只关注力减小的情况）
        forces_delta = torch.clip(
            self.prev_contact_forces - current_contact_forces, min=0
        )[
            :, self.non_termination_contact_body_ids, 2  # 只取Z轴（垂直）方向的力
        ]
        # 接触力减小越平滑，奖励越高（用指数函数放大差异）
        kbf_rew = (
            forces_delta.sum(-1)
            .mul(self.config.mimic_reward_config.component_coefficients.kbf_rew_c)
            .exp()
        )
        rew_dict["kbf_rew"] = kbf_rew

        # 6.2 功率惩罚（Power Reward）
        # 鼓励机器人使用较小的力矩，模拟真实的能量消耗
        dof_forces = self.get_dof_forces()  # 获取各个关节的力矩
        # 功率 = |力矩 × 角速度|
        power = torch.abs(torch.multiply(dof_forces, dv)).sum(dim=-1)
        pow_rew = -power  # 负奖励：功率越大，惩罚越大

        # 在刚重置后的宽限期内不计算功率惩罚
        # 因为刚重置时机器人可能需要较大的力矩来调整姿态
        has_reset_grace = (
            self.reset_track_steps.steps <= self.config.mimic_reset_track.grace_period
        )
        pow_rew[has_reset_grace] = 0

        rew_dict["pow_rew"] = pow_rew

        # ================================ 第七部分：奖励缩放与汇总 ================================
        # 将每个奖励组件乘以对应的权重，得到缩放后的奖励
        # 例如：gt_rew * gt_rew_w, rt_rew * rt_rew_w, ...
        self.last_scaled_rewards: Dict[str, Tensor] = {
            k: v * getattr(self.config.mimic_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        # 所有缩放后的奖励求和，得到总的跟踪奖励
        tracking_rew = sum(self.last_scaled_rewards.values())

        # 加上一个正常数，使奖励整体偏正（有利于训练稳定性）
        self.rew_buf = tracking_rew + self.config.mimic_reward_config.positive_constant

        # ================================ 第八部分：记录日志 ================================
        # 记录原始奖励（未缩放）的均值和标准差
        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        # 记录缩放后奖励的均值和标准差
        for rew_name, rew in self.last_scaled_rewards.items():
            self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        # ================================ 第九部分：计算误差指标（用于监控训练进度）================================
        
        # 9.1 笛卡尔误差：在机器人局部坐标系下计算位置误差
        # 将全局位置转换到以机器人朝向为基准的局部坐标系
        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(gt, inv_heading)
        # 计算相对于根部的相对位置差异
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean(-1)
        )

        # 9.2 获取Masked Mimic特定的掩码
        # 在Masked Mimic任务中，只需要跟踪部分身体部位（被掩码选中的部位）
        # 掩码的最后一项通常是朝向和速度，这里移除它
        mask = self.masked_mimic_target_bodies_masks.view(
            self.num_envs,
            self.config.mimic_target_pose.num_future_steps,
            self.num_conditionable_bodies,
            2,  # 2表示：[位置掩码, 旋转掩码]
        )[:, 0, :-1, :]  # 只取第一个时间步，移除最后一个身体部位

        translation_mask = mask[..., 0].unsqueeze(-1)  # 位置掩码
        rotation_mask = mask[..., 1]                    # 旋转掩码

        # 计算归一化系数（被掩码选中的身体部位数量）
        translation_mask_coeff = translation_mask.float().sum(1).view(-1) + 1e-6
        rotation_mask_coeff = rotation_mask.float().sum(1) + 1e-6

        # 获取可控制的身体部位索引
        active_bodies_ids = self.masked_mimic_conditionable_bodies_ids

        # 9.3 计算位置误差（只针对掩码选中的身体部位）
        gt_err = (
            (ref_gt - gt)[:, active_bodies_ids]  # 提取选中的身体部位
            .mul(translation_mask)                # 应用位置掩码
            .pow(2)                               # 平方
            .sum(-1)                              # 对xyz求和
            .sqrt()                               # 开方得到欧氏距离
            .sum(-1)                              # 对所有身体部位求和
            .div(translation_mask_coeff)          # 归一化
        )
        
        # 9.4 计算最大关节误差（用于判断最差的关节表现）
        max_joint_err = (
            (ref_gt - gt)[:, active_bodies_ids]
            .mul(translation_mask)
            .pow(2)
            .sum(-1)
            .sqrt()
            .max(-1)[0]  # 取最大值
        )

        # 9.5 如果配置要求报告全身指标，则重新计算（不使用掩码）
        if self.config.masked_mimic_obs.masked_mimic_report_full_body_metrics:
            translation_mask_coeff = self.num_bodies
            rotation_mask_coeff = self.num_bodies

            gt_err = (
                (ref_gt - gt).pow(2).sum(-1).sqrt().sum(-1).div(translation_mask_coeff)
            )
            max_joint_err = (ref_gt - gt).pow(2).sum(-1).sqrt().max(-1)[0]

        # 9.6 计算旋转误差（四元数差异）
        gr_err = (
            quat_diff_norm(gr, ref_gr, self.w_last)[:, active_bodies_ids]
            .mul(rotation_mask)
            .sum(-1)
            .div(rotation_mask_coeff)
        )
        gr_err_degrees = gr_err * 180 / torch.pi  # 转换为角度便于理解

        # ================================ 第十部分：汇总其他日志指标 ================================
        other_log_terms = {
            "tracking_rew": tracking_rew,       # 总跟踪奖励
            "total_rew": self.rew_buf,          # 最终奖励（包含常数）
            "cartesian_err": cartesian_err,     # 笛卡尔误差
            "gt_err": gt_err,                   # 位置误差
            "gr_err": gr_err,                   # 旋转误差（弧度）
            "gr_err_degrees": gr_err_degrees,   # 旋转误差（角度）
            "max_joint_err": max_joint_err,     # 最大关节误差
        }
        
        # 记录这些指标的均值和标准差到日志
        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"{rew_name}_mean"] = rew.mean()
            self.log_dict[f"{rew_name}_std"] = rew.std()

        # ================================ 第十一部分：保存奖励信息供后续使用 ================================
        self.last_unscaled_rewards: Dict[str, Tensor] = rew_dict          # 原始奖励
        self.last_scaled_rewards = self.last_scaled_rewards                # 缩放后奖励
        self.last_other_rewards = other_log_terms                          # 其他指标

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        # TODO: take env_ids as input here
        self.masked_mimic_target_poses[:] = (
            self.build_sparse_target_poses_masked_with_time(
                self.config.masked_mimic_obs.num_future_steps
            )
        )

        reshaped_masked_mimic_target_bodies_masks = (
            self.masked_mimic_target_bodies_masks.view(
                self.num_envs, self.config.masked_mimic_obs.num_future_steps, -1
            )
        )

        # Make sure only future poses that are in range are viewable
        time_offsets = (
            torch.arange(
                1,
                self.config.masked_mimic_obs.num_future_steps + 1,
                device=self.device,
                dtype=torch.long,
            )
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids.unsqueeze(-1).tile(
            [1, self.config.masked_mimic_obs.num_future_steps]
        )
        lengths = self.motion_lib.get_motion_length(motion_ids.view(-1)).view(
            self.num_envs, self.config.masked_mimic_obs.num_future_steps
        )
        in_bound_times = near_future_times <= lengths

        for i in range(self.config.masked_mimic_obs.num_future_steps):
            any_visible_joint = reshaped_masked_mimic_target_bodies_masks[:, i].any(
                dim=-1
            )
            self.masked_mimic_target_poses_masks[:, i] = (
                any_visible_joint & in_bound_times[:, i]
            )

        self.masked_mimic_target_poses_masks[:, -1:] = self.target_pose_obs_mask

        if self.total_num_objects > 0:
            env_ids_with_scenes = torch.nonzero(
                self.scene_ids >= 0, as_tuple=False
            ).reshape(-1)
            env_ids_without_scenes = torch.nonzero(
                self.scene_ids < 0, as_tuple=False
            ).reshape(-1)
            if env_ids_with_scenes.shape[0] > 0:
                self.object_bounding_box_obs[env_ids_with_scenes] = (
                    self.get_object_bounding_box_obs(env_ids_with_scenes)
                )
            if env_ids_without_scenes.shape[0] > 0:
                self.object_bounding_box_obs[env_ids_without_scenes] = 0

        hist_poses, hist_masks = self.build_historical_body_poses(env_ids)
        self.historical_pose_obs[env_ids] = hist_poses
        self.historical_pose_obs_mask[env_ids] = hist_masks.squeeze(-1)

    def build_historical_body_poses(self, env_ids):
        num_envs = len(env_ids)

        sub_sampling = (
            self.config.masked_mimic_obs.num_historical_stored_steps
            // self.config.masked_mimic_obs.num_historical_conditioned_steps
        )
        hist_gt = (
            self.body_pos_hist_buf.get_hist(env_ids)
            .permute(1, 0, 2)
            .view(
                num_envs,
                self.config.masked_mimic_obs.num_historical_stored_steps,
                self.num_bodies,
                3,
            )[:, ::sub_sampling]
        )
        hist_gr = (
            self.body_rot_hist_buf.get_hist(env_ids)
            .permute(1, 0, 2)
            .view(
                num_envs,
                self.config.masked_mimic_obs.num_historical_stored_steps,
                self.num_bodies,
                4,
            )[:, ::sub_sampling]
        )

        cur_gt = self.body_pos_hist_buf.get_current(env_ids).view(
            num_envs, self.num_bodies, 3
        )
        cur_gr = self.body_rot_hist_buf.get_current(env_ids).view(
            num_envs, self.num_bodies, 4
        )

        return (
            build_historical_body_poses(
                cur_gt=cur_gt,
                cur_gr=cur_gr,
                hist_gt=hist_gt,
                hist_gr=hist_gr,
                num_historical_stored_steps=self.config.masked_mimic_obs.num_historical_stored_steps,
                num_historical_conditioned_steps=self.config.masked_mimic_obs.num_historical_conditioned_steps,
                dt=self.dt,
                num_envs=num_envs,
                w_last=self.w_last,
            ),
            self.valid_hist_buf.get_hist(env_ids)
            .permute(1, 0, 2)
            .view(
                num_envs, self.config.masked_mimic_obs.num_historical_stored_steps, 1
            )[:, ::sub_sampling],
        )

    def build_sparse_target_poses(self, raw_future_times):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        flat_target_rot = ref_state.rb_rot
        flat_target_vel = ref_state.rb_vel

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = (
            current_state.body_pos,
            current_state.body_rot,
        )
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )

        return build_sparse_target_poses(
            cur_gt=cur_gt,
            cur_gr=cur_gr,
            flat_target_pos=flat_target_pos,
            flat_target_rot=flat_target_rot,
            flat_target_vel=flat_target_vel,
            masked_mimic_conditionable_bodies_ids=self.masked_mimic_conditionable_bodies_ids,
            num_future_steps=num_future_steps,
            num_envs=self.num_envs,
            w_last=self.w_last,
        )

    def build_sparse_target_poses_masked_with_time(self, num_future_steps):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps + 1])

        # +1 for "far future step"
        obs = self.build_sparse_target_poses(all_future_times).view(
            self.num_envs, num_future_steps + 1, self.num_conditionable_bodies, 2, 12
        )

        near_mask = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints.view(self.num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask

        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            self.num_envs, num_future_steps + 1, -1
        )

        flat_ids = motion_ids.view(-1)
        lengths = self.motion_lib.get_motion_length(flat_ids)

        times = torch.minimum(all_future_times.view(-1), lengths).view(
            self.num_envs, num_future_steps + 1, 1
        ) - self.motion_times.view(self.num_envs, 1, 1)
        ones_vec = torch.ones(
            self.num_envs, num_future_steps + 1, 1, device=self.device
        )
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(self.num_envs, -1)

    def get_object_bounding_box_obs(self, scene_env_ids):
        scene_ids = self.scene_ids[scene_env_ids]
        object_ids = self.scene_lib.scene_to_object_ids[scene_ids]

        assert (
            len(object_ids.shape) == 1 or object_ids.shape[1] == 1
        ), "This observation does not yet support multiple objects per scene."

        object_ids = object_ids.view(-1)

        num_scene_envs = scene_env_ids.shape[0]

        root_states = (
            self.get_humanoid_root_states()[scene_env_ids]
            .clone()
            .view(num_scene_envs, -1)
        )
        root_pos = root_states[:, :3]

        root_pos[:, -1] -= self.get_ground_heights(root_pos[:, :2]).view(-1)
        root_quat = root_states[:, 3:7]

        object_root_states = self.get_object_root_states()

        return get_object_bounding_box_obs(
            object_ids=object_ids,
            root_pos=root_pos,
            root_quat=root_quat,
            num_object_envs=num_scene_envs,
            object_root_states=object_root_states,
            object_root_states_offsets=self.object_root_states_offsets,
            object_bounding_box=self.object_id_to_object_bounding_box(object_ids),
            num_object_types=self.config.scene_lib.num_object_types,
            w_last=self.w_last,
        )
