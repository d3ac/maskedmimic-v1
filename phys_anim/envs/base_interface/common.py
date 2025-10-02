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

import torch


class BaseInterface(object):
    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.headless = config.headless

        self.num_envs = config.num_envs

        # 控制频率的倒数，表示每进行一次高层控制需要多少个物理仿真步（即每隔多少步执行一次控制）
        self.control_freq_inv = config.simulator.sim.control_freq_inv 

    def get_obs_size(self):
        """获取观测空间的大小。"""
        raise NotImplementedError

    def on_environment_ready(self):
        """当环境完全设置好后调用的回调函数。"""
        pass

    def step(self, actions):
        """
        在环境中执行一个时间步。

        Args:
            actions: 要应用的动作。
        """
        raise NotImplementedError

    def pre_physics_step(self, actions):
        """
        在物理模拟之前执行的步骤，通常用于应用动作。

        Args:
            actions: 要应用的动作。
        """
        raise NotImplementedError

    def reset(self, env_ids=None):
        """
        重置环境中的部分或全部智能体。

        Args:
            env_ids: 需要重置的环境ID列表。如果为None, 则重置所有环境。
        """
        raise NotImplementedError

    def physics_step(self):
        """执行一个完整的物理模拟步骤，可能包含多个子步骤。"""
        if self.isaac_pd:
            self.apply_pd_control()
        for i in range(self.control_freq_inv):
            if not self.isaac_pd:
                self.apply_motor_forces()
            self.simulate()

    def simulate(self):
        """执行单步物理模拟。"""
        raise NotImplementedError

    def post_physics_step(self):
        """在物理模拟之后执行的步骤，用于计算奖励、观测和重置。"""
        raise NotImplementedError

    def on_epoch_end(self, current_epoch: int):
        """
        在每个训练周期结束时调用的回调函数。

        Args:
            current_epoch: 当前的周期数。
        """
        pass

    def close(self):
        """清理资源并关闭环境。"""
        raise NotImplementedError
