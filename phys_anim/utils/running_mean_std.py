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


from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RunningMeanStd(nn.Module):
    """
    计算数据流的运行平均值和标准差。
    使用Welford的在线算法或其并行变体来高效地更新均值和方差。
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        epsilon: int = 1,
        shape: Tuple[int, ...] = (),
        device="cuda:0",
        clamp_value: Optional[float] = None,
    ):
        """
        初始化运行平均值和标准差的计算器。
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: 一个小的数值，用于防止除以零的算术问题，也用于初始化计数器。
        :param shape: 数据流输出的形状。
        :param device: 计算所在的设备 (例如, "cuda:0" 或 "cpu")。
        :param clamp_value: 如果提供，归一化后的值将被限制在 [-clamp_value, clamp_value] 范围内。
        """
        super().__init__()
        self.mean = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32, device=device), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.ones(shape, dtype=torch.float32, device=device), requires_grad=False
        )
        # self.count = epsilon
        self.count = nn.Parameter(
            torch.tensor(epsilon, dtype=torch.long, device=device), requires_grad=False
        )
        self.clamp_value = clamp_value

    @torch.no_grad()
    def update(self, arr: torch.tensor) -> None:
        """
        根据新的一批数据更新运行平均值和标准差。
        它首先计算这批数据的均值和方差，然后调用 update_from_moments。

        :param arr: 新的数据批次，一个torch张量。
        """
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: torch.tensor, batch_var: torch.tensor, batch_count: int
    ) -> None:
        """
        使用新批次的均值、方差和计数来更新模型的运行统计信息。
        这实现了并行算法来计算方差，允许高效地合并来自不同批次的统计数据。

        :param batch_mean: 新数据批次的均值。
        :param batch_var: 新数据批次的方差 (使用n进行归一化，即无偏估计为False)。
        :param batch_count: 新数据批次中的样本数量。
        """
        delta = batch_mean - self.mean
        new_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        self.mean[:] = new_mean
        self.var[:] = new_var
        self.count.fill_(new_count)

    def maybe_clamp(self, x: Tensor):
        """
        如果设置了 clamp_value，则将输入张量的值限制在 [-clamp_value, clamp_value] 范围内。

        :param x: 输入张量。
        :return: 如果 clamp_value 不为 None，则返回被裁剪的张量；否则返回原始张量。
        """
        if self.clamp_value is None:
            return x
        else:
            return torch.clamp(x, -self.clamp_value, self.clamp_value)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        """
        使用运行平均值和标准差对输入张量进行归一化或反归一化。

        :param arr: 要处理的输入张量。
        :param un_norm: 如果为 False (默认)，则对张量进行归一化。
                        如果为 True，则对张量进行反归一化。
        :return: 处理后的张量。
        """
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean

        return result
