import torch
from torch import Tensor

from phys_anim.utils.device_dtype_mixin import DeviceDtypeModuleMixin


class StepTracker(DeviceDtypeModuleMixin):
    """
    用于跟踪每个环境在训练过程中已经训练了多少步。
    在每个训练步骤中，StepTracker 会记录每个环境已经训练了多少步，并更新当前的最大步数。
    当环境达到最大步数时，环境会重置。
    """
    steps: Tensor

    def __init__(
        self, num_envs: int, min_steps: int, max_steps: int, device: torch.device
    ):
        super().__init__()

        self.register_buffer(
            "steps", torch.zeros(num_envs, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "cur_max_steps", torch.zeros(num_envs, dtype=torch.long), persistent=False
        )

        self.num_envs = num_envs
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.to(device)

    def advance(self):
        self.steps += 1

    def done_indices(self):
        return torch.nonzero(
            torch.greater_equal(self.steps, self.cur_max_steps), as_tuple=False
        ).squeeze(-1)

    def reset_steps(self, env_ids: Tensor = None):
        if env_ids is None:
            env_ids = torch.arange(
                0, self.num_envs, device=self.device, dtype=torch.long
            )

        n = len(env_ids)
        self.steps[env_ids] = 0
        self.cur_max_steps[env_ids] = torch.randint( # 如果所有episode都是固定长度，智能体可能会学到一些不良的周期性行为模式。
            self.min_steps,                          # 在episode快结束时采取冒险行动（因为知道马上要重置了）
            self.max_steps,                          # 形成依赖于固定时间步的策略
            size=[n],
            dtype=torch.long,
            device=self.device,
        )

    def shift_counter(self, env_ids: Tensor, shift: Tensor):
        self.steps[env_ids] -= shift
        self.cur_max_steps[env_ids] -= shift


class HistoryBuffer(DeviceDtypeModuleMixin):
    """
    Buffer that stores the past N frames of a tensor.
    Index 0 is the most recent frame.
    """

    data: Tensor

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        shape: tuple = (),
        dtype=torch.float,
        device="cpu",
    ):
        super().__init__()
        data = torch.zeros(num_steps, num_envs, *shape, dtype=dtype, device=device)
        self.register_buffer("data", data, persistent=False)
        self.to(device)

    def rotate(self):
        self.data[1:] = self.data[:-1].clone()

    @torch.no_grad()
    def update(self, fresh_data: Tensor):
        self.rotate()
        self.set_curr(fresh_data)

    @torch.no_grad()
    def set_all(self, fresh_data: Tensor, env_ids=slice(None)):
        self.data[:, env_ids] = fresh_data

    @torch.no_grad()
    def set_hist(self, fresh_data: Tensor, env_ids=slice(None)):
        self.data[1:, env_ids] = fresh_data

    @torch.no_grad()
    def set_curr(self, fresh_data: Tensor, env_ids=slice(None)):
        self.data[0, env_ids] = fresh_data

    def get_hist(self, env_ids=slice(None)):
        return self.data[1:, env_ids]

    def get_current(self, env_ids=slice(None)):
        return self.data[0, env_ids]

    def get_all(self, env_ids=slice(None)):
        return self.data[:, env_ids]

    def get_all_flattened(self, env_ids=slice(None)):
        data = self.get_all(env_ids)
        num_envs = data.shape[1]
        return data.permute(1, 0, 2).reshape(num_envs, -1)

    def get_index(self, idx: int, env_ids=slice(None)):
        return self.data[idx, env_ids]
