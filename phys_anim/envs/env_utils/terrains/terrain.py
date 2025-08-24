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

import numpy as np
import math
import torch
from scipy import ndimage

from phys_anim.envs.env_utils.terrains.subterrain import SubTerrain
from phys_anim.envs.env_utils.terrains.subterrain_generator import (
    discrete_obstacles_subterrain,
    poles_subterrain,
    pyramid_sloped_subterrain,
    pyramid_stairs_subterrain,
    random_uniform_subterrain,
    stepping_stones_subterrain,
)
from phys_anim.envs.env_utils.terrains.terrain_utils import (
    convert_heightfield_to_trimesh,
)
from phys_anim.utils.scene_lib import SceneLib

import matplotlib.pyplot as plt


class Terrain:
    def __init__(self, config, scene_lib: SceneLib, num_envs: int, device) -> None:
        self.config = config
        self.device = device
        self.num_scenes = 0
        
        # Config parameters for scene placement
        self.spacing_between_scenes = config.spacing_between_scenes  # 场景之间的间距 (米)
        self.minimal_humanoid_spacing = config.minimal_humanoid_spacing  # 人形角色之间的最小间距 (米)

        # Calculate number of scenes that can fit in one column
        # 计算一列中可以放置的场景数量
        length = config.map_length * config.num_levels  # 总地形长度
        self.num_scenes_per_column = max( # 在这个总长度上，最多可以放置多少个物体，后面假设要放沙发哪些什么，这个就是最大的间距
            math.floor(length / self.spacing_between_scenes), 1
        )

        # Terrain scaling parameters
        # 地形缩放参数
        self.horizontal_scale = config.horizontal_scale  # 水平方向缩放比例 (米/像素)
        self.vertical_scale = config.vertical_scale      # 垂直方向缩放比例 (米/像素)
        self.border_size = config.border_size            # 边界区域大小 (米)
        
        # Individual environment dimensions
        # 单个环境的尺寸
        self.env_length = config.map_length  # 单个环境的长度 (米)
        self.env_width = config.map_width    # 单个环境的宽度 (米)
        
        # Terrain type proportions (cumulative probabilities)
        # 地形类型比例 (累积概率)
        self.proportions = [                                  # 一共有8种地形
            np.sum(config.terrain_proportions[: i + 1])       # 每种地形出现的概率
            for i in range(len(config.terrain_proportions))
        ]

        # Grid layout parameters
        # 网格布局参数
        self.env_rows = config.num_levels      # 地形难度等级数 (行数)
        self.env_cols = config.num_terrains    # 地形类型数 (列数)
        self.num_maps = self.env_rows * self.env_cols  # 总地形块数
        
        # Border size in pixels
        # 边界大小 (像素单位)
        self.border = int(self.border_size / self.horizontal_scale)

        # Initialize scene library and count total scenes
        # 初始化场景库并统计总场景数
        if scene_lib is not None:
            scene_lib.call_when_terrain_init_scene_spacing(self)
            self.num_scenes = scene_lib.total_spawned_scenes  # 总生成场景数

        # Calculate object playground dimensions
        # 计算物体游乐场区域尺寸
        scene_rows = (
            0
            if self.num_scenes == 0
            else math.ceil(self.num_scenes / self.num_scenes_per_column) + 2
        )  # 场景行数 (包含2行缓冲)
        self.object_playground_depth = scene_rows * self.spacing_between_scenes  # 物体游乐场深度 (米)
        self.object_playground_buffer_size = int(5 / self.horizontal_scale)  # 5米缓冲区，转换为像素单位

        # Validate humanoid spacing
        # 验证人形角色间距是否足够
        total_size = self.num_maps * config.map_length * config.map_width * 1.0  # 总地形面积
        space_between_humanoids = total_size / num_envs  # 每个人形角色的平均空间
        assert (
            space_between_humanoids >= self.minimal_humanoid_spacing
        ), "Not enough space between humanoids, create a bigger terrain or reduce the number of envs."

        # Convert environment dimensions to pixels
        # 将环境尺寸转换为像素单位
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)   # 单个环境宽度 (像素)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale) # 单个环境长度 (像素)

        # Calculate total terrain grid dimensions
        # 计算总地形网格尺寸
        self.object_playground_cols = math.ceil(
            self.object_playground_depth / self.horizontal_scale
        )  # 物体游乐场列数 (像素)
        
        # Total columns: environments + borders + object playground
        # 总列数：环境区域 + 边界 + 物体游乐场
        self.tot_cols = (
            int(self.env_cols * self.width_per_env_pixels)  # 所有环境的总宽度
            + 2 * self.border                               # 左右边界
            + self.object_playground_cols                   # 物体游乐场区域
        )
        
        # Total rows: environments + borders
        # 总行数：环境区域 + 边界
        self.tot_rows = (
            int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        )

        # Initialize terrain height fields and maps
        # 初始化地形高度场和地图
        
        # Main height field storing terrain elevation data (scaled to int16)
        # 主高度场，存储地形高程数据 (缩放到int16)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        
        # Ceiling height field (3 meters high by default)
        # 这个应该是h上的最大高度
        self.ceiling_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        ) + (3 / self.vertical_scale)

        # Walkable area mask (0=walkable, 1=non-walkable)
        # 可行走区域掩码 (0=可行走, 1=不可行走)
        self.walkable_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        )
        
        # Flat terrain mask (1=flat by default, 0=marked as flat later)
        # 平坦地形掩码 (默认1=平坦, 后续0=标记为平坦)
        self.flat_field_raw = np.ones((self.tot_rows, self.tot_cols), dtype=np.int16)

        # Scene placement map (True=occupied by scene, False=free)
        # 场景放置地图 (True=被场景占用, False=空闲)
        self.scene_map = torch.zeros(
            (self.tot_rows, self.tot_cols), dtype=torch.bool, device=self.device
        )

        # Load or generate terrain data
        # 加载或生成地形数据
        if self.config.load_terrain:  # 是否加载预生成的地形
            print("Loading a pre-generated terrain")
            params = torch.load(self.config.terrain_path)  # 地形文件路径
            self.height_field_raw = params["height_field_raw"]
            self.walkable_field_raw = params["walkable_field_raw"]
        else:
            # Generate terrain procedurally based on configuration
            # 根据配置程序化生成地形
            self.generate_subterrains()
            
        # Create mesh representation for physics simulation
        # 为物理仿真创建网格表示
        self.heightsamples = self.height_field_raw  # 高度采样数据
        self.vertices, self.triangles = convert_heightfield_to_trimesh(   # vertices是顶点（x,y,z），triangles是三角形（三个vertices的索引）
            self.height_field_raw,
            self.horizontal_scale,
            self.vertical_scale,
            self.config.slope_threshold,  # 坡度阈值，用于确定可行走表面
        )
        
        # Compute coordinate arrays for spawn point sampling
        # 计算用于生成点采样的坐标数组
        self.compute_walkable_coords()  # 计算可行走坐标
        self.compute_flat_coords()      # 计算平坦区域坐标

        # Optionally save generated terrain for future use
        # 可选择保存生成的地形供将来使用
        if self.config.save_terrain:  # 是否保存生成的地形
            print("Saving this generated terrain")
            torch.save(
                {
                    "height_field_raw": self.height_field_raw,
                    "walkable_field_raw": self.walkable_field_raw,
                    "vertices": self.vertices,
                    "triangles": self.triangles,
                    "border_size": self.border_size,
                },
                self.config.terrain_path,  # 保存路径
            )

        # Finalize scene library initialization
        # 完成场景库初始化
        if scene_lib is not None:
            # Push all scenes to spawn at the edge of the terrain
            # 将所有场景推到地形边缘生成
            scene_y_offset = (
                self.tot_cols - self.border - self.object_playground_cols
            ) * self.horizontal_scale  # 场景Y轴偏移量 (米)
            scene_lib.call_at_terrain_done_init(scene_y_offset)

        # # Generate and show the plot
        # self.generate_terrain_plot(scene_lib)

    def generate_subterrains(self):
        if self.config.terrain_composition == "curriculum":
            self.curriculum(
                n_subterrains_per_level=self.env_cols, n_levels=self.env_rows
            )
        elif self.config.terrain_composition == "randomized_subterrains":
            self.randomized_subterrains()
        else:
            raise NotImplementedError(
                "Terrain composition configuration "
                + self.config.terrain_composition
                + " not implemented"
            )

    def compute_walkable_coords(self):
        self.walkable_field_raw[: self.border, :] = 1     # 边界区域设置为不能走
        self.walkable_field_raw[:, -(self.border + self.object_playground_cols + self.object_playground_buffer_size) :] = 1  # 边界区域设置为不能走
        self.walkable_field_raw[:, : self.border] = 1     # 边界区域设置为不能走
        self.walkable_field_raw[-self.border :, :] = 1    # 边界区域设置为不能走

        self.walkable_field = torch.tensor(self.walkable_field_raw, device=self.device)

        walkable_x_indices, walkable_y_indices = torch.where(self.walkable_field == 0)
        self.walkable_x_coords = walkable_x_indices * self.horizontal_scale   # 这里是把像素坐标转换为米坐标
        self.walkable_y_coords = walkable_y_indices * self.horizontal_scale   # 这里是把像素坐标转换为米坐标

    def compute_flat_coords(self):
        self.flat_field_raw[: self.border, :] = 1  # 还是设置的边界区域是平坦的
        self.flat_field_raw[
            :,
            -(
                self.border
                + self.object_playground_cols
                + self.object_playground_buffer_size
            ) :,
        ] = 1
        self.flat_field_raw[:, : self.border] = 1
        self.flat_field_raw[-self.border :, :] = 1

        self.flat_field_raw = torch.tensor(self.flat_field_raw, device=self.device)

        flat_x_indices, flat_y_indices = torch.where(self.flat_field_raw == 0)
        self.flat_x_coords = flat_x_indices * self.horizontal_scale
        self.flat_y_coords = flat_y_indices * self.horizontal_scale

    def sample_valid_locations(self, num_envs):
        x_loc = np.random.randint(0, self.walkable_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.walkable_y_coords.shape[0], size=num_envs)
        valid_locs = torch.stack(
            [self.walkable_x_coords[x_loc], self.walkable_y_coords[y_loc]], dim=-1
        )

        # Raise an error if any position is invalid
        assert self.is_valid_spawn_location(
            valid_locs
        ).all(), "Invalid spawn locations detected"

        return valid_locs

    def sample_flat_locations(self, num_envs):
        x_loc = np.random.randint(0, self.flat_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.flat_y_coords.shape[0], size=num_envs)
        flat_locs = torch.stack(
            [self.flat_x_coords[x_loc], self.flat_y_coords[y_loc]], dim=-1
        )

        # Raise an error if any position is invalid
        assert self.is_valid_spawn_location(
            flat_locs
        ).all(), "Invalid flat spawn locations detected"

        return flat_locs

    def randomized_subterrains(self):
        raise NotImplementedError("Randomized subterrains not properly implemented")
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            subterrain = SubTerrain(self.config, "terrain", device=self.device)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_subterrain(
                        subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3])
                    )
                    random_uniform_subterrain(
                        subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.05,
                        downsampled_scale=0.2,
                    )
                else:
                    pyramid_sloped_subterrain(
                        subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3])
                    )
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_subterrain(
                    subterrain,
                    step_width=0.31,
                    step_height=step_height,
                    platform_size=3.0,
                )
            elif choice < 1.0:
                discrete_obstacles_subterrain(
                    subterrain, 0.15, 1.0, 2.0, 40, platform_size=3.0
                )

            self.height_field_raw[start_x:end_x, start_y:end_y] = (
                subterrain.height_field_raw
            )

    def curriculum(self, n_subterrains_per_level, n_levels):
        """
        n_subterrains_per_level : cols
        n_levels : rows

        程序化生成一个包含不同难度和类型的地形课程。

        此函数将整个地形划分为一个 `n_levels` x `n_subterrains_per_level` 的网格。
        网格的每一列（由 `level_idx` 控制）代表一个难度级别，难度随着列索引的增加而增加。
        网格的每一行（由 `subterrain_idx` 控制）代表一种不同类型的子地形，例如斜坡、
        楼梯、障碍物等。具体的地形类型是根据预设的比例（`self.proportions`）来选择的。

        参数:
            n_subterrains_per_level (int): 每个难度级别中包含的子地形类型数量。
            n_levels (int): 难度级别的数量。
        """
        for subterrain_idx in range(n_subterrains_per_level):
            for level_idx in range(n_levels):
                subterrain = SubTerrain(self.config, "terrain", device=self.device)
                difficulty = level_idx / n_levels
                choice = subterrain_idx / n_subterrains_per_level

                # Heightfield coordinate system
                start_x = self.border + level_idx * self.length_per_env_pixels
                end_x = self.border + (level_idx + 1) * self.length_per_env_pixels
                start_y = self.border + subterrain_idx * self.width_per_env_pixels
                end_y = self.border + (subterrain_idx + 1) * self.width_per_env_pixels

                slope = difficulty * 0.4  # 斜坡坡度，随难度增加而增大，最大为0.4
                step_height = 0.05 + 0.175 * difficulty  # 台阶高度，基础为0.05，随难度线性增加，最大为0.225
                discrete_obstacles_height = 0.025 + difficulty * 0.15  # 离散障碍物高度，基础为0.025，随难度线性增加，最大为0.175
                stepping_stones_size = 2 - 1.8 * difficulty  # 踏脚石尺寸，随难度增加而减小，最小为0.2
                if choice < self.proportions[0]:  # 金字塔
                    if choice < 0.05:                  # 也就是说subterrain_idx == 0 的时候是倒着的金字塔
                        slope *= -1
                    pyramid_sloped_subterrain(
                        subterrain, slope=slope, platform_size=3.0
                    )
                elif choice < self.proportions[1]:  # 金字塔+随机均匀地形
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_subterrain(
                        subterrain, slope=slope, platform_size=3.0
                    )
                    random_uniform_subterrain(
                        subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.025,
                        downsampled_scale=0.2,
                    )
                elif choice < self.proportions[3]:  # 台阶，也是金字塔，但是是台阶状的金字塔
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_subterrain(
                        subterrain,
                        step_width=0.31,
                        step_height=step_height,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[4]:  # 离散障碍物
                    discrete_obstacles_subterrain(
                        subterrain,
                        discrete_obstacles_height,
                        1.0,
                        2.0,
                        40,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[5]:  # 踏脚石 （没用）
                    stepping_stones_subterrain(
                        subterrain,
                        stone_size=stepping_stones_size,
                        stone_distance=0.1,
                        max_height=0.0,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[6]:  # 杆子 （没用）
                    poles_subterrain(subterrain=subterrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (
                        subterrain.height_field_raw != 0
                    )
                elif choice < self.proportions[7]:  # 平坦地形
                    subterrain.terrain_name = "flat"

                    flat_border = int(4 / self.horizontal_scale)

                    self.flat_field_raw[
                        start_x + flat_border : end_x - flat_border,
                        start_y + flat_border : end_y - flat_border,
                    ] = 0
                    # plain walking terrain
                    pass
                self.height_field_raw[start_x:end_x, start_y:end_y] = (
                    subterrain.height_field_raw
                )

        self.walkable_field_raw = ndimage.binary_dilation(
            self.walkable_field_raw, iterations=3
        ).astype(int)

    def mark_scene_location(self, x, y):
        """
        在场景地图上标记一个场景的位置。

        参数:
            x (int): 地形地图中的X坐标。
            y (int): 地形地图中的Y坐标。
            radius (int): 场景在地形地图中的半径。
        """
        radius = (
            math.floor(self.spacing_between_scenes * 1.0 / 2 / self.horizontal_scale)
            - 1
        )
        x_min = max(0, x - radius)
        x_max = min(self.tot_rows, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(
            self.tot_cols - self.object_playground_buffer_size, y + radius + 1
        )  # Respect the buffer

        self.scene_map[x_min:x_max, y_min:y_max] = True

    def is_valid_spawn_location(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Check if locations are valid for spawning scenes.

        Args:
            locations (torch.Tensor): Tensor of shape [B, 2] containing x, y coordinates in terrain map coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape [B] indicating valid (True) or invalid (False) spawn locations.
        """
        radius = (
            math.floor(self.spacing_between_scenes * 1.0 / 2 / self.horizontal_scale)
            - 1
        )
        batch_size = locations.shape[0]

        # Calculate boundaries
        x_min = torch.clamp(locations[:, 0] - radius, min=0)
        x_max = torch.clamp(locations[:, 0] + radius + 1, max=self.tot_rows)
        y_min = torch.clamp(locations[:, 1] - radius, min=0)
        y_max = torch.clamp(locations[:, 1] + radius + 1, max=self.tot_cols)

        # Check if the area is completely outside the valid range
        valid = (x_max > x_min) & (y_max > y_min)

        # Use advanced indexing to check all valid locations in a single operation
        for i in range(batch_size):
            if valid[i]:
                valid[i] = not self.scene_map[
                    int(x_min[i]) : int(x_max[i]), int(y_min[i]) : int(y_max[i])
                ].any()

        return valid

    def generate_terrain_plot(self, scene_lib):
        # Create the figure and subplots with fixed size and layout, arranged vertically
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(8, 24), constrained_layout=True
        )

        # 1. Plot showing the height of the terrain
        height_map = ax1.imshow(self.height_field_raw, cmap="terrain", aspect="auto")
        ax1.set_title("Terrain Height")
        fig.colorbar(height_map, ax=ax1, label="Height", shrink=0.8)

        # 2. Plot highlighting the object playground area
        object_playground_map = np.zeros_like(self.height_field_raw)
        object_playground_map[:, -(self.object_playground_cols + self.border) :] = (
            1  # Mark the entire object playground area, including the border
        )

        obj_playground_plot = ax2.imshow(
            object_playground_map, cmap="binary", interpolation="nearest", aspect="auto"
        )
        ax2.set_title("Object Playground Area")
        fig.colorbar(obj_playground_plot, ax=ax2, label="Object Playground", shrink=0.8)

        # 3. Plot marking the different regions
        region_map = np.zeros_like(self.height_field_raw)

        # Object playground
        region_map[:, -(self.object_playground_cols + self.border) :] = 1

        # Buffer region
        region_map[
            :,
            -(
                self.object_playground_cols
                + self.border
                + self.object_playground_buffer_size
            ) : -(self.object_playground_cols + self.border),
        ] = 2

        # Flat region
        flat_field_cpu = self.flat_field_raw.cpu().numpy()
        flat_region = np.where(flat_field_cpu == 0)
        region_map[flat_region] = 3

        # Irregular terrain (everything else)
        irregular_region = np.where((region_map == 0) & (self.height_field_raw != 0))
        region_map[irregular_region] = 4

        cmap = plt.cm.get_cmap("viridis", 5)
        region_plot = ax3.imshow(
            region_map,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=4,
        )
        ax3.set_title("Terrain Regions")

        # Add colorbar
        cbar = fig.colorbar(region_plot, ax=ax3, ticks=[0.5, 1.5, 2.5, 3.5], shrink=0.8)
        cbar.set_ticklabels(
            ["Object Playground", "Buffer", "Flat Region", "Irregular Terrain"]
        )

        # 4. Plot showing where objects are placed using scene_map
        scene_map_cpu = self.scene_map.cpu().numpy()
        object_plot = ax4.imshow(
            scene_map_cpu, cmap="hot", interpolation="nearest", aspect="auto"
        )
        ax4.set_title("Object Placement")
        fig.colorbar(object_plot, ax=ax4, label="Object Present", shrink=0.8)

        # Remove axis ticks for cleaner look
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Save the plot to a file
        fig.savefig("terrain_plot.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
