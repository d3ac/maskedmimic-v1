# ============================================================================
# 交互式 3D 可视化骨架 - 支持 TMR (263维) 和 Isaac (SkeletonMotion) 格式
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端

# ============================================================================
# TMR/HumanML3D 格式配置 (22 关节, Y-up 坐标系)
# ============================================================================
TMR_JOINT_NAMES = [
    "Root", "R_Hip", "L_Hip", "Spine", "R_Knee", "L_Knee", "Spine1",
    "R_Ankle", "L_Ankle", "Spine2", "R_Toe", "L_Toe", "Neck",
    "R_Collar", "L_Collar", "Head", "R_Shoulder", "L_Shoulder",
    "R_Elbow", "L_Elbow", "R_Wrist", "L_Wrist"
]

# TMR 骨骼连接
TMR_KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],      # 左腿
    [0, 1, 4, 7, 10],      # 右腿
    [0, 3, 6, 9, 12, 15],  # 脊柱
    [9, 14, 17, 19, 21],   # 左臂
    [9, 13, 16, 18, 20],   # 右臂
]


# ============================================================================
# Isaac/SMPL 格式配置 (24 关节, Z-up 坐标系)
# ============================================================================
ISAAC_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 
    'Torso', 'Spine', 'Chest', 'Neck', 'Head', 
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
]

# Isaac 骨骼连接 (根据骨架树的父子关系)
ISAAC_KINEMATIC_CHAINS = [
    [0, 5, 6, 7, 8],           # 右腿: Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Toe
    [0, 1, 2, 3, 4],           # 左腿: Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Toe
    [0, 9, 10, 11, 12, 13],    # 脊柱: Pelvis -> Torso -> Spine -> Chest -> Neck -> Head
    [11, 19, 20, 21, 22, 23],  # 右臂: Chest -> R_Thorax -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand
    [11, 14, 15, 16, 17, 18],  # 左臂: Chest -> L_Thorax -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand
]

CHAIN_COLORS = ['red', 'blue', 'green', 'orange', 'purple']


# ============================================================================
# 数据格式检测
# ============================================================================
def detect_format(data):
    """
    检测数据格式，返回 'tmr' 或 'isaac'。
    
    TMR 格式: 形状为 (T, 263) 的 numpy 数组
    Isaac 格式: 包含 'rotation', 'root_translation' 等键的字典对象
    """
    # 如果是 0-d 数组且可以 .item()，则可能是字典
    if isinstance(data, np.ndarray) and data.ndim == 0:
        try:
            item = data.item()
            if isinstance(item, dict) and 'rotation' in item:
                return 'isaac'
        except (ValueError, AttributeError):
            pass
    
    # 如果是 2D 数组且第二维是 263，则是 TMR 格式
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 263:
        return 'tmr'
    
    raise ValueError(f"无法识别的数据格式。数据类型: {type(data)}, 形状: {getattr(data, 'shape', 'N/A')}")


# ============================================================================
# TMR 格式处理函数
# ============================================================================
def recover_root_position(data):
    """
    从运动特征中恢复根节点的全局位置信息。
    TMR/HumanML3D 的特征通常包含根节点的角速度和线速度，而不是绝对位置。
    该函数通过累积速度来重建根节点的轨迹。
    """
    T = data.shape[0]
    # 0:1 -> 根节点Y轴旋转角速度 (Rotational Velocity around Y-axis)
    r_rot_vel = data[:, 0:1]
    # 1:3 -> 根节点XZ平面的线速度 (Linear Velocity in XZ plane)
    r_lin_vel = data[:, 1:3]
    # 3:4 -> 根节点Y轴高度 (Root Height)
    root_y = data[:, 3:4]

    # 计算累积旋转角度
    r_rot_ang = np.cumsum(r_rot_vel, axis=0)

    # 初始化位置数组
    r_pos = np.zeros((T, 3))
    r_pos[:, 1] = root_y[:, 0]  # Y轴高度直接使用特征值

    # 逐帧累积计算XZ平面位置
    for i in range(1, T):
        angle = r_rot_ang[i-1, 0]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        local_vel_x, local_vel_z = r_lin_vel[i-1, 0], r_lin_vel[i-1, 1]
        
        # 将局部坐标系下的速度旋转到全局坐标系并累加
        # 旋转公式 (绕Y轴旋转): 
        # global_vel_x = local_vel_x * cos(a) - local_vel_z * sin(a)
        # global_vel_z = local_vel_x * sin(a) + local_vel_z * cos(a)
        r_pos[i, 0] = r_pos[i-1, 0] + cos_a * local_vel_x - sin_a * local_vel_z
        r_pos[i, 2] = r_pos[i-1, 2] + sin_a * local_vel_x + cos_a * local_vel_z
    return r_pos

def split_motion_features(data):
    """
    拆分 263 维的动作特征向量为各个组成部分。
    支持 Numpy Array 或 PyTorch Tensor。
    
    参数:
        data: 形状为 (..., 263) 的数据
        
    返回:
        一个包含各部分数据的字典:
        - 'root_info': 根节点信息 (Y角速度, XZ线速度, 高度) [4维]
        - 'ric_data': 关节局部相对位置 (Local Joint Positions) [63维]
        - 'rot_data': 关节旋转 (Continuous 6D Rotations) [126维]
        - 'local_velocity': 关节速度 (Joint Velocities) [66维]
        - 'foot_contact': 足部接触标签 (Foot Contacts) [4维]
    """
    
    # 维度定义
    dim_root = 4
    dim_ric = 63   # (22-1) * 3
    dim_rot = 126  # (22-1) * 6
    dim_vel = 66   # 22 * 3
    dim_foot = 4
    
    # 计算切片索引
    idx_root = dim_root
    idx_ric = idx_root + dim_ric
    idx_rot = idx_ric + dim_rot
    idx_vel = idx_rot + dim_vel
    # idx_foot = idx_vel + dim_foot # Should be 263
    
    # 执行切片
    root_info = data[..., :idx_root]                # (B, T, 4)   - 根节点速度与高度
    ric_data = data[..., idx_root:idx_ric]          # (B, T, 63)  - 关节局部相对位置
    rot_data = data[..., idx_ric:idx_rot]           # (B, T, 126) - 关节旋转 6D
    local_velocity = data[..., idx_rot:idx_vel]     # (B, T, 66)  - 关节速度
    foot_contact = data[..., idx_vel:]              # (B, T, 4)   - 足部接触标签
    
    return {
        "root_info": root_info,       # [0~4]
        "ric_data": ric_data,         # [4~67]
        "rot_data": rot_data,         # [67~193]
        "local_velocity": local_velocity, # [193~259]
        "foot_contact": foot_contact  # [259~263]
    }

def extract_joint_positions_tmr(data):
    """
    从 TMR 运动特征数据中提取所有关节点的全局位置。
    数据通常包含：根节点信息 + 其他关节相对于根节点的位置 (RIC: Root-Invariant Coordinates)。
    
    返回: (T, 22, 3) 关节位置，Y-up 坐标系
    """
    T = data.shape[0]
    
    # 使用 split_motion_features 分离特征
    features = split_motion_features(data)
    
    # 1. 恢复根节点位置 (传入 root_info: 4维)
    root_pos = recover_root_position(features['root_info'])
    
    # 2. 提取并重塑其他 21 个关节的相对位置数据
    ric_data = features['ric_data'].reshape(T, 21, 3)
    
    # 3. 组合根节点和其他关节
    joint_positions = np.zeros((T, 22, 3))
    joint_positions[:, 0, :] = root_pos
    joint_positions[:, 1:, :] = ric_data
    
    return joint_positions


# ============================================================================
# Isaac 格式处理函数
# ============================================================================
def quat_to_matrix(q):
    """
    将四元数 (xyzw 格式) 转换为旋转矩阵。
    
    参数:
        q: (..., 4) 形状的四元数数组
    
    返回:
        (..., 3, 3) 形状的旋转矩阵
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # 归一化四元数
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 计算旋转矩阵元素
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # 构建旋转矩阵
    mat = np.stack([
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    
    return mat


def extract_joint_positions_isaac(data):
    """
    从 Isaac SkeletonMotion 数据中提取所有关节点的全局位置。
    使用全局旋转和骨架树的局部平移来计算关节位置。
    
    返回: (T, 24, 3) 关节位置，Z-up 坐标系
    """
    item = data.item()
    
    # 获取数据
    rotations = item['rotation']['arr']      # (T, 24, 4) 全局四元数 xyzw
    root_trans = item['root_translation']['arr']  # (T, 3) 根节点位置
    skeleton = item['skeleton_tree']
    is_local = item['is_local']
    
    # 获取骨架信息
    node_names = skeleton['node_names']
    parent_indices = skeleton['parent_indices']['arr']
    local_translations = skeleton['local_translation']['arr']  # (24, 3)
    
    T = rotations.shape[0]
    num_joints = rotations.shape[1]
    
    # 如果旋转是局部的，需要转换为全局旋转来计算位置
    # 这里假设 is_local=False，即旋转已经是全局的
    if is_local:
        print("警告: 检测到局部旋转，使用前向运动学计算全局位置")
        # 需要实现前向运动学...
        raise NotImplementedError("暂不支持局部旋转的 Isaac 格式")
    
    # 计算全局关节位置
    # 对于全局旋转，位置可以通过以下方式计算:
    # global_pos[child] = global_pos[parent] + global_rot[parent] @ local_translation[child]
    
    joint_positions = np.zeros((T, num_joints, 3))
    joint_positions[:, 0, :] = root_trans  # 根节点使用 root_translation
    
    # 按照骨架层级顺序计算每个关节的全局位置
    for joint_idx in range(1, num_joints):
        parent_idx = parent_indices[joint_idx]
        local_trans = local_translations[joint_idx]  # (3,)
        
        # 获取父节点的全局旋转矩阵
        parent_rot = quat_to_matrix(rotations[:, parent_idx])  # (T, 3, 3)
        
        # 计算子节点的全局位置
        # global_pos = parent_pos + parent_rot @ local_trans
        rotated_trans = np.einsum('tij,j->ti', parent_rot, local_trans)  # (T, 3)
        joint_positions[:, joint_idx] = joint_positions[:, parent_idx] + rotated_trans
    
    return joint_positions


# ============================================================================
# 可视化函数
# ============================================================================
def interactive_plot_tmr(joint_positions, frame_idx=0):
    """TMR 格式的交互式 3D 可视化 (Y-up 坐标系)"""
    positions = joint_positions[frame_idx]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点 (Y-up: 显示时 X->X, Z->Y, Y->Z 使高度在Z轴)
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1], c='black', s=80)
    
    # 标注关节名称
    for i, name in enumerate(TMR_JOINT_NAMES):
        ax.text(positions[i, 0], positions[i, 2], positions[i, 1], f'{i}:{name}', fontsize=8)
    
    # 绘制骨骼
    for chain_idx, chain in enumerate(TMR_KINEMATIC_CHAINS):
        color = CHAIN_COLORS[chain_idx]
        for i in range(len(chain) - 1):
            s, e = chain[i], chain[i + 1]
            ax.plot([positions[s, 0], positions[e, 0]],
                    [positions[s, 2], positions[e, 2]],
                    [positions[s, 1], positions[e, 1]], c=color, linewidth=3)
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Z (Front)')
    ax.set_zlabel('Y (Up)')
    ax.set_title(f'TMR Format - Frame {frame_idx} - Drag to rotate!')
    
    # 设置相等比例
    max_range = max(positions[:, 0].ptp(), positions[:, 2].ptp(), positions[:, 1].ptp()) / 2.0
    mid = [positions[:, 0].mean(), positions[:, 2].mean(), positions[:, 1].mean()]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    plt.show()


def interactive_plot_isaac(joint_positions, frame_idx=0):
    """Isaac 格式的交互式 3D 可视化 (Z-up 坐标系)"""
    positions = joint_positions[frame_idx]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点 (Z-up: 直接使用 X, Y, Z)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='black', s=80)
    
    # 标注关节名称
    for i, name in enumerate(ISAAC_JOINT_NAMES):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], f'{i}:{name}', fontsize=8)
    
    # 绘制骨骼
    for chain_idx, chain in enumerate(ISAAC_KINEMATIC_CHAINS):
        color = CHAIN_COLORS[chain_idx]
        for i in range(len(chain) - 1):
            s, e = chain[i], chain[i + 1]
            ax.plot([positions[s, 0], positions[e, 0]],
                    [positions[s, 1], positions[e, 1]],
                    [positions[s, 2], positions[e, 2]], c=color, linewidth=3)
    
    ax.set_xlabel('X (Front)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    ax.set_title(f'Isaac Format - Frame {frame_idx} - Drag to rotate!')
    
    # 设置相等比例
    max_range = max(positions[:, 0].ptp(), positions[:, 1].ptp(), positions[:, 2].ptp()) / 2.0
    mid = [positions[:, 0].mean(), positions[:, 1].mean(), positions[:, 2].mean()]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("turn_left01_poses_isaac.npy")
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    
    print(f"加载文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    # 自动检测格式
    fmt = detect_format(data)
    print(f"检测到格式: {fmt.upper()}")
    
    if fmt == 'tmr':
        # TMR 格式处理
        joint_positions = extract_joint_positions_tmr(data)
        print(f"总帧数: {data.shape[0]}")
        print(f"显示帧: {frame_idx}")
        print("用鼠标拖动可以旋转视角!")
        interactive_plot_tmr(joint_positions, frame_idx)
    else:
        # Isaac 格式处理
        item = data.item()
        total_frames = item['rotation']['arr'].shape[0]
        joint_positions = extract_joint_positions_isaac(data)
        print(f"总帧数: {total_frames}")
        print(f"显示帧: {frame_idx}")
        print("用鼠标拖动可以旋转视角!")
        interactive_plot_isaac(joint_positions, frame_idx)
