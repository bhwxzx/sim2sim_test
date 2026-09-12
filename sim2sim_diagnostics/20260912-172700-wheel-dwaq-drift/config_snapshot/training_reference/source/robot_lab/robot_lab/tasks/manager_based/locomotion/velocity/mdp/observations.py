# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Both selectors use asset indices. Mask wheels before selecting/reordering
    # observation columns; subtraction creates a tensor independent of asset data.
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel[:, asset_cfg.joint_ids]


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def robot_pose_z_world(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose z of the robot in world"""
    asset: Articulation = env.scene[asset_cfg.name]
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w[:, 2:3]

def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force= contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase_from_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)

def get_gait_phase_from_param(env: ManagerBasedRLEnv, gait_freq: Union[float, torch.Tensor]) -> torch.Tensor:
    """获取当前的步态相位。

    步态相位通过 [sin(phase), cos(phase)] 表示以确保连续性。
    相位根据情节步数（episode length）和传入的步频（gait frequency）计算。

    参数:
        env: 环境实例。
        gait_freq: 步频参数。可以是标量（float）或形状为 (num_envs,) 的 Tensor。

    返回:
        torch.Tensor: 步态相位观测。形状为 (num_envs, 2)。
    """
    # 检查 episode_length_buf 是否可用
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # 如果 gait_freq 是标量，确保它能与 env.num_envs 匹配
    if isinstance(gait_freq, (float, int)):
        gait_freq = torch.tensor(gait_freq, device=env.device)

    # 计算步态指数 (phase = time * frequency % 1.0)
    # env.episode_length_buf * env.step_dt 得到的是当前情节持续的时间（秒）
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * gait_freq, 1.0)
    
    # 将形状调整为 (num_envs, 1) 以进行后续计算
    gait_indices = gait_indices.view(env.num_envs, 1)
    
    # 转换为正余弦表示
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_phase_from_param_with_mask(
    env: ManagerBasedRLEnv, 
    gait_freq: Union[float, torch.Tensor],
    vel_command_name: str = "base_velocity",
    vel_threshold: float = 0.1
) -> torch.Tensor:
    """获取带有速度命令遮罩的步态相位。
    
    如果速度指令模长大于 vel_threshold，则正常输出相位；
    如果小于 vel_threshold，则直接屏蔽为 [0, 0]，避免机器人在原地站立时也出现踏步动作。
    
    参数:
        env: 环境实例。
        gait_freq: 步频参数。
        vel_command_name: 获取速度指令的 command_name。
        vel_threshold: 速度阈值（默认 0.1）。
    """
    # 检查 episode_length_buf 是否可用
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # 计算标准相位
    if isinstance(gait_freq, (float, int)):
        gait_freq = torch.tensor(gait_freq, device=env.device)

    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * gait_freq, 1.0)
    gait_indices = gait_indices.view(env.num_envs, 1)
    
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)
    phase_tensor = torch.cat([sin_phase, cos_phase], dim=-1)

    # 获取速度指令进行遮罩
    vel_cmd = env.command_manager.get_command(vel_command_name)
    cmd_norm = torch.norm(vel_cmd, dim=1, keepdim=True)
    
    is_moving = (cmd_norm > vel_threshold).float()

    return phase_tensor * is_moving


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)

def feet_lin_vel_in_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足部相对于机身的速度，并转换到机身坐标系下。"""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取足部在世界系下的线性速度 (Shape: [num_envs, num_feet, 3])
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    
    base_vel_w = asset.data.root_lin_vel_w
    
    # 使用 unsqueeze(1) 将 base_vel_w 从 [N, 3] 变为 [N, 1, 3]，以便对所有足部进行广播相减
    rel_vel_w = foot_vel_w - base_vel_w.unsqueeze(1)

    base_quat_w = asset.data.root_quat_w  # [N, 4]
    
    # 因为 rel_vel_w 是 [N, num_feet, 3]，而 base_quat_w 是 [N, 4]
    # 我们需要将四元数扩展到与足部数量一致，或者利用该函数内部的 reshape 特性
    num_feet = rel_vel_w.shape[1]
    # [N, num_feet, 4]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    rel_vel_b = quat_apply_inverse(base_quat_w_expanded, rel_vel_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return rel_vel_b.view(env.num_envs, -1)

def feet_contact_bool(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足部触地状态
    
    Args:
        sensor_cfg: 触地传感器的配置，需指定传感器名称和关联的 body_ids（足部索引）。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取历史数据 (确保配置中 history_length > 1)
    # Shape: [num_envs, history_length, num_bodies_in_sensor, 3]
    net_forces_history = sensor.data.net_forces_w_history
    
    # 提取指定部位的数据
    # Shape: [num_envs, history_length, num_feet, 3]
    feet_forces_history = net_forces_history[:, :, sensor_cfg.body_ids, :]
    
    # 计算每一帧的受力模长
    # Shape: [num_envs, history_length, num_feet]
    force_norms = torch.norm(feet_forces_history, dim=-1)
    
    # 在历史维度 (dim=1) 上取最大值
    # Shape: [num_envs, num_feet]
    max_force, _ = torch.max(force_norms, dim=1)
    contact_bool = max_force > 0.5
    
    return contact_bool.float()

def feet_pos_in_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足端相对于机身的位置（在机身坐标系下表示）。"""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取足部在世界系下的位置 (Shape: [num_envs, num_feet, 3])
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # 获取机身在世界系下的位置 (Shape: [num_envs, 3])
    base_pos_w = asset.data.root_pos_w.unsqueeze(1) # 增加维度以便广播相减
    
    rel_pos_w = foot_pos_w - base_pos_w
    
    # 获取机身姿态四元数 [N, 4]
    base_quat_w = asset.data.root_quat_w
    
    # 准备四元数以匹配足部数量 [N, num_feet, 4]
    num_feet = rel_pos_w.shape[1]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    rel_pos_b = quat_apply_inverse(base_quat_w_expanded, rel_pos_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return rel_pos_b.reshape(env.num_envs, -1)

def feet_contact_forces_in_body(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足端在机身坐标系下的三维接触力。"""
    
    # 获取传感器和机器人资源对象
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    asset: Articulation = env.scene["robot"] 
    
    # 获取世界系下的接触力 [num_envs, num_feet, 3]
    if sensor_cfg.body_ids is not None:
        forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        forces_w = sensor.data.net_forces_w
        
    # 获取机身姿态四元数 [num_envs, 4]
    base_quat_w = asset.data.root_quat_w
    
    # 准备四元数以匹配足部数量
    num_feet = forces_w.shape[1]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    # 将力从世界系旋转到机身系
    forces_b = quat_apply_inverse(base_quat_w_expanded, forces_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return forces_b.reshape(env.num_envs, -1)

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]

def randomized_base_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取经过域随机化 (Domain Randomization) 后的机器人躯干实际质量。
    
    该函数直接从底层 PhysX 视图中拉取数据，确保即使在训练过程中质量发生了随机化突变，
    也能获取到最真实、最新的物理质量参数。
    
    返回:
        torch.Tensor: 真实躯干质量，形状为 (num_envs, 1)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 从 PhysX 底层直接读取最新的根刚体质量, 通常第0个索引为躯干 (Base)
    masses = asset.root_physx_view.get_masses().clone()
    
    # 提取第 0 个 link 的质量，保持形状为 [num_envs, 1]
    base_mass = masses[:, 0].unsqueeze(1)
    return base_mass.to(env.device)

def randomized_link_masses(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取经过域随机化后，指定连杆(links)的真实物理质量。
    
    参数 `asset_cfg` 可以通过 `body_ids` 过滤出特定的连杆。
    如果不指定 `body_ids`，默认返回所有连杆的质量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 从 PhysX 视图获取该机器人的所有连杆质量, 形状通常为 (num_envs, num_bodies)
    masses = asset.root_physx_view.get_masses().clone()
    
    # 如果 SceneEntityCfg 指定了 body_ids，则只提取指定连杆的质量
    if asset_cfg.body_ids is slice(None):
        pass # 获取全部
    else:
        masses = masses[:, asset_cfg.body_ids]
        
    return masses.to(env.device)

def randomized_rigid_body_material_properties(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取经过域随机化后，指定连杆（例如左右脚）的材质属性。
    
    返回静摩擦系数、动摩擦系数和恢复系数。
    如果指定了多个连杆（通过 body_names），会将它们的材质属性展平（Flatten）后拼接在一起。
    例如：指定了左右脚 (共2个连杆)，则返回形状为 (num_envs, 2 * 3 = 6) 的张量。
    注意：此函数假设每个指定的连杆只有一个 Collision Shape。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取底层物理视图中，连杆材质的属性，形状通常为 (num_envs, max_shapes, 3)
    materials = asset.root_physx_view.get_material_properties()
    
    # 根据配置提取指定连杆的材质属性
    if asset_cfg.body_ids is slice(None):
        # 如果未指定，则默认取第 0 个连杆的属性 (通常是 base)
        material_props = materials[:, 0, :]
    else:
        # 取出指定连杆的材质属性，形状变为 (num_envs, num_selected_bodies, 3)
        # 假设：选中的每个 body 对应一个主要的 shape
        material_props = materials[:, asset_cfg.body_ids, :]
        # 将多个连杆的属性展平，变成 (num_envs, num_selected_bodies * 3)
        material_props = material_props.view(env.num_envs, -1)
        
    return material_props.clone().to(env.device)

def randomized_actuator_gains(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取经过域随机化后的关节执行器（Actuators）强度。
    
    返回拼接后的 (stiffness, damping)，形状为 (num_envs, num_selected_joints * 2)。
    由于随机化使用的是比例缩放，建议在提取后，再配置文件中通过 mathematical operation 使其中心化
    （例如除以标准刚度后减1）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    
    # 初始化全局关节刚度和阻尼张量
    stiffness_all = torch.zeros((env.num_envs, num_joints), device=asset.device)
    damping_all = torch.zeros((env.num_envs, num_joints), device=asset.device)
    
    # 遍历所有 actuator 并提取其实时增益
    # (IsaacLab 在 randomize_actuator_gains 时会更新 actuator.stiffness/damping)
    for actuator in asset.actuators.values():
        indices = actuator.joint_indices
        stiffness_all[:, indices] = actuator.stiffness
        damping_all[:, indices] = actuator.damping
        
    # 如果指定了特定关节（通过 joint_names），则进行过滤
    if asset_cfg.joint_ids is not slice(None):
        stiffness_all = stiffness_all[:, asset_cfg.joint_ids]
        damping_all = damping_all[:, asset_cfg.joint_ids]
        
    # 拼接并返回
    return torch.cat([stiffness_all, damping_all], dim=-1).to(env.device)

def randomized_base_com(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取经过域随机化后的机器人质心（CoM）位置偏移。
    
    返回形状为 (num_envs, 3) 的张量，分别代表 X, Y, Z 方向上的质心坐标。
    如果指定了特定的连杆（通过 body_ids），则返回对应连杆的质心位置。默认返回躯干 (0 号连杆) 的质心。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get_coms 返回的形状为 (num_envs, num_bodies, ...)
    # 提取指定连杆的质心 (默认 0 号代表 base)
    coms = asset.root_physx_view.get_coms().clone()
    
    if asset_cfg.body_ids is slice(None):
        base_com = coms[:, 0, :3]
    else:
        # 如果指定了多个 body，展平返回
        base_com = coms[:, asset_cfg.body_ids, :3].view(env.num_envs, -1)
        
    return base_com.to(env.device)
