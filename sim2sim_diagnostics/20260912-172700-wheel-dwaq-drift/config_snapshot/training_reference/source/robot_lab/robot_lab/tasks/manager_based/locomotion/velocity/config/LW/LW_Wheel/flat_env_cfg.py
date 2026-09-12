# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
from isaaclab.utils import configclass

from .rough_env_cfg import LWWheelRoughDwaqEnvCfg, LWWheelRoughRoaEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrains.terrains_cfg import (
    DWAQ_FLAT_TERRAINS_CFG,
)

@configclass
class LWWheelFlatDwaqEnvCfg(LWWheelRoughDwaqEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = DWAQ_FLAT_TERRAINS_CFG
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        # wheel状态下承受侧向冲击能力弱
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (1.0, 1.0)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.7, 1.3)

        # Rewards
        self.rewards.base_height_l2.weight = -15.0
        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_legs.weight = -0.4
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0  # -5.0
        self.rewards.body_orientation_l2.weight = -3.0
        self.rewards.stop_motion.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.action_smoothness.weight = -0.075
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.leg_symmetry.weight = 0.5
        self.rewards.lazy_penalty.weight = -0.0
        self.rewards.same_foot_x_position.weight = -50.0
        self.rewards.feet_distance_y_exp.weight = 3.0
        self.rewards.feet_distance_penalize.weight = -150.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.50
        self.rewards.feet_distance_penalize.params["max_feet_distance"] = 0.52
        self.rewards.centrifugal_compensation.weight = 3.0 

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelFlatDwaqEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWWheelFlatDwaqEnvCfg_Play(LWWheelFlatDwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi/3, math.pi/3)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        # self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelFlatDwaqEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWWheelFlatRoaEnvCfg(LWWheelRoughRoaEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = DWAQ_FLAT_TERRAINS_CFG
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        # wheel状态下承受侧向冲击能力弱
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (1.0, 1.0)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        self.events.randomize_actuator_gains.params["distribution"] = "uniform"
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        # self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.7, 1.3)

        # Rewards
        self.rewards.base_height_l2.weight = -15.0
        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_legs.weight = -0.4
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0  # -5.0
        self.rewards.body_orientation_l2.weight = -3.0
        self.rewards.stop_motion.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.action_smoothness.weight = -0.075
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.leg_symmetry.weight = 0.5
        self.rewards.lazy_penalty.weight = -0.0
        self.rewards.same_foot_x_position.weight = -50.0
        self.rewards.feet_distance_y_exp.weight = 3.0
        self.rewards.feet_distance_penalize.weight = -150.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.50
        self.rewards.feet_distance_penalize.params["max_feet_distance"] = 0.52
        self.rewards.centrifugal_compensation.weight = 3.0 

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelFlatRoaEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWWheelFlatRoaEnvCfg_Play(LWWheelFlatRoaEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi/3, math.pi/3)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        # self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelFlatRoaEnvCfg_Play":
            self.disable_zero_weight_rewards()