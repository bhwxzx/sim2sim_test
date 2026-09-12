import math

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.LW import LW_WHEEL_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrains.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    DWAQ_ROUGH_TERRAINS_CFG
)
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg
)


@configclass
class LWWheelObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.02),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class LWWheelCommandsCfg(CommandsCfg):

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # (3.0, 15.0)
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class LWWheelActionsCfg(ActionsCfg):

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=1.0, use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class LWWheelRewardsCfg(RewardsCfg):

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "std": math.sqrt(0.5)},
    )

    same_foot_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=0.0, # changed to penalty mode
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=0.0)

    feet_distance_penalize = RewTerm(
        func=mdp.feet_distance_penalize,
        weight=0.0,
        params={"min_feet_distance": 0.115, "max_feet_distance": 0.51, "feet_links_name": [".*_wheel_link"]}
    )

    stop_motion = RewTerm(
        func=mdp.stop_motion,
        weight=0.0,
        params={"lin_threshold": 0.1, "ang_threshold": 0.1}
    )

    lazy_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=0.0,  # 负值
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.2,   # 指令总模长 > 0.2 时触发检测
            "vel_threshold": 0.1,   # 实际综合运动 < 0.1 时执行惩罚
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    keep_alive = RewTerm(
        func=mdp.stay_alive,
        weight=0.0,
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=0.0,  # 惩罚倾斜
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""), # 指定要对齐的部位名
        },
    )

    centrifugal_compensation = RewTerm(
        func=mdp.centrifugal_compensation_reward,
        weight=0.0,  # 正数
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

@configclass
class LWWheelRoughTeacherEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: LWWheelObservationsCfg = LWWheelObservationsCfg()
    commands: LWWheelCommandsCfg = LWWheelCommandsCfg()
    actions: LWWheelActionsCfg = LWWheelActionsCfg()
    rewards: LWWheelRewardsCfg = LWWheelRewardsCfg()

    base_link_name = "base_link"
    wheel_link_name = ".*_wheel_link"
    # fmt: off
    joint_names_without_wheels = [
        "right_hip_joint",
        "left_hip_joint",
        "right_thigh_joint",
        "left_thigh_joint",
        "right_shank_joint",
        "left_shank_joint",
        "right_foot_joint",
        "left_foot_joint",
    ]
    wheel_joint_names = [
        "right_wheel_joint",
        "left_wheel_joint",
    ]
    joint_names = joint_names_without_wheels + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator = BLIND_ROUGH_AND_STAIRS_TERRAINS_CFG
        # self.scene.terrain.max_init_terrain_level = 0

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # Preserve leg position noise while keeping wheel columns exactly zero.
        self.observations.policy.joint_pos.noise = mdp.GaussianJointPositionNoiseCfg(
            mean=0.0, std=0.01,
            joint_names=self.joint_names,
            wheel_joint_names=self.wheel_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        # self.events.push_robot_hard.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None

         # ------------------------------Rewards------------------------------
        # General
        self.rewards.upward.weight = 1.0 # 1.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.3 #-1.0
        self.rewards.ang_vel_xy_l2.weight = -0.3 # -0.05
        self.rewards.flat_orientation_l2.weight = -12.0 # -5.0
        self.rewards.base_height_l2.weight = -30.0
        self.rewards.base_height_l2.params["target_height"] = 0.683 # 0.647
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.6e-4 # 1.25e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_torques_wheel_l2.weight = -1.6e-4
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -1.5e-7 # -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-7 # -1.25e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_l2.weight = -0.03
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_vel_wheel_l2.weight = -5e-3
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_power.weight = -2e-5

        self.rewards.stop_motion.weight = -5.0

        # self.rewards.stand_still.weight = -3.0
        # self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_pos_penalty.weight = -0.1 # -2.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.3 # -0.01
        self.rewards.action_smoothness.weight = -0.03 

        # Contact sensorstand_still
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link", ".*wheel_link"]
        # self.rewards.contact_forces.weight = 0
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.2)
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Others
        self.rewards.leg_symmetry.weight = 0.5
        self.rewards.leg_symmetry.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.same_foot_x_position.weight = -50.0
        self.rewards.same_foot_x_position.params["asset_cfg"].body_names = [self.wheel_link_name]

        # self.rewards.feet_distance_y_exp.weight = 10.0
        # self.rewards.feet_distance_y_exp.params["stance_width"] = 0.42 # 0.42
        # self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.48
        self.rewards.feet_distance_penalize.params["max_feet_distance"] = 0.51
        self.rewards.feet_distance_penalize.params["feet_links_name"] = [self.wheel_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughTeacherEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

@configclass
class LWWheelRoughTeacherEnvCfg_Play(LWWheelRoughTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughTeacherEnvCfg_Play":
            self.disable_zero_weight_rewards()


@configclass
class LWWheelRoughStudentObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.02),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class TeacherCfg(ObsGroup):

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            noise=GaussianNoise(mean=0.0, std=0.05),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=GaussianNoise(mean=0.0, std=0.02),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class LWWheelRoughStudentEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: LWWheelRoughStudentObservationsCfg = LWWheelRoughStudentObservationsCfg()
    commands: LWWheelCommandsCfg = LWWheelCommandsCfg()
    actions: LWWheelActionsCfg = LWWheelActionsCfg()

    base_link_name = "base_link"
    wheel_link_name = ".*_wheel_link"
    # fmt: off
    joint_names_without_wheels = [
        "right_hip_joint",
        "left_hip_joint",
        "right_thigh_joint",
        "left_thigh_joint",
        "right_shank_joint",
        "left_shank_joint",
        "right_foot_joint",
        "left_foot_joint",
    ]
    wheel_joint_names = [
        "right_wheel_joint",
        "left_wheel_joint",
    ]
    joint_names = joint_names_without_wheels + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator = BLIND_ROUGH_AND_STAIRS_TERRAINS_CFG
        # self.scene.terrain.max_init_terrain_level = 0

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.teacher.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.teacher.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # Preserve leg position noise while keeping wheel columns exactly zero.
        self.observations.policy.joint_pos.noise = mdp.GaussianJointPositionNoiseCfg(
            mean=0.0, std=0.01,
            joint_names=self.joint_names,
            wheel_joint_names=self.wheel_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher.base_ang_vel.scale = 0.25
        self.observations.teacher.joint_pos.scale = 1.0
        self.observations.teacher.joint_vel.scale = 0.05
        self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.policy.height_scan = None
        self.observations.teacher.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        # self.events.push_robot_hard.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughTeacherEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

@configclass
class LWWheelRoughStudentEnvCfg_Play(LWWheelRoughStudentEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughStudentEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWWheelDwaqObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            # noise=GaussianNoise(mean=0.0, std=0.05),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            # noise=GaussianNoise(mean=0.0, std=0.025),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            # noise=GaussianNoise(mean=0.0, std=0.01),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            # noise=GaussianNoise(mean=0.0, std=0.02),
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = False
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 特权观测 给vae进行显式监督
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=2.0)
        # 环境参数 由隐变量估计
        robot_pose_z_world = ObsTerm(func=mdp.robot_pose_z_world)
        feet_contact = ObsTerm(func=mdp.feet_contact_bool, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"])})
        feet_pos_in_body = ObsTerm(func=mdp.feet_pos_in_body, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"])}, scale=1.0)
        feet_lin_vel_in_body = ObsTerm(func=mdp.feet_lin_vel_in_body, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"])},
                                       scale=1.0)
        # feet_contact_forces_in_body = ObsTerm(func=mdp.feet_contact_forces_in_body, 
        #                                       params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"])},
        #                                       scale=0.01)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class LWWheelRoughDwaqEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: LWWheelDwaqObservationsCfg = LWWheelDwaqObservationsCfg()
    commands: LWWheelCommandsCfg = LWWheelCommandsCfg()
    actions: LWWheelActionsCfg = LWWheelActionsCfg()
    rewards: LWWheelRewardsCfg = LWWheelRewardsCfg()
    # curriculum: LWLegCurriculumCfg = LWLegCurriculumCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"
    wheel_link_name = ".*_wheel_link"
    # fmt: off
    joint_names_without_wheels = [
        "right_hip_joint",
        "left_hip_joint",
        "right_thigh_joint",
        "left_thigh_joint",
        "right_shank_joint",
        "left_shank_joint",
        "right_foot_joint",
        "left_foot_joint",
    ]
    wheel_joint_names = [
        "right_wheel_joint",
        "left_wheel_joint",
    ]
    joint_names = joint_names_without_wheels + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=(1.6, 1.0))
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.base_contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # filter_prim_paths_expr 这里必须一个字符串对应一个物体，否则会报错
        self.scene.base_contact_forces.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/right_foot_link", "{ENV_REGEX_NS}/Robot/left_foot_link"]
        self.scene.terrain.terrain_generator = DWAQ_ROUGH_TERRAINS_CFG

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # Preserve leg position noise while keeping wheel columns exactly zero.
        self.observations.policy.joint_pos.noise = mdp.UniformJointPositionNoiseCfg(
            n_min=-0.01, n_max=0.01,
            joint_names=self.joint_names,
            wheel_joint_names=self.wheel_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.policy.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]

        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)}
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.add_joint_default_pos.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        
        self.events.push_robot_hard = None
        # self.events.randomize_apply_external_force_torque = None 

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0
        self.rewards.keep_alive.weight = 1.0
        # self.rewards.upward.weight = 1.0 # 1.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.5 #-1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0 # -5.0
        self.rewards.base_height_l2.weight = -10.0 # -50.0 
        self.rewards.base_height_l2.params["target_height"] = 0.67
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_orientation_l2.weight = -2.0
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.25e-5 # -1.25e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_torques_wheel_l2.weight = -1.6e-4 # -1.6e-4
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -2.5e-7 # -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-7 # -1.25e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_l2.weight = -5e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_vel_wheel_l2.weight = -5e-4 # -5e-3
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_limits.weight = -2.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_power.weight = -2e-5

        self.rewards.stop_motion.weight = -3.0

        # self.rewards.stand_still.weight = -3.0

        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.joint_deviation_ankle.weight = -1.0
        self.rewards.joint_deviation_ankle.params["asset_cfg"].joint_names = [".*_foot_joint"]
        self.rewards.joint_deviation_legs.weight = -0.02
        self.rewards.joint_deviation_legs.params["asset_cfg"].joint_names = [".*_thigh_joint",".*_shank_joint"]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.02 # -0.01 
        self.rewards.action_smoothness.weight = -0.02 # -0.15 

        # Contact sensorstand_still
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link", ".*foot_link"]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 8.0 # 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        # self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.2)
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.lazy_penalty.weight = -2.0

        # Others
        self.rewards.leg_symmetry.weight = 0.5
        self.rewards.leg_symmetry.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.same_foot_x_position.weight = -50.0
        self.rewards.same_foot_x_position.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_stumble.weight = -10.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_distance_y_exp.weight = 5.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.516
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.48
        self.rewards.feet_distance_penalize.params["max_feet_distance"] = 0.53

        self.rewards.centrifugal_compensation.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughDwaqEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi/2, math.pi/2)

@configclass
class LWWheelRoughDwaqEnvCfg_Play(LWWheelRoughDwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi/3, math.pi/3)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughDwaqEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWWheelRoaObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            # noise=GaussianNoise(mean=0.0, std=0.05),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            # noise=GaussianNoise(mean=0.0, std=0.025),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            # noise=GaussianNoise(mean=0.0, std=0.01),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            # noise=GaussianNoise(mean=0.0, std=0.02),
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = False
            self.history_length = 10

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )


        # 特权观测
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=2.0)
        base_mass = ObsTerm(func=mdp.randomized_base_mass,clip=(-100.0, 100.0),scale=0.1)
        feet_material = ObsTerm(
            func=mdp.randomized_rigid_body_material_properties,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link")},
            clip=(-100.0, 100.0),scale=1.0
        )
        actuator_gains = ObsTerm(
            func=mdp.randomized_actuator_gains,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            clip=(-100.0, 100.0),scale=0.01
        )
        base_com = ObsTerm(
            func=mdp.randomized_base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            clip=(-100.0, 100.0),scale=1.0
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=2.0)
        base_mass = ObsTerm(func=mdp.randomized_base_mass,clip=(-100.0, 100.0),scale=0.1)
        feet_material = ObsTerm(
            func=mdp.randomized_rigid_body_material_properties,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link")},
            clip=(-100.0, 100.0),scale=1.0
        )
        actuator_gains = ObsTerm(
            func=mdp.randomized_actuator_gains,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            clip=(-100.0, 100.0),scale=0.01
        )
        base_com = ObsTerm(
            func=mdp.randomized_base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            clip=(-100.0, 100.0),scale=1.0
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class LWWheelRoughRoaEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: LWWheelRoaObservationsCfg = LWWheelRoaObservationsCfg()
    commands: LWWheelCommandsCfg = LWWheelCommandsCfg()
    actions: LWWheelActionsCfg = LWWheelActionsCfg()
    rewards: LWWheelRewardsCfg = LWWheelRewardsCfg()
    # curriculum: LWLegCurriculumCfg = LWLegCurriculumCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"
    wheel_link_name = ".*_wheel_link"
    # fmt: off
    joint_names_without_wheels = [
        "right_hip_joint",
        "left_hip_joint",
        "right_thigh_joint",
        "left_thigh_joint",
        "right_shank_joint",
        "left_shank_joint",
        "right_foot_joint",
        "left_foot_joint",
    ]
    wheel_joint_names = [
        "right_wheel_joint",
        "left_wheel_joint",
    ]
    joint_names = joint_names_without_wheels + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=(1.6, 1.0))
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.base_contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # filter_prim_paths_expr 这里必须一个字符串对应一个物体，否则会报错
        self.scene.base_contact_forces.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/right_foot_link", "{ENV_REGEX_NS}/Robot/left_foot_link"]
        self.scene.terrain.terrain_generator = DWAQ_ROUGH_TERRAINS_CFG

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # Preserve leg position noise while keeping wheel columns exactly zero.
        self.observations.policy.joint_pos.noise = mdp.UniformJointPositionNoiseCfg(
            n_min=-0.01, n_max=0.01,
            joint_names=self.joint_names,
            wheel_joint_names=self.wheel_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.privileged.actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.policy.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]

        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)}
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.add_joint_default_pos.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        
        self.events.push_robot_hard = None
        # self.events.randomize_apply_external_force_torque = None 

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0
        self.rewards.keep_alive.weight = 1.0
        # self.rewards.upward.weight = 1.0 # 1.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.5 #-1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0 # -5.0
        self.rewards.base_height_l2.weight = -10.0 # -50.0 
        self.rewards.base_height_l2.params["target_height"] = 0.67
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_orientation_l2.weight = -2.0
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.25e-5 # -1.25e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_torques_wheel_l2.weight = -1.6e-4 # -1.6e-4
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -2.5e-7 # -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-7 # -1.25e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_l2.weight = -5e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_vel_wheel_l2.weight = -5e-4 # -5e-3
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_limits.weight = -2.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_power.weight = -2e-5

        self.rewards.stop_motion.weight = -3.0

        # self.rewards.stand_still.weight = -3.0

        self.rewards.joint_deviation_hip.weight = -0.3
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.joint_deviation_ankle.weight = -1.0
        self.rewards.joint_deviation_ankle.params["asset_cfg"].joint_names = [".*_foot_joint"]
        self.rewards.joint_deviation_legs.weight = -0.02
        self.rewards.joint_deviation_legs.params["asset_cfg"].joint_names = [".*_thigh_joint",".*_shank_joint"]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.02 # -0.01 
        self.rewards.action_smoothness.weight = -0.02 # -0.15 

        # Contact sensorstand_still
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link", ".*foot_link"]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 8.0 # 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        # self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.2)
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.lazy_penalty.weight = -2.0

        # Others
        self.rewards.leg_symmetry.weight = 0.5
        self.rewards.leg_symmetry.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.same_foot_x_position.weight = -50.0
        self.rewards.same_foot_x_position.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_stumble.weight = -10.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_distance_y_exp.weight = 5.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.516
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.wheel_link_name]
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.48
        self.rewards.feet_distance_penalize.params["max_feet_distance"] = 0.53

        self.rewards.centrifugal_compensation.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughRoaEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi/2, math.pi/2)

@configclass
class LWWheelRoughRoaEnvCfg_Play(LWWheelRoughRoaEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi/3, math.pi/3)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWWheelRoughRoaEnvCfg_Play":
            self.disable_zero_weight_rewards()
