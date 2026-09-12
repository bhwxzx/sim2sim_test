import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, DelayedPDActuatorCfg, ActuatorNetMLPCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

ARMATURE_10010 = 0.0547115
ARMATURE_8009 = 0.01537228935
ARMATURE_4310 = 0.027759656

VISCOUS_FRICTION_10010 = 0.1828
VISCOUS_FRICTION_8009 = 0.07152
VISCOUS_FRICTION_4310 = 0.369

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_10010 = ARMATURE_10010 * NATURAL_FREQ**2  # 215.99
STIFFNESS_8009 = ARMATURE_8009 * NATURAL_FREQ**2    
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2    # 109.59

DAMPING_10010 = 2.0 * DAMPING_RATIO * ARMATURE_10010 * NATURAL_FREQ # 13.75
DAMPING_8009 = 2.0 * DAMPING_RATIO * ARMATURE_8009 * NATURAL_FREQ  # 3.863
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ  # 6.977

LW_LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/LW/LW_description/LW.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            "right_hip_joint": 0.0,
            "left_hip_joint": 0.0,
            "right_thigh_joint": -0.4363,   # 25度
            "left_thigh_joint": 0.4363,
            "right_shank_joint": -2.234,    # 128度
            "left_shank_joint": 2.234,
            "right_wheel_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_foot_joint": -0.4712,    # 27度
            "left_foot_joint": 0.4712,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs":  DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_hip_joint",
                "left_hip_joint",
                "right_thigh_joint",
                "left_thigh_joint",
                "right_shank_joint",
                "left_shank_joint",
            ],
            effort_limit=120.0,
            velocity_limit=20.0,
            stiffness=90.0, # 70.0
            damping=3.0,  # 2.3
            armature=0.01,
            min_delay=0,
            max_delay=3
        ),
        # "legs": ActuatorNetMLPCfg(
        #     joint_names_expr=[
        #         "right_hip_joint",
        #         "left_hip_joint",
        #         "right_thigh_joint",
        #         "left_thigh_joint",
        #         "right_shank_joint",
        #         "left_shank_joint",
        #     ],
        #     effort_limit=120.0,
        #     velocity_limit=20.0,
        #     saturation_effort=120.0,
        #     armature=0.01,
        #     pos_scale=1.0,
        #     vel_scale=1.0,
        #     torque_scale=1.0,
        #     input_order="pos_vel",
        #     input_idx=[0, 4, 8],
        #     network_file="/home/young/liufengrong/robot_lab/scripts/tools/actuator_net/leg_actuator_net.pt"
        # ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_wheel_joint",
                "left_wheel_joint",
            ],
            effort_limit=40.0,
            velocity_limit=33.0,
            stiffness=0.0,
            damping=0.5, # 0.4
            armature=0.01,
            min_delay=0,
            max_delay=3
        ),
        "foots": DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_foot_joint",
                "left_foot_joint",
            ],
            effort_limit=27.0,
            velocity_limit=10.0,
            stiffness=28.0,   # 36.0
            damping=1.4,      # 1.8
            armature=0.01,
            min_delay=0,
            max_delay=3
        )
        # "foots": ActuatorNetMLPCfg(
        #     joint_names_expr=[
        #         "right_foot_joint",
        #         "left_foot_joint",
        #     ],
        #     effort_limit=27.0,
        #     velocity_limit=10.0,
        #     saturation_effort=27.0,
        #     armature=0.01,
        #     pos_scale=1.0,
        #     vel_scale=1.0,
        #     torque_scale=1.0,
        #     input_order="pos_vel",
        #     input_idx=[0, 4, 8],
        #     network_file="/home/young/liufengrong/robot_lab/scripts/tools/actuator_net/foot_actuator_net.pt"
        # )
    },
)


LW_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/LW/LW_description/LW.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.725), # 0.67 height
        joint_pos={
            "right_hip_joint": 0.0,
            "left_hip_joint": 0.0,
            "right_thigh_joint": 0.7330,  # 42度
            "left_thigh_joint": -0.7330,
            "right_shank_joint": -0.1745, # 10度
            "left_shank_joint": 0.1745,
            "right_wheel_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_foot_joint": 0.0,
            "left_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs":  DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_hip_joint",
                "left_hip_joint",
                "right_thigh_joint",
                "left_thigh_joint",
                "right_shank_joint",
                "left_shank_joint",
            ],
            effort_limit=120.0,
            velocity_limit=20.0,
            stiffness=90.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=6
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_wheel_joint",
                "left_wheel_joint",
            ],
            effort_limit=40.0,
            velocity_limit=33.0,
            stiffness=0.0,
            damping=0.5,
            armature=0.01,
            min_delay=0,
            max_delay=6
        ),
        "foots": DelayedPDActuatorCfg(
            joint_names_expr=[
                "right_foot_joint",
                "left_foot_joint",
            ],
            effort_limit=27.0,
            velocity_limit=10.0,
            stiffness=28.0, # 36.0
            damping=1.4,    # 1.8
            armature=0.01,
            min_delay=0,
            max_delay=6
        )
    },
)