#include "lw_sim_debug_message.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace
{
const std::array<const char*, LW_SIM_DEBUG_FIELD_COUNT> FIELD_NAMES = {
    "right_hip_now", "left_hip_now", "right_thigh_now", "left_thigh_now",
    "right_shank_now", "left_shank_now", "right_foot_now", "left_foot_now",
    "right_wheel_now", "left_wheel_now",
    "right_hip_target", "left_hip_target", "right_thigh_target", "left_thigh_target",
    "right_shank_target", "left_shank_target", "right_foot_target", "left_foot_target",
    "right_wheel_target", "left_wheel_target",
    "l_foot_x", "l_foot_y", "l_foot_z", "l_contact",
    "r_foot_x", "r_foot_y", "r_foot_z", "r_contact",
    "cmd_vel_x", "cmd_vel_yaw",
    "base_p_x", "base_p_y", "base_p_z",
    "base_v_x", "base_v_y", "base_v_z",
    "base_w_x", "base_w_y", "base_w_z",
    "base_q_w", "base_q_x", "base_q_y", "base_q_z"};

void ValidateConfig(const LWSimDebugMessageConfig& config)
{
    if (config.num_dofs != LW_SIM_DEBUG_NUM_DOFS)
    {
        throw std::invalid_argument(
            "LW Sim2Sim debug layout requires exactly 10 degrees of freedom");
    }
    if (config.joint_mapping.size()
        != static_cast<std::size_t>(config.num_dofs))
    {
        throw std::invalid_argument(
            "LW Sim2Sim debug joint mapping must contain 10 entries");
    }

    std::array<bool, LW_SIM_DEBUG_NUM_DOFS> mapped{};
    for (int index : config.joint_mapping)
    {
        if (index < 0 || index >= config.num_dofs)
        {
            throw std::invalid_argument(
                "LW Sim2Sim debug joint mapping is out of range");
        }
        if (mapped[static_cast<std::size_t>(index)])
        {
            throw std::invalid_argument(
                "LW Sim2Sim debug joint mapping must be a permutation");
        }
        mapped[static_cast<std::size_t>(index)] = true;
    }

    if (config.wheel_indices.size() != 2)
    {
        throw std::invalid_argument(
            "LW Sim2Sim debug layout requires exactly two wheel indices");
    }
    if (config.wheel_indices[0] == config.wheel_indices[1])
    {
        throw std::invalid_argument(
            "LW Sim2Sim debug wheel indices must be distinct");
    }
    for (int index : config.wheel_indices)
    {
        if (index < 0 || index >= config.num_dofs)
        {
            throw std::invalid_argument(
                "LW Sim2Sim debug wheel index is out of range");
        }
    }

    constexpr std::size_t gait_field_count = 8;
    constexpr std::size_t tracking_field_count = 15;
    static_assert(
        2U * LW_SIM_DEBUG_NUM_DOFS
            + gait_field_count + tracking_field_count
        == LW_SIM_DEBUG_FIELD_COUNT,
        "LW Sim2Sim debug offsets do not cover the fixed message layout");
}

bool HasRequiredState(const RobotState<float>& state) noexcept
{
    return state.motor_state.q.size() >= LW_SIM_DEBUG_NUM_DOFS
        && state.motor_state.dq.size() >= LW_SIM_DEBUG_NUM_DOFS
        && state.motor_state.tau_est.size() >= LW_SIM_DEBUG_NUM_DOFS;
}

bool HasRequiredCommand(const RobotCommand<float>& command) noexcept
{
    return command.motor_command.q.size() >= LW_SIM_DEBUG_NUM_DOFS
        && command.motor_command.dq.size() >= LW_SIM_DEBUG_NUM_DOFS;
}
} // namespace

LWSimDebugMessageCache::LWSimDebugMessageCache(
    const LWSimDebugMessageConfig& config)
{
    ValidateConfig(config);
    for (int index : config.wheel_indices)
    {
        wheel_mask_[static_cast<std::size_t>(index)] = true;
    }

    message_.name.reserve(FIELD_NAMES.size());
    for (const char* name : FIELD_NAMES)
    {
        message_.name.emplace_back(name);
    }
    message_.position.assign(FIELD_NAMES.size(), 0.0);
    message_.velocity.assign(FIELD_NAMES.size(), 0.0);
    message_.effort.assign(FIELD_NAMES.size(), 0.0);
}

bool LWSimDebugMessageCache::Populate(
    const RobotState<float>& robot_state,
    const RobotCommand<float>& robot_command,
    const LWControlSnapshot& control,
    const LWSimDebugTelemetry& telemetry) noexcept
{
    if (!HasRequiredState(robot_state) || !HasRequiredCommand(robot_command))
    {
        return false;
    }

    std::fill(message_.position.begin(), message_.position.end(), 0.0);
    std::fill(message_.velocity.begin(), message_.velocity.end(), 0.0);
    std::fill(message_.effort.begin(), message_.effort.end(), 0.0);

    for (std::size_t index = 0; index < LW_SIM_DEBUG_NUM_DOFS; ++index)
    {
        message_.position[kCurrentOffset + index] =
            robot_state.motor_state.q[index];
        message_.velocity[kCurrentOffset + index] =
            robot_state.motor_state.dq[index];
        message_.effort[kCurrentOffset + index] =
            robot_state.motor_state.tau_est[index];

        message_.position[kTargetOffset + index] =
            robot_command.motor_command.q[index];
        if (wheel_mask_[index])
        {
            message_.position[kTargetOffset + index] = 0.0;
            message_.velocity[kTargetOffset + index] =
                robot_command.motor_command.dq[index];
        }
    }

    if (telemetry.gait_available)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            message_.position[kGaitOffset + axis] =
                telemetry.left_foot_position[axis];
            message_.position[kGaitOffset + 4 + axis] =
                telemetry.right_foot_position[axis];
        }
        message_.position[kGaitOffset + 3] = telemetry.left_contact;
        message_.position[kGaitOffset + 7] = telemetry.right_contact;
    }

    message_.position[kTrackingOffset] = control.x;
    message_.position[kTrackingOffset + 1] = control.yaw;
    if (telemetry.frame_position_available)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            message_.position[kTrackingOffset + 2 + axis] =
                telemetry.frame_position[axis];
        }
    }
    if (telemetry.frame_velocity_available)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            message_.position[kTrackingOffset + 5 + axis] =
                telemetry.frame_velocity[axis];
        }
    }
    if (telemetry.angular_velocity_available)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            message_.position[kTrackingOffset + 8 + axis] =
                telemetry.angular_velocity[axis];
        }
    }
    if (telemetry.quaternion_available)
    {
        for (std::size_t component = 0; component < 4; ++component)
        {
            message_.position[kTrackingOffset + 11 + component] =
                telemetry.quaternion[component];
        }
    }
    return true;
}

sensor_msgs::msg::JointState& LWSimDebugMessageCache::message() noexcept
{
    return message_;
}

const sensor_msgs::msg::JointState& LWSimDebugMessageCache::message() const noexcept
{
    return message_;
}
