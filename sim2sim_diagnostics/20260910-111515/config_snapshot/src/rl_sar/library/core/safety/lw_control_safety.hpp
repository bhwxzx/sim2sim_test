#ifndef LW_CONTROL_SAFETY_HPP
#define LW_CONTROL_SAFETY_HPP

#include "rl_sdk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

enum class LWValidationCode
{
    Valid,
    SizeMismatch,
    NonFinite,
    NegativeGain
};

enum class LWValidationField
{
    None,
    ImuQuaternion,
    ImuGyroscope,
    MotorPosition,
    MotorVelocity,
    MotorTorqueEstimate,
    PolicyAction,
    PolicyOutputPosition,
    PolicyOutputVelocity,
    PolicyOutputTorque,
    CommandPosition,
    CommandVelocity,
    CommandTorque,
    CommandKp,
    CommandKd,
    ClipActionsUpper,
    ClipActionsLower
};

inline constexpr std::string_view LWValidationFieldName(
    LWValidationField field) noexcept
{
    switch (field)
    {
        case LWValidationField::ImuQuaternion:
            return "imu.quaternion";
        case LWValidationField::ImuGyroscope:
            return "imu.gyroscope";
        case LWValidationField::MotorPosition:
            return "motor.q";
        case LWValidationField::MotorVelocity:
            return "motor.dq";
        case LWValidationField::MotorTorqueEstimate:
            return "motor.tau_est";
        case LWValidationField::PolicyAction:
            return "policy.action";
        case LWValidationField::PolicyOutputPosition:
            return "policy.output_q";
        case LWValidationField::PolicyOutputVelocity:
            return "policy.output_dq";
        case LWValidationField::PolicyOutputTorque:
            return "policy.output_tau";
        case LWValidationField::CommandPosition:
            return "command.q";
        case LWValidationField::CommandVelocity:
            return "command.dq";
        case LWValidationField::CommandTorque:
            return "command.tau";
        case LWValidationField::CommandKp:
            return "command.kp";
        case LWValidationField::CommandKd:
            return "command.kd";
        case LWValidationField::ClipActionsUpper:
            return "clip_actions_upper";
        case LWValidationField::ClipActionsLower:
            return "clip_actions_lower";
        case LWValidationField::None:
            return {};
    }
    return {};
}

struct LWValidationResult
{
    LWValidationCode code = LWValidationCode::Valid;
    LWValidationField field = LWValidationField::None;
    size_t index = 0;
    size_t expected_size = 0;
    size_t actual_size = 0;
    float value = 0.0f;

    bool valid() const noexcept
    {
        return code == LWValidationCode::Valid;
    }

    std::string failureDescription() const
    {
        std::ostringstream stream;
        if (code == LWValidationCode::SizeMismatch)
        {
            stream << LWValidationFieldName(field)
                   << " size=" << actual_size
                   << ", expected=" << expected_size;
        }
        else if (code == LWValidationCode::NonFinite)
        {
            stream << LWValidationFieldName(field)
                   << "[" << index << "] is not finite: " << value;
        }
        else if (code == LWValidationCode::NegativeGain)
        {
            stream << LWValidationFieldName(field)
                   << "[" << index << "] is negative: " << value;
        }
        return stream.str();
    }
};

inline LWValidationResult LWValidateFiniteVector(
    LWValidationField field,
    const std::vector<float>& values,
    size_t expected_size)
{
    if (values.size() != expected_size)
    {
        LWValidationResult result;
        result.code = LWValidationCode::SizeMismatch;
        result.field = field;
        result.expected_size = expected_size;
        result.actual_size = values.size();
        return result;
    }

    for (size_t index = 0; index < values.size(); ++index)
    {
        if (!std::isfinite(values[index]))
        {
            LWValidationResult result;
            result.code = LWValidationCode::NonFinite;
            result.field = field;
            result.index = index;
            result.value = values[index];
            return result;
        }
    }
    return {};
}

inline LWValidationResult LWValidateFeedbackState(
    const RobotState<float>& state,
    size_t num_dofs)
{
    struct FieldDescriptor
    {
        LWValidationField field;
        const std::vector<float>* values;
        size_t expected_size;
    };
    const std::array<FieldDescriptor, 5> fields{{
        {LWValidationField::ImuQuaternion, &state.imu.quaternion, 4},
        {LWValidationField::ImuGyroscope, &state.imu.gyroscope, 3},
        {LWValidationField::MotorPosition, &state.motor_state.q, num_dofs},
        {LWValidationField::MotorVelocity, &state.motor_state.dq, num_dofs},
        {LWValidationField::MotorTorqueEstimate,
         &state.motor_state.tau_est,
         num_dofs},
    }};

    for (const FieldDescriptor& field : fields)
    {
        const LWValidationResult result =
            LWValidateFiniteVector(
                field.field,
                *field.values,
                field.expected_size);
        if (!result.valid())
        {
            return result;
        }
    }
    return {};
}

inline LWValidationResult LWValidatePolicyActions(
    const std::vector<float>& actions,
    size_t num_dofs)
{
    return LWValidateFiniteVector(
        LWValidationField::PolicyAction,
        actions,
        num_dofs);
}

inline LWValidationResult LWValidatePolicyOutputs(
    const std::vector<float>& positions,
    const std::vector<float>& velocities,
    const std::vector<float>& torques,
    size_t num_dofs)
{
    const LWValidationResult position_result =
        LWValidateFiniteVector(
            LWValidationField::PolicyOutputPosition,
            positions,
            num_dofs);
    if (!position_result.valid())
    {
        return position_result;
    }

    const LWValidationResult velocity_result =
        LWValidateFiniteVector(
            LWValidationField::PolicyOutputVelocity,
            velocities,
            num_dofs);
    if (!velocity_result.valid())
    {
        return velocity_result;
    }

    return LWValidateFiniteVector(
        LWValidationField::PolicyOutputTorque,
        torques,
        num_dofs);
}

inline LWValidationResult LWValidateRobotCommand(
    const RobotCommand<float>& command,
    size_t num_dofs)
{
    struct FieldDescriptor
    {
        LWValidationField field;
        const std::vector<float>* values;
    };
    const std::array<FieldDescriptor, 5> fields{{
        {LWValidationField::CommandPosition, &command.motor_command.q},
        {LWValidationField::CommandVelocity, &command.motor_command.dq},
        {LWValidationField::CommandTorque, &command.motor_command.tau},
        {LWValidationField::CommandKp, &command.motor_command.kp},
        {LWValidationField::CommandKd, &command.motor_command.kd},
    }};

    for (const FieldDescriptor& field : fields)
    {
        const LWValidationResult result =
            LWValidateFiniteVector(field.field, *field.values, num_dofs);
        if (!result.valid())
        {
            return result;
        }
    }

    for (size_t index = 0; index < num_dofs; ++index)
    {
        if (command.motor_command.kp[index] < 0.0f)
        {
            LWValidationResult result;
            result.code = LWValidationCode::NegativeGain;
            result.field = LWValidationField::CommandKp;
            result.index = index;
            result.value = command.motor_command.kp[index];
            return result;
        }
        if (command.motor_command.kd[index] < 0.0f)
        {
            LWValidationResult result;
            result.code = LWValidationCode::NegativeGain;
            result.field = LWValidationField::CommandKd;
            result.index = index;
            result.value = command.motor_command.kd[index];
            return result;
        }
    }
    return {};
}

inline void LWBuildPassiveDampingCommand(
    const RobotState<float>& state,
    RobotCommand<float>& command,
    size_t num_dofs,
    float damping_gain = 5.0f)
{
    command.motor_command.resize(num_dofs);
    for (size_t index = 0; index < num_dofs; ++index)
    {
        command.motor_command.q[index] =
            index < state.motor_state.q.size()
            ? state.motor_state.q[index]
            : 0.0f;
        command.motor_command.dq[index] = 0.0f;
        command.motor_command.tau[index] = 0.0f;
        command.motor_command.kp[index] = 0.0f;
        command.motor_command.kd[index] = damping_gain;
    }
}

inline bool LWStateUsesAttitudeProtection(const std::string& state_name)
{
    return state_name == "RLFSMStateRLLocomotion_Leg"
        || state_name == "RLFSMStateRLLocomotion_Wheel"
        || state_name == "RLFSMStateRL_LegToWheel"
        || state_name == "RLFSMStateRL_WheelToLeg";
}

struct LWAttitudeValidationResult
{
    bool protection_enabled = false;
    bool safe = true;
    float roll_degrees = 0.0f;
    float pitch_degrees = 0.0f;

    std::string failureDescription(float threshold_degrees) const
    {
        std::ostringstream stream;
        stream << "roll=" << roll_degrees
               << " deg, pitch=" << pitch_degrees
               << " deg, threshold=" << threshold_degrees << " deg";
        return stream.str();
    }
};

inline LWAttitudeValidationResult LWValidateAttitude(
    const std::string& state_name,
    const std::vector<float>& quaternion,
    float threshold_degrees)
{
    LWAttitudeValidationResult result;
    result.protection_enabled = LWStateUsesAttitudeProtection(state_name);
    if (!result.protection_enabled)
    {
        return result;
    }

    if (quaternion.size() != 4 || !std::isfinite(threshold_degrees)
        || threshold_degrees <= 0.0f)
    {
        result.safe = false;
        result.roll_degrees = std::numeric_limits<float>::quiet_NaN();
        result.pitch_degrees = std::numeric_limits<float>::quiet_NaN();
        return result;
    }

    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    const float sin_roll = 2.0f * (w * x + y * z);
    const float cos_roll = 1.0f - 2.0f * (x * x + y * y);
    const float sin_pitch =
        std::clamp(2.0f * (w * y - z * x), -1.0f, 1.0f);
    constexpr float radians_to_degrees = 57.29577951308232f;
    result.roll_degrees = std::atan2(sin_roll, cos_roll) * radians_to_degrees;
    result.pitch_degrees = std::asin(sin_pitch) * radians_to_degrees;
    result.safe =
        std::isfinite(result.roll_degrees)
        && std::isfinite(result.pitch_degrees)
        && std::fabs(result.roll_degrees) <= threshold_degrees
        && std::fabs(result.pitch_degrees) <= threshold_degrees;
    return result;
}

#endif // LW_CONTROL_SAFETY_HPP
