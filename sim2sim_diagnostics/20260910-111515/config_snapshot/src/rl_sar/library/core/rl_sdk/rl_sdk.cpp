/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"

#include <array>

const char* LWPolicyInputStatusName(LWPolicyInputStatus status) noexcept
{
    switch (status)
    {
    case LWPolicyInputStatus::Ready: return "ready";
    case LWPolicyInputStatus::Missing: return "missing";
    case LWPolicyInputStatus::GenerationMismatch:
        return "generation-mismatch";
    case LWPolicyInputStatus::Incomplete: return "incomplete";
    case LWPolicyInputStatus::Duplicate: return "duplicate";
    case LWPolicyInputStatus::Regressing: return "regressing";
    case LWPolicyInputStatus::Future: return "future";
    case LWPolicyInputStatus::Stale: return "stale";
    }
    return "unknown";
}

LWPolicyInputStatus EvaluateLWPolicyInput(
    const LWPolicyInputSnapshot* input,
    std::uint64_t active_generation,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::duration max_age,
    std::uint64_t last_generation,
    std::uint64_t last_sequence,
    std::chrono::steady_clock::time_point last_state_capture_time) noexcept
{
    if (input == nullptr)
    {
        return LWPolicyInputStatus::Missing;
    }
    if (input->generation != active_generation)
    {
        return LWPolicyInputStatus::GenerationMismatch;
    }
    if (input->sequence == 0
        || input->state_capture_time.time_since_epoch().count() == 0)
    {
        return LWPolicyInputStatus::Incomplete;
    }
    if (now < input->state_capture_time)
    {
        return LWPolicyInputStatus::Future;
    }
    if (now - input->state_capture_time > max_age)
    {
        return LWPolicyInputStatus::Stale;
    }
    if (last_generation == active_generation)
    {
        if (input->sequence < last_sequence
            || (last_state_capture_time.time_since_epoch().count() != 0
                && input->state_capture_time < last_state_capture_time))
        {
            return LWPolicyInputStatus::Regressing;
        }
        if (input->sequence == last_sequence)
        {
            return LWPolicyInputStatus::Duplicate;
        }
        if (last_state_capture_time.time_since_epoch().count() != 0
            && input->state_capture_time == last_state_capture_time)
        {
            return LWPolicyInputStatus::Regressing;
        }
    }
    return LWPolicyInputStatus::Ready;
}

bool LWPolicyOutputPayloadComplete(
    const LWPolicyOutputFrame& output,
    size_t expected_dofs) noexcept
{
    return output.dof_pos.size() == expected_dofs
        && output.dof_vel.size() == expected_dofs
        && output.dof_tau.size() == expected_dofs;
}

LWPolicyOutputStatus EvaluateLWPolicyOutput(
    const LWPolicyOutputFrame* output,
    std::uint64_t active_generation,
    size_t expected_dofs,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::duration max_age) noexcept
{
    if (output == nullptr)
    {
        return LWPolicyOutputStatus::Missing;
    }
    if (output->generation != active_generation)
    {
        return LWPolicyOutputStatus::GenerationMismatch;
    }
    if (output->sequence == 0
        || output->source_input_sequence == 0
        || output->source_state_time.time_since_epoch().count() == 0
        || !LWPolicyOutputPayloadComplete(*output, expected_dofs))
    {
        return LWPolicyOutputStatus::Incomplete;
    }
    if (now < output->source_state_time
        || now - output->source_state_time > max_age)
    {
        return LWPolicyOutputStatus::Stale;
    }
    return LWPolicyOutputStatus::Ready;
}

void LWPolicyOutputTransport::configure(size_t num_dofs)
{
    LWPolicyOutputFrame frame;
    frame.dof_pos.resize(num_dofs);
    frame.dof_vel.resize(num_dofs);
    frame.dof_tau.resize(num_dofs);
    latest_.initialize(frame);
    publish_workspace_ = frame;
    configured_dofs_ = num_dofs;
}

bool LWPolicyOutputTransport::publish(
    const LWPolicyOutputFrame& output,
    std::uint64_t active_generation,
    size_t expected_dofs)
{
    if (output.generation != active_generation
        || configured_dofs_ != expected_dofs
        || output.source_input_sequence == 0
        || output.source_state_time.time_since_epoch().count() == 0
        || !LWPolicyOutputPayloadComplete(output, expected_dofs))
    {
        return false;
    }
    if (writer_generation_ != output.generation)
    {
        writer_generation_ = output.generation;
        last_source_input_sequence_ = 0;
        last_source_state_time_ = {};
    }
    if (output.source_input_sequence <= last_source_input_sequence_
        || (last_source_state_time_.time_since_epoch().count() != 0
            && output.source_state_time <= last_source_state_time_))
    {
        return false;
    }
    publish_workspace_ = output;
    publish_workspace_.sequence = next_sequence_++;
    last_source_input_sequence_ = output.source_input_sequence;
    last_source_state_time_ = output.source_state_time;
    latest_.publish(publish_workspace_);
    return true;
}

const LWPolicyOutputFrame* LWPolicyOutputTransport::load() noexcept
{
    return latest_.readLatest();
}

void RL::StateController(
    const RobotState<float>* state,
    RobotCommand<float>* command,
    bool apply_keyboard_velocity)
{
    auto updateState = [&](FSMState* state_node)
    {
        if (auto* rl_fsm_state = dynamic_cast<RLFSMState*>(state_node))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second.get());
    }

    fsm.Run();

    this->motiontime++;

    if (apply_keyboard_velocity)
    {
        if (this->control.current_keyboard == Input::Keyboard::W)
        {
            this->control.x += 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::S)
        {
            this->control.x -= 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::A)
        {
            this->control.y += 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::D)
        {
            this->control.y -= 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::Q)
        {
            this->control.yaw += 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::E)
        {
            this->control.yaw -= 0.1f;
        }
        if (this->control.current_keyboard == Input::Keyboard::Space)
        {
            this->control.x = 0.0f;
            this->control.y = 0.0f;
            this->control.yaw = 0.0f;
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // Maintained LW simulation and real-robot paths use the body frame.
            // Keep the world-frame conversion for compatible external adapters.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "dof_pos")
        {
            // 观测的是相对默认位置值
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            // 对于轮子相对位置大小为0
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        // ============= Other Observations =============
        else if (observation == "gait_phase")
        {
            obs_list.push_back(this->obs.gait_phase);
        }
        // else if (observation == "gait_command")
        // {
        //     obs_list.push_back(this->obs.gait_command);
        // }
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->robot_name == "LW" && this->motion_loader_lw)
            {
                auto joint_pos_sdk = this->motion_loader_lw->GetJointPos();
                auto joint_vel_sdk = this->motion_loader_lw->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("motion_joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->robot_name == "LW" && this->motion_loader_lw)
            {
                std::vector<float> robot_torso_quat_w = MotionLoaderLW::ComputeTorsoQuat(this->obs.base_quat);
                std::vector<float> ref_torso_quat_w = this->motion_loader_lw->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader_lw->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

namespace
{
using Quaternion = std::array<float, 4>;

Quaternion normalizeQuaternion(const std::vector<float>& input)
{
    const float norm = std::sqrt(
        input[0] * input[0] + input[1] * input[1]
        + input[2] * input[2] + input[3] * input[3]);
    if (norm < 1e-8f)
    {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    return {
        input[0] / norm,
        input[1] / norm,
        input[2] / norm,
        input[3] / norm};
}

Quaternion multiplyQuaternion(
    const Quaternion& left,
    const Quaternion& right)
{
    return {
        left[0] * right[0] - left[1] * right[1]
            - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0]
            + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3]
            + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2]
            - left[2] * right[1] + left[3] * right[0]};
}

Quaternion asQuaternion(const std::vector<float>& input)
{
    return {input[0], input[1], input[2], input[3]};
}

void writeInverseRotatedVector(
    const std::vector<float>& quaternion,
    const std::vector<float>& vector,
    float* output)
{
    const float q_w = quaternion[0];
    const float q_x = quaternion[1];
    const float q_y = quaternion[2];
    const float q_z = quaternion[3];
    const float v_x = vector[0];
    const float v_y = vector[1];
    const float v_z = vector[2];
    const float scale = 2.0f * q_w * q_w - 1.0f;
    const float a_x = v_x * scale;
    const float a_y = v_y * scale;
    const float a_z = v_z * scale;
    const float cross_x = q_y * v_z - q_z * v_y;
    const float cross_y = q_z * v_x - q_x * v_z;
    const float cross_z = q_x * v_y - q_y * v_x;
    const float b_x = cross_x * q_w * 2.0f;
    const float b_y = cross_y * q_w * 2.0f;
    const float b_z = cross_z * q_w * 2.0f;
    const float dot = q_x * v_x + q_y * v_y + q_z * v_z;
    const float c_x = q_x * dot * 2.0f;
    const float c_y = q_y * dot * 2.0f;
    const float c_z = q_z * dot * 2.0f;
    output[0] = a_x - b_x + c_x;
    output[1] = a_y - b_y + c_y;
    output[2] = a_z - b_z + c_z;
}

void writeFirstTwoRotationColumns(
    const Quaternion& quaternion,
    float* output)
{
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;
    output[0] = 1.0f - 2.0f * (yy + zz);
    output[1] = 2.0f * (xy - wz);
    output[2] = 2.0f * (xy + wz);
    output[3] = 1.0f - 2.0f * (xx + zz);
    output[4] = 2.0f * (xz - wy);
    output[5] = 2.0f * (yz + wx);
}
} // namespace

void RL::ComputeLWObservationInto(
    const LWPolicyRuntimeConfiguration& policy_configuration,
    const Observations<float>& policy_obs,
    const LWMotionReferenceSnapshot* motion_reference,
    std::vector<float>& output) const
{
    const auto& layout = policy_configuration.observation_layout;
    const std::size_t expected_size = layout.empty()
        ? 0
        : layout.back().offset + layout.back().size;
    if (output.size() != expected_size)
    {
        throw std::invalid_argument(
            "LW observation output has the wrong size");
    }

    for (const LWObservationLayoutEntry& entry : layout)
    {
        float* destination = output.data() + entry.offset;
        switch (entry.kind)
        {
            case LWObservationKind::AngularVelocity:
                for (std::size_t i = 0; i < entry.size; ++i)
                    destination[i] = policy_obs.ang_vel[i]
                        * policy_configuration.ang_vel_scale;
                break;
            case LWObservationKind::GravityVector:
                writeInverseRotatedVector(
                    policy_obs.base_quat,
                    policy_obs.gravity_vec,
                    destination);
                break;
            case LWObservationKind::Commands:
                for (std::size_t i = 0; i < entry.size; ++i)
                    destination[i] = policy_obs.commands[i]
                        * policy_configuration.commands_scale[i];
                break;
            case LWObservationKind::DofPosition:
                for (std::size_t i = 0; i < entry.size; ++i)
                {
                    const float relative =
                        policy_configuration.wheel_mask[i] != 0
                        ? 0.0f
                        : policy_obs.dof_pos[i]
                            - policy_configuration.default_dof_pos[i];
                    destination[i] = relative
                        * policy_configuration.dof_pos_scale;
                }
                break;
            case LWObservationKind::DofVelocity:
                for (std::size_t i = 0; i < entry.size; ++i)
                    destination[i] = policy_obs.dof_vel[i]
                        * policy_configuration.dof_vel_scale;
                break;
            case LWObservationKind::Actions:
                std::copy(
                    policy_obs.actions.begin(),
                    policy_obs.actions.end(),
                    destination);
                break;
            case LWObservationKind::GaitPhase:
                std::copy(
                    policy_obs.gait_phase.begin(),
                    policy_obs.gait_phase.end(),
                    destination);
                break;
            case LWObservationKind::MotionCommand:
                if (motion_reference == nullptr)
                {
                    std::fill(destination, destination + entry.size, 0.0f);
                    break;
                }
                for (std::size_t i = 0;
                     i < policy_configuration.motion_joint_mapping.size();
                     ++i)
                {
                    const int source =
                        policy_configuration.motion_joint_mapping[i];
                    destination[i] = motion_reference->joint_pos[source];
                    destination[i + policy_configuration.num_dofs] =
                        motion_reference->joint_vel[source];
                }
                break;
            case LWObservationKind::MotionAnchorOrientation:
                if (motion_reference == nullptr)
                {
                    std::fill(destination, destination + entry.size, 0.0f);
                    break;
                }
                {
                    const Quaternion robot =
                        normalizeQuaternion(policy_obs.base_quat);
                    const Quaternion anchor = multiplyQuaternion(
                        asQuaternion(motion_reference->init_quat),
                        asQuaternion(motion_reference->anchor_quat));
                    const Quaternion robot_conjugate =
                        {robot[0], -robot[1], -robot[2], -robot[3]};
                    writeFirstTwoRotationColumns(
                        multiplyQuaternion(robot_conjugate, anchor),
                        destination);
                }
                break;
        }
    }
    const float clip = policy_configuration.clip_obs;
    for (float& value : output)
    {
        value = clamp(value, -clip, clip);
    }
}

std::vector<float> RL::ComputeLWObservation(
    const LWPolicyRuntimeConfiguration& policy_configuration,
    Observations<float>& policy_obs,
    std::vector<int>& policy_obs_dims,
    const LWMotionReferenceSnapshot* motion_reference) const
{
    policy_obs_dims.clear();
    policy_obs_dims.reserve(policy_configuration.observation_layout.size());
    std::size_t output_size = 0;
    for (const auto& entry : policy_configuration.observation_layout)
    {
        policy_obs_dims.push_back(static_cast<int>(entry.size));
        output_size += entry.size;
    }
    std::vector<float> output(output_size);
    ComputeLWObservationInto(
        policy_configuration,
        policy_obs,
        motion_reference,
        output);
    return output;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {1.0f, 0.0f, 0.0f, 0.0f};
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.gait_phase = {0.0f, 1.0f}; // sin(0) cos(0)
    // this->obs.gait_command = this->params.Get<std::vector<float>>("gait_command");
    this->ComputeObservation();
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::SetLWBaseRuntimeConfiguration(
    LWBaseRuntimeConfiguration configuration)
{
    lw_validated_base_configuration_.reset();
    LWMotionReferenceSnapshot motion_reference;
    motion_reference.joint_pos.resize(configuration.num_dofs);
    motion_reference.joint_vel.resize(configuration.num_dofs);
    motion_reference.anchor_quat.resize(4);
    motion_reference.init_quat.resize(4);
    lw_motion_reference_.initialize(motion_reference);
    lw_policy_output_transport_.configure(configuration.num_dofs);
    lw_base_runtime_configuration_ = std::move(configuration);
}

void RL::SetLWBaseRuntimeConfiguration(
    LWValidatedBaseConfiguration configuration)
{
    auto validated = std::make_unique<const LWValidatedBaseConfiguration>(
        std::move(configuration));
    SetLWBaseRuntimeConfiguration(validated->runtime());
    lw_validated_base_configuration_ = std::move(validated);
}

const LWBaseRuntimeConfiguration&
RL::GetLWBaseRuntimeConfiguration() const
{
    if (lw_base_runtime_configuration_.num_dofs == 0)
    {
        throw std::logic_error(
            "LW base runtime configuration has not been initialized");
    }
    return lw_base_runtime_configuration_;
}

void RL::SetPolicyRoot(const std::filesystem::path& policy_root)
{
    std::error_code error;
    if (policy_root.empty()
        || !std::filesystem::is_directory(policy_root, error)
        || error)
    {
        throw std::runtime_error(
            "Policy root is not a readable directory: "
            + policy_root.string());
    }
    policy_root_ = std::filesystem::canonical(policy_root, error);
    if (error)
    {
        throw std::runtime_error(
            "Failed to resolve policy root: " + error.message());
    }
}

std::string RL::ResolvePolicyPath(
    const std::string& relative_path) const
{
    if (policy_root_.empty())
    {
        throw std::runtime_error(
            "Policy root must be configured before loading resources");
    }
    const std::filesystem::path requested(relative_path);
    const std::filesystem::path normalized = requested.lexically_normal();
    if (relative_path.empty() || requested.is_absolute()
        || normalized.empty()
        || normalized.generic_string() != relative_path)
    {
        throw std::runtime_error(
            "Invalid policy-relative path: " + relative_path);
    }
    for (const auto& component : normalized)
    {
        if (component == "." || component == "..")
        {
            throw std::runtime_error(
                "Policy path escapes the configured root: "
                + relative_path);
        }
    }
    return (policy_root_ / normalized).string();
}

void RL::PreloadModel(const std::string& robot_config_path)
{
    if (!lw_validated_base_configuration_)
    {
        SetLWBaseRuntimeConfiguration(
            LWValidatedBaseConfiguration(
                this->params.config_node,
                "LW/base.yaml"));
    }
    const std::string config_path = ResolvePolicyPath(
        robot_config_path + "/config.yaml");
    YAML::Node policy_config;
    try
    {
        policy_config = YAML::LoadFile(config_path)[robot_config_path];
    }
    catch (const YAML::Exception& exception)
    {
        throw std::runtime_error(
            "Failed to load LW policy configuration '" + config_path
            + "': " + exception.what());
    }
    const LWValidatedPolicyConfiguration validated =
        lw_validated_base_configuration_->validatePolicy(
            policy_config,
            config_path);

    const std::string& model_name = validated.runtime.model_name;
    const std::string model_path = ResolvePolicyPath(
        robot_config_path + "/" + model_name);

    std::cout << LOGGER::INFO << "Preloading ONNX model into memory: " << model_path << std::endl;

    auto model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!model)
    {
        throw std::runtime_error(
            "Failed to preload LW policy model: " + model_path);
    }

    const auto start_time = std::chrono::steady_clock::now();
    ValidateLWModelContract(
        *model,
        validated.dimensions,
        model_path,
        2);
    const auto end_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> duration_ms =
        end_time - start_time;
    std::cout << LOGGER::INFO << "Validated model and warmed it up in "
              << std::fixed << std::setprecision(3) << duration_ms.count()
              << " ms." << std::endl;

    this->preloaded_lw_policy_configs_[robot_config_path] = validated;
    this->preloaded_models_[robot_config_path] =
        std::shared_ptr<InferenceRuntime::Model>(std::move(model));
    std::cout << LOGGER::INFO << "Preload & Warmup finished for: " << robot_config_path << std::endl;
}

void RL::PreloadLWPolicyContext(
    const std::string& robot_config_path)
{
    const auto config_it =
        this->preloaded_lw_policy_configs_.find(robot_config_path);
    if (config_it == this->preloaded_lw_policy_configs_.end())
    {
        throw std::runtime_error(
            "LW policy configuration was not validated: "
            + robot_config_path);
    }

    auto definition = std::make_shared<LWPolicyDefinition>();
    definition->path = robot_config_path;
    definition->params.config_node = YAML::Clone(config_it->second.merged);
    definition->runtime = config_it->second.runtime;
    definition->dimensions = config_it->second.dimensions;

    const auto model_it =
        this->preloaded_models_.find(robot_config_path);
    if (model_it == this->preloaded_models_.end()
        || !model_it->second)
    {
        throw std::runtime_error(
            "LW policy model was not preloaded: "
            + robot_config_path);
    }
    definition->model = model_it->second;

    std::unique_ptr<MotionLoaderLW> motion_player;
    if (definition->runtime.needs_motion_reference)
    {
        const std::string motion_path = ResolvePolicyPath(
            robot_config_path + "/"
            + definition->runtime.motion_file);
        definition->prepared_motion = MotionLoaderLW::Prepare(
            motion_path,
            definition->runtime.motion_fps,
            definition->runtime.motion_time_offset_frames,
            definition->runtime.num_dofs);
        motion_player = std::make_unique<MotionLoaderLW>(
            definition->prepared_motion);
    }

    this->lw_policy_definitions_[robot_config_path] =
        std::move(definition);
    if (motion_player)
    {
        this->preloaded_lw_motion_players_[robot_config_path] =
            std::move(motion_player);
    }
    else
    {
        this->preloaded_lw_motion_players_.erase(robot_config_path);
    }
}

std::shared_ptr<const LWPolicyDefinition>
RL::GetLWPolicyDefinition(
    const std::string& robot_config_path) const
{
    const auto it =
        this->lw_policy_definitions_.find(robot_config_path);
    if (it == this->lw_policy_definitions_.end())
    {
        return nullptr;
    }
    return it->second;
}

MotionLoaderLW* RL::GetPreloadedLWMotionPlayer(
    const std::string& robot_config_path) const noexcept
{
    const auto it =
        preloaded_lw_motion_players_.find(robot_config_path);
    if (it == preloaded_lw_motion_players_.end())
    {
        return nullptr;
    }
    return it->second.get();
}

std::uint64_t RL::ActivateLWPolicy(
    const std::string& robot_config_path,
    float motion_length)
{
    const auto definition =
        GetLWPolicyDefinition(robot_config_path);
    if (!definition)
    {
        throw std::runtime_error(
            "LW policy context was not preloaded: "
            + robot_config_path);
    }

    control_policy_activation_.definition = definition.get();
    control_policy_activation_.generation =
        lw_next_policy_generation_.fetch_add(
            1,
            std::memory_order_relaxed);
    control_policy_activation_.motion_length = motion_length;
    control_policy_activation_.activated_at =
        std::chrono::steady_clock::now();
    lw_policy_activation_.publish(control_policy_activation_);
    return control_policy_activation_.generation;
}

void RL::DeactivateLWPolicy()
{
    control_policy_activation_ = {};
    lw_policy_activation_.publish(control_policy_activation_);
}

const LWPolicyActivation*
RL::LoadLWPolicyActivation() const noexcept
{
    return control_policy_activation_.definition
        ? &control_policy_activation_
        : nullptr;
}

bool RL::ReadLWPolicyActivationForInference(
    LWPolicyActivation& activation) noexcept
{
    const LWPolicyActivation* latest =
        lw_policy_activation_.readLatest();
    if (!latest)
    {
        return false;
    }
    activation = *latest;
    return true;
}

void RL::PublishLWMotionReference(
    LWMotionReferenceSnapshot reference)
{
    const auto activation = LoadLWPolicyActivation();
    if (!activation
        || reference.generation != activation->generation)
    {
        return;
    }
    lw_motion_reference_.publish(reference);
}

void RL::PublishCurrentLWMotionReference(
    std::uint64_t generation)
{
    const LWPolicyActivation* activation = LoadLWPolicyActivation();
    if (!this->motion_loader_lw
        || !activation
        || generation != activation->generation)
    {
        return;
    }
    LWMotionReferenceSnapshot& reference =
        lw_motion_reference_.producerBuffer();
    reference.generation = generation;
    this->motion_loader_lw->WriteJointPos(reference.joint_pos);
    this->motion_loader_lw->WriteJointVel(reference.joint_vel);
    this->motion_loader_lw->WriteAnchorQuat(reference.anchor_quat);
    const std::vector<float>& init_quat =
        this->motion_loader_lw->GetInitQuat();
    if (init_quat.size() != reference.init_quat.size())
    {
        throw std::runtime_error(
            "LW motion initial quaternion has the wrong size");
    }
    std::copy(
        init_quat.begin(),
        init_quat.end(),
        reference.init_quat.begin());
    lw_motion_reference_.publishProducerBuffer();
}

const LWMotionReferenceSnapshot*
RL::LoadLWMotionReference() noexcept
{
    return lw_motion_reference_.readLatest();
}

void RL::PublishLWPolicyProgress(
    std::uint64_t generation,
    std::uint64_t frame)
{
    lw_policy_progress_.publish(
        LWPolicyProgressSnapshot{generation, frame});
}

const LWPolicyProgressSnapshot*
RL::LoadLWPolicyProgress() noexcept
{
    return lw_policy_progress_.readLatest();
}

void RL::PublishLWOperatorStatus(
    LWOperatorMode mode,
    float progress) noexcept
{
    lw_operator_status_.publish(
        {lw_operator_status_sequence_.fetch_add(
             1, std::memory_order_relaxed),
         mode,
         control.x,
         control.y,
         control.yaw,
         control.gait_frequency,
         progress});
}

bool RL::ReadLWOperatorStatus(
    LWOperatorStatusSnapshot& status) const noexcept
{
    return lw_operator_status_.read(status);
}

bool RL::PublishLWPolicyOutput(
    const LWPolicyOutputFrame& output,
    const LWPolicyActivation& activation)
{
    if (!activation.definition
        || output.generation != activation.generation)
    {
        return false;
    }
    const size_t expected_dofs =
        activation.definition->runtime.num_dofs;
    return lw_policy_output_transport_.publish(
        output,
        activation.generation,
        expected_dofs);
}

const LWPolicyOutputFrame* RL::LoadLWPolicyOutput() noexcept
{
    return lw_policy_output_transport_.load();
}

std::chrono::steady_clock::duration
RL::GetLWPolicyOutputMaxAge(
    const LWPolicyActivation& activation) const
{
    const float max_age_seconds =
        activation.definition->runtime.output_max_age_seconds;
    return std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
            std::chrono::duration<float>(max_age_seconds));
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");

    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history"); 
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // ==========================================
    // 直接从内存缓存中取模型
    // ==========================================
    if (this->preloaded_models_.find(robot_config_path) != this->preloaded_models_.end())
    {
        // 瞬间切换指针
        this->model = this->preloaded_models_[robot_config_path]; 
    }
    else
    {
        // 如果没预加载，就现场读
        std::cout << LOGGER::WARNING << "Model not preloaded! Loading from disk (WILL CAUSE LAG): " << robot_config_path << std::endl;
        std::string model_path = ResolvePolicyPath(
            robot_config_path + "/"
            + this->params.Get<std::string>("model_name"));
        auto loaded_model = InferenceRuntime::ModelFactory::load_model(model_path);
        this->model = std::shared_ptr<InferenceRuntime::Model>(std::move(loaded_model));
    }

    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + robot_config_path);
    }
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;
    // 这一步力矩计算好像有问题，存疑
    // output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) + this->params.Get<std::vector<float>>("rl_kd") * (vel_actions_scaled - this->obs.dof_vel);
    // output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

void RL::ComputeLWOutput(
    const LWPolicyRuntimeConfiguration& policy_configuration,
    const Observations<float>& policy_obs,
    const std::vector<float>& actions,
    std::vector<float>& output_dof_pos,
    std::vector<float>& output_dof_vel,
    std::vector<float>& output_dof_tau) const
{
    const std::size_t num_dofs = policy_configuration.num_dofs;
    if (actions.size() != num_dofs
        || output_dof_pos.size() != num_dofs
        || output_dof_vel.size() != num_dofs
        || output_dof_tau.size() != num_dofs)
    {
        throw std::invalid_argument(
            "LW output workspace has the wrong size");
    }
    for (std::size_t index = 0; index < num_dofs; ++index)
    {
        const float scaled = actions[index]
            * policy_configuration.action_scale[index];
        const bool wheel = policy_configuration.wheel_mask[index] != 0;
        const float position_action = wheel ? 0.0f : scaled;
        const float velocity_action = wheel ? scaled : 0.0f;
        const float position_target = position_action
            + policy_configuration.default_dof_pos[index];
        output_dof_pos[index] = position_target;
        output_dof_vel[index] = velocity_action;
        output_dof_tau[index] =
            policy_configuration.rl_kp[index]
                * (position_target - policy_obs.dof_pos[index])
            + policy_configuration.rl_kd[index]
                * (velocity_action - policy_obs.dof_vel[index]);
    }
}

int RL::InverseJointMapping(int idx) const
{
    const auto& joint_mapping =
        GetLWBaseRuntimeConfiguration().joint_mapping;
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

bool RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    return TorqueProtect(origin_output_dof_tau, this->params);
}

namespace
{
bool torqueProtectWithLimits(
    const std::vector<float>& origin_output_dof_tau,
    const std::vector<float>& torque_limits)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -torque_limits[i];
        float limit_upper = torque_limits[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -torque_limits[index];
            float limit_upper = torque_limits[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        // std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
    return !out_of_range_indices.empty();
}
} // namespace

bool RL::TorqueProtect(
    const std::vector<float>& origin_output_dof_tau,
    const YamlParams& policy_params) const
{
    const auto torque_limits =
        policy_params.Get<std::vector<float>>("torque_limits");
    return torqueProtectWithLimits(origin_output_dof_tau, torque_limits);
}

bool RL::TorqueProtect(
    const std::vector<float>& origin_output_dof_tau,
    const LWPolicyRuntimeConfiguration& policy_configuration) const
{
    return torqueProtectWithLimits(
        origin_output_dof_tau,
        policy_configuration.torque_limits);
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit(int input_descriptor, bool configure_terminal)
{
    static bool initialized = false;
    static termios original_term;
    static int configured_descriptor = -1;

    // Initialize terminal to non-canonical mode on first call
    if (configure_terminal && !initialized)
    {
        if (tcgetattr(input_descriptor, &original_term) == 0)
        {
            termios new_term = original_term;
            new_term.c_lflag &= ~(ICANON | ECHO);
            new_term.c_cc[VMIN] = 0;
            new_term.c_cc[VTIME] = 0;

            if (tcsetattr(input_descriptor, TCSANOW, &new_term) == 0)
            {
                configured_descriptor = input_descriptor;
                std::atexit([]() {
                    if (configured_descriptor >= 0)
                    {
                        tcsetattr(
                            configured_descriptor,
                            TCSANOW,
                            &original_term);
                    }
                });
            }
        }
        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(input_descriptor, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface(int input_descriptor, bool configure_terminal)
{
    int c = kbhit(input_descriptor, configure_terminal);
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(input_descriptor, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(input_descriptor, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = ResolvePolicyPath(
        file_path + "/" + file_name);
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = ResolvePolicyPath(robot_path + "/motor");

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains,
    LWOperatorMode lw_status_mode)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    const auto& runtime_configuration =
        rl.GetLWBaseRuntimeConfiguration();
    int required_frames = std::max(
        1,
        static_cast<int>(
            std::ceil(duration_seconds / runtime_configuration.dt)));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    const auto& kp = use_fixed_gains
        ? runtime_configuration.fixed_kp
        : runtime_configuration.rl_kp;
    const auto& kd = use_fixed_gains
        ? runtime_configuration.fixed_kd
        : runtime_configuration.rl_kd;

    for (std::size_t i = 0;
         i < runtime_configuration.num_dofs;
         ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (lw_status_mode != LWOperatorMode::Unknown)
    {
        rl.PublishLWOperatorStatus(lw_status_mode, percent);
    }
    else if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        const auto rl_kp =
            rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd =
            rl.params.Get<std::vector<float>>("rl_kd");
        for (int i = 0;
             i < rl.params.Get<int>("num_of_dofs");
             ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            fsm_command->motor_command.kp[i] = rl_kp[i];
            fsm_command->motor_command.kd[i] = rl_kd[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}

void RLFSMState::RLControlLW()
{
    const auto activation = rl.LoadLWPolicyActivation();
    if (!activation || !activation->definition)
    {
        return;
    }

    const auto output = rl.LoadLWPolicyOutput();
    const auto now = std::chrono::steady_clock::now();
    const auto max_age =
        rl.GetLWPolicyOutputMaxAge(*activation);
    const auto& policy_configuration =
        activation->definition->runtime;
    const size_t expected_dofs = policy_configuration.num_dofs;
    LWPolicyOutputStatus status = EvaluateLWPolicyOutput(
        output,
        activation->generation,
        expected_dofs,
        now,
        max_age);

    if (status == LWPolicyOutputStatus::Ready
        && last_lw_output_generation_ == activation->generation)
    {
        if (output->sequence < last_lw_output_sequence_)
        {
            status = LWPolicyOutputStatus::Stale;
        }
        else if (output->sequence > last_lw_output_sequence_
                 && (output->source_input_sequence
                         <= last_lw_source_input_sequence_
                     || output->source_state_time
                         <= last_lw_source_state_time_))
        {
            status = LWPolicyOutputStatus::Stale;
        }
    }

    if (status != LWPolicyOutputStatus::Ready)
    {
        const bool initial_wait_expired =
            now >= activation->activated_at
            && now - activation->activated_at > max_age;
        if (LWPolicyOutputRequiresFallback(
                status, initial_wait_expired))
        {
            if (!lw_output_stale_)
            {
                lw_output_stale_ = true;
                rl.HandleLWPolicyOutputFault(status);
            }
        }
        return;
    }

    if (lw_output_stale_)
    {
        lw_output_stale_ = false;
    }

    if (last_lw_output_generation_
        != activation->generation)
    {
        last_lw_output_generation_ =
            activation->generation;
        last_lw_output_sequence_ = 0;
        last_lw_source_input_sequence_ = 0;
        last_lw_source_state_time_ = {};
    }
    if (output->sequence > last_lw_output_sequence_)
    {
        last_lw_output_sequence_ = output->sequence;
        last_lw_source_input_sequence_ =
            output->source_input_sequence;
        last_lw_source_state_time_ =
            output->source_state_time;
    }

    for (size_t i = 0; i < expected_dofs; ++i)
    {
        fsm_command->motor_command.q[i] =
            output->dof_pos[i];
        fsm_command->motor_command.dq[i] =
            output->dof_vel[i];
        fsm_command->motor_command.kp[i] = policy_configuration.rl_kp[i];
        fsm_command->motor_command.kd[i] = policy_configuration.rl_kd[i];
        fsm_command->motor_command.tau[i] = 0.0f;
    }
}
