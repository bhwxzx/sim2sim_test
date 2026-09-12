#include "lw_configuration_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
[[noreturn]] void fail(
    const std::string& source,
    const std::string& message)
{
    throw std::runtime_error(
        "Invalid LW configuration '" + source + "': " + message);
}

YAML::Node requireNode(
    const YAML::Node& config,
    const std::string& key,
    const std::string& source)
{
    const YAML::Node node = config[key];
    if (!node.IsDefined() || node.IsNull())
    {
        fail(source, "missing required key '" + key + "'");
    }
    return node;
}

template<typename T>
T requireValue(
    const YAML::Node& config,
    const std::string& key,
    const std::string& source)
{
    try
    {
        return requireNode(config, key, source).as<T>();
    }
    catch (const YAML::Exception& exception)
    {
        fail(
            source,
            "key '" + key + "' has the wrong type: " + exception.what());
    }
}

float requireFinite(
    const YAML::Node& config,
    const std::string& key,
    const std::string& source)
{
    const float value = requireValue<float>(config, key, source);
    if (!std::isfinite(value))
    {
        fail(source, "key '" + key + "' must be finite");
    }
    return value;
}

float requirePositiveFinite(
    const YAML::Node& config,
    const std::string& key,
    const std::string& source)
{
    const float value = requireFinite(config, key, source);
    if (value <= 0.0f)
    {
        fail(source, "key '" + key + "' must be positive");
    }
    return value;
}

std::vector<float> requireFiniteVector(
    const YAML::Node& config,
    const std::string& key,
    std::size_t expected_size,
    const std::string& source)
{
    const auto values =
        requireValue<std::vector<float>>(config, key, source);
    if (values.size() != expected_size)
    {
        fail(
            source,
            "key '" + key + "' must contain "
                + std::to_string(expected_size) + " values, got "
                + std::to_string(values.size()));
    }
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (!std::isfinite(values[index]))
        {
            fail(
                source,
                "key '" + key + "' contains a non-finite value at index "
                    + std::to_string(index));
        }
    }
    return values;
}

void requireNonnegativeVector(
    const YAML::Node& config,
    const std::string& key,
    std::size_t expected_size,
    const std::string& source)
{
    const auto values =
        requireFiniteVector(config, key, expected_size, source);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (values[index] < 0.0f)
        {
            fail(
                source,
                "key '" + key + "' must be nonnegative at index "
                    + std::to_string(index));
        }
    }
}

void requirePositiveVector(
    const YAML::Node& config,
    const std::string& key,
    std::size_t expected_size,
    const std::string& source)
{
    const auto values =
        requireFiniteVector(config, key, expected_size, source);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (values[index] <= 0.0f)
        {
            fail(
                source,
                "key '" + key + "' must be positive at index "
                    + std::to_string(index));
        }
    }
}

std::vector<int> requireUniqueIndices(
    const YAML::Node& config,
    const std::string& key,
    std::size_t expected_size,
    std::size_t upper_bound,
    const std::string& source)
{
    const auto values =
        requireValue<std::vector<int>>(config, key, source);
    if (values.size() != expected_size)
    {
        fail(
            source,
            "key '" + key + "' must contain "
                + std::to_string(expected_size) + " indices, got "
                + std::to_string(values.size()));
    }

    std::set<int> unique;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const int value = values[index];
        if (value < 0 || static_cast<std::size_t>(value) >= upper_bound)
        {
            fail(
                source,
                "key '" + key + "' index " + std::to_string(value)
                    + " at position " + std::to_string(index)
                    + " is outside [0, "
                    + std::to_string(upper_bound) + ")");
        }
        if (!unique.insert(value).second)
        {
            fail(
                source,
                "key '" + key + "' contains duplicate index "
                    + std::to_string(value));
        }
    }
    return values;
}

void requireUniqueStrings(
    const YAML::Node& config,
    const std::string& key,
    std::size_t expected_size,
    const std::string& source)
{
    const auto values =
        requireValue<std::vector<std::string>>(config, key, source);
    if (values.size() != expected_size)
    {
        fail(
            source,
            "key '" + key + "' must contain "
                + std::to_string(expected_size) + " values, got "
                + std::to_string(values.size()));
    }
    std::set<std::string> unique;
    for (const std::string& value : values)
    {
        if (value.empty())
        {
            fail(source, "key '" + key + "' contains an empty value");
        }
        if (!unique.insert(value).second)
        {
            fail(
                source,
                "key '" + key + "' contains duplicate value '"
                    + value + "'");
        }
    }
}

YAML::Node mergeConfiguration(
    const YAML::Node& base_config,
    const YAML::Node& policy_config)
{
    YAML::Node merged = YAML::Clone(base_config);
    for (auto iterator = policy_config.begin();
         iterator != policy_config.end();
         ++iterator)
    {
        merged[iterator->first.as<std::string>()] =
            YAML::Clone(iterator->second);
    }
    return merged;
}

LWObservationKind observationKind(
    const std::string& observation,
    const std::string& source)
{
    if (observation == "ang_vel") return LWObservationKind::AngularVelocity;
    if (observation == "gravity_vec") return LWObservationKind::GravityVector;
    if (observation == "commands") return LWObservationKind::Commands;
    if (observation == "dof_pos") return LWObservationKind::DofPosition;
    if (observation == "dof_vel") return LWObservationKind::DofVelocity;
    if (observation == "actions") return LWObservationKind::Actions;
    if (observation == "gait_phase") return LWObservationKind::GaitPhase;
    if (observation == "whole_body_tracking/motion_command")
        return LWObservationKind::MotionCommand;
    if (observation == "whole_body_tracking/motion_anchor_ori_b")
        return LWObservationKind::MotionAnchorOrientation;
    fail(source, "unsupported observation '" + observation + "'");
}

std::size_t observationDimension(
    LWObservationKind kind,
    std::size_t num_dofs)
{
    switch (kind)
    {
        case LWObservationKind::AngularVelocity:
        case LWObservationKind::GravityVector:
        case LWObservationKind::Commands:
            return 3;
        case LWObservationKind::DofPosition:
        case LWObservationKind::DofVelocity:
        case LWObservationKind::Actions:
            return num_dofs;
        case LWObservationKind::GaitPhase:
            return 2;
        case LWObservationKind::MotionCommand:
            return 2 * num_dofs;
        case LWObservationKind::MotionAnchorOrientation:
            return 6;
    }
    throw std::logic_error("unsupported LW observation kind");
}

void validateTensor(
    const InferenceRuntime::TensorMetadata& tensor,
    std::size_t expected_features,
    const std::string& role,
    const std::string& source)
{
    if (tensor.element_type != InferenceRuntime::TensorElementType::Float32)
    {
        fail(source, "ONNX " + role + " tensor must use float32");
    }
    if (tensor.shape.size() != 2)
    {
        fail(
            source,
            "ONNX " + role + " tensor must have rank 2, got rank "
                + std::to_string(tensor.shape.size()));
    }
    if (tensor.shape[0] != 1)
    {
        fail(
            source,
            "ONNX " + role + " batch dimension must be fixed at 1, got "
                + std::to_string(tensor.shape[0]));
    }
    if (tensor.shape[1] <= 0)
    {
        fail(
            source,
            "ONNX " + role
                + " feature dimension must be fixed and positive");
    }
    if (tensor.shape[1] != static_cast<std::int64_t>(expected_features))
    {
        fail(
            source,
            "ONNX " + role + " feature dimension expected "
                + std::to_string(expected_features) + ", got "
                + std::to_string(tensor.shape[1]));
    }
}
} // namespace

LWBaseRuntimeConfiguration ValidateLWBaseConfiguration(
    const YAML::Node& config,
    const std::string& source)
{
    if (!config || !config.IsMap())
    {
        fail(source, "top-level robot configuration must be a map");
    }

    const int num_dofs = requireValue<int>(config, "num_of_dofs", source);
    if (num_dofs <= 0)
    {
        fail(source, "key 'num_of_dofs' must be positive");
    }
    const std::size_t dofs = static_cast<std::size_t>(num_dofs);

    requirePositiveFinite(config, "dt", source);
    const int decimation = requireValue<int>(config, "decimation", source);
    if (decimation <= 0)
    {
        fail(source, "key 'decimation' must be positive");
    }
    LWBaseRuntimeConfiguration runtime;
    runtime.sensor_timeout = requirePositiveFinite(config, "sensor_timeout", source);
    runtime.trusted_imu_timeout = requirePositiveFinite(config, "trusted_imu_timeout", source);
    runtime.imu_ahrs_pair_max_age = requirePositiveFinite(config, "imu_ahrs_pair_max_age", source);
    runtime.serial_write_timeout = requirePositiveFinite(config, "serial_write_timeout", source);

    const int cpu = requireValue<int>(config, "control_loop_cpu", source);
    if (cpu < -1)
    {
        fail(source, "key 'control_loop_cpu' must be -1 or nonnegative");
    }
    const int realtime_priority =
        requireValue<int>(config, "control_loop_realtime_priority", source);
    if (realtime_priority < 0)
    {
        fail(
            source,
            "key 'control_loop_realtime_priority' must be nonnegative");
    }
    const bool require_realtime =
        requireValue<bool>(config, "control_loop_require_realtime", source);
    if (require_realtime && realtime_priority == 0)
    {
        fail(
            source,
            "control_loop_require_realtime=true requires a positive "
            "control_loop_realtime_priority");
    }

    for (const std::string key : {
             "control_loop_degraded_consecutive_misses",
             "control_loop_fatal_consecutive_misses"})
    {
        if (requireValue<int>(config, key, source) < 0)
        {
            fail(source, "key '" + key + "' must be nonnegative");
        }
    }
    for (const std::string key : {
             "control_loop_degraded_lateness",
             "control_loop_fatal_lateness"})
    {
        if (requireFinite(config, key, source) < 0.0f)
        {
            fail(source, "key '" + key + "' must be nonnegative");
        }
    }

    requireNonnegativeVector(config, "rl_kp", dofs, source);
    requireNonnegativeVector(config, "rl_kd", dofs, source);
    requireNonnegativeVector(config, "fixed_kp", dofs, source);
    requireNonnegativeVector(config, "fixed_kd", dofs, source);
    requirePositiveVector(config, "torque_limits", dofs, source);
    requireFiniteVector(config, "default_dof_pos_leg", dofs, source);
    requireFiniteVector(config, "default_dof_pos_wheel", dofs, source);
    requireFiniteVector(config, "gait_command", 3, source);
    requireFiniteVector(config, "vel_command", 3, source);
    requireUniqueStrings(config, "joint_names", dofs, source);
    requireUniqueStrings(config, "joint_controller_names", dofs, source);
    requireUniqueIndices(config, "joint_mapping", dofs, dofs, source);
    requireUniqueIndices(config, "wheel_indices", 2, dofs, source);

    runtime.num_dofs = dofs;
    runtime.dt = config["dt"].as<float>();
    runtime.decimation = config["decimation"].as<int>();
    runtime.joint_names =
        config["joint_names"].as<std::vector<std::string>>();
    runtime.joint_mapping = config["joint_mapping"].as<std::vector<int>>();
    runtime.wheel_indices = config["wheel_indices"].as<std::vector<int>>();
    runtime.wheel_mask.assign(dofs, 0);
    for (const int index : runtime.wheel_indices)
    {
        runtime.wheel_mask[static_cast<std::size_t>(index)] = 1;
    }
    runtime.rl_kp = config["rl_kp"].as<std::vector<float>>();
    runtime.rl_kd = config["rl_kd"].as<std::vector<float>>();
    runtime.fixed_kp = config["fixed_kp"].as<std::vector<float>>();
    runtime.fixed_kd = config["fixed_kd"].as<std::vector<float>>();
    runtime.torque_limits = config["torque_limits"].as<std::vector<float>>();
    runtime.default_dof_pos_leg =
        config["default_dof_pos_leg"].as<std::vector<float>>();
    runtime.default_dof_pos_wheel =
        config["default_dof_pos_wheel"].as<std::vector<float>>();
    runtime.gait_command = config["gait_command"].as<std::vector<float>>();
    runtime.vel_command = config["vel_command"].as<std::vector<float>>();
    return runtime;
}

LWValidatedBaseConfiguration::LWValidatedBaseConfiguration(
    const YAML::Node& config, const std::string& source)
    : config_(YAML::Clone(config)),
      runtime_(ValidateLWBaseConfiguration(config_, source))
{
}

LWValidatedPolicyConfiguration ValidateLWPolicyConfiguration(
    const YAML::Node& base_config,
    const YAML::Node& policy_config,
    const std::string& source)
{
    return LWValidatedBaseConfiguration(base_config, "LW/base.yaml")
        .validatePolicy(policy_config, source);
}

LWValidatedPolicyConfiguration LWValidatedBaseConfiguration::validatePolicy(
    const YAML::Node& policy_config, const std::string& source) const
{
    if (!policy_config || !policy_config.IsMap())
    {
        fail(source, "policy configuration must be a map");
    }

    const std::vector<std::string> required_policy_keys = {
        "model_name",
        "num_observations",
        "observations",
        "observations_history",
        "observations_history_priority",
        "clip_obs",
        "clip_actions_lower",
        "clip_actions_upper",
        "rl_kp",
        "rl_kd",
        "fixed_kp",
        "fixed_kd",
        "num_of_dofs",
        "action_scale",
        "wheel_indices",
        "torque_limits",
        "default_dof_pos",
        "gait_command",
        "vel_command",
        "joint_mapping",
    };
    for (const std::string& key : required_policy_keys)
    {
        requireNode(policy_config, key, source);
    }

    YAML::Node merged = mergeConfiguration(config_, policy_config);
    const int base_num_dofs = static_cast<int>(runtime_.num_dofs);
    const int policy_num_dofs =
        requireValue<int>(policy_config, "num_of_dofs", source);
    if (policy_num_dofs != base_num_dofs)
    {
        fail(
            source,
            "num_of_dofs expected " + std::to_string(base_num_dofs)
                + ", got " + std::to_string(policy_num_dofs));
    }
    const std::size_t dofs = static_cast<std::size_t>(policy_num_dofs);

    const std::string model_name =
        requireValue<std::string>(policy_config, "model_name", source);
    if (model_name.empty())
    {
        fail(source, "key 'model_name' must not be empty");
    }

    requirePositiveFinite(merged, "dt", source);
    if (requireValue<int>(merged, "decimation", source) <= 0)
    {
        fail(source, "key 'decimation' must be positive");
    }
    requirePositiveFinite(policy_config, "clip_obs", source);
    requireNonnegativeVector(policy_config, "rl_kp", dofs, source);
    requireNonnegativeVector(policy_config, "rl_kd", dofs, source);
    requireNonnegativeVector(policy_config, "fixed_kp", dofs, source);
    requireNonnegativeVector(policy_config, "fixed_kd", dofs, source);
    requireFiniteVector(policy_config, "action_scale", dofs, source);
    requirePositiveVector(policy_config, "torque_limits", dofs, source);
    requireFiniteVector(policy_config, "default_dof_pos", dofs, source);
    requireFiniteVector(policy_config, "gait_command", 3, source);
    requireFiniteVector(policy_config, "vel_command", 3, source);
    requireUniqueIndices(policy_config, "joint_mapping", dofs, dofs, source);
    requireUniqueIndices(policy_config, "wheel_indices", 2, dofs, source);

    const auto lower = requireFiniteVector(
        policy_config, "clip_actions_lower", dofs, source);
    const auto upper = requireFiniteVector(
        policy_config, "clip_actions_upper", dofs, source);
    for (std::size_t index = 0; index < dofs; ++index)
    {
        if (lower[index] > upper[index])
        {
            fail(
                source,
                "clip_actions_lower exceeds clip_actions_upper at index "
                    + std::to_string(index));
        }
    }

    const auto observations = requireValue<std::vector<std::string>>(
        policy_config, "observations", source);
    if (observations.empty())
    {
        fail(source, "key 'observations' must not be empty");
    }
    std::set<std::string> unique_observations;
    std::size_t observation_dimension = 0;
    std::vector<LWObservationLayoutEntry> observation_layout;
    observation_layout.reserve(observations.size());
    bool needs_motion = false;
    bool needs_motion_command = false;
    bool needs_motion_anchor_orientation = false;
    for (const std::string& observation : observations)
    {
        if (!unique_observations.insert(observation).second)
        {
            fail(
                source,
                "key 'observations' contains duplicate term '"
                    + observation + "'");
        }
        const LWObservationKind kind = observationKind(observation, source);
        const std::size_t term_dimension = observationDimension(kind, dofs);
        if (observation_dimension
            > std::numeric_limits<std::size_t>::max() - term_dimension)
        {
            fail(source, "computed observation dimension overflows");
        }
        observation_layout.push_back(
            {kind, observation_dimension, term_dimension});
        observation_dimension += term_dimension;

        if (observation == "ang_vel")
        {
            requireFinite(policy_config, "ang_vel_scale", source);
        }
        else if (observation == "commands")
        {
            requireFiniteVector(policy_config, "commands_scale", 3, source);
        }
        else if (observation == "dof_pos")
        {
            requireFinite(policy_config, "dof_pos_scale", source);
        }
        else if (observation == "dof_vel")
        {
            requireFinite(policy_config, "dof_vel_scale", source);
        }
        else if (observation == "whole_body_tracking/motion_command"
                 || observation
                        == "whole_body_tracking/motion_anchor_ori_b")
        {
            needs_motion = true;
            needs_motion_command = needs_motion_command
                || kind == LWObservationKind::MotionCommand;
            needs_motion_anchor_orientation =
                needs_motion_anchor_orientation
                || kind == LWObservationKind::MotionAnchorOrientation;
        }
    }

    const int configured_observations =
        requireValue<int>(policy_config, "num_observations", source);
    if (configured_observations <= 0
        || static_cast<std::size_t>(configured_observations)
               != observation_dimension)
    {
        fail(
            source,
            "num_observations expected "
                + std::to_string(observation_dimension) + ", got "
                + std::to_string(configured_observations));
    }

    const auto history = requireValue<std::vector<int>>(
        policy_config, "observations_history", source);
    std::set<int> unique_history;
    for (std::size_t index = 0; index < history.size(); ++index)
    {
        if (history[index] < 0)
        {
            fail(
                source,
                "observations_history contains negative index at position "
                    + std::to_string(index));
        }
        if (history[index] == std::numeric_limits<int>::max())
        {
            fail(source, "observations_history index is too large");
        }
        if (!unique_history.insert(history[index]).second)
        {
            fail(
                source,
                "observations_history contains duplicate index "
                    + std::to_string(history[index]));
        }
    }
    const std::string history_priority = requireValue<std::string>(
        policy_config, "observations_history_priority", source);
    if (history_priority != "time" && history_priority != "term")
    {
        fail(
            source,
            "observations_history_priority must be 'time' or 'term', got '"
                + history_priority + "'");
    }

    if (needs_motion)
    {
        requireUniqueIndices(
            policy_config,
            "motion_joint_mapping",
            dofs,
            dofs,
            source);
        const std::string motion_file =
            requireValue<std::string>(policy_config, "motion_file", source);
        if (motion_file.empty())
        {
            fail(source, "key 'motion_file' must not be empty");
        }
        requirePositiveFinite(policy_config, "motion_fps", source);
        const int time_offset_frames = requireValue<int>(
            policy_config, "motion_time_offset_frames", source);
        if (time_offset_frames < 0)
        {
            fail(
                source,
                "key 'motion_time_offset_frames' must be nonnegative");
        }
    }

    const std::size_t history_frames = history.empty() ? 1 : history.size();
    if (observation_dimension
        > std::numeric_limits<std::size_t>::max() / history_frames)
    {
        fail(source, "computed model input dimension overflows");
    }

    LWPolicyRuntimeConfiguration runtime;
    runtime.num_dofs = dofs;
    runtime.dt = merged["dt"].as<float>();
    runtime.decimation = merged["decimation"].as<int>();
    runtime.period_seconds = runtime.dt * runtime.decimation;
    runtime.output_max_age_seconds = 3.0f * runtime.period_seconds;
    runtime.clip_obs = policy_config["clip_obs"].as<float>();
    runtime.model_name = model_name;
    runtime.observations = observations;
    runtime.observation_layout = std::move(observation_layout);
    runtime.observations_history = history;
    runtime.observations_history_priority = history_priority;
    runtime.action_scale =
        policy_config["action_scale"].as<std::vector<float>>();
    runtime.clip_actions_lower = lower;
    runtime.clip_actions_upper = upper;
    runtime.rl_kp = policy_config["rl_kp"].as<std::vector<float>>();
    runtime.rl_kd = policy_config["rl_kd"].as<std::vector<float>>();
    runtime.fixed_kp = policy_config["fixed_kp"].as<std::vector<float>>();
    runtime.fixed_kd = policy_config["fixed_kd"].as<std::vector<float>>();
    runtime.torque_limits =
        policy_config["torque_limits"].as<std::vector<float>>();
    runtime.default_dof_pos =
        policy_config["default_dof_pos"].as<std::vector<float>>();
    runtime.joint_mapping =
        policy_config["joint_mapping"].as<std::vector<int>>();
    runtime.wheel_indices =
        policy_config["wheel_indices"].as<std::vector<int>>();
    runtime.wheel_mask.assign(dofs, 0);
    for (const int index : runtime.wheel_indices)
    {
        runtime.wheel_mask[static_cast<std::size_t>(index)] = 1;
    }
    if (policy_config["ang_vel_scale"])
    {
        runtime.ang_vel_scale = policy_config["ang_vel_scale"].as<float>();
    }
    if (policy_config["commands_scale"])
    {
        runtime.commands_scale =
            policy_config["commands_scale"].as<std::vector<float>>();
    }
    if (policy_config["dof_pos_scale"])
    {
        runtime.dof_pos_scale = policy_config["dof_pos_scale"].as<float>();
    }
    if (policy_config["dof_vel_scale"])
    {
        runtime.dof_vel_scale = policy_config["dof_vel_scale"].as<float>();
    }
    runtime.needs_motion_reference = needs_motion;
    runtime.needs_motion_command = needs_motion_command;
    runtime.needs_motion_anchor_orientation =
        needs_motion_anchor_orientation;
    if (needs_motion)
    {
        runtime.motion_file = policy_config["motion_file"].as<std::string>();
        runtime.motion_fps = policy_config["motion_fps"].as<float>();
        runtime.motion_time_offset_frames =
            policy_config["motion_time_offset_frames"].as<int>();
        runtime.motion_joint_mapping =
            policy_config["motion_joint_mapping"].as<std::vector<int>>();
    }

    return {
        std::move(merged),
        {dofs,
         observation_dimension,
         observation_dimension * history_frames,
         dofs},
        std::move(runtime)};
}

void ValidateLWModelContract(
    InferenceRuntime::Model& model,
    const LWPolicyDimensions& dimensions,
    const std::string& source,
    int warmup_iterations)
{
    if (!model.is_loaded())
    {
        fail(source, "model is not loaded");
    }
    if (model.get_model_type() != "onnx")
    {
        fail(
            source,
            "LW production policy must be ONNX, got '"
                + model.get_model_type() + "'");
    }

    const auto& inputs = model.input_metadata();
    const auto& outputs = model.output_metadata();
    if (inputs.size() != 1)
    {
        fail(
            source,
            "ONNX model must expose exactly one input tensor, got "
                + std::to_string(inputs.size()));
    }
    if (outputs.size() != 1)
    {
        fail(
            source,
            "ONNX model must expose exactly one output tensor, got "
                + std::to_string(outputs.size()));
    }
    validateTensor(inputs.front(), dimensions.model_input, "input", source);
    validateTensor(outputs.front(), dimensions.model_output, "output", source);

    if (warmup_iterations < 0)
    {
        fail(source, "warmup iteration count must be nonnegative");
    }
    const std::vector<float> dummy_input(dimensions.model_input, 0.0f);
    std::vector<float> output(dimensions.model_output, 0.0f);
    const InferenceRuntime::TensorView input_view = {
        dummy_input.data(), dummy_input.size()};
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        try
        {
            model.forwardInto(
                &input_view,
                1,
                {output.data(), output.size()});
        }
        catch (const std::exception& exception)
        {
            fail(
                source,
                "ONNX warmup failed for expected input dimension "
                    + std::to_string(dimensions.model_input) + ": "
                    + exception.what());
        }
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            if (!std::isfinite(output[index]))
            {
                fail(
                    source,
                    "ONNX warmup output contains a non-finite value at index "
                        + std::to_string(index));
            }
        }
    }
}
