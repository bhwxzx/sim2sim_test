#include "lw_mujoco_control_adapter.hpp"

#include <algorithm>
#include <stdexcept>

namespace
{
[[noreturn]] void Fail(const std::string& detail)
{
    throw std::runtime_error("LW MuJoCo layout validation failed: " + detail);
}

std::string SensorBaseName(const std::string& joint_name)
{
    constexpr const char suffix[] = "_joint";
    constexpr std::size_t suffix_length = sizeof(suffix) - 1;
    if (joint_name.size() <= suffix_length
        || joint_name.compare(
            joint_name.size() - suffix_length,
            suffix_length,
            suffix) != 0)
    {
        Fail("joint name does not end in '_joint': " + joint_name);
    }
    return joint_name.substr(0, joint_name.size() - suffix_length);
}

int ResolveSensor(
    const mjModel& model,
    const std::string& name,
    mjtSensor expected_type,
    mjtObj expected_object_type,
    int expected_object_id,
    int expected_dimension)
{
    const int sensor_id = mj_name2id(&model, mjOBJ_SENSOR, name.c_str());
    if (sensor_id < 0 || sensor_id >= model.nsensor)
    {
        Fail("missing required sensor '" + name + "'");
    }
    if (model.sensor_type[sensor_id] != expected_type)
    {
        Fail("sensor '" + name + "' has the wrong type");
    }
    if (model.sensor_objtype[sensor_id] != expected_object_type
        || model.sensor_objid[sensor_id] != expected_object_id)
    {
        Fail("sensor '" + name + "' is bound to the wrong object");
    }
    if (model.sensor_dim[sensor_id] != expected_dimension)
    {
        Fail("sensor '" + name + "' has the wrong dimension");
    }
    const int address = model.sensor_adr[sensor_id];
    if (address < 0 || address + expected_dimension > model.nsensordata)
    {
        Fail("sensor '" + name + "' has an out-of-range data address");
    }

    int matching_sensors = 0;
    for (int id = 0; id < model.nsensor; ++id)
    {
        if (model.sensor_type[id] == expected_type
            && model.sensor_objtype[id] == expected_object_type
            && model.sensor_objid[id] == expected_object_id)
        {
            ++matching_sensors;
        }
    }
    if (matching_sensors != 1)
    {
        Fail("required sensor '" + name + "' is ambiguous");
    }
    return address;
}

int ResolveActuator(
    const mjModel& model,
    const std::string& name,
    int expected_joint_id)
{
    const int actuator_id = mj_name2id(&model, mjOBJ_ACTUATOR, name.c_str());
    if (actuator_id < 0 || actuator_id >= model.nu)
    {
        Fail("missing required actuator '" + name + "'");
    }
    if (model.actuator_trntype[actuator_id] != mjTRN_JOINT
        || model.actuator_trnid[2 * actuator_id] != expected_joint_id)
    {
        Fail("actuator '" + name + "' is bound to the wrong joint");
    }

    int matching_actuators = 0;
    for (int id = 0; id < model.nu; ++id)
    {
        if (model.actuator_trntype[id] == mjTRN_JOINT
            && model.actuator_trnid[2 * id] == expected_joint_id)
        {
            ++matching_actuators;
        }
    }
    if (matching_actuators != 1)
    {
        Fail("required actuator '" + name + "' is ambiguous");
    }
    return actuator_id;
}

bool HasSize(const std::vector<float>& values, std::size_t expected) noexcept
{
    return values.size() == expected;
}

bool RequiresZeroedActuators(LWSafetyAction action) noexcept
{
    return action == LWSafetyAction::HardDisable
        || action == LWSafetyAction::HardDisableAndShutdown
        || action == LWSafetyAction::AbortStartup
        || action == LWSafetyAction::OrderlyShutdown;
}

bool RequiresShutdown(LWSafetyAction action) noexcept
{
    return action == LWSafetyAction::HardDisableAndShutdown
        || action == LWSafetyAction::AbortStartup;
}
} // namespace

LWMuJoCoControlAdapter::LWMuJoCoControlAdapter(
    const mjModel& model,
    const std::vector<std::string>& joint_names,
    const std::vector<int>& joint_mapping)
    : model_(&model)
{
    if (joint_names.empty() || joint_names.size() != joint_mapping.size())
    {
        Fail("joint names and policy mapping must have the same nonzero size");
    }

    std::vector<bool> mapped(joint_names.size(), false);
    joints_.reserve(joint_mapping.size());
    for (std::size_t policy_index = 0;
         policy_index < joint_mapping.size();
         ++policy_index)
    {
        const int hardware_index = joint_mapping[policy_index];
        if (hardware_index < 0
            || static_cast<std::size_t>(hardware_index) >= joint_names.size())
        {
            Fail("policy joint mapping contains an out-of-range index");
        }
        if (mapped[static_cast<std::size_t>(hardware_index)])
        {
            Fail("policy joint mapping contains a duplicate index");
        }
        mapped[static_cast<std::size_t>(hardware_index)] = true;

        const std::string& joint_name =
            joint_names[static_cast<std::size_t>(hardware_index)];
        const int joint_id = mj_name2id(&model, mjOBJ_JOINT, joint_name.c_str());
        if (joint_id < 0 || joint_id >= model.njnt)
        {
            Fail("missing required joint '" + joint_name + "'");
        }
        if (model.jnt_type[joint_id] != mjJNT_HINGE)
        {
            Fail("joint '" + joint_name + "' is not a one-DOF hinge");
        }

        const std::string sensor_base = SensorBaseName(joint_name);
        JointCache cache;
        cache.joint_id = joint_id;
        cache.position_address = ResolveSensor(
            model,
            sensor_base + "_pos",
            mjSENS_JOINTPOS,
            mjOBJ_JOINT,
            joint_id,
            1);
        cache.velocity_address = ResolveSensor(
            model,
            sensor_base + "_vel",
            mjSENS_JOINTVEL,
            mjOBJ_JOINT,
            joint_id,
            1);
        cache.torque_address = ResolveSensor(
            model,
            sensor_base + "_torque",
            mjSENS_JOINTACTFRC,
            mjOBJ_JOINT,
            joint_id,
            1);
        cache.actuator_id = ResolveActuator(model, joint_name, joint_id);
        joints_.push_back(cache);
    }

    const int imu_site_id = mj_name2id(&model, mjOBJ_SITE, "imu");
    if (imu_site_id < 0 || imu_site_id >= model.nsite)
    {
        Fail("missing required IMU site 'imu'");
    }
    quaternion_address_ = ResolveSensor(
        model,
        "imu_quat",
        mjSENS_FRAMEQUAT,
        mjOBJ_SITE,
        imu_site_id,
        4);
    gyroscope_address_ = ResolveSensor(
        model,
        "imu_gyro",
        mjSENS_GYRO,
        mjOBJ_SITE,
        imu_site_id,
        3);
}

std::size_t LWMuJoCoControlAdapter::numDofs() const noexcept
{
    return joints_.size();
}

int LWMuJoCoControlAdapter::quaternionAddress() const noexcept
{
    return quaternion_address_;
}

int LWMuJoCoControlAdapter::gyroscopeAddress() const noexcept
{
    return gyroscope_address_;
}

bool LWMuJoCoControlAdapter::ReadState(
    const mjData& data,
    std::vector<float>& joint_positions,
    std::vector<float>& joint_velocities,
    std::vector<float>& joint_torques,
    std::vector<float>& quaternion,
    std::vector<float>& gyroscope) const noexcept
{
    const std::size_t dofs = joints_.size();
    if (!HasSize(joint_positions, dofs)
        || !HasSize(joint_velocities, dofs)
        || !HasSize(joint_torques, dofs)
        || !HasSize(quaternion, 4)
        || !HasSize(gyroscope, 3))
    {
        return false;
    }

    for (std::size_t index = 0; index < dofs; ++index)
    {
        const JointCache& cache = joints_[index];
        joint_positions[index] =
            static_cast<float>(data.sensordata[cache.position_address]);
        joint_velocities[index] =
            static_cast<float>(data.sensordata[cache.velocity_address]);
        joint_torques[index] =
            static_cast<float>(data.sensordata[cache.torque_address]);
    }
    for (std::size_t index = 0; index < quaternion.size(); ++index)
    {
        quaternion[index] = static_cast<float>(
            data.sensordata[quaternion_address_ + static_cast<int>(index)]);
    }
    for (std::size_t index = 0; index < gyroscope.size(); ++index)
    {
        gyroscope[index] = static_cast<float>(
            data.sensordata[gyroscope_address_ + static_cast<int>(index)]);
    }
    return true;
}

LWSimTorqueValidation LWMuJoCoControlAdapter::ApplyCommand(
    mjData& data,
    const std::vector<float>& desired_positions,
    const std::vector<float>& desired_velocities,
    const std::vector<float>& feedforward_torques,
    const std::vector<float>& position_gains,
    const std::vector<float>& velocity_gains,
    const std::vector<float>& torque_limits,
    std::vector<float>& torque_candidates,
    std::vector<float>& bounded_torques) const noexcept
{
    const std::size_t dofs = joints_.size();
    if (!HasSize(desired_positions, dofs)
        || !HasSize(desired_velocities, dofs)
        || !HasSize(feedforward_torques, dofs)
        || !HasSize(position_gains, dofs)
        || !HasSize(velocity_gains, dofs)
        || !HasSize(torque_limits, dofs)
        || !HasSize(torque_candidates, dofs)
        || !HasSize(bounded_torques, dofs))
    {
        return {LWSimTorqueFailure::SizeMismatch, 0};
    }

    for (std::size_t index = 0; index < dofs; ++index)
    {
        const JointCache& cache = joints_[index];
        torque_candidates[index] = feedforward_torques[index]
            + position_gains[index]
                * (desired_positions[index]
                   - static_cast<float>(
                       data.sensordata[cache.position_address]))
            + velocity_gains[index]
                * (desired_velocities[index]
                   - static_cast<float>(
                       data.sensordata[cache.velocity_address]));
    }

    const LWSimTorqueValidation validation = PrepareLWSimTorques(
        torque_candidates,
        torque_limits,
        bounded_torques);
    if (!validation.valid())
    {
        return validation;
    }
    for (std::size_t index = 0; index < dofs; ++index)
    {
        data.ctrl[joints_[index].actuator_id] = bounded_torques[index];
    }
    return validation;
}

void LWMuJoCoControlAdapter::ExecuteSafetyDecision(
    mjData& data,
    const LWSafetyDecision& decision,
    LWMuJoCoShutdownTargets shutdown_targets) const noexcept
{
    if (RequiresZeroedActuators(decision.action))
    {
        std::fill(data.ctrl, data.ctrl + model_->nu, 0.0);
    }
    if (RequiresShutdown(decision.action))
    {
        shutdown_targets.simulation_running = false;
        shutdown_targets.simulation_run = 0;
        shutdown_targets.exit_request.store(1);
    }
}
