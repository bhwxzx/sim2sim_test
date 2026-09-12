#include "lw_debug_publisher.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace
{
const std::array<std::string, MOTOR_COUNTS> CURRENT_JOINT_NAMES = {
    "right_hip_now", "left_hip_now",
    "right_thigh_now", "left_thigh_now",
    "right_shank_now", "left_shank_now",
    "right_foot_now", "left_foot_now",
    "right_wheel_now", "left_wheel_now"};

const std::array<std::string, MOTOR_COUNTS> TARGET_JOINT_NAMES = {
    "right_hip_target", "left_hip_target",
    "right_thigh_target", "left_thigh_target",
    "right_shank_target", "left_shank_target",
    "right_foot_target", "left_foot_target",
    "right_wheel_target", "left_wheel_target"};

const std::array<std::string, 6> AUXILIARY_NAMES = {
    "imu_ang_vel",
    "imu_quat_w",
    "imu_quat_x",
    "imu_quat_y",
    "imu_quat_z",
    "cmd_vel"};

void ValidateConfig(const LWDebugPublisherConfig& config)
{
    if (config.num_dofs != MOTOR_COUNTS)
    {
        throw std::invalid_argument(
            "LW debug publisher requires exactly 10 degrees of freedom");
    }
    const auto expected_size = static_cast<std::size_t>(config.num_dofs);
    if (config.joint_mapping.size() != expected_size
        || config.rl_kp.size() != expected_size
        || config.rl_kd.size() != expected_size
        || config.wheel_indices.size() != 2)
    {
        throw std::invalid_argument("invalid LW debug publisher configuration sizes");
    }
    (void)LWDebugPublishPeriod(config.publish_rate_hz);
    if (config.topic_name.empty())
    {
        throw std::invalid_argument("LW debug topic name must not be empty");
    }
    for (int index : config.joint_mapping)
    {
        if (index < 0 || index >= config.num_dofs)
        {
            throw std::invalid_argument("LW debug joint mapping is out of range");
        }
    }
    for (int index : config.wheel_indices)
    {
        if (index < 0 || index >= config.num_dofs)
        {
            throw std::invalid_argument("LW debug wheel index is out of range");
        }
    }
}

bool IsWheel(int index, const std::vector<int>& wheel_indices)
{
    return index == wheel_indices[0] || index == wheel_indices[1];
}
} // namespace

std::chrono::nanoseconds LWDebugPublishPeriod(std::int64_t publish_rate_hz)
{
    if (publish_rate_hz < LW_DEBUG_MIN_RATE_HZ
        || publish_rate_hz > LW_DEBUG_MAX_RATE_HZ)
    {
        throw std::invalid_argument(
            "LW debug publish rate must be an integer from 1 through 200 Hz");
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::seconds(1)) / publish_rate_hz;
}

std::unique_ptr<LWDebugPublisher> LWDebugPublisher::CreateIfEnabled(
    bool enabled,
    const std::shared_ptr<rclcpp::Node>& node,
    LWDebugPublisherConfig config)
{
    if (!enabled)
    {
        return nullptr;
    }
    if (!node)
    {
        throw std::invalid_argument("LW debug publisher requires a ROS node");
    }
    ValidateConfig(config);
    return std::unique_ptr<LWDebugPublisher>(
        new LWDebugPublisher(node, std::move(config)));
}

LWDebugPublisher::LWDebugPublisher(
    std::shared_ptr<rclcpp::Node> node,
    LWDebugPublisherConfig config)
    : node_(std::move(node)), config_(std::move(config))
{
    publisher_ = node_->create_publisher<sensor_msgs::msg::JointState>(
        config_.topic_name,
        rclcpp::QoS(1));
    realtime_publisher_ = std::make_shared<
        realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
        publisher_);
    initializeMessage();
    timer_ = node_->create_wall_timer(
        LWDebugPublishPeriod(config_.publish_rate_hz),
        [this]() { publishOnce(); });
}

void LWDebugPublisher::initializeMessage()
{
    realtime_publisher_->lock();
    auto& message = realtime_publisher_->msg_;
    message.name.reserve(
        CURRENT_JOINT_NAMES.size()
        + TARGET_JOINT_NAMES.size()
        + AUXILIARY_NAMES.size());
    message.name.insert(
        message.name.end(),
        CURRENT_JOINT_NAMES.begin(),
        CURRENT_JOINT_NAMES.end());
    message.name.insert(
        message.name.end(),
        TARGET_JOINT_NAMES.begin(),
        TARGET_JOINT_NAMES.end());
    message.name.insert(
        message.name.end(),
        AUXILIARY_NAMES.begin(),
        AUXILIARY_NAMES.end());
    message.position.assign(message.name.size(), 0.0);
    message.velocity.assign(message.name.size(), 0.0);
    message.effort.assign(message.name.size(), 0.0);
    realtime_publisher_->unlock();
}

bool LWDebugPublisher::publishSnapshot(const LWDebugSnapshot& snapshot)
{
    const SequencedSnapshot sequenced_snapshot{
        next_snapshot_sequence_,
        snapshot};
    if (!snapshot_.tryPublish(sequenced_snapshot))
    {
        return false;
    }
    ++next_snapshot_sequence_;
    return true;
}

bool LWDebugPublisher::publishOnce()
{
    SequencedSnapshot sequenced_snapshot;
    if (!snapshot_.read(sequenced_snapshot)
        || sequenced_snapshot.sequence <= last_published_sequence_)
    {
        return false;
    }

    if (!realtime_publisher_->trylock())
    {
        return false;
    }

    auto& message = realtime_publisher_->msg_;
    populateMessage(message, sequenced_snapshot.snapshot);
    message.header.stamp = node_->get_clock()->now();
    realtime_publisher_->unlockAndPublish();
    last_published_sequence_ = sequenced_snapshot.sequence;
    return true;
}

void LWDebugPublisher::populateMessage(
    sensor_msgs::msg::JointState& message,
    const LWDebugSnapshot& snapshot)
{
    std::fill(message.position.begin(), message.position.end(), 0.0);
    std::fill(message.velocity.begin(), message.velocity.end(), 0.0);
    std::fill(message.effort.begin(), message.effort.end(), 0.0);

    for (int index = 0; index < config_.num_dofs; ++index)
    {
        const int sdk_index = config_.joint_mapping[index];
        const auto& state = snapshot.low_state.motorState[sdk_index];
        message.velocity[index] = state.vel_now;
        message.effort[index] = state.tau_now;
        if (!IsWheel(index, config_.wheel_indices))
        {
            message.position[index] = state.pos_now;
        }

        const int target_index = config_.num_dofs + index;
        const float target = snapshot.low_command.motorCmd[sdk_index].action_set;
        if (IsWheel(index, config_.wheel_indices))
        {
            message.velocity[target_index] = target;
            message.effort[target_index] =
                config_.rl_kp[index] * (0.0f - state.pos_now)
                + config_.rl_kd[index] * (target - state.vel_now);
        }
        else
        {
            message.position[target_index] = target;
            message.effort[target_index] =
                config_.rl_kp[index] * (target - state.pos_now)
                + config_.rl_kd[index] * (0.0f - state.vel_now);
        }
    }

    const int imu_offset = 2 * config_.num_dofs;
    message.position[imu_offset] = snapshot.gyroscope[0];
    message.velocity[imu_offset] = snapshot.gyroscope[1];
    message.effort[imu_offset] = snapshot.gyroscope[2];
    for (int component = 0; component < 4; ++component)
    {
        message.position[imu_offset + 1 + component] =
            snapshot.quaternion[component];
    }

    const int command_offset = imu_offset + 5;
    message.position[command_offset] = snapshot.command_x;
    message.velocity[command_offset] = snapshot.command_y;
    message.effort[command_offset] = snapshot.command_yaw;
}
