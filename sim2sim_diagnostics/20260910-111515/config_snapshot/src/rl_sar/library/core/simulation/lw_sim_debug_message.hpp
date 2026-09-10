#ifndef LW_SIM_DEBUG_MESSAGE_HPP
#define LW_SIM_DEBUG_MESSAGE_HPP

#include "rl_sdk.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include <sensor_msgs/msg/joint_state.hpp>

inline constexpr int LW_SIM_DEBUG_NUM_DOFS = 10;
inline constexpr std::size_t LW_SIM_DEBUG_FIELD_COUNT = 43;

struct LWSimDebugMessageConfig
{
    int num_dofs = 0;
    std::vector<int> joint_mapping;
    std::vector<int> wheel_indices;
};

struct LWSimDebugTelemetry
{
    bool gait_available = false;
    std::array<double, 3> left_foot_position{};
    std::array<double, 3> right_foot_position{};
    double left_contact = 0.0;
    double right_contact = 0.0;

    bool frame_position_available = false;
    std::array<double, 3> frame_position{};
    bool frame_velocity_available = false;
    std::array<double, 3> frame_velocity{};
    bool angular_velocity_available = false;
    std::array<double, 3> angular_velocity{};
    bool quaternion_available = false;
    std::array<double, 4> quaternion{};
};

class LWSimDebugMessageCache
{
public:
    explicit LWSimDebugMessageCache(const LWSimDebugMessageConfig& config);

    bool Populate(
        const RobotState<float>& robot_state,
        const RobotCommand<float>& robot_command,
        const LWControlSnapshot& control,
        const LWSimDebugTelemetry& telemetry) noexcept;

    sensor_msgs::msg::JointState& message() noexcept;
    const sensor_msgs::msg::JointState& message() const noexcept;

private:
    static constexpr std::size_t kCurrentOffset = 0;
    static constexpr std::size_t kTargetOffset = 10;
    static constexpr std::size_t kGaitOffset = 20;
    static constexpr std::size_t kTrackingOffset = 28;

    std::array<bool, LW_SIM_DEBUG_NUM_DOFS> wheel_mask_{};
    sensor_msgs::msg::JointState message_;
};

#endif // LW_SIM_DEBUG_MESSAGE_HPP
