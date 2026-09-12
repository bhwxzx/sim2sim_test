#ifndef LW_DEBUG_PUBLISHER_HPP
#define LW_DEBUG_PUBLISHER_HPP

#include "LW_sdk.hpp"
#include "lw_runtime_sync.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

struct LWDebugSnapshot
{
    LowState low_state{};
    LowCmd low_command{};
    std::array<float, 3> gyroscope{};
    std::array<float, 4> quaternion{1.0f, 0.0f, 0.0f, 0.0f};
    float command_x = 0.0f;
    float command_y = 0.0f;
    float command_yaw = 0.0f;
};

inline constexpr std::int64_t LW_DEBUG_DEFAULT_RATE_HZ = 50;
inline constexpr std::int64_t LW_DEBUG_MIN_RATE_HZ = 1;
inline constexpr std::int64_t LW_DEBUG_MAX_RATE_HZ = 200;

struct LWDebugPublisherConfig
{
    int num_dofs = 0;
    std::vector<int> joint_mapping;
    std::vector<int> wheel_indices;
    std::vector<float> rl_kp;
    std::vector<float> rl_kd;
    std::int64_t publish_rate_hz = LW_DEBUG_DEFAULT_RATE_HZ;
    std::string topic_name{"/LW_joint_states"};
};

std::chrono::nanoseconds LWDebugPublishPeriod(std::int64_t publish_rate_hz);

class LWDebugPublisher
{
public:
    static std::unique_ptr<LWDebugPublisher> CreateIfEnabled(
        bool enabled,
        const std::shared_ptr<rclcpp::Node>& node,
        LWDebugPublisherConfig config);

    bool publishSnapshot(const LWDebugSnapshot& snapshot);
    bool publishOnce();

private:
    LWDebugPublisher(
        std::shared_ptr<rclcpp::Node> node,
        LWDebugPublisherConfig config);

    void initializeMessage();
    void populateMessage(
        sensor_msgs::msg::JointState& message,
        const LWDebugSnapshot& snapshot);

    struct SequencedSnapshot
    {
        std::uint64_t sequence = 0;
        LWDebugSnapshot snapshot{};
    };

    std::shared_ptr<rclcpp::Node> node_;
    LWDebugPublisherConfig config_;
    LWSnapshotBuffer<SequencedSnapshot> snapshot_;
    std::uint64_t next_snapshot_sequence_ = 1;
    std::uint64_t last_published_sequence_ = 0;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    std::shared_ptr<
        realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>
        realtime_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif // LW_DEBUG_PUBLISHER_HPP
