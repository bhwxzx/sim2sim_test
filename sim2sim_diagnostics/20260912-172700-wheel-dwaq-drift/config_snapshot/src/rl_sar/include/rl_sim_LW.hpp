#ifndef RL_SIM_LW_HPP
#define RL_SIM_LW_HPP

// #define ENABLE_FORWARD_LATENCY
// #define ADD_ANGVEL_NOISE
// #define ADD_JOINTVEL_NOISE
#define JOYSTICK_1
// #define JOYSTICK_2
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "lw_control_safety.hpp"
#include "lw_joystick_safety.hpp"
#include "lw_loop_config.hpp"
#include "lw_runtime_core.hpp"
#include "lw_safety_policy.hpp"
#include "lw_signal_shutdown.hpp"
#include "lw_sim_debug_message.hpp"
#include "lw_sim_plot_config.hpp"
#include "lw_sim_torque_validation.hpp"
#include "fsm_LW.hpp"

#include "LW_sdk.hpp"
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "joystick.hh"
#include <mujoco/mujoco.h>
#include "mujoco_utils.hpp"

class LWMuJoCoControlAdapter;

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real() noexcept override;

    std::shared_ptr<rclcpp::Node> ros2_node;
    std::unique_ptr<mj::Simulate> sim;
    void RequestSimulationStop() noexcept;
    void RethrowPhysicsError() const;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();
    void ApplyPendingInput();
    void HandleLoopError(const std::string& loop_name, std::exception_ptr error) noexcept;
    void HandleLoopTiming(
        const std::string& loop_name,
        LoopTimingLevel level,
        const LoopTimingSnapshot& timing) noexcept;
    void HandleLWPolicyOutputFault(
        LWPolicyOutputStatus status) noexcept override;
    void ApplySafetyEvent(
        LWSafetyEvent event,
        const std::string& reason) noexcept;
    void ExecuteSafetyDecision(
        const LWSafetyDecision& decision,
        const std::string& reason) noexcept;
    void ApplySimulationControls();
    void StartRuntimeLoopsIfReady();

    // loop
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;

    // mujoco
    mjvCamera cam;
    mjvOption opt;
    mjvPerturb pert;
    mjData *mj_data = nullptr;
    mjModel *mj_model = nullptr;
    std::unique_ptr<LWMuJoCoPhysicsLifecycle> physics_lifecycle_;
    std::unique_ptr<LWMuJoCoControlAdapter> mujoco_control_adapter_;
    std::string scene_name;
    void RefreshMuJoCoPointersLocked() noexcept;

    // LW interface
    LWSDK lw_sdk;
    LowCmd lw_low_command{};
    LowState lw_low_state{};
    LWRuntimeCore runtime_core_;
    void disable_lw_robot();

    // joystick
    std::unique_ptr<LWJoystickDevice> sys_js;
    JoystickEvent sys_js_event;

    std::array<LWJoystickButton, LW_JOYSTICK_BUTTON_COUNT> sys_js_button{};
    std::array<int, LW_JOYSTICK_AXIS_COUNT> sys_js_axis{};
    LWJoystickFaultLatch joystick_fault_latch_;
    LWInputMailbox<Input::Gamepad> joystick_input_mailbox_;
    LWInputMailbox<Input::Gamepad>::Snapshot joystick_input_snapshot_{};
    std::uint64_t consumed_gamepad_sequence_ = 0;
    bool sys_js_active = false;
    float axis_deadzone = 0.05f;
    int sys_js_max_value = (1 << (16 - 1)); // 即 2的15次方 = 32768
    void SetupSysJoystick(const std::string& device, int bits);
    void GetSysJoystick();
    void LatchJoystickFault(const LWJoystickSampleResult& result) noexcept;
    void ApplyJoystickFaultGate() noexcept;

    // Imu
    // sensor_msgs::msg::Imu imu;
    // rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    // void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg);

    // Optional high-rate Sim2Sim plot telemetry. The buffer, publisher, timer,
    // and control-cycle callback are absent unless explicitly enabled.
    struct SimDebugSnapshot
    {
        RobotState<float> robot_state;
        RobotCommand<float> robot_command;
        LWControlSnapshot control;
    };
    struct SimDebugMuJoCoCache
    {
        int left_foot_site = -1;
        int right_foot_site = -1;
        int left_foot_force_address = -1;
        int right_foot_force_address = -1;
        int frame_position_address = -1;
        int frame_velocity_address = -1;
        int angular_velocity_address = -1;
        int quaternion_address = -1;
    };
    LWSimPlotConfiguration plot_configuration_;
    std::unique_ptr<LWSnapshotBuffer<SimDebugSnapshot>> plot_snapshot_;
    std::unique_ptr<SimDebugSnapshot> plot_write_snapshot_;
    std::unique_ptr<SimDebugSnapshot> plot_read_snapshot_;
    std::unique_ptr<LWSimDebugMessageCache> plot_message_cache_;
    SimDebugMuJoCoCache plot_mujoco_cache_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointstate_plot_publisher_;
    rclcpp::TimerBase::SharedPtr plot_timer_;
    rclcpp::TimerBase::SharedPtr operator_status_timer_;
    void jointstate_plot_callback(void);
    void InitializePlotDebugResourcesLocked();
    LWSimDebugTelemetry ReadPlotTelemetryLocked() const noexcept;
    void OperatorStatusCallback();
    std::uint64_t last_operator_status_sequence_ = 0;
    bool operator_status_seen_ = false;

    // Final MuJoCo torque frame. Candidates are validated before ctrl mutation.
    std::vector<float> mujoco_tau_candidates_;
    std::vector<float> mujoco_tau_bounded_;

    // others
    void disable_robot(void);

};

#endif // RL_SIM_LW_HPP
