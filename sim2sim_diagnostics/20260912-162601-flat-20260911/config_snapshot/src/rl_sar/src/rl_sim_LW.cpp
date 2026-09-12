
#include "rl_sim_LW.hpp"
#include "lw_mujoco_control_adapter.hpp"

#include <iomanip>
#include <sstream>

using namespace std::chrono_literals;

namespace
{
const char* LWOperatorModeName(LWOperatorMode mode)
{
    switch (mode)
    {
    case LWOperatorMode::Passive: return "passive";
    case LWOperatorMode::GetUpLeg: return "get-up-leg";
    case LWOperatorMode::GetUpWheel: return "get-up-wheel";
    case LWOperatorMode::GetDown: return "get-down";
    case LWOperatorMode::LegLocomotion: return "leg-locomotion";
    case LWOperatorMode::WheelLocomotion: return "wheel-locomotion";
    case LWOperatorMode::LegToWheel: return "leg-to-wheel";
    case LWOperatorMode::WheelToLeg: return "wheel-to-leg";
    case LWOperatorMode::Unknown: return "unknown";
    }
    return "unknown";
}

bool LWOperatorModeHasProgress(LWOperatorMode mode)
{
    return mode == LWOperatorMode::GetUpLeg
        || mode == LWOperatorMode::GetUpWheel
        || mode == LWOperatorMode::GetDown
        || mode == LWOperatorMode::LegToWheel
        || mode == LWOperatorMode::WheelToLeg;
}
} // namespace

// // ===================== [定义抗冲击实验状态变量] =====================
// static double g_impact_start_time = -1.0;  // 记录冲击开始的物理时间
// static double g_impact_force_x = 0.0;      // X方向(前后)的冲击力 (N)
// static double g_impact_force_y = 0.0;      // Y方向(侧向)的冲击力 (N)
// static double g_impact_duration = 0.2;     // 冲击持续时间 (例如 0.2秒 = 200ms)
// // ============================================================================

RL_Real::RL_Real(int argc, char **argv)
    : plot_configuration_(ParseLWSimPlotConfiguration(argc, argv))
{
    runtime_core_.bind(
        *this,
        [this](const LWSafetyDecision& decision, const std::string& reason)
        {
            ExecuteSafetyDecision(decision, reason);
        });
    std::filesystem::path policy_root = POLICY_DIR;
    for (int index = 1; index < argc; ++index)
    {
        if (std::string(argv[index]) == "--policy-root")
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error(
                    "--policy-root requires a directory argument");
            }
            policy_root = argv[++index];
        }
    }
    SetPolicyRoot(policy_root);
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_LW_node");
    // this->imu_subscriber_ = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
    //     "/imu", rclcpp::SystemDefaultsQoS(),
    //     [this] (const sensor_msgs::msg::Imu::SharedPtr imu_msg) {this->ImuCallback(imu_msg);}
    // );

    this->robot_name = "LW";
    this->scene_name = "scene"; // "scene" "scene_terrain"
    this->ang_vel_axis = "body";
    this->ReadYaml(this->robot_name, "base.yaml");
    SetLWBaseRuntimeConfiguration(
        LWValidatedBaseConfiguration(
            this->params.config_node,
            this->ResolvePolicyPath(this->robot_name + "/base.yaml")));
    const auto& runtime_configuration = GetLWBaseRuntimeConfiguration();

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjv_defaultCamera(&this->cam);

    mjv_defaultOption(&this->opt);

    mjv_defaultPerturb(&this->pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    physics_lifecycle_ = std::make_unique<LWMuJoCoPhysicsLifecycle>(*sim);
    try
    {
        physics_lifecycle_->Start(
            filename,
            [this, &runtime_configuration](const mjModel& model, mjData&)
            {
                mujoco_control_adapter_ =
                    std::make_unique<LWMuJoCoControlAdapter>(
                        model,
                        runtime_configuration.joint_names,
                        runtime_configuration.joint_mapping);
            });
    }
    catch (...)
    {
        mujoco_control_adapter_.reset();
        throw;
    }
    {
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
        RefreshMuJoCoPointersLocked();
    }
    std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;

    this->SetupSysJoystick("/dev/input/js1", 16);

    // 提前加载所有的模型到内存
    this->PreloadModel(this->robot_name + "/robot_lab/leg_loco");
    this->PreloadModel(this->robot_name + "/robot_lab/wheel_loco");
    this->PreloadModel(this->robot_name + "/robot_lab/leg_to_wheel");
    this->PreloadModel(this->robot_name + "/robot_lab/wheel_to_leg");
    this->PreloadLWPolicyContext(this->robot_name + "/robot_lab/leg_loco");
    this->PreloadLWPolicyContext(this->robot_name + "/robot_lab/wheel_loco");
    this->PreloadLWPolicyContext(this->robot_name + "/robot_lab/leg_to_wheel");
    this->PreloadLWPolicyContext(this->robot_name + "/robot_lab/wheel_to_leg");

    const std::size_t num_dofs = runtime_configuration.num_dofs;
    mujoco_tau_candidates_.assign(num_dofs, 0.0f);
    mujoco_tau_bounded_.assign(num_dofs, 0.0f);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->lw_sdk.InitCmdData(this->lw_low_command);
    this->InitJointNum(num_dofs);
    this->InitOutputs();
    this->InitControl();
    this->control.gait_frequency = runtime_configuration.gait_command[0];
    this->gait_phase_time = 0.0f;
    runtime_core_.publishInitialPolicyInput();

    if (plot_configuration_.enabled)
    {
        plot_snapshot_ =
            std::make_unique<LWSnapshotBuffer<SimDebugSnapshot>>();
        plot_write_snapshot_ = std::make_unique<SimDebugSnapshot>();
        plot_read_snapshot_ = std::make_unique<SimDebugSnapshot>();
        plot_write_snapshot_->robot_state.motor_state.resize(
            static_cast<std::size_t>(runtime_configuration.num_dofs));
        plot_write_snapshot_->robot_command.motor_command.resize(
            static_cast<std::size_t>(runtime_configuration.num_dofs));
        plot_read_snapshot_->robot_state.motor_state.resize(
            static_cast<std::size_t>(runtime_configuration.num_dofs));
        plot_read_snapshot_->robot_command.motor_command.resize(
            static_cast<std::size_t>(runtime_configuration.num_dofs));
        plot_message_cache_ = std::make_unique<LWSimDebugMessageCache>(
            LWSimDebugMessageConfig{
                static_cast<int>(runtime_configuration.num_dofs),
                runtime_configuration.joint_mapping,
                runtime_configuration.wheel_indices});
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
        RefreshMuJoCoPointersLocked();
        InitializePlotDebugResourcesLocked();
    }

    // loop
    const auto loop_error_handler = [this](
        const std::string& loop_name,
        std::exception_ptr error)
    {
        HandleLoopError(loop_name, error);
    };
    const auto loop_timing_handler = [this](
        const std::string& loop_name,
        LoopTimingLevel level,
        const LoopTimingSnapshot& timing)
    {
        HandleLoopTiming(loop_name, level, timing);
    };
    this->loop_joystick = std::make_shared<LoopFunc>(
        "loop_joystick",
        0.01,
        std::bind(&RL_Real::GetSysJoystick, this),
        -1,
        loop_error_handler);
    this->loop_control = std::make_shared<LoopFunc>(
        "loop_control",
        BuildLWControlLoopConfig(this->params),
        std::bind(&RL_Real::RobotControl, this),
        loop_error_handler,
        loop_timing_handler);
    this->loop_rl = std::make_shared<LoopFunc>(
        "loop_rl",
        this->params.Get<float>("dt") * this->params.Get<int>("decimation"),
        std::bind(&RL_Real::RunModel, this),
        -1,
        loop_error_handler);

    // Match the real-robot lifecycle: finish every fallible resource setup
    // before the ROS executor can start runtime worker callbacks. Physics
    // initialization alone cannot authorize runtime startup.
    this->operator_status_timer_ = ros2_node->create_wall_timer(
        100ms,
        std::bind(&RL_Real::OperatorStatusCallback, this));
    if (plot_configuration_.enabled)
    {
        this->jointstate_plot_publisher_ =
            ros2_node->create_publisher<sensor_msgs::msg::JointState>(
                "/LW_joint_states",
                rclcpp::SystemDefaultsQoS());
        this->plot_timer_ = ros2_node->create_wall_timer(
            LWSimPlotPeriod(plot_configuration_),
            std::bind(&RL_Real::jointstate_plot_callback, this));
        std::cout << LOGGER::INFO
                  << "[Sim2Sim] Plot publishing enabled: topic=/LW_joint_states, rate="
                  << plot_configuration_.rate_hz << " Hz" << std::endl;
    }
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    // Runtime loops start from the ROS executor after render/physics readiness.
}

void RL_Real::StartRuntimeLoopsIfReady()
{
    try
    {
        physics_lifecycle_->startup().StartIfReady(
            [this](int stage)
            {
                if (stage == 0) this->loop_joystick->start();
                else if (stage == 1) this->loop_rl->start();
                else this->loop_control->start();
            },
            [this]() noexcept
            {
                this->loop_control->shutdown();
                this->loop_rl->shutdown();
                this->loop_joystick->shutdown();
            },
            [this]() { return !rclcpp::ok() || sim->exitrequest.load() != 0; });
    }
    catch (...)
    {
        // StartIfReady has joined partial workers before touching runtime safety.
        runtime_core_.reportSafetyEvent(
            LWSafetyEvent::StartupLoopStartFailed,
            "[Safety] Failed to start Sim2Sim worker loops");
        RequestSimulationStop();
        throw;
    }
}

RL_Real::~RL_Real() noexcept
{
    physics_lifecycle_->startup().Cancel();
    runtime_core_.reportSafetyEvent(LWSafetyEvent::NormalShutdown);
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    this->loop_joystick->shutdown();
    if (physics_lifecycle_)
    {
        physics_lifecycle_->Stop();
    }
    mj_data = nullptr;
    mj_model = nullptr;
    //disable_lw_robot();
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::RequestSimulationStop() noexcept
{
    if (physics_lifecycle_) physics_lifecycle_->startup().Cancel();
    if (sim)
    {
        sim->RequestExit();
    }
}

void RL_Real::RethrowPhysicsError() const
{
    if (physics_lifecycle_)
    {
        physics_lifecycle_->RethrowWorkerError();
    }
}

void RL_Real::RefreshMuJoCoPointersLocked() noexcept
{
    mj_model = physics_lifecycle_ ? physics_lifecycle_->model() : nullptr;
    mj_data = physics_lifecycle_ ? physics_lifecycle_->data() : nullptr;
}

void RL_Real::InitializePlotDebugResourcesLocked()
{
    if (!mj_model || !mujoco_control_adapter_)
    {
        throw std::runtime_error(
            "LW Sim2Sim debug setup requires an initialized MuJoCo model");
    }

    const auto site_id = [this](const char* name)
    {
        const int id = mj_name2id(mj_model, mjOBJ_SITE, name);
        if (id >= mj_model->nsite)
        {
            throw std::runtime_error(
                std::string("LW Sim2Sim debug site is out of range: ") + name);
        }
        return id;
    };
    const auto sensor_address = [this](const char* name, int required_dimension)
    {
        const int id = mj_name2id(mj_model, mjOBJ_SENSOR, name);
        if (id < 0)
        {
            return -1;
        }
        if (id >= mj_model->nsensor
            || mj_model->sensor_dim[id] < required_dimension)
        {
            throw std::runtime_error(
                std::string("LW Sim2Sim debug sensor has invalid dimension: ")
                + name);
        }
        const int address = mj_model->sensor_adr[id];
        if (address < 0
            || address + required_dimension > mj_model->nsensordata)
        {
            throw std::runtime_error(
                std::string("LW Sim2Sim debug sensor is out of range: ") + name);
        }
        return address;
    };

    plot_mujoco_cache_.left_foot_site = site_id("left_foot_site");
    plot_mujoco_cache_.right_foot_site = site_id("right_foot_site");
    plot_mujoco_cache_.left_foot_force_address =
        sensor_address("left_foot_force_sensor", 3);
    plot_mujoco_cache_.right_foot_force_address =
        sensor_address("right_foot_force_sensor", 3);
    plot_mujoco_cache_.frame_position_address =
        sensor_address("frame_pos", 3);
    plot_mujoco_cache_.frame_velocity_address =
        sensor_address("frame_vel", 3);
    plot_mujoco_cache_.angular_velocity_address =
        mujoco_control_adapter_->gyroscopeAddress();
    plot_mujoco_cache_.quaternion_address =
        mujoco_control_adapter_->quaternionAddress();
}

LWSimDebugTelemetry RL_Real::ReadPlotTelemetryLocked() const noexcept
{
    LWSimDebugTelemetry telemetry;
    const auto& cache = plot_mujoco_cache_;
    telemetry.gait_available =
        cache.left_foot_site >= 0
        && cache.right_foot_site >= 0
        && cache.left_foot_force_address >= 0
        && cache.right_foot_force_address >= 0;
    if (telemetry.gait_available)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            telemetry.left_foot_position[axis] =
                mj_data->site_xpos[3 * cache.left_foot_site + axis];
            telemetry.right_foot_position[axis] =
                mj_data->site_xpos[3 * cache.right_foot_site + axis];
        }
        constexpr double force_threshold = 10.0;
        telemetry.left_contact =
            mj_data->sensordata[cache.left_foot_force_address + 2]
                    > force_threshold
                ? 1.0 : 0.0;
        telemetry.right_contact =
            mj_data->sensordata[cache.right_foot_force_address + 2]
                    > force_threshold
                ? 1.0 : 0.0;
    }

    const auto read_three = [this](
        int address,
        bool& available,
        std::array<double, 3>& values)
    {
        available = address >= 0;
        if (available)
        {
            for (std::size_t axis = 0; axis < values.size(); ++axis)
            {
                values[axis] = mj_data->sensordata[address + axis];
            }
        }
    };
    read_three(
        cache.frame_position_address,
        telemetry.frame_position_available,
        telemetry.frame_position);
    read_three(
        cache.frame_velocity_address,
        telemetry.frame_velocity_available,
        telemetry.frame_velocity);
    read_three(
        cache.angular_velocity_address,
        telemetry.angular_velocity_available,
        telemetry.angular_velocity);

    telemetry.quaternion_available = cache.quaternion_address >= 0;
    if (telemetry.quaternion_available)
    {
        for (std::size_t component = 0;
             component < telemetry.quaternion.size();
             ++component)
        {
            telemetry.quaternion[component] =
                mj_data->sensordata[cache.quaternion_address + component];
        }
    }
    return telemetry;
}

void RL_Real::jointstate_plot_callback(void)
{
    if (!plot_snapshot_ || !plot_read_snapshot_ || !plot_message_cache_
        || !jointstate_plot_publisher_)
    {
        return;
    }
    if (!plot_snapshot_->read(*plot_read_snapshot_))
    {
        return;
    }
    if (!sim)
    {
        return;
    }
    const std::unique_lock<std::recursive_mutex> simulation_lock(sim->mtx);
    RefreshMuJoCoPointersLocked();
    if (!mj_model || !mj_data)
    {
        return;
    }
    const LWSimDebugTelemetry telemetry = ReadPlotTelemetryLocked();
    if (!plot_message_cache_->Populate(
            plot_read_snapshot_->robot_state,
            plot_read_snapshot_->robot_command,
            plot_read_snapshot_->control,
            telemetry))
    {
        return;
    }
    auto& message = plot_message_cache_->message();
    message.header.stamp = this->ros2_node->get_clock()->now();
    this->jointstate_plot_publisher_->publish(message);
}

void RL_Real::OperatorStatusCallback()
{
    StartRuntimeLoopsIfReady();
    LWOperatorStatusSnapshot status;
    if (!ReadLWOperatorStatus(status)
        || (operator_status_seen_
            && status.sequence == last_operator_status_sequence_))
    {
        return;
    }

    operator_status_seen_ = true;
    last_operator_status_sequence_ = status.sequence;
    std::ostringstream message;
    message << LOGGER::INFO << "[LW Sim2Sim] mode="
            << LWOperatorModeName(status.mode)
            << std::fixed << std::setprecision(3)
            << ", x=" << status.x
            << ", y=" << status.y
            << ", yaw=" << status.yaw
            << ", gait=" << status.gait_frequency;
    if (LWOperatorModeHasProgress(status.mode))
    {
        message << std::setprecision(1)
                << ", progress=" << status.progress * 100.0f << '%';
    }
    std::cout << message.str() << std::endl;
}

void RL_Real::HandleLoopError(
    const std::string& loop_name,
    std::exception_ptr error) noexcept
{
    runtime_core_.handleLoopError(
        loop_name,
        error,
        [this]()
        {
            LatchJoystickFault(
                {LWJoystickSampleStatus::Error, EFAULT, -1});
        });
}

void RL_Real::HandleLoopTiming(
    const std::string& loop_name,
    LoopTimingLevel level,
    const LoopTimingSnapshot& timing) noexcept
{
    if (loop_name == "loop_control"
        && (level == LoopTimingLevel::Degraded
            || level == LoopTimingLevel::Fatal))
    {
        runtime_core_.handleControlTiming(
            level == LoopTimingLevel::Fatal,
            timing.missed_deadlines);
    }
}

void RL_Real::HandleLWPolicyOutputFault(
    LWPolicyOutputStatus status) noexcept
{
    runtime_core_.handlePolicyOutputFault(status);
}

void RL_Real::ApplySafetyEvent(
    LWSafetyEvent event,
    const std::string& reason) noexcept
{
    runtime_core_.reportSafetyEvent(event, reason);
}

void RL_Real::ExecuteSafetyDecision(
    const LWSafetyDecision& decision,
    const std::string& reason) noexcept
{
    if (!reason.empty())
    {
        std::ostream& output =
            decision.severity >= LWSafetySeverity::InputDegraded
            ? std::cerr
            : std::cout;
        output << (decision.severity >= LWSafetySeverity::InputDegraded
                       ? LOGGER::WARNING
                       : LOGGER::INFO)
               << "[LW Sim2Sim Safety] " << reason
               << ", action=" << LWSafetyActionName(decision.action)
               << std::endl;
    }

    const bool requires_shutdown =
        decision.action == LWSafetyAction::HardDisableAndShutdown
        || decision.action == LWSafetyAction::AbortStartup;
    if (sim)
    {
        try
        {
            const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
            RefreshMuJoCoPointersLocked();
            if (mj_data && mujoco_control_adapter_)
            {
                mujoco_control_adapter_->ExecuteSafetyDecision(
                    *mj_data,
                    decision,
                    {simulation_running, sim->run, sim->exitrequest});
                return;
            }
        }
        catch (...)
        {
        }
    }
    if (requires_shutdown)
    {
        simulation_running = false;
        if (sim)
        {
            try
            {
                sim->RequestExit();
            }
            catch (...)
            {
            }
        }
    }
}

void RL_Real::disable_lw_robot(void)
{
    LowCmd cmd;
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
    {
        cmd.motorCmd[i].action_set = 0.0f;
        cmd.motorCmd[i].Kp = 0.0f;
        cmd.motorCmd[i].Kd = 0.0f;
    }
    cmd.motors_disable = true;
    this->lw_sdk.SendCmdData(cmd);
}

void RL_Real::RobotControl()
{
    LWControlCycleHooks hooks{
        [this]()
        {
            ApplyPendingInput();
            ApplyJoystickFaultGate();
        },
        [this]() { KeyboardInterface(); },
        []() { return true; },
        []() { return true; },
        [this]() { ApplySimulationControls(); },
        {}};
    if (plot_snapshot_)
    {
        hooks.after_command_delivery = [this]()
        {
            plot_write_snapshot_->robot_state = robot_state;
            plot_write_snapshot_->robot_command = robot_command;
            plot_write_snapshot_->control = {
                control.x,
                control.y,
                control.yaw,
                control.gait_frequency};
            plot_snapshot_->publish(*plot_write_snapshot_);
        };
    }
    runtime_core_.runControlCycle(hooks);
}

void RL_Real::ApplySimulationControls()
{
    if (!sim)
    {
        return;
    }
    const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
    RefreshMuJoCoPointersLocked();
    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {   
            if(this->robot_name == "LW")
            {
                int id = mj_name2id(this->mj_model, mjOBJ_KEY, "home_leg");
                mj_resetDataKeyframe(this->mj_model, this->mj_data, id);
            }
            else
            {
                mj_resetData(this->mj_model, this->mj_data);
            }

            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::T || this->control.current_gamepad == Input::Gamepad::RB_A)
    {
        if (this->mj_model && this->mj_data)
        {   
            if(this->robot_name == "LW")
            {
                int id = mj_name2id(this->mj_model, mjOBJ_KEY, "home_wheel");
                mj_resetDataKeyframe(this->mj_model, this->mj_data, id);
            }
            else
            {
                mj_resetData(this->mj_model, this->mj_data);
            }

            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }
}

void RL_Real::RunModel()
{
    LWInferenceCycleHooks hooks;
#if defined(ADD_ANGVEL_NOISE) || defined(ADD_JOINTVEL_NOISE)
    hooks.mutate_observation = [](
        Observations<float>& observations,
        const RobotState<float>& state)
    {
#ifdef ADD_ANGVEL_NOISE
        static std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, 0.4f);
        observations.ang_vel.resize(state.imu.gyroscope.size());
        for (size_t index = 0; index < state.imu.gyroscope.size(); ++index)
        {
            observations.ang_vel[index] =
                state.imu.gyroscope[index] + distribution(generator);
        }
#endif
#ifdef ADD_JOINTVEL_NOISE
        static std::default_random_engine velocity_generator;
        std::normal_distribution<float> velocity_distribution(0.0f, 1.5f);
        observations.dof_vel.resize(state.motor_state.dq.size());
        for (size_t index = 0; index < state.motor_state.dq.size(); ++index)
        {
            observations.dof_vel[index] =
                state.motor_state.dq[index]
                + velocity_distribution(velocity_generator);
        }
#endif
    };
#endif
#ifdef ENABLE_FORWARD_LATENCY
    hooks.after_forward = []()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
    };
#endif
#ifdef CSV_LOGGER
    hooks.after_publish = [this](
        const std::vector<float>& torque,
        const RobotState<float>& state,
        const Observations<float>& observations,
        const std::vector<float>& positions,
        const std::vector<float>&)
    {
        CSVLogger(
            torque,
            state.motor_state.tau_est,
            observations.dof_pos,
            positions,
            observations.dof_vel);
    };
#endif
    runtime_core_.runInferenceCycle(
        joystick_fault_latch_.faulted(),
        hooks);
}

std::vector<float> RL_Real::Forward()
{
    return runtime_core_.forward();
}

// void RL_Real::ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
// {
//     this->imu = *msg;
// }

void RL_Real::GetState(RobotState<float> *state)
{
    if (state == nullptr || !sim)
    {
        ApplySafetyEvent(
            LWSafetyEvent::FeedbackReadFailed,
            "[Safety] MuJoCo state source is unavailable");
        return;
    }
    bool state_read = false;
    {
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
        RefreshMuJoCoPointersLocked();
        if (mj_data && mujoco_control_adapter_)
        {
            state_read = mujoco_control_adapter_->ReadState(
                *mj_data,
                state->motor_state.q,
                state->motor_state.dq,
                state->motor_state.tau_est,
                state->imu.quaternion,
                state->imu.gyroscope);
        }
    }
    if (!state_read)
    {
        ApplySafetyEvent(
            LWSafetyEvent::FeedbackReadFailed,
            "[Safety] MuJoCo state layout is unavailable");
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    if (command == nullptr)
    {
        ApplySafetyEvent(
            LWSafetyEvent::NullRobotCommand,
            "[Safety] Null LW command");
        return;
    }
    const auto& runtime_configuration = GetLWBaseRuntimeConfiguration();
    const auto& joint_mapping = runtime_configuration.joint_mapping;
    const auto& wheel_mask = runtime_configuration.wheel_mask;
    const auto& torque_limits = runtime_configuration.torque_limits;
    const std::size_t num_dofs = runtime_configuration.num_dofs;

    for (std::size_t i = 0; i < num_dofs; ++i)
    {
        const int motor_id = joint_mapping[i];
        
        this->lw_low_command.motorCmd[motor_id].Kp = command->motor_command.kp[i];
        this->lw_low_command.motorCmd[motor_id].Kd = command->motor_command.kd[i];

        if (wheel_mask[i] != 0U) {

            this->lw_low_command.motorCmd[motor_id].action_set = command->motor_command.dq[i];

        } else {
            this->lw_low_command.motorCmd[motor_id].action_set = command->motor_command.q[i];
        }
    }
    
    this->lw_low_command.motors_disable = false;
    // this->lw_sdk.SendCmdData(this->lw_low_command);

    if (sim)
    {
        bool sink_available = false;
        LWSimTorqueValidation validation{
            LWSimTorqueFailure::SizeMismatch,
            0};
        {
            const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
            RefreshMuJoCoPointersLocked();
            sink_available = mj_data && mujoco_control_adapter_;
            if (sink_available)
            {
                validation = mujoco_control_adapter_->ApplyCommand(
                    *mj_data,
                    command->motor_command.q,
                    command->motor_command.dq,
                    command->motor_command.tau,
                    command->motor_command.kp,
                    command->motor_command.kd,
                    torque_limits,
                    mujoco_tau_candidates_,
                    mujoco_tau_bounded_);
            }
        }
        if (!sink_available)
        {
            ApplySafetyEvent(
                LWSafetyEvent::ControlCommandIncomplete,
                "[Safety] MuJoCo command sink is unavailable");
            return;
        }
        if (!validation.valid())
        {
            ApplySafetyEvent(
                LWSafetyEvent::SimulationActuatorCommandInvalid,
                std::string("[Safety] Invalid final MuJoCo actuator torque: ")
                    + validation.failureName()
                    + " at joint " + std::to_string(validation.index));
            return;
        }
    }
}

void RL_Real::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<LWJoystickDevice>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        LatchJoystickFault(
            {LWJoystickSampleStatus::Disconnected, EBADF, -1});
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Real::LatchJoystickFault(
    const LWJoystickSampleResult& result) noexcept
{
    runtime_core_.reportSafetyEvent(LWSafetyEvent::JoystickUnavailable);
    LWClearJoystickState(this->sys_js_button, this->sys_js_axis);
    joystick_input_mailbox_.clear(Input::Gamepad::None);
    this->sys_js_active = false;

    if (joystick_fault_latch_.latch())
    {
        std::cerr << LOGGER::ERROR
                  << "[Safety] Joystick input latched unavailable"
                  << ", status=" << static_cast<int>(result.status)
                  << ", errno=" << result.error_number
                  << ", bytes=" << result.bytes_read
                  << ". Velocity commands will remain zero and the current "
                     "simulation FSM will remain active. Restart is required "
                     "to restore joystick input."
                  << std::endl;
    }
}

void RL_Real::ApplyPendingInput()
{
    (void)joystick_input_mailbox_.read(joystick_input_snapshot_);
    const auto& input = joystick_input_snapshot_;
    this->control.x = input.x;
    this->control.y = input.y;
    this->control.yaw = input.yaw;
    if (input.event_sequence != consumed_gamepad_sequence_)
    {
        consumed_gamepad_sequence_ = input.event_sequence;
        this->control.SetGamepad(input.event);
    }
}

void RL_Real::ApplyJoystickFaultGate() noexcept
{
    if (!joystick_fault_latch_.faulted())
    {
        return;
    }

    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
    this->control.current_gamepad = Input::Gamepad::None;
    this->control.last_gamepad = Input::Gamepad::None;
}

void RL_Real::GetSysJoystick()
{
    LWBeginJoystickCycle(this->sys_js_button);

    if (joystick_fault_latch_.faulted())
    {
        return;
    }

    if (!this->sys_js || !this->sys_js->isFound())
    {
        LatchJoystickFault(
            {LWJoystickSampleStatus::Disconnected, EBADF, -1});
        return;
    }

    while (true)
    {
        const LWJoystickSampleResult sample_result =
            this->sys_js->sample(&this->sys_js_event);
        if (sample_result.status == LWJoystickSampleStatus::NoData)
        {
            break;
        }
        if (!sample_result.hasEvent())
        {
            LatchJoystickFault(sample_result);
            return;
        }

        const LWJoystickEventResult event_result =
            LWApplyJoystickEvent(
                this->sys_js_event,
                this->sys_js_button,
                this->sys_js_axis,
                this->sys_js_max_value,
                this->axis_deadzone);
        if (event_result == LWJoystickEventResult::ButtonOutOfRange
            || event_result == LWJoystickEventResult::AxisOutOfRange)
        {
            static size_t invalid_event_count = 0;
            ++invalid_event_count;
            if (invalid_event_count == 1 || invalid_event_count % 100 == 0)
            {
                std::cerr << LOGGER::WARNING
                          << "[Safety] Ignoring out-of-range joystick event"
                          << ", type=" << static_cast<int>(this->sys_js_event.type)
                          << ", number=" << static_cast<int>(this->sys_js_event.number)
                          << ", occurrences=" << invalid_event_count
                          << std::endl;
            }
        }
        else if (event_result == LWJoystickEventResult::InvalidConfiguration)
        {
            LatchJoystickFault(
                {LWJoystickSampleStatus::Error, EINVAL, 0});
            return;
        }
    }

#ifdef JOYSTICK_1
    // 修改为自己的手柄映射
    if (this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::B);
    if (this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::X);
    if (this->sys_js_button[4].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::Y);
    if (this->sys_js_button[6].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB);
    if (this->sys_js_button[7].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB);
    if (this->sys_js_button[13].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LStick);
    if (this->sys_js_button[14].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadRight);
    if (this->sys_js_button[6].pressed && this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_A);
    if (this->sys_js_button[6].pressed && this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_B);
    if (this->sys_js_button[6].pressed && this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_X);
    if (this->sys_js_button[6].pressed && this->sys_js_button[4].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_Y);
    if (this->sys_js_button[6].pressed && this->sys_js_button[13].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[6].pressed && this->sys_js_button[14].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[6].pressed && this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[6].pressed && this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[6].pressed && this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[6].pressed && this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[7].pressed && this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_A);
    if (this->sys_js_button[7].pressed && this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_B);
    if (this->sys_js_button[7].pressed && this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_X);
    if (this->sys_js_button[7].pressed && this->sys_js_button[4].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_Y);
    if (this->sys_js_button[7].pressed && this->sys_js_button[13].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[7].pressed && this->sys_js_button[14].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[7].pressed && this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[7].pressed && this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[7].pressed && this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[7].pressed && this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[6].pressed && this->sys_js_button[7].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_RB);

    // 通过sys_js_max_value将各项指令归一化
    const auto& velocity_limits =
        GetLWBaseRuntimeConfiguration().vel_command;
    float ly = (-float(this->sys_js_axis[1]) / float(this->sys_js_max_value)) * velocity_limits[0];
    float lx = (-float(this->sys_js_axis[0]) / float(this->sys_js_max_value)) * velocity_limits[1];
    float rx = (-float(this->sys_js_axis[2]) / float(this->sys_js_max_value)) * velocity_limits[2];

#endif

#ifdef JOYSTICK_2
    // 修改为自己的手柄映射
    if (this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) joystick_input_mailbox_.publishEvent(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) joystick_input_mailbox_.publishEvent(Input::Gamepad::LB_RB);

    // 通过sys_js_max_value将各项指令归一化
    // float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    // float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    // float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    const auto& velocity_limits =
        GetLWBaseRuntimeConfiguration().vel_command;
    float ly = (-float(this->sys_js_axis[1]) / float(this->sys_js_max_value)) * velocity_limits[0];
    float lx = (-float(this->sys_js_axis[0]) / float(this->sys_js_max_value)) * velocity_limits[1];
    float rx = (-float(this->sys_js_axis[3]) / float(this->sys_js_max_value)) * velocity_limits[2];
#endif

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        joystick_input_mailbox_.publishVelocity(ly, lx, rx);
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        joystick_input_mailbox_.publishVelocity(
            0.0f,
            0.0f,
            0.0f);
        this->sys_js_active = false;
    }

    // // =====================[手柄一键触发速度追踪测试] =====================
    // static double test_start_time = -1.0;

    // // 设定触发按键
    // // 当你按下该键，且当前没有在测试中时，触发实验
    // if (this->sys_js_button[3].on_press && test_start_time < 0) {
    //     test_start_time = this->mj_data->time; // 记录测试起点的绝对仿真时间
    //     std::cout << "\n[INFO] 🚀 自动速度追踪实验开始！" << std::endl;
    // }

    // if (test_start_time > 0) {
    //     // 计算实验已经进行的时间
    //     double t = this->mj_data->time - test_start_time;

    //     // 设计经典的多段阶跃曲线 (Staircase Command)
    //     if (t < 1.0) {
    //         this->control.x = 0.0f;   // 原地站立预备
    //     } 
    //     else if (t < 4.0) {
    //         this->control.x = 0.3f;   // 前向巡航加速 
    //     }
    //     else if (t < 7.0) {
    //         this->control.x = 0.6f;   // 前向巡航加速 
    //     }
    //     else if (t < 10.0) {
    //         this->control.x = 0.9f;   // 前向巡航加速 
    //     }
    //     else if (t < 13.0) {
    //         this->control.x = 1.2f;   // 前向巡航加速 
    //     }
    //     else if (t < 16.0) {
    //         this->control.x = 1.5f;   // 前向巡航加速 
    //     }
    //     else {
    //         this->control.x = 0.0f;   // 实验结束
    //         this->control.yaw = 0.0f;
    //         test_start_time = -1.0;   // 状态复位，允许按键再次触发
    //         std::cout << "[INFO] ✅ 自动速度追踪实验结束！" << std::endl;
    //     }

    //     this->control.y = 0.0f;
    //     // this->control.yaw = 0.0f;
    //     this->sys_js_active = true;
    // } else {
    //     // 非测试状态下，保持静止
    //     this->control.x = 0.0f;
    //     this->control.y = 0.0f;
    //     this->control.yaw = 0.0f;
    //     this->sys_js_active = false;
    // }
    
    // // =====================[手柄一键触发冲击测试] =====================
    // // 设定冲击力量级 (例如 150 牛顿)
    // double impact_magnitude = 200.0; 
    
    // // // 触发右侧向冲击 
    // // if (this->sys_js_button[3].on_press && g_impact_start_time < 0) {
    // //     g_impact_start_time = this->mj_data->time;
    // //     g_impact_force_x = 0.0;
    // //     g_impact_force_y = impact_magnitude;  
    // //     std::cout << "\n[INFO] 💥 触发侧向(Y+)冲击力: " << impact_magnitude << " N" << std::endl;
    // // }
    // // // 触发左侧向冲击 
    // // if (this->sys_js_button[3].on_press && g_impact_start_time < 0) {
    // //     g_impact_start_time = this->mj_data->time;
    // //     g_impact_force_x = 0.0;
    // //     g_impact_force_y = -impact_magnitude;  
    // //     std::cout << "\n[INFO] 💥 触发侧向(Y-)冲击力: " << -impact_magnitude << " N" << std::endl;
    // // }
    // // 触发前方冲击
    // if (this->sys_js_button[3].on_press && g_impact_start_time < 0) {
    //     g_impact_start_time = this->mj_data->time;
    //     g_impact_force_x = -impact_magnitude;
    //     g_impact_force_y = 0.0;  
    //     std::cout << "\n[INFO] 💥 触发后向(X-)冲击力: " << -impact_magnitude << " N" << std::endl;
    // }
    // ============================================================================

    // if (this->control.current_gamepad == Input::Gamepad::DPadUp )
    // {
    //     if (!this->control.dpad_handled) { // 如果还没处理过
    //         this->control.gait_frequency += 0.1f;
    //         this->control.dpad_handled = true; // 标记为已处理
    //     }
    // }
    // else if (this->control.current_gamepad == Input::Gamepad::DPadDown )
    // {
    //     if (!this->control.dpad_handled) { // 如果还没处理过
    //         this->control.gait_frequency -= 0.1f;
    //         this->control.dpad_handled = true; // 标记为已处理
    //     }
    // }
    // else 
    // {
    //     this->control.dpad_handled = false;
    // }
}

int main(int argc, char **argv)
{
    try
    {
        LWSimShutdownCoordinator shutdown_coordinator;
        LWSigintWaiter sigint_waiter(
            [&shutdown_coordinator]() { shutdown_coordinator.Request(); });
        rclcpp::init(
            argc,
            argv,
            rclcpp::InitOptions(),
            rclcpp::SignalHandlerOptions::SigTerm);
        auto rl_sar = std::make_shared<RL_Real>(argc, argv);
        shutdown_coordinator.Bind(
            [weak_rl_sar = std::weak_ptr<RL_Real>(rl_sar)]()
            {
                if (const std::shared_ptr<RL_Real> locked = weak_rl_sar.lock())
                {
                    locked->RequestSimulationStop();
                }
            });
        std::exception_ptr ros_error;
        std::thread ros_thread([&]() {
            ros_error = RunLWSimShutdownBoundWorker(
                shutdown_coordinator,
                [&]() { rclcpp::spin(rl_sar->ros2_node); });
        });

        std::exception_ptr main_thread_error;
        try
        {
            if (rl_sar->sim)
            {
                rl_sar->sim->RenderLoop();
                rl_sar->RequestSimulationStop();
            }
            if (shutdown_coordinator.requested())
            {
                std::cout << LOGGER::INFO
                          << "Shutdown requested, exiting Sim2Sim..."
                          << std::endl;
            }
        }
        catch (...)
        {
            main_thread_error = std::current_exception();
            shutdown_coordinator.Request();
        }

        try
        {
            rclcpp::shutdown();
        }
        catch (...)
        {
            if (!main_thread_error)
            {
                main_thread_error = std::current_exception();
            }
            shutdown_coordinator.Request();
        }
        if (ros_thread.joinable())
        {
            ros_thread.join();
        }
        sigint_waiter.ShutdownAndKeepBlocked();
        shutdown_coordinator.Unbind();
        sigint_waiter.RethrowWaitError();
        rl_sar->RethrowPhysicsError();
        if (ros_error)
        {
            std::rethrow_exception(ros_error);
        }
        if (main_thread_error)
        {
            std::rethrow_exception(main_thread_error);
        }
    }
    catch (const std::exception& error)
    {
        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }
        std::cerr << LOGGER::ERROR
                  << "[LW Sim2Sim] Fatal lifecycle error: "
                  << error.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }
        std::cerr << LOGGER::ERROR
                  << "[LW Sim2Sim] Unknown lifecycle error" << std::endl;
        return 1;
    }
    return 0;
}
