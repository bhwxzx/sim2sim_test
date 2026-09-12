#ifndef LW_RUNTIME_CORE_HPP
#define LW_RUNTIME_CORE_HPP

#include "lw_control_safety.hpp"
#include "lw_safety_policy.hpp"
#include "rl_sdk.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct LWRuntimeSafetySnapshot
{
    LWSafetySnapshot decision;
    bool terminal_latched = false;
    bool shutdown_requested = false;
};

struct LWInferenceTraceSnapshot
{
    std::uint64_t generation = 0;
    std::uint64_t frame = 0;
    std::uint64_t source_input_sequence = 0;
    std::chrono::steady_clock::time_point source_state_time{};
    Observations<float> observations;
    std::vector<float> output_dof_pos;
    std::vector<float> output_dof_vel;
    std::vector<float> output_dof_tau;
};

struct LWControlCycleHooks
{
    std::function<void()> apply_input;
    std::function<void()> apply_keyboard;
    std::function<bool()> platform_ready;
    std::function<bool()> platform_precheck;
    std::function<void()> adapter_controls;
    std::function<void()> after_command_delivery;
};

struct LWInferenceCycleHooks
{
    std::function<void(
        Observations<float>&,
        const RobotState<float>&)> mutate_observation;
    std::function<void()> after_forward;
    std::function<void(
        const std::vector<float>&,
        const RobotState<float>&,
        const Observations<float>&,
        const std::vector<float>&,
        const std::vector<float>&)> after_publish;
};

// Platform-neutral LW control, inference, validation, and safety orchestration.
// Real hardware and MuJoCo supply only lifecycle and I/O hooks.
class LWRuntimeCore
{
public:
    using SafetySink = std::function<void(
        const LWSafetyDecision&,
        const std::string&)>;

    void bind(RL& rl, SafetySink safety_sink)
    {
        rl_ = &rl;
        safety_sink_ = std::move(safety_sink);
    }

    const LWSafetyDecision& reportSafetyEvent(
        LWSafetyEvent event,
        const std::string& reason = {}) noexcept
    {
        const auto& decision = safety_supervisor_.report(event);
        if (decision.action == LWSafetyAction::HardDisable
            || decision.action == LWSafetyAction::HardDisableAndShutdown
            || decision.action == LWSafetyAction::AbortStartup)
        {
            terminal_latched_.store(true, std::memory_order_release);
        }
        if (decision.action == LWSafetyAction::HardDisableAndShutdown
            || decision.action == LWSafetyAction::AbortStartup
            || decision.action == LWSafetyAction::OrderlyShutdown)
        {
            shutdown_requested_.store(true, std::memory_order_release);
        }

        if (safety_sink_)
        {
            try
            {
                safety_sink_(decision, reason);
            }
            catch (...)
            {
                // A safety adapter must never unwind into a real-time loop.
            }
        }
        return decision;
    }

    void handlePolicyOutputFault(LWPolicyOutputStatus) noexcept
    {
        reportSafetyEvent(
            LWSafetyEvent::PolicyOutputUnavailable,
            "[Safety] LW policy output became stale or incomplete");
    }

    void handlePolicyInputFault(LWPolicyInputStatus status) noexcept
    {
        reportSafetyEvent(
            LWSafetyEvent::PolicyInputUnavailable,
            std::string("[Safety] LW policy input provenance is ")
                + LWPolicyInputStatusName(status));
    }

    void handleLoopError(
        const std::string& loop_name,
        std::exception_ptr error,
        const std::function<void()>& joystick_fault = {}) noexcept
    {
        std::string message = "unknown exception";
        try
        {
            if (error)
            {
                std::rethrow_exception(error);
            }
        }
        catch (const std::exception& exception)
        {
            message = exception.what();
        }
        catch (...)
        {
        }

        if (loop_name == "loop_joystick")
        {
            if (joystick_fault)
            {
                try
                {
                    joystick_fault();
                }
                catch (...)
                {
                }
            }
            reportSafetyEvent(LWSafetyEvent::JoystickLoopException);
            return;
        }
        if (loop_name == "loop_rl")
        {
            reportSafetyEvent(
                LWSafetyEvent::InferenceLoopException,
                "[Loop] Inference callback exception - error: " + message);
            return;
        }
        reportSafetyEvent(
            loop_name == "loop_control"
                ? LWSafetyEvent::ControlLoopException
                : LWSafetyEvent::UnknownLoopException,
            "[Loop] Fatal callback exception - name: " + loop_name
                + ", error: " + message);
    }

    void handleControlTiming(
        bool fatal,
        std::uint64_t missed_deadlines) noexcept
    {
        if (!fatal)
        {
            reportSafetyEvent(LWSafetyEvent::ControlTimingDegraded);
            return;
        }
        reportSafetyEvent(
            LWSafetyEvent::ControlTimingFatal,
            "[Timing] Fatal control-loop timing fault after "
                + std::to_string(missed_deadlines)
                + " missed deadlines; hard-disable threshold was explicitly enabled");
    }

    bool inputInhibited() const noexcept
    {
        return static_cast<int>(safety_supervisor_.snapshot().highest_severity)
            >= static_cast<int>(LWSafetySeverity::InputDegraded);
    }

    bool controlledFallbackLatched() const noexcept
    {
        return safety_supervisor_.controlledFallbackLatched();
    }

    bool terminalLatched() const noexcept
    {
        return terminal_latched_.load(std::memory_order_acquire);
    }

    LWRuntimeSafetySnapshot safetySnapshot() const noexcept
    {
        return {
            safety_supervisor_.snapshot(),
            terminalLatched(),
            shutdown_requested_.load(std::memory_order_acquire)};
    }

    LWSafetySupervisor& safetySupervisor() noexcept
    {
        return safety_supervisor_;
    }

    bool readInferenceTrace(LWInferenceTraceSnapshot& trace) const
    {
        return inference_trace_.read(trace);
    }

    bool acceptPolicyActions(
        const std::vector<float>& actions,
        size_t num_dofs)
    {
        const LWValidationResult result =
            LWValidatePolicyActions(actions, num_dofs);
        if (result.valid())
        {
            return true;
        }
        reportSafetyEvent(
            LWSafetyEvent::PolicyActionInvalid,
            "[Safety] Invalid LW policy action: "
                + result.failureDescription());
        return false;
    }

    bool acceptPolicyOutputs(
        const std::vector<float>& positions,
        const std::vector<float>& velocities,
        const std::vector<float>& torques,
        size_t num_dofs)
    {
        const LWValidationResult result = LWValidatePolicyOutputs(
            positions,
            velocities,
            torques,
            num_dofs);
        if (result.valid())
        {
            return true;
        }
        reportSafetyEvent(
            LWSafetyEvent::PolicyOutputInvalid,
            "[Safety] Invalid LW policy output: "
                + result.failureDescription());
        return false;
    }

    void publishInitialPolicyInput()
    {
        requireBound();
        publishPolicyInput(std::chrono::steady_clock::now());
    }

    void runControlCycle(const LWControlCycleHooks& hooks)
    {
        requireBound();
        call(hooks.apply_input);
        call(hooks.apply_keyboard);
        inhibitVelocityIfRequired();

        rl_->GetState(&rl_->robot_state);
        const auto state_capture_time = std::chrono::steady_clock::now();
        if (terminalLatched())
        {
            return;
        }
        if (hooks.platform_ready && !hooks.platform_ready())
        {
            return;
        }
        // Attitude S2 can continue to damping; invalid feedback/FSM is terminal.
        if (!validateFeedbackAndAttitude() && terminalLatched())
        {
            return;
        }
        if (hooks.platform_precheck && !hooks.platform_precheck())
        {
            return;
        }

        const bool fallback_before_controller = controlledFallbackLatched();
        if (fallback_before_controller)
        {
            applyControlledFallbackCommand();
        }
        else
        {
            rl_->StateController(
                &rl_->robot_state,
                &rl_->robot_command,
                false);
        }
        if (controlledFallbackLatched())
        {
            applyControlledFallbackCommand();
        }
        inhibitVelocityIfRequired();
        publishPolicyInput(state_capture_time);

        call(hooks.adapter_controls);
        if (!validateFeedbackAndAttitude())
        {
            if (terminalLatched())
            {
                return;
            }
            // A transition or adapter update exposed an excessive attitude.
            // Replace its command before final validation and delivery.
            applyControlledFallbackCommand();
        }
        if (!validateCommandForSend(rl_->robot_command) || terminalLatched())
        {
            return;
        }

        rl_->control.ClearInput();
        rl_->SetCommand(&rl_->robot_command);
        call(hooks.after_command_delivery);
    }

    void runInferenceCycle(
        bool external_input_fault,
        const LWInferenceCycleHooks& hooks = {})
    {
        requireBound();
        if (terminalLatched() || controlledFallbackLatched())
        {
            return;
        }
        LWPolicyActivation published_activation;
        if (!rl_->ReadLWPolicyActivationForInference(
                published_activation)
            || !published_activation.definition)
        {
            inference_activation_ = {};
            inference_motion_reference_ = nullptr;
            return;
        }
        if (!inference_activation_.definition
            || inference_activation_.generation
                != published_activation.generation)
        {
            inference_activation_ = published_activation;
            resetInferenceWorkspace(inference_activation_);
        }
        const LWPolicyActivation* activation = &inference_activation_;

        if (!policy_input_snapshot_.read(inference_policy_input_))
        {
            return;
        }
        const LWPolicyInputSnapshot& policy_input =
            inference_policy_input_;
        const auto policy_data_max_age =
            rl_->GetLWPolicyOutputMaxAge(*activation);
        const auto input_now = std::chrono::steady_clock::now();
        const LWPolicyInputStatus input_status = EvaluateLWPolicyInput(
            &policy_input,
            activation->generation,
            input_now,
            policy_data_max_age,
            last_inference_input_generation_,
            last_inference_input_sequence_,
            last_inference_state_capture_time_);
        if (input_status != LWPolicyInputStatus::Ready)
        {
            if (LWPolicyInputRequiresFallback(input_status))
            {
                handlePolicyInputFault(input_status);
            }
            return;
        }
        const RobotState<float>& local_state = policy_input.robot_state;
        const LWControlSnapshot& local_control = policy_input.control;
        LWControlSnapshot effective_control = local_control;
        if (external_input_fault || inputInhibited())
        {
            effective_control.x = 0.0f;
            effective_control.y = 0.0f;
            effective_control.yaw = 0.0f;
        }

        const auto& policy_configuration =
            activation->definition->runtime;
        const bool needs_motion_reference =
            policy_configuration.needs_motion_reference;
        inference_motion_reference_ = nullptr;
        if (needs_motion_reference)
        {
            inference_motion_reference_ =
                rl_->LoadLWMotionReference();
            if (!inference_motion_reference_
                || inference_motion_reference_->generation
                    != activation->generation)
            {
                return;
            }
            if (policy_configuration.needs_motion_command
                && (inference_motion_reference_->joint_pos.size()
                        != policy_configuration.num_dofs
                    || inference_motion_reference_->joint_vel.size()
                        != policy_configuration.num_dofs))
            {
                return;
            }
            if (policy_configuration.needs_motion_anchor_orientation
                && (inference_motion_reference_->anchor_quat.size() != 4
                    || inference_motion_reference_->init_quat.size() != 4))
            {
                return;
            }
        }
        last_inference_input_generation_ = policy_input.generation;
        last_inference_input_sequence_ = policy_input.sequence;
        last_inference_state_capture_time_ =
            policy_input.state_capture_time;
        ++inference_frame_;
        inference_obs_.ang_vel = local_state.imu.gyroscope;
        inference_obs_.commands = {
            effective_control.x,
            effective_control.y,
            effective_control.yaw};
        inference_obs_.base_quat = local_state.imu.quaternion;
        inference_obs_.dof_pos = local_state.motor_state.q;
        inference_obs_.dof_vel = local_state.motor_state.dq;
        if (hooks.mutate_observation)
        {
            hooks.mutate_observation(inference_obs_, local_state);
        }

        inference_gait_phase_time_ +=
            policy_configuration.period_seconds
            * effective_control.gait_frequency;
        while (inference_gait_phase_time_ >= 1.0f)
        {
            inference_gait_phase_time_ -= 1.0f;
        }
        const float command_norm = std::sqrt(
            effective_control.x * effective_control.x
            + effective_control.y * effective_control.y
            + effective_control.yaw * effective_control.yaw);
        const float is_moving = command_norm > 0.1f ? 1.0f : 0.0f;
        constexpr float pi = 3.14159265358979323846f;
        inference_obs_.gait_phase = {
            is_moving * std::sin(2.0f * pi * inference_gait_phase_time_),
            is_moving * std::cos(2.0f * pi * inference_gait_phase_time_)};

        const bool forward_succeeded = runForwardIntoActions();
        call(hooks.after_forward);
        if (!forward_succeeded
            || terminalLatched()
            || controlledFallbackLatched())
        {
            return;
        }

        rl_->ComputeLWOutput(
            policy_configuration,
            inference_obs_,
            inference_obs_.actions,
            inference_output_dof_pos_,
            inference_output_dof_vel_,
            inference_output_dof_tau_);
        const size_t num_dofs = policy_configuration.num_dofs;
        if (!acceptPolicyOutputs(
                inference_output_dof_pos_,
                inference_output_dof_vel_,
                inference_output_dof_tau_,
                num_dofs))
        {
            return;
        }

        if (rl_->TorqueProtect(
                inference_output_dof_tau_,
                policy_configuration))
        {
            reportSafetyEvent(LWSafetyEvent::TorqueLimitWarning);
        }
        const auto inference_completed_at =
            std::chrono::steady_clock::now();
        if (inference_completed_at < policy_input.state_capture_time
            || inference_completed_at - policy_input.state_capture_time
                > policy_data_max_age)
        {
            handlePolicyInputFault(
                inference_completed_at < policy_input.state_capture_time
                    ? LWPolicyInputStatus::Future
                    : LWPolicyInputStatus::Stale);
            return;
        }
        inference_output_frame_.generation = policy_input.generation;
        inference_output_frame_.sequence = 0;
        inference_output_frame_.frame = inference_frame_;
        inference_output_frame_.source_input_sequence = policy_input.sequence;
        inference_output_frame_.source_state_time =
            policy_input.state_capture_time;
        inference_output_frame_.dof_pos = inference_output_dof_pos_;
        inference_output_frame_.dof_vel = inference_output_dof_vel_;
        inference_output_frame_.dof_tau = inference_output_dof_tau_;
        if (!rl_->PublishLWPolicyOutput(
            inference_output_frame_, *activation))
        {
            return;
        }
        rl_->PublishLWPolicyProgress(
            activation->generation,
            inference_frame_);
        inference_trace_frame_.generation = activation->generation;
        inference_trace_frame_.frame = inference_frame_;
        inference_trace_frame_.source_input_sequence = policy_input.sequence;
        inference_trace_frame_.source_state_time =
            policy_input.state_capture_time;
        inference_trace_frame_.observations = inference_obs_;
        inference_trace_frame_.output_dof_pos = inference_output_dof_pos_;
        inference_trace_frame_.output_dof_vel = inference_output_dof_vel_;
        inference_trace_frame_.output_dof_tau = inference_output_dof_tau_;
        inference_trace_.publish(inference_trace_frame_);
        if (hooks.after_publish)
        {
            hooks.after_publish(
                inference_output_dof_tau_,
                local_state,
                inference_obs_,
                inference_output_dof_pos_,
                inference_output_dof_vel_);
        }
    }

    std::vector<float> forward()
    {
        if (!runForwardIntoActions())
        {
            return {};
        }
        return inference_obs_.actions;
    }

private:
    bool runForwardIntoActions()
    {
        requireBound();
        if (!inference_activation_.definition
            || !inference_activation_.definition->model)
        {
            return false;
        }
        const auto& definition = *inference_activation_.definition;
        const auto& policy_configuration = definition.runtime;
        rl_->ComputeLWObservationInto(
            policy_configuration,
            inference_obs_,
            inference_motion_reference_,
            inference_flat_obs_);

        const std::vector<float>* model_input = &inference_flat_obs_;
        const auto& history_indices =
            policy_configuration.observations_history;
        if (!history_indices.empty())
        {
            if (inference_frame_ == 1)
            {
                inference_history_obs_buf_.resetAll(inference_flat_obs_);
            }
            inference_history_obs_buf_.insert(inference_flat_obs_);
            inference_history_obs_buf_.getObsInto(
                history_indices,
                inference_history_obs_);
            model_input = &inference_history_obs_;
        }
        const InferenceRuntime::TensorView input_view = {
            model_input->data(), model_input->size()};
        definition.model->forwardInto(
            &input_view,
            1,
            {inference_obs_.actions.data(),
             inference_obs_.actions.size()});

        const size_t num_dofs = policy_configuration.num_dofs;
        if (!acceptPolicyActions(inference_obs_.actions, num_dofs))
        {
            return false;
        }

        const auto& upper =
            policy_configuration.clip_actions_upper;
        const auto& lower =
            policy_configuration.clip_actions_lower;
        // Bounds belong to the immutable policy definition and were validated
        // by ValidateLWPolicyConfiguration before preloading/activation.
        for (std::size_t index = 0; index < num_dofs; ++index)
        {
            inference_obs_.actions[index] = clamp(
                inference_obs_.actions[index],
                lower[index],
                upper[index]);
        }
        return true;
    }
    static void call(const std::function<void()>& hook)
    {
        if (hook)
        {
            hook();
        }
    }

    void requireBound() const
    {
        if (!rl_)
        {
            throw std::logic_error("LW runtime core is not bound");
        }
    }

    void inhibitVelocityIfRequired()
    {
        if (!inputInhibited())
        {
            return;
        }
        rl_->control.x = 0.0f;
        rl_->control.y = 0.0f;
        rl_->control.yaw = 0.0f;
    }

    void publishPolicyInput(
        std::chrono::steady_clock::time_point state_capture_time)
    {
        const auto activation = rl_->LoadLWPolicyActivation();
        control_policy_input_.generation =
            activation ? activation->generation : 0;
        control_policy_input_.sequence =
            next_policy_input_sequence_;
        control_policy_input_.state_capture_time = state_capture_time;
        control_policy_input_.robot_state = rl_->robot_state;
        control_policy_input_.control =
            {rl_->control.x,
             rl_->control.y,
             rl_->control.yaw,
             rl_->control.gait_frequency};
        if (policy_input_snapshot_.tryPublish(control_policy_input_))
        {
            ++next_policy_input_sequence_;
        }
    }

    bool validateFeedbackAndAttitude()
    {
        const size_t num_dofs =
            rl_->GetLWBaseRuntimeConfiguration().num_dofs;
        const LWValidationResult feedback_result =
            LWValidateFeedbackState(rl_->robot_state, num_dofs);
        if (!feedback_result.valid())
        {
            reportSafetyEvent(
                LWSafetyEvent::FeedbackInvalid,
                "[Safety] Invalid LW feedback: "
                    + feedback_result.failureDescription());
            return false;
        }
        if (!rl_->fsm.current_state_)
        {
            reportSafetyEvent(
                LWSafetyEvent::FsmStateMissing,
                "[Safety] LW FSM has no current state");
            return false;
        }

        constexpr float attitude_threshold_degrees = 75.0f;
        const std::string& state_name =
            rl_->fsm.current_state_->GetStateName();
        const LWAttitudeValidationResult attitude_result = LWValidateAttitude(
            state_name,
            rl_->robot_state.imu.quaternion,
            attitude_threshold_degrees);
        if (!attitude_result.safe)
        {
            reportSafetyEvent(
                LWSafetyEvent::AttitudeLimitExceeded,
                "[Safety] LW attitude limit exceeded in " + state_name + ": "
                    + attitude_result.failureDescription(
                        attitude_threshold_degrees));
            return false;
        }
        return true;
    }

    bool validateCommandForSend(const RobotCommand<float>& command)
    {
        const size_t num_dofs =
            rl_->GetLWBaseRuntimeConfiguration().num_dofs;
        const LWValidationResult command_result =
            LWValidateRobotCommand(command, num_dofs);
        if (!command_result.valid())
        {
            reportSafetyEvent(
                LWSafetyEvent::RobotCommandInvalid,
                "[Safety] Invalid LW command: "
                    + command_result.failureDescription());
            return false;
        }
        return true;
    }

    void applyControlledFallbackCommand()
    {
        rl_->control.x = 0.0f;
        rl_->control.y = 0.0f;
        rl_->control.yaw = 0.0f;
        rl_->control.ClearInput();
        if (!controlled_fallback_applied_.exchange(
                true, std::memory_order_acq_rel))
        {
            rl_->DeactivateLWPolicy();
        }
        if (rl_->fsm.current_state_
            && rl_->fsm.current_state_->GetStateName()
                != "RLFSMStatePassive")
        {
            rl_->fsm.RequestStateChange("RLFSMStatePassive");
            rl_->StateController(
                &rl_->robot_state,
                &rl_->robot_command,
                false);
        }
        LWBuildPassiveDampingCommand(
            rl_->robot_state,
            rl_->robot_command,
            rl_->GetLWBaseRuntimeConfiguration().num_dofs);
    }

    void resetInferenceWorkspace(const LWPolicyActivation& activation)
    {
        const auto& policy_configuration = activation.definition->runtime;
        const size_t num_dofs = policy_configuration.num_dofs;
        inference_frame_ = 0;
        last_inference_input_generation_ = 0;
        last_inference_input_sequence_ = 0;
        last_inference_state_capture_time_ = {};
        inference_gait_phase_time_ = 0.0f;
        inference_motion_reference_ = nullptr;
        inference_obs_ = {};
        inference_obs_.lin_vel = {0.0f, 0.0f, 0.0f};
        inference_obs_.ang_vel = {0.0f, 0.0f, 0.0f};
        inference_obs_.gravity_vec = {0.0f, 0.0f, -1.0f};
        inference_obs_.commands = {0.0f, 0.0f, 0.0f};
        inference_obs_.base_quat = {1.0f, 0.0f, 0.0f, 0.0f};
        inference_obs_.dof_pos =
            policy_configuration.default_dof_pos;
        inference_obs_.dof_vel.assign(num_dofs, 0.0f);
        inference_obs_.actions.assign(num_dofs, 0.0f);
        inference_obs_.gait_phase = {0.0f, 1.0f};
        inference_output_dof_pos_ = inference_obs_.dof_pos;
        inference_output_dof_vel_.assign(num_dofs, 0.0f);
        inference_output_dof_tau_.assign(num_dofs, 0.0f);
        inference_output_frame_.dof_pos.assign(num_dofs, 0.0f);
        inference_output_frame_.dof_vel.assign(num_dofs, 0.0f);
        inference_output_frame_.dof_tau.assign(num_dofs, 0.0f);
        inference_trace_frame_.observations = inference_obs_;
        inference_trace_frame_.output_dof_pos.assign(num_dofs, 0.0f);
        inference_trace_frame_.output_dof_vel.assign(num_dofs, 0.0f);
        inference_trace_frame_.output_dof_tau.assign(num_dofs, 0.0f);
        inference_flat_obs_.assign(
            activation.definition->dimensions.observation,
            0.0f);
        inference_obs_dims_.clear();
        inference_obs_dims_.reserve(
            policy_configuration.observation_layout.size());
        for (const auto& entry : policy_configuration.observation_layout)
        {
            inference_obs_dims_.push_back(
                static_cast<int>(entry.size));
        }
        const auto& history_indices =
            policy_configuration.observations_history;
        if (!history_indices.empty())
        {
            inference_history_obs_.assign(
                activation.definition->dimensions.model_input,
                0.0f);
            const int history_length =
                *std::max_element(
                    history_indices.begin(),
                    history_indices.end())
                + 1;
            inference_history_obs_buf_ = ObservationBuffer(
                1,
                inference_obs_dims_,
                history_length,
                policy_configuration.observations_history_priority);
        }
        else
        {
            inference_history_obs_.clear();
            inference_history_obs_buf_ = ObservationBuffer();
        }
    }

    RL* rl_ = nullptr;
    SafetySink safety_sink_;
    LWSafetySupervisor safety_supervisor_;
    std::atomic<bool> terminal_latched_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> controlled_fallback_applied_{false};
    LWSnapshotBuffer<LWPolicyInputSnapshot> policy_input_snapshot_;
    LWSnapshotBuffer<LWInferenceTraceSnapshot> inference_trace_;

    LWPolicyActivation inference_activation_{};
    LWPolicyInputSnapshot control_policy_input_;
    LWPolicyInputSnapshot inference_policy_input_;
    const LWMotionReferenceSnapshot* inference_motion_reference_ = nullptr;
    Observations<float> inference_obs_;
    std::vector<int> inference_obs_dims_;
    std::vector<float> inference_flat_obs_;
    ObservationBuffer inference_history_obs_buf_;
    std::vector<float> inference_history_obs_;
    std::vector<float> inference_output_dof_pos_;
    std::vector<float> inference_output_dof_vel_;
    std::vector<float> inference_output_dof_tau_;
    LWPolicyOutputFrame inference_output_frame_;
    LWInferenceTraceSnapshot inference_trace_frame_;
    std::uint64_t inference_frame_ = 0;
    std::uint64_t next_policy_input_sequence_ = 1;
    std::uint64_t last_inference_input_generation_ = 0;
    std::uint64_t last_inference_input_sequence_ = 0;
    std::chrono::steady_clock::time_point
        last_inference_state_capture_time_{};
    float inference_gait_phase_time_ = 0.0f;
};

#endif // LW_RUNTIME_CORE_HPP
