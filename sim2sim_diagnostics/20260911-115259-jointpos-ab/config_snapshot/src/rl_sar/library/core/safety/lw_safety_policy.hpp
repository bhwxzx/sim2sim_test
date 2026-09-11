#ifndef LW_SAFETY_POLICY_HPP
#define LW_SAFETY_POLICY_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

enum class LWSafetySeverity : int
{
    Diagnostic = 0,
    InputDegraded = 1,
    ControlledFallback = 2,
    HardDisable = 3,
    FatalShutdown = 4
};

enum class LWSafetyAction
{
    Observe,
    InhibitInput,
    PassiveDamping,
    HardDisable,
    HardDisableAndShutdown,
    AbortStartup,
    OrderlyShutdown
};

enum class LWSafetyEvent : std::size_t
{
    FeedbackParserError,
    TorqueLimitWarning,
    JoystickUnavailable,
    JoystickLoopException,
    ControlTimingDegraded,
    InferenceLoopException,
    PolicyActionInvalid,
    PolicyOutputInvalid,
    PolicyConfigurationInvalid,
    PolicyInputUnavailable,
    PolicyOutputUnavailable,
    MotorHardwareFault,
    ControlLoopException,
    UnknownLoopException,
    ControlTimingFatal,
    SensorDataStale,
    WaitingDisableIncomplete,
    FeedbackInvalid,
    FsmStateMissing,
    AttitudeLimitExceeded,
    RobotCommandInvalid,
    NullRobotCommand,
    FeedbackReadFailed,
    ControlCommandIncomplete,
    SimulationActuatorCommandInvalid,
    StartupSerialInitializationFailed,
    StartupInitialDisableIncomplete,
    StartupLoopStartFailed,
    NormalShutdown,
    Count
};

struct LWSafetyDecision
{
    LWSafetyEvent event;
    LWSafetySeverity severity;
    LWSafetyAction action;
    const char* name;
    bool latched;
    bool restart_required;
};

constexpr std::array<LWSafetyDecision,
                     static_cast<std::size_t>(LWSafetyEvent::Count)>
    kLWSafetyDecisions{{
        {LWSafetyEvent::FeedbackParserError, LWSafetySeverity::Diagnostic,
         LWSafetyAction::Observe, "feedback-parser-error", false, false},
        {LWSafetyEvent::TorqueLimitWarning, LWSafetySeverity::Diagnostic,
         LWSafetyAction::Observe, "torque-limit-warning", false, false},
        {LWSafetyEvent::JoystickUnavailable, LWSafetySeverity::InputDegraded,
         LWSafetyAction::InhibitInput, "joystick-unavailable", true, true},
        {LWSafetyEvent::JoystickLoopException, LWSafetySeverity::InputDegraded,
         LWSafetyAction::InhibitInput, "joystick-loop-exception", true, true},
        {LWSafetyEvent::ControlTimingDegraded, LWSafetySeverity::InputDegraded,
         LWSafetyAction::InhibitInput, "control-timing-degraded", true, true},
        {LWSafetyEvent::InferenceLoopException, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "inference-loop-exception", true, true},
        {LWSafetyEvent::PolicyActionInvalid, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "policy-action-invalid", true, true},
        {LWSafetyEvent::PolicyOutputInvalid, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "policy-output-invalid", true, true},
        {LWSafetyEvent::PolicyConfigurationInvalid, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "policy-configuration-invalid", true, true},
        {LWSafetyEvent::PolicyInputUnavailable, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "policy-input-unavailable", true, true},
        {LWSafetyEvent::PolicyOutputUnavailable, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "policy-output-unavailable", true, true},
        {LWSafetyEvent::MotorHardwareFault, LWSafetySeverity::HardDisable,
         LWSafetyAction::HardDisable, "motor-hardware-fault", true, true},
        {LWSafetyEvent::ControlLoopException, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "control-loop-exception", true, true},
        {LWSafetyEvent::UnknownLoopException, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "unknown-loop-exception", true, true},
        {LWSafetyEvent::ControlTimingFatal, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "control-timing-fatal", true, true},
        {LWSafetyEvent::SensorDataStale, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "sensor-data-stale", true, true},
        {LWSafetyEvent::WaitingDisableIncomplete, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "waiting-disable-incomplete", true, true},
        {LWSafetyEvent::FeedbackInvalid, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "feedback-invalid", true, true},
        {LWSafetyEvent::FsmStateMissing, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "fsm-state-missing", true, true},
        {LWSafetyEvent::AttitudeLimitExceeded, LWSafetySeverity::ControlledFallback,
         LWSafetyAction::PassiveDamping, "attitude-limit-exceeded", true, true},
        {LWSafetyEvent::RobotCommandInvalid, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "robot-command-invalid", true, true},
        {LWSafetyEvent::NullRobotCommand, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "null-robot-command", true, true},
        {LWSafetyEvent::FeedbackReadFailed, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "feedback-read-failed", true, true},
        {LWSafetyEvent::ControlCommandIncomplete, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "control-command-incomplete", true, true},
        {LWSafetyEvent::SimulationActuatorCommandInvalid, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::HardDisableAndShutdown, "simulation-actuator-command-invalid", true, true},
        {LWSafetyEvent::StartupSerialInitializationFailed, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::AbortStartup, "startup-serial-initialization-failed", true, true},
        {LWSafetyEvent::StartupInitialDisableIncomplete, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::AbortStartup, "startup-initial-disable-incomplete", true, true},
        {LWSafetyEvent::StartupLoopStartFailed, LWSafetySeverity::FatalShutdown,
         LWSafetyAction::AbortStartup, "startup-loop-start-failed", true, true},
        {LWSafetyEvent::NormalShutdown, LWSafetySeverity::Diagnostic,
         LWSafetyAction::OrderlyShutdown, "normal-shutdown", false, false},
    }};

static_assert(
    kLWSafetyDecisions.size() == static_cast<std::size_t>(LWSafetyEvent::Count),
    "Every LW safety event must have an explicit decision");

constexpr const LWSafetyDecision& LWSafetyDecisionFor(
    LWSafetyEvent event) noexcept
{
    return kLWSafetyDecisions[static_cast<std::size_t>(event)];
}

constexpr const char* LWSafetySeverityName(LWSafetySeverity severity) noexcept
{
    switch (severity)
    {
    case LWSafetySeverity::Diagnostic: return "S0-diagnostic";
    case LWSafetySeverity::InputDegraded: return "S1-input-degraded";
    case LWSafetySeverity::ControlledFallback: return "S2-controlled-fallback";
    case LWSafetySeverity::HardDisable: return "S3-hard-disable";
    case LWSafetySeverity::FatalShutdown: return "S4-fatal-shutdown";
    }
    return "unknown";
}

constexpr const char* LWSafetyActionName(LWSafetyAction action) noexcept
{
    switch (action)
    {
    case LWSafetyAction::Observe: return "observe";
    case LWSafetyAction::InhibitInput: return "inhibit-input";
    case LWSafetyAction::PassiveDamping: return "passive-damping";
    case LWSafetyAction::HardDisable: return "hard-disable";
    case LWSafetyAction::HardDisableAndShutdown:
        return "hard-disable-and-shutdown";
    case LWSafetyAction::AbortStartup: return "abort-startup";
    case LWSafetyAction::OrderlyShutdown: return "orderly-shutdown";
    }
    return "unknown";
}

struct LWSafetySnapshot
{
    std::uint64_t sequence = 0;
    LWSafetyEvent latest_event = LWSafetyEvent::Count;
    LWSafetySeverity highest_severity = LWSafetySeverity::Diagnostic;
    bool controlled_fallback_latched = false;
};

class LWSafetySupervisor
{
public:
    const LWSafetyDecision& report(LWSafetyEvent event) noexcept
    {
        const auto& decision = LWSafetyDecisionFor(event);
        const LWSafetyEvent previous = latest_event_.exchange(
            event, std::memory_order_acq_rel);
        if (previous != event)
        {
            sequence_.fetch_add(1, std::memory_order_release);
        }

        int observed = highest_severity_.load(std::memory_order_relaxed);
        const int requested = static_cast<int>(decision.severity);
        while (observed < requested
               && !highest_severity_.compare_exchange_weak(
                   observed,
                   requested,
                   std::memory_order_release,
                   std::memory_order_relaxed))
        {
        }
        if (decision.action == LWSafetyAction::PassiveDamping)
        {
            controlled_fallback_latched_.store(true, std::memory_order_release);
        }
        return decision;
    }

    bool controlledFallbackLatched() const noexcept
    {
        return controlled_fallback_latched_.load(std::memory_order_acquire);
    }

    LWSafetySnapshot snapshot() const noexcept
    {
        LWSafetySnapshot result;
        result.sequence = sequence_.load(std::memory_order_acquire);
        result.latest_event = latest_event_.load(std::memory_order_relaxed);
        result.highest_severity = static_cast<LWSafetySeverity>(
            highest_severity_.load(std::memory_order_acquire));
        result.controlled_fallback_latched = controlledFallbackLatched();
        return result;
    }

private:
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<LWSafetyEvent> latest_event_{LWSafetyEvent::Count};
    std::atomic<int> highest_severity_{
        static_cast<int>(LWSafetySeverity::Diagnostic)};
    std::atomic<bool> controlled_fallback_latched_{false};
};

#endif // LW_SAFETY_POLICY_HPP
