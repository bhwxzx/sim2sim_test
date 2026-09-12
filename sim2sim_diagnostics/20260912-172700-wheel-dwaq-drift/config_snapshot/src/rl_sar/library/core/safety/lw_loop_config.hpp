#ifndef LW_LOOP_CONFIG_HPP
#define LW_LOOP_CONFIG_HPP

#include "loop.hpp"
#include "rl_sdk.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

inline std::chrono::nanoseconds LWReadNonnegativeDuration(
    const YamlParams& params,
    const std::string& key,
    float default_seconds)
{
    const float seconds = params.Get<float>(key, default_seconds);
    if (!std::isfinite(seconds) || seconds < 0.0f)
    {
        throw std::runtime_error("LW " + key + " must be finite and nonnegative");
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(seconds));
}

inline std::uint64_t LWReadNonnegativeCount(
    const YamlParams& params,
    const std::string& key,
    int default_value)
{
    const int value = params.Get<int>(key, default_value);
    if (value < 0)
    {
        throw std::runtime_error("LW " + key + " must be nonnegative");
    }
    return static_cast<std::uint64_t>(value);
}

inline LoopConfig BuildLWControlLoopConfig(const YamlParams& params)
{
    LoopConfig config = LoopConfig::FromSeconds(params.Get<float>("dt"));
    config.cpu_affinity = params.Get<int>("control_loop_cpu", -1);
    config.realtime_priority =
        params.Get<int>("control_loop_realtime_priority", 0);
    config.require_realtime =
        params.Get<bool>("control_loop_require_realtime", false);
    config.timing_policy.degraded_consecutive_misses =
        LWReadNonnegativeCount(
            params,
            "control_loop_degraded_consecutive_misses",
            3);
    config.timing_policy.degraded_lateness = LWReadNonnegativeDuration(
        params,
        "control_loop_degraded_lateness",
        0.02f);
    config.timing_policy.fatal_consecutive_misses =
        LWReadNonnegativeCount(
            params,
            "control_loop_fatal_consecutive_misses",
            0);
    config.timing_policy.fatal_lateness = LWReadNonnegativeDuration(
        params,
        "control_loop_fatal_lateness",
        0.0f);
    return config;
}

#endif // LW_LOOP_CONFIG_HPP
