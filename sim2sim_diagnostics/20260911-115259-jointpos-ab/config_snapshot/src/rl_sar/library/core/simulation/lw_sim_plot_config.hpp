#ifndef LW_SIM_PLOT_CONFIG_HPP
#define LW_SIM_PLOT_CONFIG_HPP

#include <charconv>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

constexpr int kLWSimDefaultPlotRateHz = 100;
constexpr int kLWSimMinimumPlotRateHz = 1;
constexpr int kLWSimMaximumPlotRateHz = 200;

struct LWSimPlotConfiguration
{
    bool enabled = false;
    int rate_hz = kLWSimDefaultPlotRateHz;
};

inline LWSimPlotConfiguration ParseLWSimPlotConfiguration(
    int argc,
    char* const argv[])
{
    LWSimPlotConfiguration configuration;
    bool rate_specified = false;
    for (int index = 1; index < argc; ++index)
    {
        if (argv[index] == nullptr)
        {
            continue;
        }
        const std::string_view argument(argv[index]);
        if (argument == "--use_actuator_net")
        {
            throw std::runtime_error(
                "--use_actuator_net has been removed; rl_sim_LW always uses "
                "MuJoCo PD plus feedforward torque");
        }
        if (argument == "--enable-plot")
        {
            configuration.enabled = true;
            continue;
        }
        if (argument != "--plot-rate-hz")
        {
            continue;
        }
        if (index + 1 >= argc || argv[index + 1] == nullptr)
        {
            throw std::runtime_error(
                "--plot-rate-hz requires an integer argument");
        }

        const std::string_view value(argv[++index]);
        int parsed_rate = 0;
        const auto parsed = std::from_chars(
            value.data(),
            value.data() + value.size(),
            parsed_rate);
        if (parsed.ec != std::errc()
            || parsed.ptr != value.data() + value.size())
        {
            throw std::runtime_error(
                "Invalid --plot-rate-hz value: " + std::string(value));
        }
        if (parsed_rate < kLWSimMinimumPlotRateHz
            || parsed_rate > kLWSimMaximumPlotRateHz)
        {
            throw std::runtime_error(
                "--plot-rate-hz must be between 1 and 200: "
                + std::to_string(parsed_rate));
        }
        if (rate_specified && parsed_rate != configuration.rate_hz)
        {
            throw std::runtime_error(
                "Conflicting --plot-rate-hz values: "
                + std::to_string(configuration.rate_hz) + " and "
                + std::to_string(parsed_rate));
        }
        rate_specified = true;
        configuration.rate_hz = parsed_rate;
    }
    if (rate_specified && !configuration.enabled)
    {
        throw std::runtime_error(
            "--plot-rate-hz requires --enable-plot");
    }
    return configuration;
}

inline std::chrono::nanoseconds LWSimPlotPeriod(
    const LWSimPlotConfiguration& configuration)
{
    if (configuration.rate_hz < kLWSimMinimumPlotRateHz
        || configuration.rate_hz > kLWSimMaximumPlotRateHz)
    {
        throw std::logic_error("Invalid Sim2Sim plot-rate configuration");
    }
    return std::chrono::nanoseconds(std::chrono::seconds(1))
        / configuration.rate_hz;
}

#endif // LW_SIM_PLOT_CONFIG_HPP
