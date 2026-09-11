#ifndef LW_STARTUP_DISABLE_HPP
#define LW_STARTUP_DISABLE_HPP

#include "LW_sdk.hpp"
#include "command_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

struct LWStartupDisableConfig
{
    std::string right_port = "/dev/ttyLegRight";
    std::string left_port = "/dev/ttyLegLeft";
    LWSDK::Duration bootstrap_write_timeout = std::chrono::milliseconds(2);
    std::size_t initial_disable_attempts = 20;
    std::size_t final_disable_attempts = 20;
    std::chrono::steady_clock::duration disable_interval =
        std::chrono::milliseconds(5);
};

class LWStartupDisableGuard
{
public:
    explicit LWStartupDisableGuard(LWStartupDisableConfig config = {})
        : config_(std::move(config))
    {
        validateConfig();
        sdk_.SetWriteTimeout(config_.bootstrap_write_timeout);
        sdk_.InitCmdData(disable_command_);
        disable_command_.motors_disable = true;
        serial_attempted_ = true;

        try
        {
            const LWSerialInitStatus status = sdk_.InitSerial(
                config_.right_port.c_str(),
                config_.left_port.c_str());
            if (!status.bothInitialized())
            {
                command_gate_.close();
                throw std::runtime_error(
                    "failed to initialize both LW serial ports: "
                    + status.failureSummary());
            }

            for (std::size_t attempt = 0;
                 attempt < config_.initial_disable_attempts;
                 ++attempt)
            {
                const LWSendResult result = sendDisable(false);
                if (!result.complete())
                {
                    throw std::runtime_error(
                        "initial LW disable command was incomplete: "
                        + result.failureSummary());
                }
                ++initial_complete_writes_;
                sleepBetweenAttempts(attempt, config_.initial_disable_attempts);
            }
            initial_disable_established_ = true;
            startKeepalive();
        }
        catch (...)
        {
            finalize();
            throw;
        }
    }

    ~LWStartupDisableGuard()
    {
        finalize();
    }

    LWStartupDisableGuard(const LWStartupDisableGuard&) = delete;
    LWStartupDisableGuard& operator=(const LWStartupDisableGuard&) = delete;

    LWSDK& sdk() noexcept
    {
        return sdk_;
    }

    CommandGate& commandGate() noexcept
    {
        return command_gate_;
    }

    bool initialDisableEstablished() const noexcept
    {
        return initial_disable_established_;
    }

    std::size_t initialCompleteWrites() const noexcept
    {
        return initial_complete_writes_;
    }

    bool keepaliveStarted() const noexcept
    {
        return keepalive_started_;
    }

    std::size_t keepaliveCompleteWrites() const noexcept
    {
        return keepalive_complete_writes_.load(std::memory_order_acquire);
    }

    void requireHealthy() const
    {
        if (!initial_disable_established_)
        {
            throw std::runtime_error(
                "LW startup disable output was not established");
        }
        if (keepalive_failed_.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            throw std::runtime_error(
                "LW startup disable keepalive failed"
                + (keepalive_failure_.empty()
                       ? std::string()
                       : ": " + keepalive_failure_));
        }
        if (finalized_.load(std::memory_order_acquire))
        {
            throw std::runtime_error(
                "LW startup disable output was already finalized");
        }
        if (command_gate_.isClosed())
        {
            throw std::runtime_error(
                "LW startup command delivery was already latched closed");
        }
    }

    void handOffToRuntime(LWSDK::Duration runtime_write_timeout)
    {
        stopKeepalive();
        requireHealthy();
        sdk_.SetWriteTimeout(runtime_write_timeout);
        const LWSendResult result = sendDisable(false);
        if (!result.complete())
        {
            command_gate_.close();
            throw std::runtime_error(
                "LW disable command was incomplete during runtime handoff: "
                + result.failureSummary());
        }
        runtime_handoff_complete_ = true;
    }

    bool runtimeHandoffComplete() const noexcept
    {
        return runtime_handoff_complete_;
    }

    LWSendResult sendDisable(bool latch_commands = false)
    {
        LWSendResult result;
        const auto send = [this, &result]()
        {
            result = sdk_.SendCmdData(disable_command_);
        };
        if (latch_commands)
        {
            command_gate_.closeAndSend(send);
        }
        else
        {
            command_gate_.sendSerialized(send);
        }
        return result;
    }

    void sendEmergencyDisableBurst() noexcept
    {
        command_gate_.close();
        stopKeepalive();
        sendBurstNoThrow(config_.final_disable_attempts, "emergency shutdown");
    }

    void finalize() noexcept
    {
        bool expected = false;
        if (!finalized_.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            return;
        }

        command_gate_.close();
        stopKeepalive();
        if (serial_attempted_)
        {
            sendBurstNoThrow(config_.final_disable_attempts, "final shutdown");
        }
    }

private:
    void validateConfig() const
    {
        if (config_.right_port.empty() || config_.left_port.empty())
        {
            throw std::invalid_argument("LW serial port paths must not be empty");
        }
        if (config_.bootstrap_write_timeout <= LWSDK::Duration::zero())
        {
            throw std::invalid_argument(
                "LW bootstrap serial write timeout must be positive");
        }
        if (config_.initial_disable_attempts == 0
            || config_.final_disable_attempts == 0)
        {
            throw std::invalid_argument(
                "LW disable attempt counts must be positive");
        }
        if (config_.disable_interval <= std::chrono::steady_clock::duration::zero())
        {
            throw std::invalid_argument(
                "LW disable interval must be positive");
        }
    }

    void startKeepalive()
    {
        keepalive_stop_.store(false, std::memory_order_release);
        keepalive_thread_ = std::thread([this]()
        {
            auto next = std::chrono::steady_clock::now();
            while (!keepalive_stop_.load(std::memory_order_acquire))
            {
                try
                {
                    const LWSendResult result = sendDisable(false);
                    if (!result.complete())
                    {
                        const std::string summary = result.failureSummary();
                        recordKeepaliveFailure(summary.c_str());
                        break;
                    }
                    keepalive_complete_writes_.fetch_add(
                        1,
                        std::memory_order_release);
                }
                catch (const std::exception& exception)
                {
                    recordKeepaliveFailure(exception.what());
                    break;
                }
                catch (...)
                {
                    recordKeepaliveFailure("unexpected keepalive exception");
                    break;
                }
                next += config_.disable_interval;
                std::this_thread::sleep_until(next);
            }
        });
        keepalive_started_ = true;
    }

    void recordKeepaliveFailure(const char* message) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            keepalive_failure_ = message == nullptr ? "unknown failure" : message;
        }
        catch (...)
        {
        }
        keepalive_failed_.store(true, std::memory_order_release);
        command_gate_.close();
    }

    void stopKeepalive() noexcept
    {
        keepalive_stop_.store(true, std::memory_order_release);
        if (keepalive_thread_.joinable())
        {
            keepalive_thread_.join();
        }
    }

    void sleepBetweenAttempts(
        std::size_t attempt,
        std::size_t attempts) const
    {
        if (attempt + 1 < attempts)
        {
            std::this_thread::sleep_for(config_.disable_interval);
        }
    }

    void sendBurstNoThrow(
        std::size_t attempts,
        const char* context) noexcept
    {
        bool failure_logged = false;
        for (std::size_t attempt = 0; attempt < attempts; ++attempt)
        {
            try
            {
                const LWSendResult result = sendDisable(true);
                if (!result.complete() && !failure_logged)
                {
                    std::cerr << "[Safety] LW disable burst was incomplete during "
                              << context << ": " << result.failureSummary()
                              << std::endl;
                    failure_logged = true;
                }
            }
            catch (const std::exception& exception)
            {
                if (!failure_logged)
                {
                    std::cerr << "[Safety] LW disable burst failed during "
                              << context << ": " << exception.what()
                              << std::endl;
                    failure_logged = true;
                }
            }
            catch (...)
            {
                if (!failure_logged)
                {
                    std::cerr << "[Safety] LW disable burst failed during "
                              << context << std::endl;
                    failure_logged = true;
                }
            }
            sleepBetweenAttempts(attempt, attempts);
        }
    }

    LWStartupDisableConfig config_;
    LWSDK sdk_;
    LowCmd disable_command_{};
    CommandGate command_gate_;
    bool serial_attempted_ = false;
    bool initial_disable_established_ = false;
    std::size_t initial_complete_writes_ = 0;
    bool runtime_handoff_complete_ = false;

    std::atomic<bool> keepalive_stop_{false};
    std::atomic<bool> keepalive_failed_{false};
    std::atomic<std::size_t> keepalive_complete_writes_{0};
    bool keepalive_started_ = false;
    std::thread keepalive_thread_;
    mutable std::mutex failure_mutex_;
    std::string keepalive_failure_;
    std::atomic<bool> finalized_{false};
};

#endif // LW_STARTUP_DISABLE_HPP
