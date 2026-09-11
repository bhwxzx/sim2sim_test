/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LOOP_H
#define LOOP_H

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "logger.hpp"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

enum class LoopTimingLevel : int
{
    Normal = 0,
    Degraded = 1,
    Fatal = 2,
};

struct LoopTimingPolicy
{
    std::uint64_t degraded_consecutive_misses = 0;
    std::chrono::nanoseconds degraded_lateness{0};
    std::uint64_t fatal_consecutive_misses = 0;
    std::chrono::nanoseconds fatal_lateness{0};
};

struct LoopConfig
{
    std::chrono::nanoseconds period{0};
    int cpu_affinity = -1;
    int realtime_priority = 0;
    bool require_realtime = false;
    LoopTimingPolicy timing_policy;

    static LoopConfig FromSeconds(float seconds, int cpu_affinity = -1)
    {
        if (!std::isfinite(seconds) || seconds <= 0.0f)
        {
            throw std::invalid_argument("loop period must be a finite positive value");
        }

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(seconds));
        if (period.count() <= 0)
        {
            throw std::invalid_argument("loop period is below clock resolution");
        }

        LoopConfig config;
        config.period = period;
        config.cpu_affinity = cpu_affinity;
        return config;
    }
};

struct LoopTimingSnapshot
{
    std::uint64_t cycles = 0;
    std::uint64_t missed_deadlines = 0;
    std::uint64_t skipped_periods = 0;
    std::uint64_t consecutive_misses = 0;
    std::chrono::nanoseconds last_wakeup_lateness{0};
    std::chrono::nanoseconds max_wakeup_lateness{0};
    std::chrono::nanoseconds total_wakeup_lateness{0};
    std::chrono::nanoseconds last_deadline_lateness{0};
    std::chrono::nanoseconds max_deadline_lateness{0};
    std::chrono::nanoseconds last_execution_time{0};
    std::chrono::nanoseconds max_execution_time{0};
    LoopTimingLevel level = LoopTimingLevel::Normal;
};

struct LoopStartupSnapshot
{
    int requested_cpu_affinity = -1;
    int requested_realtime_priority = 0;
    bool affinity_applied = false;
    bool realtime_applied = false;
    int realtime_error = 0;
};

class LoopFunc
{
public:
    using ErrorCallback =
        std::function<void(const std::string&, std::exception_ptr)>;
    using TimingCallback =
        std::function<void(
            const std::string&,
            LoopTimingLevel,
            const LoopTimingSnapshot&)>;

    LoopFunc(
        const std::string& name,
        float period,
        std::function<void()> func,
        int bind_cpu = -1,
        ErrorCallback error_callback = nullptr)
        : LoopFunc(
              name,
              LoopConfig::FromSeconds(period, bind_cpu),
              std::move(func),
              std::move(error_callback),
              nullptr)
    {
    }

    LoopFunc(
        const std::string& name,
        LoopConfig config,
        std::function<void()> func,
        ErrorCallback error_callback = nullptr,
        TimingCallback timing_callback = nullptr)
        : name_(name),
          config_(std::move(config)),
          func_(std::move(func)),
          error_callback_(std::move(error_callback)),
          timing_callback_(std::move(timing_callback))
    {
        validateConfig();
    }

    ~LoopFunc()
    {
        shutdown();
    }

    LoopFunc(const LoopFunc&) = delete;
    LoopFunc& operator=(const LoopFunc&) = delete;

    void start()
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (thread_.joinable() || running_.load(std::memory_order_acquire))
        {
            throw std::logic_error("Loop '" + name_ + "' is already running");
        }

        {
            std::lock_guard<std::mutex> startup_lock(startup_mutex_);
            startup_complete_ = false;
            startup_error_ = nullptr;
        }
        resetStatistics();
        running_.store(true, std::memory_order_release);

        try
        {
            thread_ = std::thread(&LoopFunc::threadMain, this);
        }
        catch (...)
        {
            running_.store(false, std::memory_order_release);
            condition_.notify_all();
            throw;
        }

        std::exception_ptr startup_error;
        {
            std::unique_lock<std::mutex> startup_lock(startup_mutex_);
            startup_condition_.wait(
                startup_lock,
                [this]() { return startup_complete_; });
            startup_error = startup_error_;
        }

        if (startup_error)
        {
            running_.store(false, std::memory_order_release);
            condition_.notify_all();
            if (thread_.joinable())
            {
                thread_.join();
            }
            std::rethrow_exception(startup_error);
        }

        const LoopStartupSnapshot startup = startupSnapshot();
        std::cout << LOGGER::INFO << "[Loop] Loop start - name: " << name_
                  << ", period: " << formatPeriod() << "ms"
                  << (startup.affinity_applied
                          ? ", cpu: " + std::to_string(startup.requested_cpu_affinity)
                          : ", cpu: unspecified")
                  << (startup.realtime_applied
                          ? ", SCHED_FIFO priority: "
                                + std::to_string(startup.requested_realtime_priority)
                          : ", scheduling: SCHED_OTHER")
                  << std::endl;
    }

    void shutdown() noexcept
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        const bool had_thread = thread_.joinable();

        running_.store(false, std::memory_order_release);
        condition_.notify_all();

        if (had_thread)
        {
            if (thread_.get_id() == std::this_thread::get_id())
            {
                std::cerr << LOGGER::ERROR
                          << "[Loop] Loop cannot join itself - name: "
                          << name_ << std::endl;
                std::terminate();
            }
            thread_.join();
            std::cout << LOGGER::INFO << "[Loop] Loop end - name: "
                      << name_ << std::endl;
        }
    }

    LoopTimingSnapshot timingSnapshot() const noexcept
    {
        LoopTimingSnapshot snapshot;
        snapshot.cycles = cycles_.load(std::memory_order_relaxed);
        snapshot.missed_deadlines =
            missed_deadlines_.load(std::memory_order_relaxed);
        snapshot.skipped_periods =
            skipped_periods_.load(std::memory_order_relaxed);
        snapshot.consecutive_misses =
            consecutive_misses_.load(std::memory_order_relaxed);
        snapshot.last_wakeup_lateness = std::chrono::nanoseconds(
            last_wakeup_lateness_ns_.load(std::memory_order_relaxed));
        snapshot.max_wakeup_lateness = std::chrono::nanoseconds(
            max_wakeup_lateness_ns_.load(std::memory_order_relaxed));
        snapshot.total_wakeup_lateness = std::chrono::nanoseconds(
            total_wakeup_lateness_ns_.load(std::memory_order_relaxed));
        snapshot.last_deadline_lateness = std::chrono::nanoseconds(
            last_deadline_lateness_ns_.load(std::memory_order_relaxed));
        snapshot.max_deadline_lateness = std::chrono::nanoseconds(
            max_deadline_lateness_ns_.load(std::memory_order_relaxed));
        snapshot.last_execution_time = std::chrono::nanoseconds(
            last_execution_time_ns_.load(std::memory_order_relaxed));
        snapshot.max_execution_time = std::chrono::nanoseconds(
            max_execution_time_ns_.load(std::memory_order_relaxed));
        snapshot.level = static_cast<LoopTimingLevel>(
            timing_level_.load(std::memory_order_acquire));
        return snapshot;
    }

    LoopStartupSnapshot startupSnapshot() const noexcept
    {
        LoopStartupSnapshot snapshot;
        snapshot.requested_cpu_affinity = config_.cpu_affinity;
        snapshot.requested_realtime_priority = config_.realtime_priority;
        snapshot.affinity_applied =
            affinity_applied_.load(std::memory_order_acquire);
        snapshot.realtime_applied =
            realtime_applied_.load(std::memory_order_acquire);
        snapshot.realtime_error =
            realtime_error_.load(std::memory_order_acquire);
        return snapshot;
    }

private:
    using Clock = std::chrono::steady_clock;

    std::string name_;
    LoopConfig config_;
    std::function<void()> func_;
    ErrorCallback error_callback_;
    TimingCallback timing_callback_;
    std::atomic<bool> running_{false};
    std::mutex wait_mutex_;
    std::condition_variable condition_;
    std::mutex lifecycle_mutex_;
    std::thread thread_;

    std::mutex startup_mutex_;
    std::condition_variable startup_condition_;
    bool startup_complete_ = false;
    std::exception_ptr startup_error_;
    std::atomic<bool> affinity_applied_{false};
    std::atomic<bool> realtime_applied_{false};
    std::atomic<int> realtime_error_{0};

    std::atomic<std::uint64_t> cycles_{0};
    std::atomic<std::uint64_t> missed_deadlines_{0};
    std::atomic<std::uint64_t> skipped_periods_{0};
    std::atomic<std::uint64_t> consecutive_misses_{0};
    std::atomic<std::int64_t> last_wakeup_lateness_ns_{0};
    std::atomic<std::int64_t> max_wakeup_lateness_ns_{0};
    std::atomic<std::int64_t> total_wakeup_lateness_ns_{0};
    std::atomic<std::int64_t> last_deadline_lateness_ns_{0};
    std::atomic<std::int64_t> max_deadline_lateness_ns_{0};
    std::atomic<std::int64_t> last_execution_time_ns_{0};
    std::atomic<std::int64_t> max_execution_time_ns_{0};
    std::atomic<int> timing_level_{
        static_cast<int>(LoopTimingLevel::Normal)};

    void validateConfig() const
    {
        if (config_.period.count() <= 0)
        {
            throw std::invalid_argument("loop period must be positive");
        }
        if (config_.cpu_affinity < -1)
        {
            throw std::invalid_argument("loop CPU affinity must be -1 or nonnegative");
        }
        if (config_.realtime_priority < 0)
        {
            throw std::invalid_argument("loop real-time priority must be nonnegative");
        }
        if (config_.require_realtime && config_.realtime_priority == 0)
        {
            throw std::invalid_argument(
                "required real-time scheduling needs a positive priority");
        }
        validateTimingDuration(
            config_.timing_policy.degraded_lateness,
            "degraded lateness");
        validateTimingDuration(
            config_.timing_policy.fatal_lateness,
            "fatal lateness");
    }

    static void validateTimingDuration(
        std::chrono::nanoseconds duration,
        const char* name)
    {
        if (duration.count() < 0)
        {
            throw std::invalid_argument(
                std::string("loop ") + name + " must be nonnegative");
        }
    }

    void resetStatistics() noexcept
    {
        cycles_.store(0, std::memory_order_relaxed);
        missed_deadlines_.store(0, std::memory_order_relaxed);
        skipped_periods_.store(0, std::memory_order_relaxed);
        consecutive_misses_.store(0, std::memory_order_relaxed);
        last_wakeup_lateness_ns_.store(0, std::memory_order_relaxed);
        max_wakeup_lateness_ns_.store(0, std::memory_order_relaxed);
        total_wakeup_lateness_ns_.store(0, std::memory_order_relaxed);
        last_deadline_lateness_ns_.store(0, std::memory_order_relaxed);
        max_deadline_lateness_ns_.store(0, std::memory_order_relaxed);
        last_execution_time_ns_.store(0, std::memory_order_relaxed);
        max_execution_time_ns_.store(0, std::memory_order_relaxed);
        timing_level_.store(
            static_cast<int>(LoopTimingLevel::Normal),
            std::memory_order_release);
        affinity_applied_.store(false, std::memory_order_release);
        realtime_applied_.store(false, std::memory_order_release);
        realtime_error_.store(0, std::memory_order_release);
    }

    void signalStartup(std::exception_ptr error = nullptr) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(startup_mutex_);
            startup_error_ = std::move(error);
            startup_complete_ = true;
        }
        startup_condition_.notify_all();
    }

    void threadMain() noexcept
    {
        try
        {
            configureCurrentThread();
            signalStartup();
        }
        catch (...)
        {
            running_.store(false, std::memory_order_release);
            signalStartup(std::current_exception());
            return;
        }

        try
        {
            runLoop();
        }
        catch (...)
        {
            reportCallbackException(std::current_exception());
        }
    }

    void runLoop()
    {
        Clock::time_point scheduled_start = Clock::now();

        while (running_.load(std::memory_order_acquire))
        {
            const Clock::time_point actual_start = Clock::now();
            const auto raw_lateness = actual_start > scheduled_start
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                      actual_start - scheduled_start)
                : std::chrono::nanoseconds(0);

            std::uint64_t skipped_before_callback = 0;
            if (raw_lateness >= config_.period)
            {
                skipped_before_callback = static_cast<std::uint64_t>(
                    raw_lateness.count() / config_.period.count());
                scheduled_start += multiplyDuration(
                    config_.period,
                    skipped_before_callback);
            }

            const Clock::time_point callback_start = Clock::now();
            func_();
            const Clock::time_point callback_end = Clock::now();
            const auto execution_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    callback_end - callback_start);

            Clock::time_point next_deadline = scheduled_start + config_.period;
            const auto completion_lateness = callback_end > next_deadline
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                      callback_end - next_deadline)
                : std::chrono::nanoseconds(0);
            const auto deadline_lateness =
                std::max(raw_lateness, completion_lateness);
            std::uint64_t skipped_after_callback = 0;
            if (callback_end >= next_deadline)
            {
                const auto overrun =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        callback_end - next_deadline);
                skipped_after_callback = static_cast<std::uint64_t>(
                    overrun.count() / config_.period.count()) + 1;
                next_deadline += multiplyDuration(
                    config_.period,
                    skipped_after_callback);
            }

            const std::uint64_t missed =
                skipped_before_callback + skipped_after_callback;
            updateStatistics(
                raw_lateness, deadline_lateness, execution_time, missed);
            if (running_.load(std::memory_order_acquire))
            {
                evaluateTimingPolicy(deadline_lateness);
            }
            scheduled_start = next_deadline;

            std::unique_lock<std::mutex> lock(wait_mutex_);
            if (condition_.wait_until(
                    lock,
                    scheduled_start,
                    [this]() {
                        return !running_.load(std::memory_order_acquire);
                    }))
            {
                break;
            }
        }
    }

    static Clock::duration multiplyDuration(
        std::chrono::nanoseconds duration,
        std::uint64_t multiplier)
    {
        using Rep = std::chrono::nanoseconds::rep;
        const auto max = static_cast<std::uint64_t>(
            std::numeric_limits<Rep>::max());
        if (multiplier != 0
            && static_cast<std::uint64_t>(duration.count()) > max / multiplier)
        {
            throw std::overflow_error("loop deadline duration overflow");
        }
        return std::chrono::nanoseconds(
            duration.count() * static_cast<Rep>(multiplier));
    }

    void updateStatistics(
        std::chrono::nanoseconds wakeup_lateness,
        std::chrono::nanoseconds deadline_lateness,
        std::chrono::nanoseconds execution_time,
        std::uint64_t missed) noexcept
    {
        cycles_.fetch_add(1, std::memory_order_relaxed);
        last_wakeup_lateness_ns_.store(
            wakeup_lateness.count(), std::memory_order_relaxed);
        total_wakeup_lateness_ns_.fetch_add(
            wakeup_lateness.count(), std::memory_order_relaxed);
        updateMaximum(max_wakeup_lateness_ns_, wakeup_lateness.count());
        last_deadline_lateness_ns_.store(
            deadline_lateness.count(), std::memory_order_relaxed);
        updateMaximum(
            max_deadline_lateness_ns_, deadline_lateness.count());
        last_execution_time_ns_.store(
            execution_time.count(), std::memory_order_relaxed);
        updateMaximum(max_execution_time_ns_, execution_time.count());

        if (missed > 0)
        {
            missed_deadlines_.fetch_add(missed, std::memory_order_relaxed);
            skipped_periods_.fetch_add(missed, std::memory_order_relaxed);
            consecutive_misses_.fetch_add(missed, std::memory_order_relaxed);
        }
        else
        {
            consecutive_misses_.store(0, std::memory_order_relaxed);
        }
    }

    static void updateMaximum(
        std::atomic<std::int64_t>& maximum,
        std::int64_t value) noexcept
    {
        std::int64_t observed = maximum.load(std::memory_order_relaxed);
        while (observed < value
               && !maximum.compare_exchange_weak(
                   observed,
                   value,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed))
        {
        }
    }

    void evaluateTimingPolicy(std::chrono::nanoseconds deadline_lateness)
    {
        const std::uint64_t consecutive =
            consecutive_misses_.load(std::memory_order_relaxed);
        const LoopTimingPolicy& policy = config_.timing_policy;

        const bool fatal =
            (policy.fatal_consecutive_misses > 0
             && consecutive >= policy.fatal_consecutive_misses)
            || (policy.fatal_lateness.count() > 0
                && deadline_lateness >= policy.fatal_lateness);
        if (fatal)
        {
            promoteTimingLevel(LoopTimingLevel::Fatal);
            return;
        }

        const bool degraded =
            (policy.degraded_consecutive_misses > 0
             && consecutive >= policy.degraded_consecutive_misses)
            || (policy.degraded_lateness.count() > 0
                && deadline_lateness >= policy.degraded_lateness);
        if (degraded)
        {
            promoteTimingLevel(LoopTimingLevel::Degraded);
        }
    }

    void promoteTimingLevel(LoopTimingLevel requested)
    {
        int observed = timing_level_.load(std::memory_order_acquire);
        const int desired = static_cast<int>(requested);
        while (observed < desired)
        {
            if (timing_level_.compare_exchange_weak(
                    observed,
                    desired,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                if (timing_callback_)
                {
                    timing_callback_(name_, requested, timingSnapshot());
                }
                return;
            }
        }
    }

    void reportCallbackException(std::exception_ptr error) noexcept
    {
        running_.store(false, std::memory_order_release);
        condition_.notify_all();

        if (error_callback_)
        {
            try
            {
                error_callback_(name_, error);
            }
            catch (const std::exception& callback_error)
            {
                std::cerr << LOGGER::ERROR
                          << "[Loop] Error callback failed for '" << name_
                          << "': " << callback_error.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << LOGGER::ERROR
                          << "[Loop] Error callback failed for '" << name_
                          << "' with an unknown exception" << std::endl;
            }
        }
        else
        {
            std::cerr << LOGGER::ERROR
                      << "[Loop] Unhandled callback exception - name: "
                      << name_ << std::endl;
        }
    }

    void configureCurrentThread()
    {
        if (config_.cpu_affinity != -1)
        {
            applyThreadAffinity(config_.cpu_affinity);
            affinity_applied_.store(true, std::memory_order_release);
        }

        if (config_.realtime_priority > 0)
        {
            const int result = applyRealtimePriority(config_.realtime_priority);
            if (result == 0)
            {
                realtime_applied_.store(true, std::memory_order_release);
            }
            else
            {
                realtime_error_.store(result, std::memory_order_release);
                if (config_.require_realtime)
                {
                    throw std::runtime_error(
                        "failed to apply SCHED_FIFO priority "
                        + std::to_string(config_.realtime_priority)
                        + " to loop '" + name_ + "' (error "
                        + std::to_string(result) + ")");
                }
            }
        }
    }

    static void applyThreadAffinity(int cpu_id)
    {
#ifdef __linux__
        if (cpu_id < 0 || cpu_id >= CPU_SETSIZE)
        {
            throw std::invalid_argument("loop CPU affinity is outside CPU_SETSIZE");
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        const int result = pthread_setaffinity_np(
            pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (result != 0)
        {
            throw std::runtime_error(
                "failed to set loop CPU affinity to CPU "
                + std::to_string(cpu_id) + " (error "
                + std::to_string(result) + ")");
        }
#else
        (void)cpu_id;
        throw std::runtime_error("thread affinity is not supported on this platform");
#endif
    }

    static int applyRealtimePriority(int priority)
    {
#ifdef __linux__
        const int minimum = sched_get_priority_min(SCHED_FIFO);
        const int maximum = sched_get_priority_max(SCHED_FIFO);
        if (minimum == -1 || maximum == -1)
        {
            return errno != 0 ? errno : EINVAL;
        }
        if (priority < minimum || priority > maximum)
        {
            throw std::invalid_argument(
                "SCHED_FIFO priority must be in ["
                + std::to_string(minimum) + ", "
                + std::to_string(maximum) + "]");
        }

        sched_param parameters{};
        parameters.sched_priority = priority;
        return pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters);
#else
        (void)priority;
        return ENOTSUP;
#endif
    }

    std::string formatPeriod() const
    {
        const double milliseconds =
            std::chrono::duration<double, std::milli>(config_.period).count();
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << milliseconds;
        return stream.str();
    }
};

#endif // LOOP_H
