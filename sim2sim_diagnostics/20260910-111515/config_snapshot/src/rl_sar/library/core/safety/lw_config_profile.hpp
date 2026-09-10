#ifndef LW_CONFIG_PROFILE_HPP
#define LW_CONFIG_PROFILE_HPP

#include "loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

struct LWProfileDistributionSnapshot
{
    std::uint64_t count = 0;
    std::size_t retained = 0;
    double minimum_us = 0.0;
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double p999_us = 0.0;
    double maximum_us = 0.0;
};

struct LWProfileTimedSourceSnapshot
{
    bool seen = false;
    double first_sample_delay_us = 0.0;
    double final_age_us = 0.0;
    LWProfileDistributionSnapshot gaps;
};

// Keeps aggregate extrema/mean for the complete run and a bounded rolling
// sample window for percentile estimates. Long profiles therefore cannot grow
// memory without bound, and the report states how many samples were retained.
class LWProfileDistribution
{
public:
    explicit LWProfileDistribution(std::size_t capacity = 200000)
        : samples_(capacity, 0.0), capacity_(capacity)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument(
                "LW profile distribution capacity must be positive");
        }
    }

    template <typename Rep, typename Period>
    void record(std::chrono::duration<Rep, Period> duration)
    {
        recordMicroseconds(
            std::chrono::duration<double, std::micro>(duration).count());
    }

    void recordMicroseconds(double value)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            throw std::invalid_argument(
                "LW profile samples must be finite and nonnegative");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0)
        {
            minimum_us_ = value;
            maximum_us_ = value;
        }
        else
        {
            minimum_us_ = std::min(minimum_us_, value);
            maximum_us_ = std::max(maximum_us_, value);
        }
        total_us_ += static_cast<long double>(value);
        samples_[static_cast<std::size_t>(count_ % capacity_)] = value;
        ++count_;
    }

    LWProfileDistributionSnapshot snapshot() const
    {
        std::vector<double> retained_samples;
        LWProfileDistributionSnapshot result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result.count = count_;
            result.retained = static_cast<std::size_t>(
                std::min<std::uint64_t>(count_, capacity_));
            if (count_ == 0)
            {
                return result;
            }
            result.minimum_us = minimum_us_;
            result.maximum_us = maximum_us_;
            result.mean_us = static_cast<double>(
                total_us_ / static_cast<long double>(count_));
            retained_samples.assign(
                samples_.begin(),
                samples_.begin()
                    + static_cast<std::ptrdiff_t>(result.retained));
        }
        std::sort(retained_samples.begin(), retained_samples.end());
        result.p50_us = percentile(retained_samples, 0.50);
        result.p95_us = percentile(retained_samples, 0.95);
        result.p99_us = percentile(retained_samples, 0.99);
        result.p999_us = percentile(retained_samples, 0.999);
        return result;
    }

private:
    static double percentile(
        const std::vector<double>& sorted,
        double fraction)
    {
        if (sorted.empty())
        {
            return 0.0;
        }
        const double position =
            fraction * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(std::floor(position));
        const auto upper = static_cast<std::size_t>(std::ceil(position));
        const double blend = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - blend) + sorted[upper] * blend;
    }

    mutable std::mutex mutex_;
    std::vector<double> samples_;
    std::uint64_t capacity_ = 0;
    std::uint64_t count_ = 0;
    long double total_us_ = 0.0;
    double minimum_us_ = 0.0;
    double maximum_us_ = 0.0;
};

struct LWProfileTimedSource
{
    void mark(std::chrono::steady_clock::time_point now)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (seen)
        {
            gaps.record(now - last);
        }
        else
        {
            first = now;
        }
        last = now;
        seen = true;
    }

    bool hasBeenSeen() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return seen;
    }

    LWProfileDistributionSnapshot snapshot() const
    {
        return gaps.snapshot();
    }

    LWProfileTimedSourceSnapshot snapshotSince(
        std::chrono::steady_clock::time_point started) const
    {
        std::lock_guard<std::mutex> lock(mutex);
        LWProfileTimedSourceSnapshot result;
        result.seen = seen;
        result.gaps = gaps.snapshot();
        if (!seen)
        {
            return result;
        }
        const auto now = std::chrono::steady_clock::now();
        result.first_sample_delay_us =
            std::chrono::duration<double, std::micro>(first - started).count();
        result.final_age_us =
            std::chrono::duration<double, std::micro>(now - last).count();
        return result;
    }

    mutable std::mutex mutex;
    bool seen = false;
    std::chrono::steady_clock::time_point first{};
    std::chrono::steady_clock::time_point last{};
    LWProfileDistribution gaps;
};

#endif // LW_CONFIG_PROFILE_HPP
