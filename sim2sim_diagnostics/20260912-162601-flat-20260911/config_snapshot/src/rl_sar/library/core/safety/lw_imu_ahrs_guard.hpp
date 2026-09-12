#ifndef LW_IMU_AHRS_GUARD_HPP
#define LW_IMU_AHRS_GUARD_HPP

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

enum class LWImuAhrsGuardStatus
{
    Accepted,
    MissingAhrs,
    StaleAhrs,
    InvalidAhrs,
    InvalidImu,
};

struct LWImuAhrsGuardDecision
{
    LWImuAhrsGuardStatus status = LWImuAhrsGuardStatus::MissingAhrs;
    std::chrono::steady_clock::duration pair_age{};
    bool pair_age_observed = false;
    double quaternion_norm = 0.0;

    bool accepted() const noexcept
    {
        return status == LWImuAhrsGuardStatus::Accepted;
    }
};

// FDLink publishes /imu from an IMU packet while taking its orientation from
// the most recently cached AHRS packet. A finite AHRS event therefore grants
// exactly one, short-lived authorization to the next valid IMU message.
class LWImuAhrsGuard
{
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

    explicit LWImuAhrsGuard(Duration pair_max_age)
    {
        setPairMaxAge(pair_max_age);
    }

    void setPairMaxAge(Duration pair_max_age)
    {
        if (pair_max_age <= Duration::zero())
        {
            throw std::invalid_argument(
                "IMU/AHRS pair maximum age must be positive");
        }
        pair_max_age_ = pair_max_age;
    }

    bool observeAhrs(
        TimePoint received_at,
        const std::array<double, 3>& euler) noexcept
    {
        authorization_available_ = false;
        if (!allFinite(euler))
        {
            last_ahrs_valid_ = false;
            return false;
        }
        last_ahrs_received_at_ = received_at;
        last_ahrs_valid_ = true;
        authorization_available_ = true;
        return true;
    }

    LWImuAhrsGuardDecision observeImu(
        TimePoint received_at,
        const std::array<double, 4>& quaternion,
        const std::array<double, 3>& angular_velocity) noexcept
    {
        LWImuAhrsGuardDecision decision;
        if (!last_ahrs_valid_)
        {
            decision.status = LWImuAhrsGuardStatus::InvalidAhrs;
            return decision;
        }
        if (!authorization_available_)
        {
            decision.status = LWImuAhrsGuardStatus::MissingAhrs;
            return decision;
        }

        // One AHRS event can never authorize a later retry or repeated IMU.
        authorization_available_ = false;
        if (received_at < last_ahrs_received_at_)
        {
            decision.status = LWImuAhrsGuardStatus::StaleAhrs;
            return decision;
        }
        decision.pair_age = received_at - last_ahrs_received_at_;
        decision.pair_age_observed = true;
        if (decision.pair_age > pair_max_age_)
        {
            decision.status = LWImuAhrsGuardStatus::StaleAhrs;
            return decision;
        }
        if (!allFinite(quaternion) || !allFinite(angular_velocity))
        {
            decision.status = LWImuAhrsGuardStatus::InvalidImu;
            return decision;
        }

        const double norm_squared =
            quaternion[0] * quaternion[0]
            + quaternion[1] * quaternion[1]
            + quaternion[2] * quaternion[2]
            + quaternion[3] * quaternion[3];
        if (!std::isfinite(norm_squared) || norm_squared <= 0.0)
        {
            decision.status = LWImuAhrsGuardStatus::InvalidImu;
            return decision;
        }
        decision.quaternion_norm = std::sqrt(norm_squared);
        if (decision.quaternion_norm < kMinimumQuaternionNorm
            || decision.quaternion_norm > kMaximumQuaternionNorm)
        {
            decision.status = LWImuAhrsGuardStatus::InvalidImu;
            return decision;
        }

        decision.status = LWImuAhrsGuardStatus::Accepted;
        return decision;
    }

private:
    template <std::size_t Size>
    static bool allFinite(const std::array<double, Size>& values) noexcept
    {
        for (const double value : values)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
        return true;
    }

    // Accepted samples are normalized before use. This range rejects zero and
    // clearly corrupted scaling while allowing normal sensor round-off.
    static constexpr double kMinimumQuaternionNorm = 0.9;
    static constexpr double kMaximumQuaternionNorm = 1.1;

    Duration pair_max_age_{};
    TimePoint last_ahrs_received_at_{};
    bool last_ahrs_valid_ = false;
    bool authorization_available_ = false;
};

#endif // LW_IMU_AHRS_GUARD_HPP
