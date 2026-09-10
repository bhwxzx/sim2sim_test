#ifndef SENSOR_READINESS_HPP
#define SENSOR_READINESS_HPP

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>

enum class SensorReadinessDecision
{
    Waiting,
    Ready,
    FaultLatched,
};

struct SensorReadinessStatus
{
    SensorReadinessDecision decision = SensorReadinessDecision::Waiting;
    bool imu_fresh = false;
    bool right_feedback_fresh = false;
    bool left_feedback_fresh = false;

    bool allFresh() const noexcept
    {
        return imu_fresh && right_feedback_fresh && left_feedback_fresh;
    }

    std::string missingSources() const
    {
        std::string result;
        const auto append = [&result](const std::string& source)
        {
            if (!result.empty())
            {
                result += ", ";
            }
            result += source;
        };

        if (!imu_fresh)
        {
            append("IMU");
        }
        if (!right_feedback_fresh)
        {
            append("right motor board");
        }
        if (!left_feedback_fresh)
        {
            append("left motor board");
        }
        return result;
    }
};

class SensorReadinessMonitor
{
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

    explicit SensorReadinessMonitor(Duration timeout)
    {
        setTimeout(timeout);
    }

    void setTimeout(Duration timeout)
    {
        if (timeout <= Duration::zero())
        {
            throw std::invalid_argument("sensor freshness timeout must be positive");
        }
        imu_timeout_ = timeout;
        motor_feedback_timeout_ = timeout;
    }

    void setImuTimeout(Duration timeout)
    {
        requirePositiveTimeout(timeout);
        imu_timeout_ = timeout;
    }

    void setMotorFeedbackTimeout(Duration timeout)
    {
        requirePositiveTimeout(timeout);
        motor_feedback_timeout_ = timeout;
    }

    void markImuReceived(TimePoint received_at) noexcept
    {
        last_imu_ = received_at;
    }

    void markRightFeedbackReceived(TimePoint received_at) noexcept
    {
        last_right_feedback_ = received_at;
    }

    void markLeftFeedbackReceived(TimePoint received_at) noexcept
    {
        last_left_feedback_ = received_at;
    }

    SensorReadinessStatus evaluate(TimePoint now) noexcept
    {
        SensorReadinessStatus status;
        status.imu_fresh = isFresh(last_imu_, now, imu_timeout_);
        status.right_feedback_fresh =
            isFresh(last_right_feedback_, now, motor_feedback_timeout_);
        status.left_feedback_fresh =
            isFresh(last_left_feedback_, now, motor_feedback_timeout_);

        if (fault_latched_)
        {
            status.decision = SensorReadinessDecision::FaultLatched;
        }
        else if (status.allFresh())
        {
            was_ready_ = true;
            status.decision = SensorReadinessDecision::Ready;
        }
        else if (was_ready_)
        {
            fault_latched_ = true;
            status.decision = SensorReadinessDecision::FaultLatched;
        }
        else
        {
            status.decision = SensorReadinessDecision::Waiting;
        }

        return status;
    }

    bool wasReady() const noexcept
    {
        return was_ready_;
    }

    bool faultLatched() const noexcept
    {
        return fault_latched_;
    }

private:
    static void requirePositiveTimeout(Duration timeout)
    {
        if (timeout <= Duration::zero())
        {
            throw std::invalid_argument(
                "sensor freshness timeout must be positive");
        }
    }

    bool isFresh(
        const std::optional<TimePoint>& timestamp,
        TimePoint now,
        Duration timeout) const noexcept
    {
        if (!timestamp)
        {
            return false;
        }
        if (*timestamp > now)
        {
            return true;
        }
        return now - *timestamp <= timeout;
    }

    Duration imu_timeout_{};
    Duration motor_feedback_timeout_{};
    std::optional<TimePoint> last_imu_;
    std::optional<TimePoint> last_right_feedback_;
    std::optional<TimePoint> last_left_feedback_;
    bool was_ready_ = false;
    bool fault_latched_ = false;
};

#endif // SENSOR_READINESS_HPP
