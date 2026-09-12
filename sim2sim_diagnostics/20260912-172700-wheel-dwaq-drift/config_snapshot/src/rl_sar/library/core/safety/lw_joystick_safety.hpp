#ifndef LW_JOYSTICK_SAFETY_HPP
#define LW_JOYSTICK_SAFETY_HPP

#include "joystick.hh"

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <fcntl.h>
#include <string>
#include <unistd.h>

constexpr size_t LW_JOYSTICK_BUTTON_COUNT = 20;
constexpr size_t LW_JOYSTICK_AXIS_COUNT = 10;

enum class LWJoystickSampleStatus
{
    Event,
    NoData,
    Disconnected,
    Error
};

struct LWJoystickSampleResult
{
    LWJoystickSampleStatus status = LWJoystickSampleStatus::NoData;
    int error_number = 0;
    int bytes_read = 0;

    bool hasEvent() const noexcept
    {
        return status == LWJoystickSampleStatus::Event;
    }
};

class LWJoystickDevice
{
public:
    explicit LWJoystickDevice(const std::string& device_path) noexcept
        : fd_(::open(device_path.c_str(), O_RDONLY | O_NONBLOCK))
    {
    }

    ~LWJoystickDevice()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    LWJoystickDevice(const LWJoystickDevice&) = delete;
    LWJoystickDevice& operator=(const LWJoystickDevice&) = delete;

    bool isFound() const noexcept
    {
        return fd_ >= 0;
    }

    LWJoystickSampleResult sample(JoystickEvent* event) noexcept
    {
        if (event == nullptr)
        {
            return {LWJoystickSampleStatus::Error, EINVAL, 0};
        }

        errno = 0;
        const ssize_t bytes = ::read(fd_, event, sizeof(*event));
        if (bytes == static_cast<ssize_t>(sizeof(*event)))
        {
            return {
                LWJoystickSampleStatus::Event,
                0,
                static_cast<int>(bytes)
            };
        }
        if (bytes == 0)
        {
            return {LWJoystickSampleStatus::Disconnected, 0, 0};
        }
        if (bytes < 0)
        {
            const int read_error = errno;
            if (read_error == EAGAIN
                || read_error == EWOULDBLOCK
                || read_error == EINTR)
            {
                return {LWJoystickSampleStatus::NoData, read_error, -1};
            }
            if (read_error == ENODEV
                || read_error == EIO
                || read_error == EBADF)
            {
                return {
                    LWJoystickSampleStatus::Disconnected,
                    read_error,
                    -1
                };
            }
            return {LWJoystickSampleStatus::Error, read_error, -1};
        }

        return {
            LWJoystickSampleStatus::Error,
            EPROTO,
            static_cast<int>(bytes)
        };
    }

private:
    int fd_ = -1;
};

struct LWJoystickButton
{
    void update(bool state) noexcept
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    void clearTransient() noexcept
    {
        on_press = false;
        on_release = false;
    }

    void clear() noexcept
    {
        pressed = false;
        clearTransient();
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

enum class LWJoystickEventResult
{
    Applied,
    Unsupported,
    ButtonOutOfRange,
    AxisOutOfRange,
    InvalidConfiguration
};

inline LWJoystickEventResult LWApplyJoystickEvent(
    const JoystickEvent& event,
    std::array<LWJoystickButton, LW_JOYSTICK_BUTTON_COUNT>& buttons,
    std::array<int, LW_JOYSTICK_AXIS_COUNT>& axes,
    int max_axis_value,
    float axis_deadzone) noexcept
{
    if (max_axis_value <= 0
        || !std::isfinite(axis_deadzone)
        || axis_deadzone < 0.0f
        || axis_deadzone >= 1.0f)
    {
        return LWJoystickEventResult::InvalidConfiguration;
    }

    if ((event.type & JS_EVENT_BUTTON) != 0)
    {
        const size_t index = static_cast<size_t>(event.number);
        if (index >= buttons.size())
        {
            return LWJoystickEventResult::ButtonOutOfRange;
        }
        buttons[index].update(event.value != 0);
        return LWJoystickEventResult::Applied;
    }

    if ((event.type & JS_EVENT_AXIS) != 0)
    {
        const size_t index = static_cast<size_t>(event.number);
        if (index >= axes.size())
        {
            return LWJoystickEventResult::AxisOutOfRange;
        }

        const double normalized =
            static_cast<double>(event.value)
            / static_cast<double>(max_axis_value);
        axes[index] =
            std::abs(normalized) < static_cast<double>(axis_deadzone)
            ? 0
            : event.value;
        return LWJoystickEventResult::Applied;
    }

    return LWJoystickEventResult::Unsupported;
}

inline void LWBeginJoystickCycle(
    std::array<LWJoystickButton, LW_JOYSTICK_BUTTON_COUNT>& buttons) noexcept
{
    for (auto& button : buttons)
    {
        button.clearTransient();
    }
}

inline void LWClearJoystickState(
    std::array<LWJoystickButton, LW_JOYSTICK_BUTTON_COUNT>& buttons,
    std::array<int, LW_JOYSTICK_AXIS_COUNT>& axes) noexcept
{
    for (auto& button : buttons)
    {
        button.clear();
    }
    axes.fill(0);
}

class LWJoystickFaultLatch
{
public:
    bool latch() noexcept
    {
        bool expected = false;
        return faulted_.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    bool faulted() const noexcept
    {
        return faulted_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> faulted_{false};
};

#endif // LW_JOYSTICK_SAFETY_HPP
