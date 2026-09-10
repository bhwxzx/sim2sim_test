#ifndef LW_TERMINAL_KEYBOARD_HPP
#define LW_TERMINAL_KEYBOARD_HPP

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

class LWTerminalKeyboard
{
public:
    explicit LWTerminalKeyboard(const std::string& device_path = "/dev/tty")
    {
        descriptor_ = ::open(
            device_path.c_str(),
            O_RDWR | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
        if (descriptor_ < 0)
        {
            throw std::runtime_error(
                "cannot open terminal keyboard " + device_path + ": "
                + std::strerror(errno));
        }

        if (::tcgetattr(descriptor_, &original_termios_) != 0)
        {
            const int error_number = errno;
            closeDescriptor();
            throw std::runtime_error(
                "cannot read terminal keyboard settings for " + device_path
                + ": " + std::strerror(error_number));
        }

        termios keyboard_termios = original_termios_;
        keyboard_termios.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        keyboard_termios.c_cc[VMIN] = 0;
        keyboard_termios.c_cc[VTIME] = 0;
        if (::tcsetattr(descriptor_, TCSANOW, &keyboard_termios) != 0)
        {
            const int error_number = errno;
            closeDescriptor();
            throw std::runtime_error(
                "cannot configure terminal keyboard " + device_path + ": "
                + std::strerror(error_number));
        }
        configured_ = true;
    }

    ~LWTerminalKeyboard()
    {
        restore();
    }

    LWTerminalKeyboard(const LWTerminalKeyboard&) = delete;
    LWTerminalKeyboard& operator=(const LWTerminalKeyboard&) = delete;
    LWTerminalKeyboard(LWTerminalKeyboard&&) = delete;
    LWTerminalKeyboard& operator=(LWTerminalKeyboard&&) = delete;

    int descriptor() const noexcept
    {
        return descriptor_;
    }

private:
    void closeDescriptor() noexcept
    {
        if (descriptor_ >= 0)
        {
            ::close(descriptor_);
            descriptor_ = -1;
        }
    }

    void restore() noexcept
    {
        if (configured_ && descriptor_ >= 0)
        {
            ::tcsetattr(descriptor_, TCSANOW, &original_termios_);
            configured_ = false;
        }
        closeDescriptor();
    }

    int descriptor_ = -1;
    termios original_termios_{};
    bool configured_ = false;
};

#endif // LW_TERMINAL_KEYBOARD_HPP
