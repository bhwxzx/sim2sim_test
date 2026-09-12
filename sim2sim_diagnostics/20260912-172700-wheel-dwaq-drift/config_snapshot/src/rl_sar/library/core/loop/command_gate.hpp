#ifndef COMMAND_GATE_HPP
#define COMMAND_GATE_HPP

#include <atomic>
#include <mutex>
#include <utility>

class CommandGate
{
public:
    CommandGate() = default;
    CommandGate(const CommandGate&) = delete;
    CommandGate& operator=(const CommandGate&) = delete;

    void close() noexcept
    {
        closed_.store(true, std::memory_order_release);
    }

    bool isClosed() const noexcept
    {
        return closed_.load(std::memory_order_acquire);
    }

    template <typename SendFunction>
    bool sendIfOpen(SendFunction&& send_function)
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        if (closed_.load(std::memory_order_acquire))
        {
            return false;
        }

        std::forward<SendFunction>(send_function)();
        return true;
    }

    template <typename SendFunction>
    void sendSerialized(SendFunction&& send_function)
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        std::forward<SendFunction>(send_function)();
    }

    template <typename SendFunction>
    void closeAndSend(SendFunction&& send_function)
    {
        close();
        std::lock_guard<std::mutex> lock(send_mutex_);
        std::forward<SendFunction>(send_function)();
    }

private:
    std::atomic<bool> closed_{false};
    std::mutex send_mutex_;
};

#endif // COMMAND_GATE_HPP
