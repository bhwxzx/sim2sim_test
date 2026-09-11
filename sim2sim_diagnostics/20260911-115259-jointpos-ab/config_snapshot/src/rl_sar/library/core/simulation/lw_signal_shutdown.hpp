#ifndef LW_SIGNAL_SHUTDOWN_HPP
#define LW_SIGNAL_SHUTDOWN_HPP

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <pthread.h>

class LWSimShutdownCoordinator
{
public:
    using Callback = std::function<void()>;

    void Bind(Callback callback)
    {
        Callback callback_to_run;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_ = std::move(callback);
            if (requested_)
            {
                callback_to_run = callback_;
            }
        }
        if (callback_to_run)
        {
            callback_to_run();
        }
    }

    void Unbind() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = nullptr;
    }

    void Request()
    {
        Callback callback_to_run;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (requested_)
            {
                return;
            }
            requested_ = true;
            callback_to_run = callback_;
        }
        if (callback_to_run)
        {
            callback_to_run();
        }
    }

    bool requested() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return requested_;
    }

private:
    mutable std::mutex mutex_;
    Callback callback_;
    bool requested_ = false;
};

template<typename Worker>
std::exception_ptr RunLWSimShutdownBoundWorker(
    LWSimShutdownCoordinator& coordinator,
    Worker&& worker) noexcept
{
    std::exception_ptr error;
    try
    {
        std::forward<Worker>(worker)();
    }
    catch (...)
    {
        error = std::current_exception();
    }

    try
    {
        coordinator.Request();
    }
    catch (...)
    {
        if (!error)
        {
            error = std::current_exception();
        }
    }
    return error;
}

class LWSigintWaiter
{
public:
    using Callback = std::function<void()>;

    explicit LWSigintWaiter(Callback callback)
        : callback_(std::move(callback))
    {
        if (!callback_)
        {
            throw std::invalid_argument("SIGINT waiter callback is required");
        }
        sigemptyset(&signal_set_);
        sigaddset(&signal_set_, SIGINT);
        const int mask_result = pthread_sigmask(
            SIG_BLOCK, &signal_set_, &previous_mask_);
        if (mask_result != 0)
        {
            throw std::runtime_error(
                "failed to block SIGINT: " + std::string(std::strerror(mask_result)));
        }
        mask_installed_ = true;
        try
        {
            thread_ = std::thread([this]() { Wait(); });
        }
        catch (...)
        {
            RestoreMask();
            throw;
        }
    }

    ~LWSigintWaiter()
    {
        Shutdown();
        DrainPendingSignal();
        RestoreMask();
    }

    LWSigintWaiter(const LWSigintWaiter&) = delete;
    LWSigintWaiter& operator=(const LWSigintWaiter&) = delete;

    void Shutdown() noexcept
    {
        stop_requested_.store(true);
        if (thread_.joinable())
        {
            thread_.join();
        }
    }

    void ShutdownAndKeepBlocked() noexcept
    {
        restore_mask_on_destroy_ = false;
        Shutdown();
        DrainPendingSignal();
    }

    void RethrowWaitError() const
    {
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            error = wait_error_;
        }
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

private:
    void Wait() noexcept
    {
        constexpr long kPollNanoseconds = 10L * 1000L * 1000L;
        const timespec timeout{0, kPollNanoseconds};
        while (!stop_requested_.load())
        {
            errno = 0;
            const int received = sigtimedwait(&signal_set_, nullptr, &timeout);
            if (received == SIGINT)
            {
                if (!callback_delivered_.exchange(true))
                {
                    try
                    {
                        callback_();
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(error_mutex_);
                        wait_error_ = std::current_exception();
                    }
                }
                continue;
            }
            if (received == -1 && (errno == EAGAIN || errno == EINTR))
            {
                continue;
            }
            std::lock_guard<std::mutex> lock(error_mutex_);
            wait_error_ = std::make_exception_ptr(std::runtime_error(
                "failed while waiting for SIGINT: "
                + std::string(std::strerror(errno))));
            return;
        }
    }

    void DrainPendingSignal() noexcept
    {
        const timespec no_wait{0, 0};
        while (sigtimedwait(&signal_set_, nullptr, &no_wait) == SIGINT)
        {
        }
    }

    void RestoreMask() noexcept
    {
        if (mask_installed_ && restore_mask_on_destroy_)
        {
            pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr);
        }
        mask_installed_ = false;
    }

    Callback callback_;
    sigset_t signal_set_{};
    sigset_t previous_mask_{};
    bool mask_installed_ = false;
    bool restore_mask_on_destroy_ = true;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> callback_delivered_{false};
    mutable std::mutex error_mutex_;
    std::exception_ptr wait_error_;
    std::thread thread_;
};

#endif
