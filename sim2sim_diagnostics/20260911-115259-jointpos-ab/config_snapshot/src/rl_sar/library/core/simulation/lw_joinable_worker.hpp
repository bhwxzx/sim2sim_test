#ifndef LW_JOINABLE_WORKER_HPP
#define LW_JOINABLE_WORKER_HPP

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

class LWJoinableWorker
{
public:
    using Callback = std::function<void()>;

    explicit LWJoinableWorker(Callback request_stop)
        : request_stop_(std::move(request_stop))
    {
    }

    ~LWJoinableWorker()
    {
        shutdown();
    }

    LWJoinableWorker(const LWJoinableWorker&) = delete;
    LWJoinableWorker& operator=(const LWJoinableWorker&) = delete;

    void start(
        Callback prepare,
        Callback run,
        std::chrono::milliseconds startup_timeout)
    {
        std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (thread_.joinable())
        {
            throw std::logic_error("worker is already running");
        }
        if (startup_timeout <= std::chrono::milliseconds::zero())
        {
            throw std::invalid_argument("worker startup timeout must be positive");
        }

        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            startup_complete_ = false;
            startup_error_ = nullptr;
            worker_error_ = nullptr;
        }

        thread_ = std::thread(
            [this, prepare = std::move(prepare), run = std::move(run)]()
            {
                try
                {
                    if (prepare)
                    {
                        prepare();
                    }
                    {
                        std::lock_guard<std::mutex> state_lock(state_mutex_);
                        startup_complete_ = true;
                    }
                    state_condition_.notify_all();
                    if (run)
                    {
                        run();
                    }
                }
                catch (...)
                {
                    {
                        std::lock_guard<std::mutex> state_lock(state_mutex_);
                        if (!startup_complete_)
                        {
                            startup_error_ = std::current_exception();
                            startup_complete_ = true;
                            state_condition_.notify_all();
                            return;
                        }
                        worker_error_ = std::current_exception();
                    }
                    if (request_stop_)
                    {
                        try
                        {
                            request_stop_();
                        }
                        catch (...)
                        {
                        }
                    }
                }
            });

        std::exception_ptr startup_error;
        {
            std::unique_lock<std::mutex> state_lock(state_mutex_);
            if (!state_condition_.wait_for(
                    state_lock,
                    startup_timeout,
                    [this]() { return startup_complete_; }))
            {
                state_lock.unlock();
                lifecycle_lock.unlock();
                shutdown();
                throw std::runtime_error("worker startup timed out");
            }
            startup_error = startup_error_;
        }

        if (startup_error)
        {
            lifecycle_lock.unlock();
            shutdown();
            std::rethrow_exception(startup_error);
        }
    }

    void shutdown() noexcept
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (request_stop_)
        {
            try
            {
                request_stop_();
            }
            catch (...)
            {
            }
        }
        if (!thread_.joinable())
        {
            return;
        }
        if (thread_.get_id() == std::this_thread::get_id())
        {
            std::terminate();
        }
        thread_.join();
    }

    void rethrowWorkerError() const
    {
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            error = worker_error_;
        }
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

    bool joinable() const noexcept
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        return thread_.joinable();
    }

private:
    Callback request_stop_;
    mutable std::mutex lifecycle_mutex_;
    mutable std::mutex state_mutex_;
    std::condition_variable state_condition_;
    std::thread thread_;
    bool startup_complete_ = false;
    std::exception_ptr startup_error_;
    std::exception_ptr worker_error_;
};

#endif
