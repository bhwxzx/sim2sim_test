#ifndef LW_SIM_STARTUP_HPP
#define LW_SIM_STARTUP_HPP

#include <atomic>
#include <chrono>
#include <thread>

// One initialization publisher, one runtime starter, and concurrent cancellation.
// Cancellation is terminal: neither late readiness nor startup completion revives it.
class LWSimStartup
{
public:
    enum class State { Preparing, Ready, Starting, Running, Cancelled };

    void PublishReady() noexcept
    {
        State expected = State::Preparing;
        state_.compare_exchange_strong(expected, State::Ready);
    }

    void Cancel() noexcept { state_.store(State::Cancelled); }
    State state() const noexcept { return state_.load(); }

    // Called only by the ROS executor. rollback must stop all partially started
    // loops, including a loop whose start operation threw after creating a thread.
    template<class StartStage, class Rollback, class ExitRequested>
    void StartIfReady(StartStage start, Rollback rollback, ExitRequested exiting)
    {
        if (exiting()) Cancel();
        State expected = State::Ready;
        if (!state_.compare_exchange_strong(expected, State::Starting)) return;
        try
        {
            for (int stage = 0; stage < 3; ++stage)
            {
                if (state() == State::Cancelled || exiting())
                {
                    Cancel();
                    rollback();
                    return;
                }
                start(stage);
            }
            if (exiting()) Cancel();
            expected = State::Starting;
            if (!state_.compare_exchange_strong(expected, State::Running)) rollback();
        }
        catch (...)
        {
            Cancel();
            rollback();
            throw;
        }
    }

    // No simulation mutex is held here. External window/ROS shutdown can be
    // observed even when it did not call Cancel directly. Polling is bounded
    // to 1 ms sleeps; this is not a startup deadline or a control-loop threshold.
    template<class ExitRequested>
    bool WaitForRuntime(ExitRequested exiting)
    {
        for (;;)
        {
            if (exiting()) Cancel();
            const State current = state();
            if (current == State::Cancelled) return false;
            if (current == State::Running) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    std::atomic<State> state_{State::Preparing};
};

#endif
