#ifndef LW_RUNTIME_SYNC_HPP
#define LW_RUNTIME_SYNC_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

template <typename T>
class LWSnapshotBuffer
{
public:
    bool tryPublish(const T& value)
    {
        std::unique_lock<std::mutex> lock(
            mutex_,
            std::try_to_lock);
        if (!lock.owns_lock())
        {
            return false;
        }
        value_ = value;
        valid_ = true;
        return true;
    }

    void publish(const T& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = value;
        valid_ = true;
    }

    bool read(T& value) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_)
        {
            return false;
        }
        value = value_;
        return true;
    }

    bool valid() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return valid_;
    }

private:
    mutable std::mutex mutex_;
    T value_{};
    bool valid_ = false;
};

template <typename T>
class LWSpscLatestValue
{
public:
    // One producer and one consumer own their indices exclusively. The tagged
    // middle index is the only shared state, so neither side can access the
    // slot currently being copied by the other side. Call initialize() only
    // before either concurrent endpoint starts.
    static_assert(
        std::atomic<std::uint32_t>::is_always_lock_free,
        "LW SPSC snapshot requires lock-free 32-bit atomics");

    T& producerBuffer() noexcept
    {
        return slots_[producer_index_];
    }

    void publishProducerBuffer() noexcept
    {
        const std::uint32_t previous = middle_.exchange(
            producer_index_ | kPublished,
            std::memory_order_acq_rel);
        producer_index_ = previous & kIndexMask;
    }

    void publish(const T& value)
    {
        producerBuffer() = value;
        publishProducerBuffer();
    }

    const T* readLatest() noexcept
    {
        const std::uint32_t available =
            middle_.load(std::memory_order_acquire);
        if ((available & kPublished) != 0U)
        {
            const std::uint32_t previous = middle_.exchange(
                consumer_index_,
                std::memory_order_acq_rel);
            consumer_index_ = previous & kIndexMask;
            consumer_valid_ = true;
        }
        return consumer_valid_ ? &slots_[consumer_index_] : nullptr;
    }

    void initialize(const T& value)
    {
        for (T& slot : slots_)
        {
            slot = value;
        }
    }

private:
    static constexpr std::uint32_t kIndexMask = 0x3U;
    static constexpr std::uint32_t kPublished = 0x4U;
    std::array<T, 3> slots_{};
    std::uint32_t producer_index_ = 0;
    std::uint32_t consumer_index_ = 2;
    std::atomic<std::uint32_t> middle_{1};
    bool consumer_valid_ = false;
};

template <typename Event>
struct LWInputSnapshot
{
    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
    Event event{};
    std::uint64_t event_sequence = 0;
};

template <typename Event>
class LWInputMailbox
{
public:
    using Snapshot = LWInputSnapshot<Event>;

    void publishVelocity(float x, float y, float yaw)
    {
        producer_snapshot_.x = x;
        producer_snapshot_.y = y;
        producer_snapshot_.yaw = yaw;
        snapshots_.publish(producer_snapshot_);
    }

    void publishEvent(Event event)
    {
        producer_snapshot_.event = event;
        ++producer_snapshot_.event_sequence;
        snapshots_.publish(producer_snapshot_);
    }

    void clear(Event none)
    {
        producer_snapshot_.x = 0.0f;
        producer_snapshot_.y = 0.0f;
        producer_snapshot_.yaw = 0.0f;
        producer_snapshot_.event = none;
        ++producer_snapshot_.event_sequence;
        snapshots_.publish(producer_snapshot_);
    }

    bool read(Snapshot& snapshot) noexcept
    {
        const Snapshot* latest = snapshots_.readLatest();
        if (!latest)
        {
            return false;
        }
        snapshot = *latest;
        return true;
    }

private:
    Snapshot producer_snapshot_{};
    LWSpscLatestValue<Snapshot> snapshots_;
};

enum class LWOperatorMode : int
{
    Unknown = 0,
    Passive,
    GetUpLeg,
    GetUpWheel,
    GetDown,
    LegLocomotion,
    WheelLocomotion,
    LegToWheel,
    WheelToLeg,
};

struct LWOperatorStatusSnapshot
{
    std::uint64_t sequence = 0;
    LWOperatorMode mode = LWOperatorMode::Unknown;
    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
    float gait_frequency = 0.0f;
    float progress = 0.0f;
};

class LWOperatorStatusMailbox
{
public:
    // One control-loop writer publishes status while ROS diagnostics may read
    // concurrently. Sequentially consistent atomics make the revision check a
    // coherent, nonblocking snapshot boundary without a control-path mutex.
    void publish(const LWOperatorStatusSnapshot& status) noexcept
    {
        revision_.fetch_add(1);
        sequence_.store(status.sequence);
        mode_.store(static_cast<int>(status.mode));
        x_.store(status.x);
        y_.store(status.y);
        yaw_.store(status.yaw);
        gait_frequency_.store(status.gait_frequency);
        progress_.store(status.progress);
        revision_.fetch_add(1);
    }

    bool read(LWOperatorStatusSnapshot& status) const noexcept
    {
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            const std::uint64_t before = revision_.load();
            if ((before & 1U) != 0U)
            {
                continue;
            }

            LWOperatorStatusSnapshot candidate;
            candidate.sequence = sequence_.load();
            candidate.mode = static_cast<LWOperatorMode>(mode_.load());
            candidate.x = x_.load();
            candidate.y = y_.load();
            candidate.yaw = yaw_.load();
            candidate.gait_frequency = gait_frequency_.load();
            candidate.progress = progress_.load();

            const std::uint64_t after = revision_.load();
            if (before == after && (after & 1U) == 0U && after != 0)
            {
                status = candidate;
                return true;
            }
        }
        return false;
    }

private:
    std::atomic<std::uint64_t> revision_{0};
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<int> mode_{static_cast<int>(LWOperatorMode::Unknown)};
    std::atomic<float> x_{0.0f};
    std::atomic<float> y_{0.0f};
    std::atomic<float> yaw_{0.0f};
    std::atomic<float> gait_frequency_{0.0f};
    std::atomic<float> progress_{0.0f};
};

#endif // LW_RUNTIME_SYNC_HPP
