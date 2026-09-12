#ifndef LW_SIM_TORQUE_VALIDATION_HPP
#define LW_SIM_TORQUE_VALIDATION_HPP

#include <cstddef>
#include <vector>

enum class LWSimTorqueFailure
{
    None,
    SizeMismatch,
    NonFiniteCandidate,
    InvalidLimit
};

struct LWSimTorqueValidation
{
    LWSimTorqueFailure failure = LWSimTorqueFailure::None;
    std::size_t index = 0;

    bool valid() const noexcept
    {
        return failure == LWSimTorqueFailure::None;
    }

    const char* failureName() const noexcept;
};

LWSimTorqueValidation PrepareLWSimTorques(
    const std::vector<float>& candidates,
    const std::vector<float>& limits,
    std::vector<float>& bounded) noexcept;

#endif // LW_SIM_TORQUE_VALIDATION_HPP
