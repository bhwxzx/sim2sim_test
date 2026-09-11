#include "lw_sim_torque_validation.hpp"

#include <algorithm>
#include <cmath>

const char* LWSimTorqueValidation::failureName() const noexcept
{
    switch (failure)
    {
    case LWSimTorqueFailure::None: return "none";
    case LWSimTorqueFailure::SizeMismatch: return "size-mismatch";
    case LWSimTorqueFailure::NonFiniteCandidate:
        return "non-finite-candidate";
    case LWSimTorqueFailure::InvalidLimit: return "invalid-limit";
    }
    return "unknown";
}

LWSimTorqueValidation PrepareLWSimTorques(
    const std::vector<float>& candidates,
    const std::vector<float>& limits,
    std::vector<float>& bounded) noexcept
{
    if (candidates.size() != limits.size()
        || candidates.size() != bounded.size())
    {
        return {LWSimTorqueFailure::SizeMismatch, 0};
    }

    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        if (!std::isfinite(candidates[index]))
        {
            return {LWSimTorqueFailure::NonFiniteCandidate, index};
        }
        if (!std::isfinite(limits[index]) || limits[index] < 0.0f)
        {
            return {LWSimTorqueFailure::InvalidLimit, index};
        }
    }

    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        bounded[index] = std::clamp(
            candidates[index], -limits[index], limits[index]);
    }
    return {};
}
