/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "observation_buffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>

ObservationBuffer::ObservationBuffer()
{
}

ObservationBuffer::ObservationBuffer(int num_envs,
                                     const std::vector<int>& obs_dims,
                                     int history_length,
                                     const std::string& priority)
    : num_envs(num_envs),
      obs_dims(obs_dims),
      priority(priority),
      history_length(history_length)
{
    if (num_envs <= 0 || history_length <= 0)
    {
        throw std::invalid_argument("num_envs and history_length must be positive");
    }

    for (int dim : obs_dims)
    {
        if (dim <= 0)
        {
            throw std::invalid_argument("All observation dimensions must be positive");
        }
        if (num_obs > std::numeric_limits<int>::max() - dim)
        {
            throw std::overflow_error("Total observation dimension overflows int");
        }
        num_obs += dim;
    }

    num_obs_total = num_obs;
    if (num_obs_total <= 0)
    {
        throw std::runtime_error("Invalid total observation dimension");
    }

    const std::size_t environment_count =
        static_cast<std::size_t>(num_envs);
    const std::size_t frame_count =
        static_cast<std::size_t>(history_length);
    const std::size_t frame_width =
        static_cast<std::size_t>(num_obs_total);
    if (frame_count > std::numeric_limits<std::size_t>::max() / frame_width
        || environment_count
               > std::numeric_limits<std::size_t>::max()
                   / (frame_count * frame_width))
    {
        throw std::length_error("Observation buffer size overflows");
    }
    obs_buf.assign(environment_count * frame_count * frame_width, 0.0f);
}

std::size_t ObservationBuffer::frameOffset(
    int env_idx,
    int logical_step) const
{
    const int physical_step =
        (newest_slot + logical_step) % history_length;
    return (
        static_cast<std::size_t>(env_idx)
            * static_cast<std::size_t>(history_length)
        + static_cast<std::size_t>(physical_step))
        * static_cast<std::size_t>(num_obs_total);
}

void ObservationBuffer::reset(
    const std::vector<int>& reset_idxs,
    const std::vector<float>& new_obs)
{
    if (obs_buf.empty() || new_obs.size() != static_cast<size_t>(num_obs_total))
    {
        return;
    }
    for (int env_idx : reset_idxs)
    {
        if (env_idx >= 0 && env_idx < num_envs)
        {
            for (int t = 0; t < history_length; ++t)
            {
                const auto offset = frameOffset(env_idx, t);
                std::copy(
                    new_obs.begin(),
                    new_obs.end(),
                    obs_buf.begin() + static_cast<std::ptrdiff_t>(offset));
            }
        }
    }
}

void ObservationBuffer::resetAll(const std::vector<float>& new_obs)
{
    if (obs_buf.empty() || new_obs.size() != static_cast<size_t>(num_obs_total))
    {
        return;
    }
    for (int env_idx = 0; env_idx < num_envs; ++env_idx)
    {
        for (int t = 0; t < history_length; ++t)
        {
            const auto offset = frameOffset(env_idx, t);
            std::copy(
                new_obs.begin(),
                new_obs.end(),
                obs_buf.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }
}

void ObservationBuffer::insert(const std::vector<float>& new_obs)
{
    if (obs_buf.empty() || new_obs.size() != static_cast<size_t>(num_obs_total))
    {
        return;
    }

    newest_slot = (newest_slot + history_length - 1) % history_length;
    for (int env_idx = 0; env_idx < num_envs; ++env_idx)
    {
        const auto offset = frameOffset(env_idx, 0);
        std::copy(
            new_obs.begin(),
            new_obs.end(),
            obs_buf.begin() + static_cast<std::ptrdiff_t>(offset));
    }
}

/**
 * @brief Gets history of observations indexed by obs_ids.
 *
 * @param obs_ids An array of integers with which to index the desired
 *                observations, where 0 is the latest observation and
 *                history_length - 1 is the oldest observation.
 * @return A vector containing the concatenated observations.
 */
std::size_t ObservationBuffer::requestedOutputSize(
    const std::vector<int>& obs_ids) const
{
    if (obs_buf.empty() || obs_ids.empty())
    {
        return 0;
    }

    // obs_ids are history-frame indices, not observation-term indices.
    for (int obs_id : obs_ids)
    {
        if (obs_id < 0 || obs_id >= history_length)
        {
            throw std::out_of_range(
                "Observation history index " + std::to_string(obs_id)
                + " is outside [0, " + std::to_string(history_length) + ")");
        }
    }

    const std::size_t environment_count =
        static_cast<std::size_t>(num_envs);
    const std::size_t selected_frame_count = obs_ids.size();
    const std::size_t frame_width =
        static_cast<std::size_t>(num_obs_total);
    if (selected_frame_count
        > std::numeric_limits<std::size_t>::max() / frame_width)
    {
        throw std::length_error("Observation history output size overflows");
    }
    const std::size_t selected_width = selected_frame_count * frame_width;
    if (environment_count
        > std::numeric_limits<std::size_t>::max() / selected_width)
    {
        throw std::length_error("Observation history output size overflows");
    }
    return environment_count * selected_width;
}

void ObservationBuffer::getObsInto(
    const std::vector<int>& obs_ids,
    std::vector<float>& output) const
{
    const std::size_t output_size = requestedOutputSize(obs_ids);
    if (output.size() != output_size)
    {
        throw std::invalid_argument(
            "Observation history output expected "
            + std::to_string(output_size) + " values, got "
            + std::to_string(output.size()));
    }
    std::size_t output_offset = 0;

    if (this->priority == "time")
    {
        // Time priority: iterate environments first, then time steps, finally observation dimensions
        for (int env_idx = 0; env_idx < num_envs; ++env_idx)
        {
            for (int obs_id : obs_ids)
            {
                // obs_id=0 is newest (at index 0), obs_id=N is older.
                const auto offset = frameOffset(env_idx, obs_id);
                std::copy_n(
                    obs_buf.begin() + static_cast<std::ptrdiff_t>(offset),
                    num_obs_total,
                    output.begin()
                        + static_cast<std::ptrdiff_t>(output_offset));
                output_offset += static_cast<std::size_t>(num_obs_total);
            }
        }
    }
    else if (this->priority == "term")
    {
        // Term priority: iterate environments first, then observation terms, finally time steps
        for (int env_idx = 0; env_idx < num_envs; ++env_idx)
        {
            int obs_offset = 0;
            for (size_t i = 0; i < obs_dims.size(); ++i)
            {
                int dim = obs_dims[i];
                for (int step : obs_ids)
                {
                    // step=0 is newest (at index 0), step=N is older.
                    for (int j = 0; j < dim; ++j)
                    {
                        output[output_offset++] = obs_buf[
                            frameOffset(env_idx, step)
                            + static_cast<std::size_t>(obs_offset + j)];
                    }
                }
                obs_offset += dim;
            }
        }
    }

}

std::vector<float> ObservationBuffer::get_obs_vec(
    const std::vector<int>& obs_ids) const
{
    std::vector<float> output(requestedOutputSize(obs_ids));
    getObsInto(obs_ids, output);
    return output;
}
