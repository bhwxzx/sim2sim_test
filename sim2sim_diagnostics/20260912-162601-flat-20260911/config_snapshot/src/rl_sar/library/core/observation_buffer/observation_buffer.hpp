/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBSERVATION_BUFFER_HPP
#define OBSERVATION_BUFFER_HPP

#include <vector>
#include <string>

/**
 * @brief Observation buffer for storing historical observations
 * 
 * Manages a circular buffer of observations with configurable history length
 * and supports different output priority modes (time/term)
 */
class ObservationBuffer
{
public:
    /**
     * @brief Constructor with parameters
     * @param num_envs Number of environments
     * @param obs_dims Dimensions of each observation component
     * @param history_length Length of observation history to maintain
     * @param priority Output priority mode ("time" or "term")
     */
    ObservationBuffer(int num_envs, const std::vector<int>& obs_dims, int history_length, const std::string& priority);
    
    /**
     * @brief Default constructor
     */
    ObservationBuffer();

    /**
     * @brief Reset specified environments with new observations
     * @param reset_idxs Indices of environments to reset
     * @param new_obs New observation data to fill the buffer
     */
    void reset(
        const std::vector<int>& reset_idxs,
        const std::vector<float>& new_obs);

    void resetAll(const std::vector<float>& new_obs);
    
    /**
     * @brief Insert new observation into buffer
     * @param new_obs New observation data to insert
     */
    void insert(const std::vector<float>& new_obs);
    
    /**
     * @brief Get observation history based on specified frame indices
     * @param obs_ids History-frame indices, where 0 is the newest frame
     * @return Concatenated observation vector
     * @throws std::out_of_range if an index is outside the history buffer
     * @throws std::length_error if the requested output size overflows
     */
    std::vector<float> get_obs_vec(const std::vector<int>& obs_ids) const;

    void getObsInto(
        const std::vector<int>& obs_ids,
        std::vector<float>& output) const;

private:
    int num_envs;                                           ///< Number of environments
    std::vector<int> obs_dims;                              ///< Dimensions of observation components
    std::string priority;                                   ///< Output priority mode
    int num_obs = 0;                                        ///< Total observation dimension
    int history_length = 0;                                 ///< History buffer length
    int num_obs_total = 0;                                  ///< Total observation size
    int newest_slot = 0;
    std::vector<float> obs_buf;                             ///< Contiguous buffer [env][physical time][obs]

    std::size_t frameOffset(int env_idx, int logical_step) const;
    std::size_t requestedOutputSize(
        const std::vector<int>& obs_ids) const;
};

#endif // OBSERVATION_BUFFER_HPP
