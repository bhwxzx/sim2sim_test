#ifndef LW_MUJOCO_CONTROL_ADAPTER_HPP
#define LW_MUJOCO_CONTROL_ADAPTER_HPP

#include "lw_safety_policy.hpp"
#include "lw_sim_torque_validation.hpp"

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

struct LWMuJoCoShutdownTargets
{
    bool& simulation_running;
    int& simulation_run;
    std::atomic<int>& exit_request;
};

class LWMuJoCoControlAdapter
{
public:
    LWMuJoCoControlAdapter(
        const mjModel& model,
        const std::vector<std::string>& joint_names,
        const std::vector<int>& joint_mapping);

    std::size_t numDofs() const noexcept;
    int quaternionAddress() const noexcept;
    int gyroscopeAddress() const noexcept;

    bool ReadState(
        const mjData& data,
        std::vector<float>& joint_positions,
        std::vector<float>& joint_velocities,
        std::vector<float>& joint_torques,
        std::vector<float>& quaternion,
        std::vector<float>& gyroscope) const noexcept;

    LWSimTorqueValidation ApplyCommand(
        mjData& data,
        const std::vector<float>& desired_positions,
        const std::vector<float>& desired_velocities,
        const std::vector<float>& feedforward_torques,
        const std::vector<float>& position_gains,
        const std::vector<float>& velocity_gains,
        const std::vector<float>& torque_limits,
        std::vector<float>& torque_candidates,
        std::vector<float>& bounded_torques) const noexcept;

    void ExecuteSafetyDecision(
        mjData& data,
        const LWSafetyDecision& decision,
        LWMuJoCoShutdownTargets shutdown_targets) const noexcept;

private:
    struct JointCache
    {
        int joint_id = -1;
        int position_address = -1;
        int velocity_address = -1;
        int torque_address = -1;
        int actuator_id = -1;
    };

    const mjModel* model_ = nullptr;
    std::vector<JointCache> joints_;
    int quaternion_address_ = -1;
    int gyroscope_address_ = -1;
};

#endif // LW_MUJOCO_CONTROL_ADAPTER_HPP
