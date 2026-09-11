
#ifndef MOTION_LOADER_LW_HPP
#define MOTION_LOADER_LW_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vector_math.hpp"

/**
 * @brief Motion data loader for mimic/dance tasks
 *
 * Loads motion capture data from CSV files and provides interpolated
 * joint positions and velocities for a given time.
 *
 * CSV Format (per row):
 * root_pos_x, root_pos_y, root_pos_z, root_quat_x, root_quat_y, root_quat_z, root_quat_w,
 * joint_0, joint_1, ..., joint_N
 */
class MotionLoaderLW
{
public:
    struct PreparedMotion;
    using PreparedMotionPtr = std::shared_ptr<const PreparedMotion>;

    /**
     * @brief Parse and validate an immutable motion clip.
     *
     * This is the only operation that accesses the CSV file or computes the
     * complete velocity table. Runtime playback should construct a cursor from
     * the returned prepared motion.
     */
    static PreparedMotionPtr Prepare(
        const std::string& motion_file,
        float fps,
        int time_offset_frames,
        std::size_t expected_num_joints);

    /**
     * @brief Constructor
     * @param motion_file Path to CSV motion file
     * @param fps Frames per second of the motion data
     * @param time_offset_frames Source-frame index represented by the first CSV row
     * @param expected_num_joints Required number of joint columns
     */
    MotionLoaderLW(
        const std::string& motion_file,
        float fps,
        int time_offset_frames,
        std::size_t expected_num_joints);

    /**
     * @brief Construct a lightweight playback cursor over prepared data.
     */
    explicit MotionLoaderLW(PreparedMotionPtr prepared_motion);

    /**
     * @brief Update motion to a specific time
     * @param time Current time in seconds
     */
    void Update(float time);

    /**
     * @brief Reset motion to start with yaw alignment
     * @param robot_base_quat Current robot base quaternion [w, x, y, z]
     */
    void Reset(const std::vector<float>& robot_base_quat);

    /**
     * @brief Get interpolated joint positions at current time
     * @return Joint positions vector
     */
    std::vector<float> GetJointPos() const;

    /** Fill a pre-sized joint-position buffer without allocating. */
    void WriteJointPos(std::vector<float>& result) const;

    /**
     * @brief Get interpolated joint velocities at current time
     * @return Joint velocities vector
     */
    std::vector<float> GetJointVel() const;

    /** Fill a pre-sized joint-velocity buffer without allocating. */
    void WriteJointVel(std::vector<float>& result) const;

    /**
     * @brief Get interpolated root quaternion at current time
     * @return Root quaternion [w, x, y, z]
     */
    std::vector<float> GetRootQuat() const;

    /** Fill a four-element root-quaternion buffer without allocating. */
    void WriteRootQuat(std::vector<float>& result) const;

    /**
     * @brief Get anchor (torso) quaternion at current time
     * @return Anchor quaternion [w, x, y, z]
     */
    std::vector<float> GetAnchorQuat() const;

    /** Fill a four-element anchor-quaternion buffer without allocating. */
    void WriteAnchorQuat(std::vector<float>& result) const;

    /**
     * @brief Get motion duration in seconds
     */
    float GetDuration() const;

    /**
     * @brief Get world to init transformation quaternion (yaw alignment)
     */
    const std::vector<float>& GetInitQuat() const noexcept
    {
        return world_to_init_;
    }

    /**
     * @brief Compute torso quaternion from base quaternion
     * @param base_quat Base (pelvis) quaternion [w, x, y, z]
     * @return Torso quaternion [w, x, y, z]
     */
    static std::vector<float> ComputeTorsoQuat(const std::vector<float>& base_quat);

    /**
     * @brief Compute initial yaw alignment quaternion
     * @param robot_torso_quat Robot's torso quaternion [w, x, y, z]
     * @param motion_torso_quat Motion's torso quaternion [w, x, y, z]
     * @return Yaw alignment quaternion [w, x, y, z]
     */
    static std::vector<float> ComputeYawAlignment(const std::vector<float>& robot_torso_quat, const std::vector<float>& motion_torso_quat);

private:
    /**
     * @brief Spherical linear interpolation between two quaternions
     * @param q0 First quaternion [w, x, y, z]
     * @param q1 Second quaternion [w, x, y, z]
     * @param t Interpolation parameter [0, 1]
     * @return Interpolated quaternion [w, x, y, z]
     */
    void WriteSlerp(
        const std::vector<float>& q0,
        const std::vector<float>& q1,
        float t,
        std::vector<float>& result) const;

    PreparedMotionPtr prepared_motion_;

    // Current interpolation state
    std::size_t index_0_; // Current frame index
    std::size_t index_1_; // Next frame index
    float blend_;        // Interpolation factor [0, 1]

    // Coordinate transformation
    std::vector<float> world_to_init_;  // For yaw alignment between robot and motion [w, x, y, z]
};

#endif // MOTION_LOADER_LW_HPP
