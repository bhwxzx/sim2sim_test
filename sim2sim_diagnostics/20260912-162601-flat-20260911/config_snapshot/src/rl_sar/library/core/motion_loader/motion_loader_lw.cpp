#include "motion_loader_lw.hpp"

#include "logger.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
std::string trim(const std::string& value)
{
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](unsigned char character) {
            return std::isspace(character) != 0;
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](unsigned char character) {
            return std::isspace(character) != 0;
        }).base();
    return first < last ? std::string(first, last) : std::string();
}

[[noreturn]] void csvError(
    const std::string& filename,
    std::size_t row,
    const std::string& message)
{
    throw std::runtime_error(
        "Invalid LW motion CSV '" + filename + "' at row "
        + std::to_string(row) + ": " + message);
}

float parseFiniteValue(
    const std::string& token,
    const std::string& filename,
    std::size_t row,
    std::size_t column)
{
    const std::string trimmed = trim(token);
    if (trimmed.empty())
    {
        csvError(
            filename,
            row,
            "column " + std::to_string(column) + " is empty");
    }

    std::size_t consumed = 0;
    float value = 0.0f;
    try
    {
        value = std::stof(trimmed, &consumed);
    }
    catch (const std::exception& exception)
    {
        csvError(
            filename,
            row,
            "column " + std::to_string(column)
                + " is not a valid float ('" + trimmed + "'): "
                + exception.what());
    }
    if (consumed != trimmed.size())
    {
        csvError(
            filename,
            row,
            "column " + std::to_string(column)
                + " contains trailing text: '" + trimmed + "'");
    }
    if (!std::isfinite(value))
    {
        csvError(
            filename,
            row,
            "column " + std::to_string(column) + " must be finite");
    }
    return value;
}
} // namespace

struct MotionLoaderLW::PreparedMotion
{
    std::vector<std::vector<float>> root_positions;
    std::vector<std::vector<float>> root_quaternions;
    std::vector<std::vector<float>> joint_positions;
    std::vector<std::vector<float>> joint_velocities;
    std::size_t num_frames = 0;
    std::size_t num_joints = 0;
    int time_offset_frames = 0;
    float dt = 0.0f;
    float duration = 0.0f;
};

MotionLoaderLW::PreparedMotionPtr MotionLoaderLW::Prepare(
    const std::string& motion_file,
    float fps,
    int time_offset_frames,
    std::size_t expected_num_joints)
{
    if (!std::isfinite(fps) || fps <= 0.0f)
    {
        throw std::invalid_argument(
            "LW motion fps must be finite and positive");
    }
    if (time_offset_frames < 0)
    {
        throw std::invalid_argument(
            "LW motion time_offset_frames must be nonnegative");
    }
    if (expected_num_joints == 0)
    {
        throw std::invalid_argument(
            "LW motion expected_num_joints must be positive");
    }

    auto prepared = std::make_shared<PreparedMotion>();
    prepared->num_joints = expected_num_joints;
    prepared->time_offset_frames = time_offset_frames;
    prepared->dt = 1.0f / fps;

    std::ifstream file(motion_file);
    if (!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open motion file: " + motion_file);
    }

    const std::size_t expected_columns = 7 + expected_num_joints;
    std::vector<std::vector<float>> rows;
    std::string line;
    std::size_t row_number = 0;
    while (std::getline(file, line))
    {
        ++row_number;
        if (trim(line).empty())
        {
            csvError(motion_file, row_number, "row is empty");
        }

        std::stringstream stream(line);
        std::string token;
        std::vector<float> row;
        std::size_t column = 0;
        while (std::getline(stream, token, ','))
        {
            ++column;
            row.push_back(
                parseFiniteValue(
                    token, motion_file, row_number, column));
        }
        if (!line.empty() && line.back() == ',')
        {
            csvError(
                motion_file,
                row_number,
                "column " + std::to_string(column + 1) + " is empty");
        }
        if (row.size() != expected_columns)
        {
            csvError(
                motion_file,
                row_number,
                "expected " + std::to_string(expected_columns)
                    + " columns (7 root + "
                    + std::to_string(expected_num_joints) + " joints), got "
                    + std::to_string(row.size()));
        }

        const float quaternion_norm_squared =
            row[3] * row[3] + row[4] * row[4]
            + row[5] * row[5] + row[6] * row[6];
        if (!std::isfinite(quaternion_norm_squared)
            || quaternion_norm_squared <= 1.0e-12f)
        {
            csvError(
                motion_file,
                row_number,
                "root quaternion norm is too small");
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty())
    {
        throw std::runtime_error(
            "LW motion CSV contains no frames: " + motion_file);
    }
    if (rows.size() < 2)
    {
        throw std::runtime_error(
            "LW motion CSV requires at least two frames: " + motion_file);
    }

    prepared->root_positions.reserve(rows.size());
    prepared->root_quaternions.reserve(rows.size());
    prepared->joint_positions.reserve(rows.size());
    for (const auto& row : rows)
    {
        prepared->root_positions.push_back({row[0], row[1], row[2]});
        prepared->root_quaternions.push_back(
            QuaternionNormalize({row[6], row[3], row[4], row[5]}));
        prepared->joint_positions.emplace_back(row.begin() + 7, row.end());
    }

    prepared->joint_velocities.assign(
        prepared->joint_positions.size(),
        std::vector<float>(expected_num_joints, 0.0f));
    for (std::size_t frame = 0;
         frame + 1 < prepared->joint_positions.size();
         ++frame)
    {
        for (std::size_t joint = 0; joint < expected_num_joints; ++joint)
        {
            prepared->joint_velocities[frame][joint] =
                (prepared->joint_positions[frame + 1][joint]
                 - prepared->joint_positions[frame][joint])
                / prepared->dt;
        }
    }
    prepared->joint_velocities.back() =
        prepared->joint_velocities[prepared->joint_velocities.size() - 2];

    prepared->num_frames = prepared->root_positions.size();
    const double last_source_frame =
        static_cast<double>(prepared->time_offset_frames)
        + static_cast<double>(prepared->num_frames - 1);
    const double duration =
        last_source_frame * static_cast<double>(prepared->dt);
    if (!std::isfinite(duration)
        || duration > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::overflow_error("LW motion duration overflows float");
    }
    prepared->duration = static_cast<float>(duration);
    if (!std::isfinite(prepared->duration) || prepared->duration <= 0.0f)
    {
        throw std::runtime_error(
            "LW motion duration must be finite and positive");
    }

    std::cout << LOGGER::INFO << "MotionLoaderLW: Prepared "
              << prepared->num_frames
              << " frames, " << prepared->num_joints
              << " joints, first-frame-time="
              << static_cast<float>(prepared->time_offset_frames)
                    * prepared->dt
              << "s, duration=" << prepared->duration << "s"
              << std::endl;
    return prepared;
}

MotionLoaderLW::MotionLoaderLW(
    const std::string& motion_file,
    float fps,
    int time_offset_frames,
    std::size_t expected_num_joints)
    : MotionLoaderLW(
          Prepare(
              motion_file,
              fps,
              time_offset_frames,
              expected_num_joints))
{
}

MotionLoaderLW::MotionLoaderLW(PreparedMotionPtr prepared_motion)
    : prepared_motion_(std::move(prepared_motion)),
      index_0_(0),
      index_1_(0),
      blend_(0.0f),
      world_to_init_{1.0f, 0.0f, 0.0f, 0.0f}
{
    if (!prepared_motion_)
    {
        throw std::invalid_argument(
            "LW prepared motion must not be null");
    }
}

float MotionLoaderLW::GetDuration() const
{
    return prepared_motion_->duration;
}

void MotionLoaderLW::Update(float time)
{
    if (!std::isfinite(time))
    {
        throw std::invalid_argument("LW motion update time must be finite");
    }
    const float clamped_time =
        std::clamp(time, 0.0f, prepared_motion_->duration);
    const float last_frame =
        static_cast<float>(prepared_motion_->num_frames - 1);
    const float frame_float = std::clamp(
        clamped_time / prepared_motion_->dt
            - static_cast<float>(prepared_motion_->time_offset_frames),
        0.0f,
        last_frame);

    index_0_ = static_cast<std::size_t>(std::floor(frame_float));
    index_1_ = std::min(
        index_0_ + 1,
        prepared_motion_->num_frames - 1);
    blend_ = frame_float - static_cast<float>(index_0_);
}

void MotionLoaderLW::Reset(const std::vector<float>& robot_base_quat)
{
    Update(0.0f);

    const std::vector<float> robot_torso =
        ComputeTorsoQuat(robot_base_quat);
    const std::vector<float> motion_torso = GetAnchorQuat();
    world_to_init_ = ComputeYawAlignment(robot_torso, motion_torso);
}

std::vector<float> MotionLoaderLW::GetJointPos() const
{
    std::vector<float> result(prepared_motion_->num_joints);
    WriteJointPos(result);
    return result;
}

void MotionLoaderLW::WriteJointPos(std::vector<float>& result) const
{
    if (result.size() != prepared_motion_->num_joints)
    {
        throw std::invalid_argument(
            "LW joint-position output has the wrong size");
    }
    const auto& pos0 = prepared_motion_->joint_positions[index_0_];
    const auto& pos1 = prepared_motion_->joint_positions[index_1_];

    for (std::size_t joint = 0;
         joint < prepared_motion_->num_joints;
         ++joint)
    {
        result[joint] =
            pos0[joint] * (1.0f - blend_)
            + pos1[joint] * blend_;
    }
}

std::vector<float> MotionLoaderLW::GetJointVel() const
{
    std::vector<float> result(prepared_motion_->num_joints);
    WriteJointVel(result);
    return result;
}

void MotionLoaderLW::WriteJointVel(std::vector<float>& result) const
{
    if (result.size() != prepared_motion_->num_joints)
    {
        throw std::invalid_argument(
            "LW joint-velocity output has the wrong size");
    }
    const auto& vel0 = prepared_motion_->joint_velocities[index_0_];
    const auto& vel1 = prepared_motion_->joint_velocities[index_1_];

    for (std::size_t joint = 0;
         joint < prepared_motion_->num_joints;
         ++joint)
    {
        result[joint] =
            vel0[joint] * (1.0f - blend_)
            + vel1[joint] * blend_;
    }
}

std::vector<float> MotionLoaderLW::GetRootQuat() const
{
    std::vector<float> result(4);
    WriteRootQuat(result);
    return result;
}

void MotionLoaderLW::WriteRootQuat(std::vector<float>& result) const
{
    WriteSlerp(
        prepared_motion_->root_quaternions[index_0_],
        prepared_motion_->root_quaternions[index_1_],
        blend_,
        result);
}

std::vector<float> MotionLoaderLW::ComputeTorsoQuat(
    const std::vector<float>& base_quat)
{
    return QuaternionNormalize(base_quat);
}

std::vector<float> MotionLoaderLW::ComputeYawAlignment(
    const std::vector<float>& robot_torso_quat,
    const std::vector<float>& motion_torso_quat)
{
    const std::vector<float> robot_yaw =
        QuaternionYawOnly(robot_torso_quat);
    const std::vector<float> motion_yaw =
        QuaternionYawOnly(motion_torso_quat);
    return QuaternionMultiply(
        robot_yaw, QuaternionConjugate(motion_yaw));
}

std::vector<float> MotionLoaderLW::GetAnchorQuat() const
{
    std::vector<float> result(4);
    WriteAnchorQuat(result);
    return result;
}

void MotionLoaderLW::WriteAnchorQuat(std::vector<float>& result) const
{
    WriteRootQuat(result);
    const float norm = std::sqrt(
        result[0] * result[0]
        + result[1] * result[1]
        + result[2] * result[2]
        + result[3] * result[3]);
    if (norm < 1e-8f)
    {
        result[0] = 1.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        return;
    }
    for (float& value : result)
    {
        value /= norm;
    }
}

void MotionLoaderLW::WriteSlerp(
    const std::vector<float>& q0,
    const std::vector<float>& q1,
    float t,
    std::vector<float>& result) const
{
    if (q0.size() != 4 || q1.size() != 4 || result.size() != 4)
    {
        throw std::invalid_argument(
            "LW quaternion interpolation requires four-element buffers");
    }
    float dot = q0[0] * q1[0] + q0[1] * q1[1]
        + q0[2] * q1[2] + q0[3] * q1[3];

    float q1_sign = 1.0f;
    if (dot < 0.0f)
    {
        q1_sign = -1.0f;
        dot = -dot;
    }

    if (dot > 0.9995f)
    {
        for (std::size_t index = 0; index < 4; ++index)
        {
            const float adjusted = q1_sign * q1[index];
            result[index] = q0[index] + t * (adjusted - q0[index]);
        }
    }
    else
    {
        const float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
        const float sin_theta = std::sin(theta);
        const float weight_0 = std::sin((1.0f - t) * theta) / sin_theta;
        const float weight_1 = std::sin(t * theta) / sin_theta;
        for (std::size_t index = 0; index < 4; ++index)
        {
            result[index] = q0[index] * weight_0
                + q1_sign * q1[index] * weight_1;
        }
    }

    const float norm = std::sqrt(
        result[0] * result[0]
        + result[1] * result[1]
        + result[2] * result[2]
        + result[3] * result[3]);
    if (norm < 1e-8f)
    {
        result[0] = 1.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        return;
    }
    for (float& value : result)
    {
        value /= norm;
    }
}
