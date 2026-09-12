#ifndef LW_DEPLOYMENT_BUNDLE_HPP
#define LW_DEPLOYMENT_BUNDLE_HPP

#include <filesystem>
#include <string>
#include <vector>

struct LWDeploymentFileRecord
{
    std::string path;
    std::string sha256;
};

struct LWDeploymentBundleInfo
{
    std::filesystem::path package_prefix;
    std::filesystem::path bundle_root;
    std::filesystem::path policy_root;
    std::filesystem::path manifest_path;
    std::filesystem::path executable_path;
    std::string source_commit;
    std::string build_type;
    std::string executable_sha256;
    std::string onnx_runtime_version;
    std::string onnx_runtime_architecture;
    std::string onnx_runtime_archive_name;
    std::string onnx_runtime_archive_url;
    std::string onnx_runtime_archive_sha256;
    LWDeploymentFileRecord onnx_runtime_provenance;
    std::vector<LWDeploymentFileRecord> onnx_runtime_libraries;
    std::vector<LWDeploymentFileRecord> files;
    std::vector<LWDeploymentFileRecord> runtime_files;
};

class LWDeploymentBundle
{
public:
    static LWDeploymentBundleInfo Verify(
        const std::filesystem::path& package_share,
        const std::filesystem::path& running_executable,
        const std::string& compiled_source_commit);

    static std::string Sha256File(
        const std::filesystem::path& path);
};

#endif
