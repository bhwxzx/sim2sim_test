#include "lw_deployment_bundle.hpp"

#include <openssl/evp.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace
{
namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& message);

const std::set<std::string>& requiredPolicyFiles()
{
    static const std::set<std::string> files = {
        "policy/LW/base.yaml",
        "policy/LW/robot_lab/leg_loco/config.yaml",
        "policy/LW/robot_lab/leg_loco/policy.onnx",
        "policy/LW/robot_lab/leg_to_wheel/config.yaml",
        "policy/LW/robot_lab/leg_to_wheel/leg_to_wheel_transform_60hz.csv",
        "policy/LW/robot_lab/leg_to_wheel/policy.onnx",
        "policy/LW/robot_lab/wheel_loco/config.yaml",
        "policy/LW/robot_lab/wheel_loco/policy.onnx",
        "policy/LW/robot_lab/wheel_to_leg/config.yaml",
        "policy/LW/robot_lab/wheel_to_leg/policy.onnx",
        "policy/LW/robot_lab/wheel_to_leg/wheel_to_leg_transform_60hz.csv",
    };
    return files;
}

const std::set<std::string>& requiredRuntimeFiles()
{
    static const std::set<std::string> files = {
        "lib/libserial.a",
        "lib/fdilink_ahrs/ahrs_driver_node",
        "lib/rl_sar/lw_config_profiler",
        "lib/rl_sar/profile_lw_runtime_config.py",
        "share/ament_index/resource_index/packages/fdilink_ahrs",
        "share/ament_index/resource_index/packages/rl_sar",
        "share/ament_index/resource_index/packages/serial",
        "share/fdilink_ahrs/launch/ahrs_driver.launch.py",
        "share/fdilink_ahrs/package.xml",
        "share/fdilink_ahrs/wheeltec_udev.sh",
        "share/rl_sar/launch/rl_real_LW.launch.py",
        "share/serial/package.xml",
    };
    return files;
}

const std::set<std::string>& requiredProductionLaunchFiles()
{
    static const std::set<std::string> files = {
        "share/rl_sar/launch/rl_real_LW.launch.py",
    };
    return files;
}

const std::set<std::string>& requiredOnnxRuntimeFiles()
{
    static const std::set<std::string> files = {
        "lib/rl_sar/onnxruntime/libonnxruntime.so.1",
        "lib/rl_sar/onnxruntime/libonnxruntime_providers_shared.so",
    };
    return files;
}

const std::set<std::string>& requiredOnnxRuntimeDirectoryFiles()
{
    static const std::set<std::string> files = {
        "lib/rl_sar/onnxruntime/libonnxruntime.so.1",
        "lib/rl_sar/onnxruntime/libonnxruntime_providers_shared.so",
        "lib/rl_sar/onnxruntime/origin.json",
    };
    return files;
}

struct ApprovedOnnxArchive
{
    const char* version;
    const char* name;
    const char* url;
    const char* sha256;
    const char* main_library_sha256;
    const char* provider_library_sha256;
};

const ApprovedOnnxArchive& approvedOnnxArchive()
{
#if defined(__x86_64__) || defined(_M_X64)
    static const ApprovedOnnxArchive archive{
        "1.22.0",
        "onnxruntime-linux-x64-1.22.0.tgz",
        "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz",
        "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3",
#ifdef RL_SAR_TEST_APPROVED_ONNX_MAIN_SHA256
        RL_SAR_TEST_APPROVED_ONNX_MAIN_SHA256,
        RL_SAR_TEST_APPROVED_ONNX_PROVIDER_SHA256};
#else
        "3da6146e14e7b8aaec625dde11d6114c7457c87a5f93d744897da8781e35c673",
        "e4706ea02be3999167f000bb1ff7391c86e95d2d96f78eb33225c14b172a8451"};
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    static const ApprovedOnnxArchive archive{
        "1.22.0",
        "onnxruntime-linux-aarch64-1.22.0.tgz",
        "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz",
        "bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702",
#ifdef RL_SAR_TEST_APPROVED_ONNX_MAIN_SHA256
        RL_SAR_TEST_APPROVED_ONNX_MAIN_SHA256,
        RL_SAR_TEST_APPROVED_ONNX_PROVIDER_SHA256};
#else
        "0afd69a0ae38c5099fd0e8604dda398ac43dee67cd9c6394b5142b19e82528de",
        "3b53a2c1f3ecb87dedaee20e2a6d4d9aaec133e3ec280e2b9cc34efbed7de423"};
#endif
#else
    fail("unsupported compiled architecture");
#endif
    return archive;
}

std::string approvedOnnxLibrarySha256(
    const ApprovedOnnxArchive& archive,
    const std::string& relative_path)
{
    if (relative_path
        == "lib/rl_sar/onnxruntime/libonnxruntime.so.1")
    {
        return archive.main_library_sha256;
    }
    if (relative_path
        == "lib/rl_sar/onnxruntime/libonnxruntime_providers_shared.so")
    {
        return archive.provider_library_sha256;
    }
    return {};
}

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(
        "LW deployment validation failed: " + message);
}

bool isLowerHex(const std::string& value, std::size_t length)
{
    if (value.size() != length)
    {
        return false;
    }
    for (const unsigned char character : value)
    {
        if (!std::isdigit(character)
            && !(character >= 'a' && character <= 'f'))
        {
            return false;
        }
    }
    return true;
}

bool isSemanticVersion(const std::string& value)
{
    int component_count = 1;
    bool component_has_digit = false;
    for (const unsigned char character : value)
    {
        if (std::isdigit(character))
        {
            component_has_digit = true;
            continue;
        }
        if (character != '.' || !component_has_digit || component_count == 3)
        {
            return false;
        }
        ++component_count;
        component_has_digit = false;
    }
    return component_count == 3 && component_has_digit;
}

std::string compiledArchitecture()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    fail("unsupported compiled architecture");
#endif
}

std::uint16_t expectedElfMachine(const std::string& architecture)
{
    if (architecture == "x86_64")
    {
        return 62;
    }
    if (architecture == "aarch64")
    {
        return 183;
    }
    fail("unsupported ONNX Runtime architecture: " + architecture);
}

void requireElfArchitecture(
    const fs::path& path,
    const std::string& architecture)
{
    std::ifstream input(path, std::ios::binary);
    std::array<unsigned char, 20> header{};
    input.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())
        || header[0] != 0x7f || header[1] != 'E' || header[2] != 'L'
        || header[3] != 'F' || header[4] != 2)
    {
        fail("ONNX Runtime library is not ELF64: " + path.string());
    }

    std::uint16_t machine = 0;
    if (header[5] == 1)
    {
        machine = static_cast<std::uint16_t>(header[18])
            | static_cast<std::uint16_t>(header[19] << 8);
    }
    else if (header[5] == 2)
    {
        machine = static_cast<std::uint16_t>(header[18] << 8)
            | static_cast<std::uint16_t>(header[19]);
    }
    else
    {
        fail("ONNX Runtime library has invalid ELF encoding: "
             + path.string());
    }
    if (machine != expectedElfMachine(architecture))
    {
        fail("ONNX Runtime architecture mismatch for: " + path.string());
    }
}

bool isWithin(const fs::path& root, const fs::path& candidate)
{
    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    while (root_it != root.end() && candidate_it != candidate.end())
    {
        if (*root_it != *candidate_it)
        {
            return false;
        }
        ++root_it;
        ++candidate_it;
    }
    return root_it == root.end();
}

fs::path canonicalDirectory(
    const fs::path& path,
    const std::string& description)
{
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_directory(status))
    {
        fail(description + " is not a real directory: " + path.string());
    }
    const fs::path canonical_path = fs::canonical(path, error);
    if (error)
    {
        fail("cannot resolve " + description + ": " + error.message());
    }
    return canonical_path;
}

void requireRegularNonSymlink(
    const fs::path& path,
    const std::string& description)
{
    std::error_code error;
    const fs::file_status link_status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(link_status)
        || !fs::is_regular_file(link_status))
    {
        fail(description + " must be a regular non-symlink file: "
             + path.string());
    }
}

void requireNoSymlinkComponents(
    const fs::path& root,
    const fs::path& relative_path)
{
    fs::path current = root;
    for (const auto& component : relative_path)
    {
        current /= component;
        std::error_code error;
        const fs::file_status status = fs::symlink_status(current, error);
        if (error == std::errc::no_such_file_or_directory)
        {
            return;
        }
        if (error)
        {
            fail("cannot inspect bundle path " + current.string()
                 + ": " + error.message());
        }
        if (fs::is_symlink(status))
        {
            fail("bundle path contains a symbolic link: "
                 + current.string());
        }
    }
}

fs::path validateRelativePath(const std::string& value)
{
    const fs::path path(value);
    if (value.empty() || path.is_absolute())
    {
        fail("manifest path must be non-empty and relative: " + value);
    }
    const fs::path normalized = path.lexically_normal();
    if (normalized.generic_string() != value)
    {
        fail("manifest path is not normalized: " + value);
    }
    for (const auto& component : normalized)
    {
        if (component == "." || component == "..")
        {
            fail("manifest path escapes the bundle: " + value);
        }
    }
    return normalized;
}

std::string requiredScalar(
    const YAML::Node& node,
    const std::string& key)
{
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        fail("manifest field '" + key + "' is missing or not scalar");
    }
    return value.as<std::string>();
}
}

std::string LWDeploymentBundle::Sha256File(const fs::path& path)
{
    requireRegularNonSymlink(path, "hashed file");

    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        fail("cannot open file for hashing: " + path.string());
    }

    using ContextPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    ContextPtr context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    {
        fail("cannot initialize SHA-256");
    }

    std::array<char, 64 * 1024> buffer{};
    while (input)
    {
        input.read(buffer.data(), buffer.size());
        const std::streamsize count = input.gcount();
        if (count > 0
            && EVP_DigestUpdate(
                context.get(),
                buffer.data(),
                static_cast<std::size_t>(count)) != 1)
        {
            fail("cannot update SHA-256 for: " + path.string());
        }
    }
    if (!input.eof())
    {
        fail("failed while reading file for hashing: " + path.string());
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_length = 0;
    if (EVP_DigestFinal_ex(
            context.get(), digest.data(), &digest_length) != 1
        || digest_length != 32)
    {
        fail("cannot finalize SHA-256 for: " + path.string());
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_length; ++index)
    {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

LWDeploymentBundleInfo LWDeploymentBundle::Verify(
    const fs::path& package_share,
    const fs::path& running_executable,
    const std::string& compiled_source_commit)
{
    if (!isLowerHex(compiled_source_commit, 40))
    {
        fail("binary was not built in verified production mode");
    }

    const fs::path share = canonicalDirectory(
        package_share, "package share directory");
    if (share.filename() != "rl_sar"
        || share.parent_path().filename() != "share")
    {
        fail("unexpected package share layout: " + share.string());
    }
    const fs::path prefix = share.parent_path().parent_path();
    const fs::path bundle_root = canonicalDirectory(
        share / "deployment" / "LW", "LW deployment bundle");
    if (!isWithin(share, bundle_root))
    {
        fail("LW deployment bundle escapes the package share directory");
    }

    const fs::path manifest_path = bundle_root / "manifest.yaml";
    requireNoSymlinkComponents(bundle_root, "manifest.yaml");
    requireRegularNonSymlink(manifest_path, "deployment manifest");

    YAML::Node manifest;
    try
    {
        manifest = YAML::LoadFile(manifest_path.string());
    }
    catch (const std::exception& exception)
    {
        fail("cannot parse manifest: " + std::string(exception.what()));
    }
    if (!manifest || !manifest.IsMap())
    {
        fail("manifest root must be a mapping");
    }
    const YAML::Node schema = manifest["schema_version"];
    if (!schema || !schema.IsScalar() || schema.as<int>() != 4)
    {
        fail("unsupported manifest schema_version");
    }

    const std::string source_commit = requiredScalar(manifest, "source_commit");
    if (!isLowerHex(source_commit, 40)
        || source_commit != compiled_source_commit)
    {
        fail("manifest source_commit does not match the running binary");
    }
    const std::string build_type = requiredScalar(manifest, "build_type");
    if (build_type != "Release")
    {
        fail("production bundle build_type must be Release");
    }

    const YAML::Node executable = manifest["executable"];
    if (!executable || !executable.IsMap()
        || requiredScalar(executable, "name") != "rl_real_LW")
    {
        fail("manifest executable identity is invalid");
    }
    const std::string executable_sha256 =
        requiredScalar(executable, "sha256");
    if (!isLowerHex(executable_sha256, 64))
    {
        fail("manifest executable SHA-256 is invalid");
    }

    const fs::path installed_executable =
        prefix / "lib" / "rl_sar" / "rl_real_LW";
    requireRegularNonSymlink(installed_executable, "installed executable");
    std::error_code error;
    const fs::path installed_canonical =
        fs::canonical(installed_executable, error);
    if (error)
    {
        fail("cannot resolve installed executable: " + error.message());
    }
    const fs::path running_canonical = fs::canonical(running_executable, error);
    if (error || running_canonical != installed_canonical)
    {
        fail("running executable is not the verified package executable");
    }
    if (Sha256File(running_canonical) != executable_sha256)
    {
        fail("running executable SHA-256 mismatch");
    }

    const YAML::Node onnx_runtime = manifest["onnx_runtime"];
    if (!onnx_runtime || !onnx_runtime.IsMap())
    {
        fail("manifest onnx_runtime field must be a mapping");
    }
    const std::string onnx_runtime_version =
        requiredScalar(onnx_runtime, "version");
    if (!isSemanticVersion(onnx_runtime_version))
    {
        fail("manifest ONNX Runtime version is invalid");
    }
    const std::string onnx_runtime_architecture =
        requiredScalar(onnx_runtime, "architecture");
    if (onnx_runtime_architecture != compiledArchitecture())
    {
        fail("manifest ONNX Runtime architecture does not match the binary");
    }
    const ApprovedOnnxArchive& approved_archive = approvedOnnxArchive();
    if (onnx_runtime_version != approved_archive.version)
    {
        fail("manifest ONNX Runtime version is not approved");
    }
    const YAML::Node onnx_archive = onnx_runtime["archive"];
    if (!onnx_archive || !onnx_archive.IsMap())
    {
        fail("manifest ONNX Runtime archive field must be a mapping");
    }
    const std::string onnx_archive_name =
        requiredScalar(onnx_archive, "name");
    const std::string onnx_archive_url =
        requiredScalar(onnx_archive, "url");
    const std::string onnx_archive_sha256 =
        requiredScalar(onnx_archive, "sha256");
    if (onnx_archive_name != approved_archive.name
        || onnx_archive_url != approved_archive.url
        || onnx_archive_sha256 != approved_archive.sha256)
    {
        fail("manifest ONNX Runtime archive provenance is not approved");
    }

    const YAML::Node onnx_provenance = onnx_runtime["provenance"];
    if (!onnx_provenance || !onnx_provenance.IsMap())
    {
        fail("manifest ONNX Runtime provenance field must be a mapping");
    }
    const std::string provenance_path =
        requiredScalar(onnx_provenance, "path");
    const std::string provenance_sha256 =
        requiredScalar(onnx_provenance, "sha256");
    if (provenance_path != "lib/rl_sar/onnxruntime/origin.json"
        || !isLowerHex(provenance_sha256, 64))
    {
        fail("manifest ONNX Runtime provenance record is invalid");
    }
    const fs::path provenance_relative = validateRelativePath(provenance_path);
    requireNoSymlinkComponents(prefix, provenance_relative);
    const fs::path provenance_file = prefix / provenance_relative;
    requireRegularNonSymlink(provenance_file, "ONNX Runtime provenance");
    if (Sha256File(provenance_file) != provenance_sha256)
    {
        fail("ONNX Runtime provenance SHA-256 mismatch");
    }
    const YAML::Node onnx_libraries = onnx_runtime["libraries"];
    if (!onnx_libraries || !onnx_libraries.IsSequence())
    {
        fail("manifest ONNX Runtime libraries field must be a sequence");
    }

    std::set<std::string> seen_onnx_paths;
    std::vector<LWDeploymentFileRecord> onnx_records;
    onnx_records.reserve(onnx_libraries.size());
    for (const YAML::Node& entry : onnx_libraries)
    {
        if (!entry || !entry.IsMap())
        {
            fail("manifest ONNX Runtime library entry must be a mapping");
        }
        const std::string relative_string = requiredScalar(entry, "path");
        const fs::path relative_path = validateRelativePath(relative_string);
        if (!seen_onnx_paths.insert(relative_string).second)
        {
            fail("duplicate ONNX Runtime manifest path: " + relative_string);
        }
        const std::string expected_sha256 = requiredScalar(entry, "sha256");
        if (!isLowerHex(expected_sha256, 64))
        {
            fail("invalid ONNX Runtime SHA-256 for: " + relative_string);
        }
        if (expected_sha256
            != approvedOnnxLibrarySha256(approved_archive, relative_string))
        {
            fail(
                "ONNX Runtime library SHA-256 is not approved for: "
                + relative_string);
        }

        requireNoSymlinkComponents(prefix, relative_path);
        const fs::path library = prefix / relative_path;
        requireRegularNonSymlink(library, "ONNX Runtime library");
        const fs::path canonical_library = fs::canonical(library, error);
        if (error || !isWithin(prefix, canonical_library))
        {
            fail("ONNX Runtime library escapes install prefix: "
                 + relative_string);
        }
        requireElfArchitecture(library, onnx_runtime_architecture);
        if (Sha256File(library) != expected_sha256)
        {
            fail("ONNX Runtime SHA-256 mismatch for: " + relative_string);
        }
        onnx_records.push_back({relative_string, expected_sha256});
    }
    if (seen_onnx_paths != requiredOnnxRuntimeFiles())
    {
        fail("manifest does not contain the exact approved ONNX Runtime set");
    }
    const fs::path onnx_directory = canonicalDirectory(
        prefix / "lib" / "rl_sar" / "onnxruntime",
        "ONNX Runtime directory");
    std::set<std::string> actual_onnx_paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(onnx_directory))
    {
        const fs::file_status status = entry.symlink_status(error);
        if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        {
            fail("ONNX Runtime directory contains a non-regular entry: "
                 + entry.path().string());
        }
        actual_onnx_paths.insert(
            entry.path().lexically_relative(prefix).generic_string());
    }
    if (actual_onnx_paths != requiredOnnxRuntimeDirectoryFiles())
    {
        fail("ONNX Runtime directory does not contain the exact approved library set");
    }

    const YAML::Node files = manifest["files"];
    if (!files || !files.IsSequence())
    {
        fail("manifest files field must be a sequence");
    }

    std::set<std::string> seen_paths;
    std::vector<LWDeploymentFileRecord> records;
    records.reserve(files.size());
    for (const YAML::Node& entry : files)
    {
        if (!entry || !entry.IsMap())
        {
            fail("manifest file entry must be a mapping");
        }
        const std::string relative_string = requiredScalar(entry, "path");
        const fs::path relative_path = validateRelativePath(relative_string);
        if (!seen_paths.insert(relative_string).second)
        {
            fail("duplicate manifest path: " + relative_string);
        }
        const std::string expected_sha256 = requiredScalar(entry, "sha256");
        if (!isLowerHex(expected_sha256, 64))
        {
            fail("invalid SHA-256 for: " + relative_string);
        }

        requireNoSymlinkComponents(bundle_root, relative_path);
        const fs::path asset = bundle_root / relative_path;
        requireRegularNonSymlink(asset, "deployment asset");
        const fs::path canonical_asset = fs::canonical(asset, error);
        if (error || !isWithin(bundle_root, canonical_asset))
        {
            fail("deployment asset escapes bundle: " + relative_string);
        }
        if (Sha256File(asset) != expected_sha256)
        {
            fail("SHA-256 mismatch for: " + relative_string);
        }
        records.push_back({relative_string, expected_sha256});
    }

    if (seen_paths != requiredPolicyFiles())
    {
        fail("manifest does not contain the exact approved LW policy file set");
    }

    const YAML::Node runtime_files = manifest["runtime_files"];
    if (!runtime_files || !runtime_files.IsSequence())
    {
        fail("manifest runtime_files field must be a sequence");
    }

    std::set<std::string> seen_runtime_paths;
    std::vector<LWDeploymentFileRecord> runtime_records;
    runtime_records.reserve(runtime_files.size());
    for (const YAML::Node& entry : runtime_files)
    {
        if (!entry || !entry.IsMap())
        {
            fail("manifest runtime file entry must be a mapping");
        }
        const std::string relative_string = requiredScalar(entry, "path");
        const fs::path relative_path = validateRelativePath(relative_string);
        if (!seen_runtime_paths.insert(relative_string).second)
        {
            fail("duplicate runtime manifest path: " + relative_string);
        }
        const std::string expected_sha256 = requiredScalar(entry, "sha256");
        if (!isLowerHex(expected_sha256, 64))
        {
            fail("invalid runtime SHA-256 for: " + relative_string);
        }

        requireNoSymlinkComponents(prefix, relative_path);
        const fs::path asset = prefix / relative_path;
        requireRegularNonSymlink(asset, "runtime dependency");
        const fs::path canonical_asset = fs::canonical(asset, error);
        if (error || !isWithin(prefix, canonical_asset))
        {
            fail("runtime dependency escapes install prefix: "
                 + relative_string);
        }
        if (Sha256File(asset) != expected_sha256)
        {
            fail("runtime SHA-256 mismatch for: " + relative_string);
        }
        runtime_records.push_back({relative_string, expected_sha256});
    }

    if (seen_runtime_paths != requiredRuntimeFiles())
    {
        fail("manifest does not contain the exact approved runtime file set");
    }

    const fs::path launch_directory = canonicalDirectory(
        prefix / "share" / "rl_sar" / "launch",
        "production launch directory");
    std::set<std::string> actual_launch_paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(launch_directory))
    {
        const fs::file_status status = entry.symlink_status(error);
        if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        {
            fail("production launch directory contains a non-regular entry: "
                 + entry.path().string());
        }
        actual_launch_paths.insert(
            entry.path().lexically_relative(prefix).generic_string());
    }
    if (actual_launch_paths != requiredProductionLaunchFiles())
    {
        fail("production launch directory does not contain the exact approved file set");
    }

    const fs::path policy_root = canonicalDirectory(
        bundle_root / "policy", "LW deployment policy root");
    return {
        prefix,
        bundle_root,
        policy_root,
        manifest_path,
        running_canonical,
        source_commit,
        build_type,
        executable_sha256,
        onnx_runtime_version,
        onnx_runtime_architecture,
        onnx_archive_name,
        onnx_archive_url,
        onnx_archive_sha256,
        {provenance_path, provenance_sha256},
        onnx_records,
        records,
        runtime_records,
    };
}
