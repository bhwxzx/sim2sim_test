/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_runtime.hpp"
#include <limits>
#include <stdexcept>
#include <iostream>

namespace InferenceRuntime
{

namespace
{
const std::vector<TensorMetadata>& emptyTensorMetadata()
{
    static const std::vector<TensorMetadata> empty;
    return empty;
}

std::size_t checkedElementCount(
    const std::vector<int64_t>& shape,
    const std::string& description)
{
    std::size_t count = 1;
    for (const int64_t dimension : shape)
    {
        if (dimension <= 0)
        {
            throw std::runtime_error(
                description + " contains a non-positive dimension");
        }
        const auto unsigned_dimension =
            static_cast<std::uint64_t>(dimension);
        if (unsigned_dimension
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() / count))
        {
            throw std::runtime_error(
                description + " element count overflows size_t");
        }
        count *= static_cast<std::size_t>(unsigned_dimension);
    }
    return count;
}

#ifdef USE_ONNX
std::shared_ptr<const Ort::Env> sharedOnnxEnvironment()
{
    static const std::shared_ptr<const Ort::Env> environment =
        std::make_shared<Ort::Env>(
            ORT_LOGGING_LEVEL_WARNING,
            "rl_sar");
    return environment;
}

void requireStaticSingleSampleTensor(
    const TensorMetadata& tensor,
    const std::string& role)
{
    if (tensor.element_type != TensorElementType::Float32)
    {
        throw std::runtime_error(
            "ONNX " + role + " tensor must use float32");
    }
    if (tensor.shape.size() != 2)
    {
        throw std::runtime_error(
            "ONNX " + role + " tensor must have rank 2");
    }
    if (tensor.shape[0] != 1)
    {
        throw std::runtime_error(
            "ONNX " + role + " batch dimension must be fixed at 1");
    }
    if (tensor.shape[1] <= 0)
    {
        throw std::runtime_error(
            "ONNX " + role
            + " feature dimension must be fixed and positive");
    }
    static_cast<void>(checkedElementCount(
        tensor.shape, "ONNX " + role + " shape"));
}
#endif
} // namespace

const std::vector<TensorMetadata>& Model::input_metadata() const
{
    return emptyTensorMetadata();
}

const std::vector<TensorMetadata>& Model::output_metadata() const
{
    return emptyTensorMetadata();
}

std::vector<float> Model::forward(
    const std::vector<std::vector<float>>& inputs)
{
    if (!is_loaded())
    {
        throw std::runtime_error("Model not loaded");
    }
    if (output_metadata().size() != 1)
    {
        throw std::logic_error(
            "Model compatibility forward requires one output tensor");
    }
    std::vector<TensorView> input_views;
    input_views.reserve(inputs.size());
    for (const auto& input : inputs)
    {
        input_views.push_back({input.data(), input.size()});
    }
    std::vector<float> output(checkedElementCount(
        output_metadata().front().shape,
        "model output shape"));
    forwardInto(
        input_views.data(),
        input_views.size(),
        {output.data(), output.size()});
    return output;
}

// ============================================================================
// ONNXModel Implementation
// ============================================================================

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : environment_(sharedOnnxEnvironment()),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
}

ONNXModel::~ONNXModel() = default;

bool ONNXModel::load(const std::string& model_path)
{
    reset_loaded_state();
    try
    {
#ifdef USE_ONNX
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create inference session
        session_ = std::make_unique<Ort::Session>(
            *environment_,
            model_path.c_str(),
            session_options);

        // Setup input/output information
        setup_input_output_info();

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
#else
        static_cast<void>(model_path);
        std::cout << LOGGER::WARNING << "ONNX support not compiled. Please define USE_ONNX." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load ONNX model: " << e.what() << std::endl;
        reset_loaded_state();
        return false;
    }
}

void ONNXModel::forwardInto(
    const TensorView* inputs,
    std::size_t input_count,
    MutableTensorView output)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_ONNX
    try
    {
        if (input_count != 1)
        {
            throw std::invalid_argument(
                "ONNX inference requires exactly one input tensor");
        }
        if (inputs == nullptr || inputs[0].data == nullptr)
        {
            throw std::invalid_argument("ONNX input tensor view is null");
        }
        if (output.data == nullptr)
        {
            throw std::invalid_argument("ONNX output tensor view is null");
        }
        const TensorView& input = inputs[0];
        const auto& input_shape = input_metadata_.front().shape;
        const std::size_t expected_input_count = input_element_count_;
        if (input.size != expected_input_count)
        {
            throw std::invalid_argument(
                "ONNX input element count must be "
                + std::to_string(expected_input_count) + ", got "
                + std::to_string(input.size));
        }

        const auto& output_shape = output_metadata_.front().shape;
        const std::size_t expected_output_count = output_element_count_;
        if (output.size != expected_output_count)
        {
            throw std::invalid_argument(
                "ONNX output element count must be "
                + std::to_string(expected_output_count) + ", got "
                + std::to_string(output.size));
        }

        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input.data),
            input.size,
            input_shape.data(),
            input_shape.size()
        );
        auto output_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            output.data,
            output.size,
            output_shape.data(),
            output_shape.size());

        // Prepare input/output names
        const char* input_names[] = {input_node_names_[0].c_str()};
        const char* output_names[] = {output_node_names_[0].c_str()};

        // Execute inference
        session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            &output_tensor,
            1);

        validateOutput(output_tensor);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX inference error: " << e.what() << std::endl;
        throw;
    }
#else
    static_cast<void>(inputs);
    static_cast<void>(input_count);
    static_cast<void>(output);
    throw std::runtime_error("ONNX support not compiled");
#endif
}

void ONNXModel::reset_loaded_state() noexcept
{
    loaded_ = false;
    model_path_.clear();
#ifdef USE_ONNX
    session_.reset();
    input_node_names_.clear();
    output_node_names_.clear();
#endif
    input_metadata_.clear();
    output_metadata_.clear();
    input_element_count_ = 0;
    output_element_count_ = 0;
}

#ifdef USE_ONNX
void ONNXModel::setup_input_output_info()
{
    input_node_names_.clear();
    output_node_names_.clear();
    input_metadata_.clear();
    output_metadata_.clear();

    const size_t num_input_nodes = session_->GetInputCount();
    const size_t num_output_nodes = session_->GetOutputCount();
    if (num_input_nodes != 1)
    {
        throw std::runtime_error(
            "ONNX model must expose exactly one input tensor, got "
            + std::to_string(num_input_nodes));
    }
    if (num_output_nodes != 1)
    {
        throw std::runtime_error(
            "ONNX model must expose exactly one output tensor, got "
            + std::to_string(num_output_nodes));
    }

    input_node_names_.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        // Get input name
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_node_names_.push_back(std::string(input_name.get()));

        // Get input shape
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        if (input_type_info.GetONNXType() != ONNX_TYPE_TENSOR)
        {
            throw std::runtime_error(
                "ONNX input must be a tensor");
        }
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        input_metadata_.push_back(
            {input_node_names_.back(),
             input_dims,
             input_tensor_info.GetElementType()
                 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                 ? TensorElementType::Float32
                 : TensorElementType::Unknown});
        requireStaticSingleSampleTensor(input_metadata_.back(), "input");
        input_element_count_ = checkedElementCount(
            input_metadata_.back().shape,
            "cached ONNX input shape");
    }

    output_node_names_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        // Get output name
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_node_names_.push_back(std::string(output_name.get()));

        // Get output shape
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        if (output_type_info.GetONNXType() != ONNX_TYPE_TENSOR)
        {
            throw std::runtime_error(
                "ONNX output must be a tensor");
        }
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        output_metadata_.push_back(
            {output_node_names_.back(),
             output_dims,
             output_tensor_info.GetElementType()
                 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                 ? TensorElementType::Float32
                 : TensorElementType::Unknown});
        requireStaticSingleSampleTensor(output_metadata_.back(), "output");
        output_element_count_ = checkedElementCount(
            output_metadata_.back().shape,
            "cached ONNX output shape");
    }

    // Establish the private cache invariant before load() publishes loaded_.
    if (input_metadata_.size() != 1
        || output_metadata_.size() != 1
        || input_node_names_.size() != 1
        || output_node_names_.size() != 1)
    {
        throw std::logic_error(
            "ONNX model cache is inconsistent with the single-tensor contract");
    }
}

void ONNXModel::validateOutput(const Ort::Value& output) const
{
    if (!output.IsTensor())
    {
        throw std::runtime_error("ONNX Runtime output is not a tensor");
    }

    const auto output_info = output.GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType()
        != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error(
            "ONNX Runtime output tensor must use float32");
    }
    if (output_info.GetElementCount() != output_element_count_)
    {
        throw std::runtime_error(
            "ONNX Runtime output element count differs from the loaded contract");
    }

}
#endif

// ============================================================================
// ModelFactory Implementation
// ============================================================================

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
        case ModelType::ONNX:
            return std::make_unique<ONNXModel>();
        default:
            return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    // Extract file extension from path
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Determine model type based on extension
    if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    else
    {
        throw std::runtime_error(
            "Unknown model file extension: " + extension
            + ". Supported: .onnx");
    }
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path, ModelType type)
{
    // If type is AUTO, automatically detect model type
    if (type == ModelType::AUTO)
    {
        type = detect_model_type(model_path);
    }

    // Create and load model
    auto model = create_model(type);
    if (model && model->load(model_path))
    {
        return model;
    }
    return nullptr;
}

} // namespace InferenceRuntime
