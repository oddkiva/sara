#pragma once

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <fstream>
#include <sstream>


namespace DO::Sara::TensorRT {

  inline auto
  load_from_model_weights(nvinfer1::IRuntime& runtime,
                          const std::string& trt_model_weights_filepath)
      -> CudaEngineUniquePtr
  {
    auto model_weights_file = std::ifstream{trt_model_weights_filepath};
    if (!model_weights_file)
      throw std::runtime_error{"Failed to open model weights file!"};

    auto model_weights_stream = std::stringstream{};
    model_weights_stream << model_weights_file.rdbuf();

    // Count the number of bytes.
    model_weights_stream.seekg(0, std::ios::end);
    const auto model_weights_byte_size = model_weights_stream.tellg();

    // Rewind to the beginning of the file.
    model_weights_stream.seekg(0, std::ios::beg);

    // Read the file and transfer the data to the array of the bytes.
    auto model_weights = std::vector<char>(model_weights_byte_size);
    model_weights_stream.read(model_weights.data(), model_weights.size());

    // Deserialize the model weights data to initialize the CUDA inference
    // engine.
    return CudaEngineUniquePtr{runtime.deserializeCudaEngine(
                                   model_weights.data(), model_weights.size()),
                               &engine_deleter};
  }

  inline auto save_model_weights(nvinfer1::ICudaEngine* engine,
                                 const std::string& model_weights_filepath)
      -> void
  {
    // Serialize the model weights into the following data buffer.
    auto model_weights =
        HostMemoryUniquePtr{engine->serialize(), &host_memory_deleter};

    // Save in the disk.
    auto model_weights_stream = std::stringstream{};
    model_weights_stream.seekg(0, model_weights_stream.beg);
    model_weights_stream.write(static_cast<const char*>(model_weights->data()),
                               model_weights->size());

    auto model_weights_file = std::ofstream{model_weights_filepath};
    if (!model_weights_file)
      throw std::runtime_error{"Failed to create model weights file!"};
    model_weights_file << model_weights_stream.rdbuf();
  }


}  // namespace DO::Sara::TensorRT
