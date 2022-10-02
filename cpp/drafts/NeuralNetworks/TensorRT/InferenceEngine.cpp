// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <drafts/NeuralNetworks/TensorRT/InferenceEngine.hpp>

#include <fstream>


namespace DO::Sara::TensorRT {

  InferenceEngine::InferenceEngine(
      const HostMemoryUniquePtr& serialized_network)
  {
    // Create a runtime.
    _runtime = {nvinfer1::createInferRuntime(Logger::instance()),
                &runtime_deleter};

    // Create or load an engine.
    _engine = {_runtime->deserializeCudaEngine(serialized_network->data(),
                                               serialized_network->size()),
               &engine_deleter};

    // Create an execution context.
    _context = {_engine->createExecutionContext(), &context_deleter};
  }

  auto InferenceEngine::operator()(const PinnedTensor<float, 3>& in,
                                   PinnedTensor<float, 3>& out,
                                   const bool synchronize) const -> void
  {
    const auto device_tensors = std::array{
        const_cast<void*>(reinterpret_cast<const void*>(in.data())),  //
        reinterpret_cast<void*>(out.data())                           //
    };

    // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
    if (!_context->enqueueV2(device_tensors.data(), *_cuda_stream, nullptr))
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;

    // Wait for the completion of GPU operations.
    if (synchronize)
      cudaStreamSynchronize(*_cuda_stream);
  }

  auto InferenceEngine::operator()(  //
      const PinnedTensor<float, 3>& in,
      std::vector<PinnedTensor<float, 3>>& out,  //
      const bool synchronize) const -> void
  {
    auto device_tensors = std::vector{
        const_cast<void*>(reinterpret_cast<const void*>(in.data())),  //
    };
    for (auto& o : out)
      device_tensors.push_back(reinterpret_cast<void*>(o.data()));

    // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
    if (!_context->enqueueV2(device_tensors.data(), *_cuda_stream, nullptr))
    {
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;
    }

    // Wait for the completion of GPU operations.
    if (synchronize)
      cudaStreamSynchronize(*_cuda_stream);
  }

  auto InferenceEngine::load_from_plan_file(const std::string& plan_filepath)
      -> void
  {
    // Create a runtime.
    if (_runtime.get() == nullptr)
    {
      SARA_DEBUG << "Creating a CUDA runtime...\n";
      _runtime = {nvinfer1::createInferRuntime(Logger::instance()),
                  &runtime_deleter};
    }

    // Create an execution context.
    SARA_DEBUG << "Opening TensorRT plan file...\n";
    auto model_weights_file =
        std::ifstream{plan_filepath, std::ifstream::in | std::ifstream::binary};
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
    SARA_DEBUG << "Deserializing TensorRT plan file...\n";
    _engine = {_runtime->deserializeCudaEngine(model_weights.data(),
                                               model_weights.size()),
               &engine_deleter};

    // Create an execution context.
    if (_context.get() == nullptr)
    {
      SARA_DEBUG << "Creating inference context...\n";
      _context = {_engine->createExecutionContext(), &context_deleter};
    }
  }

}  // namespace DO::Sara::TensorRT
