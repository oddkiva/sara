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

#include <drafts/NeuralNetworks/TensorRT/InferenceExecutor.hpp>


namespace DO::Sara::TensorRT {

  InferenceExecutor::InferenceExecutor(
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

  auto InferenceExecutor::operator()(const PinnedTensor<float, 3>& in,
                                     PinnedTensor<float, 3>& out,
                                     const bool synchronize) const -> void
  {
    const auto device_tensors = std::array{
        const_cast<void*>(reinterpret_cast<const void*>(in.data())),  //
        reinterpret_cast<void*>(out.data())                           //
    };

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

  auto InferenceExecutor::operator()(  //
      const PinnedTensor<float, 3>& in,
      std::array<PinnedTensor<float, 3>, 2>& out,  //
      const bool synchronize) const -> void
  {
    const auto device_tensors = std::array{
        const_cast<void*>(reinterpret_cast<const void*>(in.data())),  //
        reinterpret_cast<void*>(out[0].data()),                       //
        reinterpret_cast<void*>(out[1].data())                        //
    };

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

}  // namespace DO::Sara::TensorRT
