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

#pragma once

#include <NvInfer.h>

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>

#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace DO::Sara::TensorRT {

  //! @ingroup NeuralNetworks
  //! @defgroup TensorRT TensorRT helper functions.
  //! @{

  class Logger : public nvinfer1::ILogger
  {
  public:
    static inline auto instance() -> Logger&
    {
      static Logger instance;
      return instance;
    }

    inline auto log(Severity severity, const char* msg) noexcept
        -> void override
    {
      if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
  };


  inline auto delete_cuda_stream(cudaStream_t* cuda_stream)
  {
    if (cuda_stream == nullptr)
      return;
    SHAKTI_SAFE_CUDA_CALL(cudaStreamDestroy(*cuda_stream));
    delete cuda_stream;
    cuda_stream = nullptr;
  }

  template <typename NVInferObject>
  inline auto delete_nvinfer_object(NVInferObject* object) -> void
  {
    if (object != nullptr)
      delete object;
    object = nullptr;
  }

  inline auto delete_network_def(nvinfer1::INetworkDefinition* network_def)
  {
    delete_nvinfer_object(network_def);
  }

  inline auto delete_builder(nvinfer1::IBuilder* builder)
  {
    delete_nvinfer_object(builder);
  };

  inline auto make_cuda_stream()
      -> std::unique_ptr<cudaStream_t, decltype(&delete_cuda_stream)>
  {
    auto cuda_stream_ptr = new cudaStream_t{};
    SHAKTI_SAFE_CUDA_CALL(cudaStreamCreate(cuda_stream_ptr));
    return {cuda_stream_ptr, &delete_cuda_stream};
  }

  inline auto make_builder()
      -> std::unique_ptr<nvinfer1::IBuilder, decltype(&delete_builder)>
  {
    return {nvinfer1::createInferBuilder(Logger::instance()), &delete_builder};
  }

  inline auto make_network(nvinfer1::IBuilder* builder)
      -> std::unique_ptr<nvinfer1::INetworkDefinition,
                         decltype(&delete_network_def)>
  {
    static constexpr nvinfer1::NetworkDefinitionCreationFlags flags =
        1u << static_cast<std::uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    return {builder->createNetworkV2(flags), &delete_network_def};
  }

  //! @}


  inline auto engine_deleter(nvinfer1::ICudaEngine* engine) -> void
  {
    delete_nvinfer_object(engine);
  }

  inline auto runtime_deleter(nvinfer1::IRuntime* runtime) -> void
  {
    delete_nvinfer_object(runtime);
  }

  inline auto config_deleter(nvinfer1::IBuilderConfig* config) -> void
  {
    delete_nvinfer_object(config);
  }

  inline auto context_deleter(nvinfer1::IExecutionContext* context) -> void
  {
    delete_nvinfer_object(context);
  };

  inline auto host_memory_deleter(nvinfer1::IHostMemory* memory) -> void
  {
    delete_nvinfer_object(memory);
  }

  using BuilderUniquePtr =
      std::unique_ptr<nvinfer1::IBuilder, decltype(&delete_builder)>;
  using NetworkUniquePtr = std::unique_ptr<nvinfer1::INetworkDefinition,
                                           decltype(&delete_network_def)>;
  using HostMemoryUniquePtr =
      std::unique_ptr<nvinfer1::IHostMemory, decltype(&host_memory_deleter)>;
  using CudaStreamUniquePtr =
      std::unique_ptr<cudaStream_t, decltype(&delete_cuda_stream)>;
  using CudaEngineUniquePtr =
      std::unique_ptr<nvinfer1::ICudaEngine, decltype(&engine_deleter)>;
  using RuntimeUniquePtr =
      std::unique_ptr<nvinfer1::IRuntime, decltype(&runtime_deleter)>;
  using ConfigUniquePtr =
      std::unique_ptr<nvinfer1::IBuilderConfig, decltype(&config_deleter)>;
  using ContextUniquePtr =
      std::unique_ptr<nvinfer1::IExecutionContext, decltype(&context_deleter)>;

}  // namespace DO::Sara::TensorRT
