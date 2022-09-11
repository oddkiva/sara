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
    SHAKTI_STDOUT << "DELETING CUDA STREAM" << std::endl;
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
    {
#ifdef DEBUG
      std::cout << "Deleting " << typeid(object).name() << " " << object
                << std::endl;
#endif
      delete object;
    }
    object = nullptr;
  }

  inline auto delete_network_def(nvinfer1::INetworkDefinition* network_def)
  {
    SHAKTI_STDOUT << "DELETING NETWORK DEFINITION" << std::endl;
    delete_nvinfer_object(network_def);
  }

  inline auto delete_builder(nvinfer1::IBuilder* builder)
  {
    SHAKTI_STDOUT << "DELETING BUILDER" << std::endl;
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

}  // namespace DO::Sara::TensorRT
