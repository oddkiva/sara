#pragma once

#if (defined(_WIN32) || defined(_WIN32_WCE)) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/TensorRT/InferenceEngine.hpp>

#include <cstdint>


namespace trt = DO::Shakti::TensorRT;


using CudaManagedTensor3ub =
    trt::InferenceEngine::ManagedTensor<std::uint8_t, 3>;
using CudaManagedTensor3f = trt::InferenceEngine::ManagedTensor<float, 3>;


auto naive_downsample_and_transpose(CudaManagedTensor3f& tensor_chw_resized_32f,
                                    const CudaManagedTensor3ub& tensor_hwc_8u)
    -> void;