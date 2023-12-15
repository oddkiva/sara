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

#if (defined(_WIN32) || defined(_WIN32_WCE)) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/MultiArray/PinnedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/TensorRT/Helpers.hpp>


namespace DO::Shakti::TensorRT {

  class DO_SARA_EXPORT InferenceExecutor
  {
  public:
    template <typename T, int N>
    using PinnedTensor = Sara::Tensor_<T, N, PinnedMemoryAllocator>;

    template <typename T, int N>
    using ManagedTensor = Sara::Tensor_<T, N, ManagedMemoryAllocator>;

    InferenceExecutor() = default;

    explicit InferenceExecutor(const HostMemoryUniquePtr& serialized_network);

    auto operator()(const PinnedTensor<float, 3>& in,
                    PinnedTensor<float, 3>& out,  //
                    const bool synchronize = true) const -> void;

    auto operator()(const PinnedTensor<float, 3>& in,
                    std::array<PinnedTensor<float, 3>, 2>& out,  //
                    const bool synchronize = true) const -> void;

    auto operator()(const ManagedTensor<float, 3>& in,
                    std::array<PinnedTensor<float, 3>, 2>& out,  //
                    const bool synchronize = true) const -> void;

    // private:
    CudaStreamUniquePtr _cuda_stream = make_cuda_stream();
    RuntimeUniquePtr _runtime = {nullptr, &runtime_deleter};
    CudaEngineUniquePtr _engine = {nullptr, &engine_deleter};
    ContextUniquePtr _context = {nullptr, &context_deleter};
  };

}  // namespace DO::Shakti::TensorRT
