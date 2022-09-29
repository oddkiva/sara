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

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Cuda/MultiArray/PinnedMemoryAllocator.hpp>


namespace DO::Sara::TensorRT {

  class InferenceExecutor
  {
  public:
    template <typename T, int N>
    using PinnedTensor = Tensor_<T, N, Shakti::PinnedMemoryAllocator>;

    InferenceExecutor() = default;

    InferenceExecutor(const HostMemoryUniquePtr& serialized_network);

    auto operator()(const PinnedTensor<float, 3>& in,
                    PinnedTensor<float, 3>& out,  //
                    const bool synchronize = false) -> void;

    CudaStreamUniquePtr _cuda_stream = make_cuda_stream();
    RuntimeUniquePtr _runtime = {nullptr, &runtime_deleter};
    CudaEngineUniquePtr _engine = {nullptr, &engine_deleter};
    ContextUniquePtr _context = {nullptr, &context_deleter};
  };

}  // namespace DO::Sara::TensorRT
