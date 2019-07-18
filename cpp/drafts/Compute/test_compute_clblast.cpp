// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Compute/CLBlast Backend"

#include "CLBlast.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_clblast_gemm_batched)
{
  using namespace DO::Sara;

  // OpenCL platform/device settings.
  const auto platform_id = 0;
  const auto device_id = 0;

  // List the available compute platforms for OpenCL.
  auto platforms = get_platforms();
  auto platform = platforms[platform_id];

  // List the available compute (GPU/CPU) devices for the selected platform.
  auto devices = get_compute_devices(platform);
  auto device = devices[device_id];
  
  // Initialize the compute context (resources + compute).
  const auto device_as_vector = std::vector<cl::Device>{device};
  const auto context = cl::Context(device_as_vector);
  const auto queue = cl::CommandQueue(context, device);

  const auto batch_count = 3;
  for (int i = 2; i < 1000; ++i)
  {
    SARA_DEBUG << "iteration = " << i << std::endl;
    const auto m = i;
    const auto n = i;
    const auto k = i;

    // Initialize the data.
    auto range = Tensor_<float, 1>{m * k};
    std::iota(std::begin(range), std::end(range), 0);
    range.flat_array() /= range.flat_array().maxCoeff();

    auto A = Tensor_<float, 3>{batch_count, m, k};
    for (int b = 0; b < batch_count; ++b)
      A[b].matrix() = range.reshape(Vector2i{m, k}).matrix();

    auto B = Tensor_<float, 3>{batch_count, k, n};
    for (int b = 0; b < batch_count; ++b)
      B[b].flat_array() = A[b].flat_array() * 1.2345f * (b + 1);

    auto C = Tensor_<float, 3>{batch_count, m, n};
    C.flat_array().fill(0.f);

    auto alpha = Tensor_<float, 1>{batch_count};
    alpha.flat_array().fill(1.f);
    auto beta = Tensor_<float, 1>{batch_count};
    beta.flat_array().fill(0.f);

    // Run the batch multiplication.
    //
    // N.B.: the GEMM wrapper as it is implemented is like a tensorflow
    // graph where we immediately run the session with the selected device and
    // with the data that is living in the host memory.
    gemm_batched(alpha, A, B, beta, C, context, queue);

    // Check matrix multiplication results.
    for (int b = 0; b < C.size(0); ++b)
    {
      const MatrixXf true_C_b = A[b].matrix() * B[b].matrix();
      const double rel_err =
          (C[b].matrix() - true_C_b).norm() / true_C_b.norm();
      BOOST_CHECK_SMALL(rel_err, 1e-6);
    }
  }
}
