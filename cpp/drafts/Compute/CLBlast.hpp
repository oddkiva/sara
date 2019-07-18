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

#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
# define CL_SILENCE_DEPRECATION
# include <OpenCL/cl.hpp>
#else
# include <CL/cl.hpp>
#endif

#include <clblast.h>


namespace DO::Sara {

inline auto get_platforms()
{
  auto platforms = std::vector<cl::Platform>{};

  cl::Platform::get(&platforms);
  if (platforms.empty())
    throw std::runtime_error{"No available OpenCL platforms!"};

  return platforms;
}

inline auto get_compute_devices(const cl::Platform& platform)
{
  auto devices = std::vector<cl::Device>{};

  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.empty())
    throw std::runtime_error{"No available OpenCL compute devices!"};

  return devices;
}

inline auto gemm_batched(const TensorView_<float, 1>& alpha,
                         const TensorView_<float, 3>& A,
                         const TensorView_<float, 3>& B,
                         const TensorView_<float, 1>& beta,
                         TensorView_<float, 3>& C, const cl::Context& context,
                         const cl::CommandQueue& queue)
{
  auto calculate_offsets = [](const auto& batched_matrices) {
    auto batch_range = Tensor_<int, 1>(batched_matrices.size(0));
    auto offsets = Tensor_<std::size_t, 1>(batched_matrices.size(0));
    std::iota(std::begin(batch_range), std::end(batch_range), 0);
    std::transform(std::begin(batch_range), std::end(batch_range),
                   std::begin(offsets),
                   [&](int n) { return n * batched_matrices[n].size(); });
    return offsets;
  };
  const auto A_offsets = calculate_offsets(A);
  const auto B_offsets = calculate_offsets(B);
  const auto C_offsets = calculate_offsets(C);
  
  // Copy the matrices to the compute device.
  auto A_device_buffer =
      cl::Buffer(context, CL_MEM_READ_WRITE, A.size() * sizeof(float));
  auto B_device_buffer =
      cl::Buffer(context, CL_MEM_READ_WRITE, B.size() * sizeof(float));
  auto C_device_buffer =
      cl::Buffer(context, CL_MEM_READ_WRITE, C.size() * sizeof(float));

  queue.enqueueWriteBuffer(A_device_buffer, /* blocking */ CL_TRUE,
                           /* offset */ 0, A.size() * sizeof(float),
                           A.data());
  queue.enqueueWriteBuffer(B_device_buffer, /* blocking */ CL_TRUE,
                           /* offset */ 0, B.size() * sizeof(float),
                           B.data());
  queue.enqueueWriteBuffer(C_device_buffer, /* blocking */ CL_TRUE,
                           /* offset */ 0, C.size() * sizeof(float),
                           C.data());

  auto timer = Timer{};
  timer.restart();
  {
    auto queue_plain = queue();
    auto event = cl_event{nullptr};
    
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto n = B.size(1);
    const auto k = A.size(2);

    const auto status = clblast::GemmBatched(
        clblast::Layout::kRowMajor,                        //
        clblast::Transpose::kNo, clblast::Transpose::kNo,  //
        m, n, k,                                           //
        alpha.data(),                                      //
        A_device_buffer(), A_offsets.data(), A.stride(1),  //
        B_device_buffer(), B_offsets.data(), B.stride(1),  //
        beta.data(),                                       //
        C_device_buffer(), C_offsets.data(), C.stride(1),  //
        batch_size,                                        //
        &queue_plain, &event);

    if (status != clblast::StatusCode::kSuccess)
      throw std::runtime_error{
          format("Batched GEMM operation failed! Status code = %d", status)};

    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }
  const auto elapsed_time = timer.elapsed_ms();
  SARA_DEBUG << format("Completed batched SGEMM in %.3lf ms", elapsed_time)
             << std::endl;

  {
    const auto status = queue.enqueueReadBuffer(
        C_device_buffer, /* blocking */ true,
        /* offset */ 0, C.size() * sizeof(float), C.data());

    if (status)
      throw std::runtime_error{format(
          "Error: Failed to read buffer from device to host! Status code = %d",
          status)};
  }
}

} /* namespace DO::Sara */
