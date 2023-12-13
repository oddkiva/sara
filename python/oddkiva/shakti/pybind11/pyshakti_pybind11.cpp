// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "shakti_enlarge_gpu.h"
#include "shakti_gaussian_convolution_cpu.h"
#include "shakti_gaussian_convolution_gpu.h"
#include "shakti_gradient_2d_32f_gpu_v2.h"
#include "shakti_gray32f_to_rgb8u_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"
#include "shakti_polar_gradient_2d_32f_gpu_v2.h"
#include "shakti_reduce_32f_gpu.h"
#include "shakti_scale_32f_gpu.h"

#include <DO/Sara/Core.hpp>

#ifdef USE_SHAKTI_CUDA_LIBRARIES
#  include <DO/Shakti/Cuda/ImageProcessing.hpp>
#endif
#include <DO/Shakti/Halide/MyHalide.hpp>

#include "do/sara/pybind11/Utilities.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>


namespace py = pybind11;


// ========================================================================== //
// Conversion from Numpy array to Halide runtime buffer.
// ========================================================================== //
template <typename T>
inline auto as_runtime_buffer_2d(py::array_t<T> image)
{
  if (image.ndim() != 2)
    throw std::runtime_error{"Invalid image shape!"};

  const auto height = static_cast<int>(image.shape(0));
  const auto width = static_cast<int>(image.shape(1));
  auto data = const_cast<T*>(image.data());
  auto buffer = Halide::Runtime::Buffer<T>{data, width, height};
  return buffer;
}

template <typename T>
inline auto as_runtime_buffer_4d(py::array_t<T>& image)
{
  if (image.ndim() != 2)
    throw std::runtime_error{"Invalid image shape!"};

  const auto height = static_cast<int>(image.shape(0));
  const auto width = static_cast<int>(image.shape(1));
  auto data = const_cast<T*>(image.data());

  return Halide::Runtime::Buffer<T>(data, width, height, 1, 1);
}

template <typename T>
inline auto as_interleaved_runtime_buffer_2d(py::array_t<T>& image)
{
  if (image.ndim() != 3 && image.shape(2) != 2)
    throw std::runtime_error{"Invalid image shape!"};

  const auto height = image.shape(0);
  const auto width = image.shape(1);
  const auto channels = image.shape(2);

  return Halide::Runtime::Buffer<T, 3>::make_interleaved(
      const_cast<T*>(image.data()), width, height, channels);
}


// ========================================================================== //
// Color conversion.
// ========================================================================== //
auto convert_rgb8_to_gray32f_cpu(py::array_t<std::uint8_t> src,
                                 py::array_t<float> dst)
{
  auto src_buffer = as_interleaved_runtime_buffer_2d(src);
  auto dst_buffer = as_runtime_buffer_2d(dst);
  shakti_rgb8u_to_gray32f_cpu(src_buffer, dst_buffer);
}


auto convert_gray32f_to_rgb8_cpu(py::array_t<float> src,
                                 py::array_t<std::uint8_t> dst)
{
  auto src_buffer = as_runtime_buffer_2d(src);
  auto dst_buffer = as_interleaved_runtime_buffer_2d(dst);
  shakti_gray32f_to_rgb8u_cpu(src_buffer, dst_buffer);
}


// ========================================================================== //
// Differential Calculus.
// ========================================================================== //
auto gradient_2d_32f(py::array_t<float> src, py::array_t<float> grad_x,
                     py::array_t<float> grad_y)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto grad_x_buffer = as_runtime_buffer_4d(grad_x);
  auto grad_y_buffer = as_runtime_buffer_4d(grad_y);

  src_buffer.set_host_dirty();
  shakti_gradient_2d_32f_gpu_v2(src_buffer, grad_x_buffer, grad_y_buffer);
  grad_x_buffer.copy_to_host();
  grad_y_buffer.copy_to_host();
}


auto polar_gradient_2d_32f(py::array_t<float> src, py::array_t<float> grad_x,
                           py::array_t<float> grad_y)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto grad_x_buffer = as_runtime_buffer_4d(grad_x);
  auto grad_y_buffer = as_runtime_buffer_4d(grad_y);

  src_buffer.set_host_dirty();
  shakti_polar_gradient_2d_32f_gpu_v2(src_buffer, grad_x_buffer, grad_y_buffer);
  grad_x_buffer.copy_to_host();
  grad_y_buffer.copy_to_host();
}


// ========================================================================== //
// Convolution
// ========================================================================== //
auto gaussian_convolution(py::array_t<float> src, py::array_t<float> dst,
                          float sigma, int truncation_factor, bool gpu)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto dst_buffer = as_runtime_buffer_4d(dst);

  src_buffer.set_host_dirty();
  if (gpu)
    shakti_gaussian_convolution_gpu(src_buffer, sigma, truncation_factor,
                                    dst_buffer);
  else
    shakti_gaussian_convolution_cpu(src_buffer, sigma, truncation_factor,
                                    dst_buffer);
  dst_buffer.copy_to_host();
}


// ========================================================================== //
// Resize operations.
// ========================================================================== //
auto scale(py::array_t<float> src, py::array_t<float> dst)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto dst_buffer = as_runtime_buffer_4d(dst);

  src_buffer.set_host_dirty();
  dst_buffer.set_host_dirty();
  shakti_scale_32f_gpu(src_buffer, dst_buffer.width(), dst_buffer.height(),
                       dst_buffer);
  dst_buffer.copy_to_host();
}

auto reduce(py::array_t<float> src, py::array_t<float> dst)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto dst_buffer = as_runtime_buffer_4d(dst);

  src_buffer.set_host_dirty();
  dst_buffer.set_host_dirty();
  shakti_reduce_32f_gpu(src_buffer, dst_buffer.width(), dst_buffer.height(),
                        dst_buffer);
  dst_buffer.copy_to_host();
}

auto enlarge(py::array_t<float> src, py::array_t<float> dst)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto dst_buffer = as_runtime_buffer_4d(dst);

  src_buffer.set_host_dirty();
  dst_buffer.set_host_dirty();
  shakti_enlarge_gpu(src_buffer,                               //
                     src_buffer.width(), src_buffer.height(),  //
                     dst_buffer.width(), dst_buffer.height(),  //
                     dst_buffer);
  dst_buffer.copy_to_host();
}


// ========================================================================== //
// Gaussian filtering.
// ========================================================================== //
#ifdef USE_SHAKTI_CUDA_LIBRARIES
struct CudaGaussianFilter : public DO::Shakti::GaussianFilter
{
  CudaGaussianFilter(float sigma, int gauss_trunc_factor)
    : DO::Shakti::GaussianFilter(sigma, gauss_trunc_factor)
  {
  }

  void apply(const DO::Sara::ImageView<float>& src,
             DO::Sara::ImageView<float>& dst) const
  {
    this->operator()(dst.data(), src.data(), src.sizes().data());
  }
};
#endif


PYBIND11_MODULE(pyshakti_pybind11, m)
{
  // Halide
  m.def("convert_rgb8_to_gray32f_cpu", &convert_rgb8_to_gray32f_cpu,
        "convert a RGB image buffer to a gray single-precision floating point "
        "image buffer");

  m.def("convert_gray32f_to_rgb8_cpu", &convert_gray32f_to_rgb8_cpu,
        "convert a gray single-precision floating point image buffer to a RGB "
        "image buffer");

  m.def("gaussian_convolution", &gaussian_convolution,
        "Convolve a gray single-precision floating point image buffer with a "
        "truncated Gaussian kernel");

  m.def("gradient_2d_32f", &gradient_2d_32f,
        "Calculate the 2D image gradients");
  m.def("polar_gradient_2d_32f", &polar_gradient_2d_32f,
        "Calculate the 2D image gradients");

  // CUDA.
#ifdef USE_SHAKTI_CUDA_LIBRARIES
  py::class_<CudaGaussianFilter>(m, "CudaGaussianFilter")
      .def(py::init<float, int>())
      .def("apply", &CudaGaussianFilter::apply);
#endif
}
