#include "python/do/sara/pybind11/Utilities.hpp"

#include "shakti_gradient_2d_32f_v2.h"
#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_rgb_to_gray.h"
#include "shakti_polar_gradient_2d_32f_v2.h"

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>
#include <DO/Shakti/Halide/GaussianConvolution.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>


namespace py = pybind11;
namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


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


auto convert_rgb8_to_gray32f_cpu(py::array_t<std::uint8_t> src,
                                 py::array_t<float> dst)
{
  auto src_buffer = as_interleaved_runtime_buffer_2d(src);
  auto dst_buffer = as_runtime_buffer_2d(dst);
  shakti_halide_rgb_to_gray(src_buffer, dst_buffer);
}


auto convert_gray32f_to_rgb8_cpu(py::array_t<float> src,
                                 py::array_t<std::uint8_t> dst)
{
  auto src_buffer = as_runtime_buffer_2d(src);
  auto dst_buffer = as_interleaved_runtime_buffer_2d(dst);
  shakti_halide_gray32f_to_rgb(src_buffer, dst_buffer);
}

auto gaussian_convolution(py::array_t<float> src, py::array_t<float> dst,
                          float sigma, int truncation_factor)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto dst_buffer = as_runtime_buffer_4d(dst);

  src_buffer.set_host_dirty();
  halide::gaussian_convolution(src_buffer, dst_buffer, sigma, truncation_factor);
  dst_buffer.copy_to_host();
}

auto polar_gradient_2d_32f(py::array_t<float> src, py::array_t<float> grad_x, py::array_t<float> grad_y)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto grad_x_buffer = as_runtime_buffer_4d(grad_x);
  auto grad_y_buffer = as_runtime_buffer_4d(grad_y);

  src_buffer.set_host_dirty();
  shakti_polar_gradient_2d_32f_v2(src_buffer, grad_x_buffer, grad_y_buffer);
  grad_x_buffer.copy_to_host();
  grad_y_buffer.copy_to_host();
}

auto gradient_2d_32f(py::array_t<float> src, py::array_t<float> grad_x, py::array_t<float> grad_y)
{
  auto src_buffer = as_runtime_buffer_4d(src);
  auto grad_x_buffer = as_runtime_buffer_4d(grad_x);
  auto grad_y_buffer = as_runtime_buffer_4d(grad_y);

  src_buffer.set_host_dirty();
  shakti_gradient_2d_32f_v2(src_buffer, grad_x_buffer, grad_y_buffer);
  grad_x_buffer.copy_to_host();
  grad_y_buffer.copy_to_host();
}


PYBIND11_MODULE(pyshakti_pybind11, m)
{
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
}
