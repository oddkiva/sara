#include "python/do/sara/pybind11/Utilities.hpp"

#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_rgb_to_gray.h"

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>
#include <DO/Shakti/Halide/GaussianConvolution.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;

namespace py = pybind11;


auto convert_rgb8_to_gray32f_cpu(py::array_t<std::uint8_t> src,
                                 py::array_t<float> dst)
{
  // To Sara API.
  auto src_view = to_interleaved_rgb_image_view<std::uint8_t>(src);
  auto dst_view = to_image_view(dst);

  // To Halide API.
  auto src_buffer = halide::as_interleaved_runtime_buffer(src_view);
  auto dst_buffer = halide::as_runtime_buffer(dst_view);

  // Call the Halide-optimized function.
  shakti_halide_rgb_to_gray(src_buffer, dst_buffer);
}


auto convert_gray32f_to_rgb8_cpu(py::array_t<float> src,
                                 py::array_t<std::uint8_t> dst)
{
  // To Sara API.
  auto src_view = to_image_view(src);
  auto dst_view = to_interleaved_rgb_image_view<std::uint8_t>(dst);

  // To Halide API.
  auto src_buffer = halide::as_runtime_buffer(src_view);
  auto dst_buffer = halide::as_interleaved_runtime_buffer(dst_view);

  // Call the Halide-optimized function.
  shakti_halide_gray32f_to_rgb(src_buffer, dst_buffer);
}

auto gaussian_convolution(py::array_t<float> src, py::array_t<float> dst,
                          float sigma, int truncation_factor)
{
  // To Sara API.
  auto src_view = to_image_view(src);
  auto dst_view = to_image_view(dst);
  halide::gaussian_convolution(src_view, dst_view, sigma, truncation_factor);
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
}
