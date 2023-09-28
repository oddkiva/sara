// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// Convolving in batch the input image does not seem very fast.
// Rather convolving sequentially seems much faster if we base ourselves from
// the computation time spent in the Halide implementation.
//
// THE TIMING on Titan 1080 Xp
//
// No memory error according to the memcheck tool:
// $> compute-sanitizer --tool memcheck
// ./bin/shakti_test_gaussian_convolution
//
// [gauss-octave][7680x4320x6] 21.212 ms
// [gauss-octave][3840x2160x6]  5.384 ms
// [gauss-octave][1920x1080x6]  1.416 ms
// [gauss-octave][ 960x 540x6]  0.436 ms
// [gauss-octave][ 480x 270x6]  0.151 ms
// [gauss-octave][ 240x 135x6]  0.091 ms
//
// 4K Gaussian pyramid = 7.797 ms
// 8K Gaussian pyramid = 29.538 ms


#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Gaussian Convolution"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/StringFormat.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/SmallGaussianConvolutionFP32.hpp>
#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;
namespace scg = sc::Gaussian;


BOOST_AUTO_TEST_CASE(test_convolve)
{
// #define CHECK_IMPL
#ifdef CHECK_IMPL
  const auto w0 = 2 * 3840;
  const auto h0 = 2 * 2160;
#else
  const auto w0 = 2 * 3840;
  const auto h0 = 2 * 2160;
#endif
  auto w = w0;
  auto h = h0;

  auto h_arr = sara::Image<float, 2>{w, h};
  h_arr.flat_array().fill(0);
  h_arr(w / 2, h / 2) = 1.f;

  // shakti::tic();
  auto d_in = shakti::MultiArray<float, 2>{h_arr.data(), {w, h}};
  // shakti::toc("Copy from host memory");

  const auto host_kernels = sc::GaussianOctaveKernels<float>{};
  auto device_kernels = scg::DeviceGaussianFilterBank{host_kernels};
  device_kernels.copy_filters_to_device_constant_memory();
  device_kernels.peek_filters_in_device_constant_memory();

  // Timing stuff.
  auto pyramid_compute_time = double{};
  auto elapsed = double{};
  auto timer = shakti::Timer{};

  do
  {
    // Initialize the octave CUDA surface.
    auto h_arr = sara::Image<float, 2>{w, h};
    h_arr.flat_array().fill(0);
    h_arr(w / 2, h / 2) = 1.f;

    // shakti::tic();
    auto d_in = shakti::MultiArray<float, 2>{h_arr.data(), {w, h}};
    // shakti::toc("Copy from host memory");

    // shakti::tic();
    auto d_convx = shakti::MultiArray<float, 2>{{w, h}};
    auto d_convy = shakti::MultiArray<float, 2>{{w, h}};
    // shakti::toc("Resize device buffers");

    auto octave_compute_time = double{};
#ifdef CHECK_IMPL
    const auto kernel_index = 0u;
#else  // BENCHMARK
    for (auto kernel_index = 0u; kernel_index < host_kernels.scales.size();
         ++kernel_index)
#endif
    {
      auto gauss_compute_time = double{};
      timer.restart();
      {
        device_kernels(d_in, d_convx, d_convy, kernel_index);
      }
      elapsed = timer.elapsed_ms();
      gauss_compute_time += elapsed;
      SARA_DEBUG << sara::format("[gaussf][s=%d] %0.3f ms", kernel_index,
                                 gauss_compute_time)
                 << std::endl;
      octave_compute_time += gauss_compute_time;

      if (w < 9 && h < 9)
      {
        auto h_convx = h_arr;
        auto h_convy = h_arr;
        d_convx.copy_to_host(h_convx.data());
        d_convy.copy_to_host(h_convy.data());

        const auto HeavyFmt =
            Eigen::IOFormat(4, 0, ", ", ",\n", "[", "]", "[", "]");

        std::cout << h_convx.matrix().format(HeavyFmt) << std::endl;
        std::cout << std::endl;
        std::cout << h_convy.matrix().format(HeavyFmt) << std::endl;
      }
    }

    SARA_DEBUG << sara::format("[octave][%4dx%4dx%u] %0.3f ms",  //
                               w, h, host_kernels.scales.size(),
                               octave_compute_time)
               << std::endl;

    pyramid_compute_time += octave_compute_time;

    w /= 2;
    h /= 2;

    SARA_CHECK(w);

  } while (w >= 32 && h >= 16);

  SARA_DEBUG << sara::format("[pyramid][%4dx%4dx%u] %0.3f ms",  //
                             w0, h0, host_kernels.scales.size(),
                             pyramid_compute_time)
             << std::endl;
}
