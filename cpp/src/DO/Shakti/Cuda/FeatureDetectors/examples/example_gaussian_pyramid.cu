// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/SmallGaussianConvolutionFP32.hpp>
#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>

#include <cstdio>
#include <cstdlib>
#include <signal.h>

auto do_shutdown = sig_atomic_t{};
void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  do_shutdown = 1;
}


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = shakti::Cuda;
namespace scg = sc::Gaussian;


struct GaussianOctaveComputer
{
  inline GaussianOctaveComputer(int w, int h, int scale_count = 3)
    : host_kernels{scale_count}
    , device_kernels{host_kernels}
    , d_convx{{w, h}}
    , d_octave{sc::make_gaussian_octave<float>(w, h, scale_count)}
  {
    device_kernels.copy_filters_to_device_constant_memory();
  }

  inline auto
  operator()(shakti::MultiArrayView<float, 2, shakti::RowMajorStrides>& d_in)
      -> void
  {
    shakti::tic(d_timer);
    device_kernels(d_in, d_convx, d_octave);
    shakti::toc(d_timer, "Gaussian Octave");
  }

  inline auto copy_to_host() -> void
  {
    if (!h_octave.has_value())
      h_octave = sara::Image<float, 3, shakti::PinnedMemoryAllocator>{
          d_octave.width(), d_octave.height(), d_octave.scale_count()};

    // This is an extremely expensive copy.
    shakti::tic(d_timer);
    d_octave.array().copy_to(*h_octave);
    shakti::toc(d_timer, "Device to Host");
  }

  // Gaussian kernels.
  sc::GaussianOctaveKernels<float> host_kernels;
  scg::DeviceGaussianFilterBank device_kernels;

  // Device work buffer for intermediate x-convolution results.
  shakti::MultiArray<float, 2> d_convx;
  // Device buffer for the Gaussian octave.
  sc::Octave<float> d_octave;

  // Optional host buffer to read back the result.
  std::optional<sara::Image<float, 3, shakti::PinnedMemoryAllocator>> h_octave;

  // Profile.
  shakti::Timer d_timer;
};


auto example_1() -> void
{
  auto devices = shakti::get_devices();
  auto& device = devices.back();
  device.make_current_device();

  constexpr auto video_filepath =
#ifdef _WIN32
      "C:/Users/David/Desktop/GOPR0542.MP4"
#else
      "/home/david/Desktop/Datasets/sfm/oddkiva/bali-excursion.mp4"
  // "/home/david/Desktop/Datasets/sfm/Family.mp4"
#endif
      ;
  std::cout << video_filepath << std::endl;
  sara::VideoStream video_stream{video_filepath};
  const auto w = video_stream.width();
  const auto h = video_stream.height();

  auto frame_index = int{-1};
  auto frame = video_stream.frame();
  // Use pinned memory, it's much much faster.
  auto frame_gray32f =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};
  auto frame_blurred =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};

  static constexpr auto scale_count = 4;
  const auto host_kernels = sc::GaussianOctaveKernels<float>{scale_count};
  auto device_kernels = scg::DeviceGaussianFilterBank{host_kernels};
  device_kernels.copy_filters_to_device_constant_memory();
  device_kernels.peek_filters_in_device_constant_memory();


  auto d_in = shakti::MultiArray<float, 2, shakti::RowMajorStrides>{
      frame_gray32f.data(), {w, h}};
  auto d_convx = shakti::MultiArray<float, 2>{{w, h}};
  auto d_convy = shakti::MultiArray<float, 2>{{w, h}};

  auto d_timer = shakti::Timer{};

  // Display.
  sara::create_window(video_stream.sizes());
  while (video_stream.read())
  {
    ++frame_index;
    std::cout << "[Read frame] " << frame_index << "" << std::endl;

    sara::tic();
    sara::from_rgb8_to_gray32f(frame, frame_gray32f);
    sara::toc("Grayscale");

    shakti::tic(d_timer);
    d_in.copy_from_host(frame_gray32f.data(), w, h);
    shakti::toc(d_timer, "Host to Device");

    for (auto kernel_index = 0u; kernel_index < host_kernels.sigmas.size();
         ++kernel_index)
    {
      shakti::tic(d_timer);
      device_kernels(d_in, d_convx, d_convy, kernel_index);
      shakti::toc(d_timer, "GF");

      shakti::tic(d_timer);
      d_convy.copy_to_host(frame_blurred.data());
      shakti::toc(d_timer, "Device To Host");

#define CHECK_WITH_REFERENCE_IMPL
#ifdef CHECK_WITH_REFERENCE_IMPL
      const auto gt =
          sara::gaussian(frame_gray32f, host_kernels.sigmas[kernel_index]);
      SARA_CHECK(kernel_index);
      SARA_DEBUG
          << "L-inf norm = "
          << (gt.matrix() - frame_blurred.matrix()).lpNorm<Eigen::Infinity>()
          << std::endl;
      SARA_DEBUG << "L2 norm = "
                 << (gt.matrix() - frame_blurred.matrix()).norm() << std::endl;
      SARA_DEBUG << "Relative error = "
                 << (gt.matrix() - frame_blurred.matrix()).norm() /
                        gt.matrix().norm()
                 << std::endl;

      // TODO: double-check again...
      // L-inf norm = 0.000273228
      // L2 norm = 0.154609
      // Relative error = 0.000268316
    }

    sara::tic();
    sara::display(frame_blurred);
    sara::toc("Display");


    // device_kernels.peek_filters_in_device_constant_memory();
#endif
  }
}

auto example_2() -> void
{
  struct sigaction sig_int_handler;
  {
    sig_int_handler.sa_handler = my_handler;
    sigemptyset(&sig_int_handler.sa_mask);
    sig_int_handler.sa_flags = 0;
    sigaction(SIGINT, &sig_int_handler, nullptr);
  }

  auto devices = shakti::get_devices();
  auto& device = devices.back();
  device.make_current_device();

  constexpr auto video_filepath =
#ifdef _WIN32
      "C:/Users/David/Desktop/GOPR0542.MP4"
#else
        // "/home/david/Desktop/Datasets/sfm/oddkiva/bali-excursion.mp4"
        "/home/david/Desktop/Datasets/sfm/Family.mp4"
#endif
      ;
  std::cout << video_filepath << std::endl;
  sara::VideoStream video_stream{video_filepath};
  const auto w = video_stream.width();
  const auto h = video_stream.height();

  // Use pinned memory, it's much much faster.
  auto frame_gray32f =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};
  auto frame_blurred =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};

  static constexpr auto scale_count = 3;
  auto goc = GaussianOctaveComputer{w, h, scale_count};

  // Host and device input grayscale data.
  auto& h_in = frame_gray32f;
  auto d_in = shakti::MultiArray<float, 2, shakti::RowMajorStrides>{
      frame_gray32f.data(), {w, h}};

  // Profile.
  auto d_timer = shakti::Timer{};

  // Display.
  sara::create_window(video_stream.sizes());
  auto frame_index = int{-1};
  auto frame = video_stream.frame();
  while (video_stream.read())
  {
    ++frame_index;
    std::cout << "[Read frame] " << frame_index << "" << std::endl;

    sara::tic();
    sara::from_rgb8_to_gray32f(frame, frame_gray32f);
    sara::toc("Grayscale");

    shakti::tic(d_timer);
    d_in.copy_from_host(h_in.data(), w, h);
    shakti::toc(d_timer, "Transfer to Device");

    goc(d_in);
    goc.copy_to_host();


#ifdef INSPECT_ALL
    for (auto s = 0; s < goc.d_octave.scale_count(); ++s)
#else
      const auto s = goc.d_octave.scale_count() - 1;
#endif
    {
      const auto layer_s =
          sara::image_view(sara::tensor_view(*goc.h_octave)[s]);

      sara::tic();
      sara::display(layer_s);
      sara::toc("Display");
    }

    if (do_shutdown)
    {
      SARA_DEBUG << "CTRL+C triggered: quitting cleanly..." << std::endl;
      break;
    }
  }
}


GRAPHICS_MAIN()
{
  example_2();
  return 0;
}
