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

#include <DO/Shakti/Cuda/FeatureDetectors/DoG.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/GaussianOctaveComputer.hpp>
#include <DO/Shakti/Cuda/Utilities/DeviceInfo.hpp>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <signal.h>

namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = shakti::Cuda;
namespace scg = sc::Gaussian;


auto do_shutdown = sig_atomic_t{};
void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  do_shutdown = 1;
}


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

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " VIDEO_FILE" << std::endl;
    return 1;
  }

  omp_set_num_threads(omp_get_max_threads());

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

  const auto video_filepath = argv[1];
  sara::VideoStream video_stream{video_filepath};
  const auto w = video_stream.width();
  const auto h = video_stream.height();
  SARA_CHECK(video_filepath);

  // Use pinned memory, it's much much faster.
  auto frame_gray32f =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};

  static constexpr auto scale_count = 3;
  auto goc = sc::GaussianOctaveComputer{w, h, scale_count};

  // Host and device input grayscale data.
  auto& h_in = frame_gray32f;
  auto d_in = shakti::MultiArray<float, 2, shakti::RowMajorStrides>{
      frame_gray32f.data(), {w, h}};

  auto d_gaussian_octave = sc::make_gaussian_octave<float>(w, h, scale_count);
  auto d_dog_octave = sc::make_DoG_octave<float>(w, h, scale_count);
  auto h_dog_octave = sara::Image<float, 3, shakti::PinnedMemoryAllocator>{
      w, h, d_dog_octave.scale_count()};

  auto d_extremum_map = shakti::MultiArray<std::int8_t, 3>(  //
      {w, h, d_dog_octave.scale_count()});
  auto h_extremum_map =
      sara::Image<std::int8_t, 3, shakti::PinnedMemoryAllocator>{
          w, h, d_extremum_map.depth()};

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
    shakti::toc(d_timer, "Host to Device");

    goc(d_in, d_gaussian_octave);

    shakti::tic(d_timer);
    sc::compute_dog_octave(d_gaussian_octave, d_dog_octave);
    shakti::toc(d_timer, "DoG");

    shakti::tic(d_timer);
    static constexpr auto min_extremum_abs_value = 0.03f;
    sc::compute_scale_space_extremum_map(d_dog_octave, d_extremum_map,
                                         min_extremum_abs_value);
    shakti::toc(d_timer, "Extremum Map");

#define INSPECT
#ifdef INSPECT
    // shakti::tic(d_timer);
    // d_dog_octave.array().copy_to(h_dog_octave);
    // shakti::toc(d_timer, "Device To Host");

    shakti::tic(d_timer);
    d_extremum_map.copy_to_host(h_extremum_map.data());
    shakti::toc(d_timer, "Device To Host");

    // // #define INSPECT_ALL
    // #  ifdef INSPECT_ALL
    //     for (auto s = 0; s < d_dog_octave.scale_count(); ++s)
    // #  else
    //     const auto s = d_dog_octave.scale_count() - 1;
    // #  endif
    //     {
    //       const auto layer_s =
    //       sara::image_view(sara::tensor_view(h_dog_octave)[s]);
    //
    //       sara::tic();
    //       sara::display(sara::color_rescale(layer_s));
    //       sara::toc("Display");
    //     }
    // #endif

    sara::tic();
#pragma omp parallel for
    for (auto s = 1; s < h_extremum_map.depth() - 1; ++s)
      for (auto y = 1; y < h_extremum_map.height() - 1; ++y)
        for (auto x = 1; x < h_extremum_map.width() - 1; ++x)
        {
          if (h_extremum_map(x, y, s) == 1)
            sara::draw_circle(frame, x, y, 4, sara::Red8, 3);
          else if (h_extremum_map(x, y, s) == -1)
            sara::draw_circle(frame, x, y, 4, sara::Blue8, 3);
        }
    sara::display(frame);
    sara::toc("Display");
#endif

    if (do_shutdown)
    {
      SARA_DEBUG << "CTRL+C triggered: quitting cleanly..." << std::endl;
      break;
    }
  }

  return 0;
}
