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
#include <DO/Shakti/Cuda/FeatureDetectors/DominantOrientations.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/Gradient.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/ScaleSpaceExtremum.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/GaussianOctaveComputer.hpp>
#include <DO/Shakti/Cuda/Utilities/DeviceInfo.hpp>

#include <omp.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <signal.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "AsyncDisplayTask.hpp"
#include "OctaveVisualization.hpp"


#ifndef _WIN32
auto do_shutdown = sig_atomic_t{};
void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  do_shutdown = 1;
}
#endif


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

  const auto video_filepath = argv[1];
  const auto scale_count = argc < 3 ? 3 : std::stoi(argv[2]);
  const auto min_extremum_abs_value = argc < 4 ? 0.01f : std::stof(argv[3]);
  const auto sync_display = argc > 4;

  // if (scale_count < 3)
  // {
  //   SARA_DEBUG << "Our implementation requires 3 scales in the octave at
  //   least!"
  //              << std::endl;
  //   return 1;
  // }

  omp_set_num_threads(omp_get_max_threads());

#ifndef _WIN32
  struct sigaction sig_int_handler;
  {
    sig_int_handler.sa_handler = my_handler;
    sigemptyset(&sig_int_handler.sa_mask);
    sig_int_handler.sa_flags = 0;
    sigaction(SIGINT, &sig_int_handler, nullptr);
  }
#endif

  auto display_queue = sara::SafeQueue<DisplayTask>{};
  auto frame_index = std::atomic_int32_t{-1};
  auto last_frame_shown = std::atomic_int32_t{-1};
  auto video_stream_end = std::atomic_bool{false};
  auto display_async_task = std::thread{
      [&display_queue, &last_frame_shown, &frame_index, &video_stream_end] {
        while (!video_stream_end)
        {
          auto task = display_queue.dequeue();
          if (task.index < last_frame_shown || task.index + 3 < frame_index)
            continue;
          last_frame_shown = task.index;
          task.run();
        }
      }};

  auto devices = shakti::get_devices();
  auto& device = devices.back();
  device.make_current_device();

  sara::VideoStream video_stream{video_filepath};
  const auto w = video_stream.width();
  const auto h = video_stream.height();
  SARA_CHECK(video_filepath);

  // Use pinned memory, it's much much faster.
  auto frame_gray32f =
      sara::Image<float, 2, shakti::PinnedMemoryAllocator>{w, h};

  auto goc = sc::GaussianOctaveComputer{w, h, scale_count};

  // Host and device input grayscale data.
  auto& h_in = frame_gray32f;
  auto d_in = shakti::MultiArray<float, 2, shakti::RowMajorStrides>{
      frame_gray32f.data(), {w, h}};

  auto d_gaussian_octave = sc::make_gaussian_octave<float>(w, h, scale_count);
  auto d_dog_octave = sc::make_DoG_octave<float>(w, h, scale_count);
  auto d_grad_mag = sc::make_gaussian_octave<float>(w, h, scale_count);
  auto d_grad_ori = sc::make_gaussian_octave<float>(w, h, scale_count);

  auto d_orientations = shakti::MultiArray<float, 2, shakti::RowMajorStrides>{};

  // TODO: because we need to pass it to thrust, so it cannot be pitched
  // memory.
  auto d_extremum_flat_map =
      shakti::MultiArray<std::int8_t, 1>(w * h * d_dog_octave.scale_count());
  auto d_extrema = sc::DeviceExtrema{};

  // Profile.
  auto d_timer = shakti::Timer{};

  // Display.
  sara::create_window(video_stream.sizes());
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
    sc::compute_scale_space_extremum_map(d_dog_octave, d_extremum_flat_map,
                                         min_extremum_abs_value);
    shakti::toc(d_timer, "Extremum Map");

    shakti::tic(d_timer);
    d_extrema = sc::compress_extremum_map(d_extremum_flat_map);
    shakti::toc(d_timer, "Stream Compaction");

    shakti::tic(d_timer);
    sc::initialize_extrema(d_extrema, w, h, d_dog_octave.scale_count());
    shakti::toc(d_timer, "Extrema Init");

    shakti::tic(d_timer);
    sc::refine_extrema(d_dog_octave, d_extrema, goc.host_kernels.scale_initial,
                       goc.host_kernels.scale_factor);
    shakti::toc(d_timer, "Extrema Refinement");

    shakti::tic(d_timer);
#define DENSE_GRADIENT
#ifdef DENSE_GRADIENT
    sc::compute_polar_gradient_octave(d_gaussian_octave, d_grad_mag,
                                      d_grad_ori);
#else
    sc::compute_histogram_of_gradients(d_gaussian_octave, d_extrema.x,
                                       d_extrema.y, d_extrema.s);
#endif
    shakti::toc(d_timer, "Gradient");

    shakti::tic(d_timer);
    auto h_extrema = d_extrema.copy_to_host();
    shakti::toc(d_timer, "Extrema Copy to Host");

    sara::tic();
    auto task = DisplayTask{frame, std::move(h_extrema), frame_index};
    if (sync_display)
    {
      auto quit = false;

#ifdef INSPECT_STEPS
      view_octave(d_gaussian_octave, quit);
      if (quit)
        break;
      view_octave(d_dog_octave, quit, true);
      if (quit)
        break;
      view_octave(d_grad_mag, quit, true);
      if (quit)
        break;
      view_octave(d_grad_ori, quit, true);
      if (quit)
        break;
#endif

      task.run();
      quit = sara::get_key() == sara::KEY_ESCAPE;
      if (quit)
        break;
    }
    else
      display_queue.enqueue(std::move(task));
    sara::toc("Display Enqueue");

#ifndef _WIN32
    if (do_shutdown)
    {
      SARA_DEBUG << "CTRL+C triggered: quitting cleanly..." << std::endl;
      break;
    }
#endif
  }
  video_stream_end = true;

  display_async_task.join();

  std::cout << "Finished" << std::endl;

  return 0;
}
