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

#include "Utilities.hpp"


auto do_shutdown = sig_atomic_t{};
void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  do_shutdown = 1;
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
    static constexpr auto min_extremum_abs_value = 0.04f;
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
    sc::refine_extrema(d_dog_octave, d_extrema);
    shakti::toc(d_timer, "Extrema Refinement");

    shakti::tic(d_timer);
    auto h_extrema = d_extrema.copy_to_host();
    shakti::toc(d_timer, "Extrema Copy to Host");

    sara::tic();
#ifdef ASYNC
    display_queue.enqueue({frame, std::move(h_extrema), frame_index});
#else
    auto task = DisplayTask{frame, std::move(h_extrema), frame_index};
    task.run();
    sara::get_key();
#endif
    sara::toc("Display Enqueue");

    if (do_shutdown)
    {
      SARA_DEBUG << "CTRL+C triggered: quitting cleanly..." << std::endl;
      break;
    }
  }
  video_stream_end = true;

  display_async_task.join();

  std::cout << "Finished" << std::endl;

  return 0;
}
