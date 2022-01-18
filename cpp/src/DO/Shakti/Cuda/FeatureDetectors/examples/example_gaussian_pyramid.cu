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

#include <drafts/Taskflow/SafeQueue.hpp>

#include <omp.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <signal.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


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

struct IsExtremum
{
  __host__ __device__ inline auto operator()(std::int8_t val) -> bool
  {
    return val != 0;
  }
};

struct DisplayTask
{
  sara::Image<sara::Rgb8> image;
  sc::QuantizedExtrema data;
  int index = -1;

  inline DisplayTask() = default;

  inline DisplayTask(sara::Image<sara::Rgb8> im, sc::QuantizedExtrema&& data,
                     int id)
    : image{std::move(im)}
    , data{std::move(data)}
    , index{id}
  {
  }

  inline DisplayTask(const DisplayTask& task) = default;

  inline DisplayTask(DisplayTask&& task) noexcept
    : image{std::move(task.image)}
    , data{std::move(task.data)}
    , index{task.index}
  {
  }

  inline ~DisplayTask() = default;

  inline auto run() -> void
  {
    if (index == -1 || image.data() == nullptr)
      return;

    const auto w = image.width();
    const auto h = image.height();

    sara::draw_text(image, 100, 50, std::to_string(index), sara::White8, 30);
    const auto num_extrema = static_cast<int>(data.indices.size());

#pragma omp parallel for
    for (auto k = 0; k < num_extrema; ++k)
    {
      const auto& i = data.indices[k];

      const auto s = i / (w * h);
      const auto y = (i - s * w * h) / w;
      const auto x = i - s * w * h - y * w;

      if (data.types[k] == 1)
        sara::draw_circle(image, x, y, 4, sara::Red8, 3);
      else if (data.types[k] == -1)
        sara::draw_circle(image, x, y, 4, sara::Blue8, 3);
    }

    sara::display(image);

    std::cout << "[" << index << "] " << num_extrema << " keypoints"
              << std::endl;
  }
};


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
        std::cout << "Finished display task" << std::endl;
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
      shakti::MultiArray<std::int8_t, 1>{w * h * d_dog_octave.scale_count()};
  auto h_extremum_map =
      sara::Image<std::int8_t, 3, shakti::PinnedMemoryAllocator>{
          w, h, d_dog_octave.scale_count()};

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
    const auto [d_indices, d_extremum_sparse_map] =
        sc::compress_extremum_map(d_extremum_flat_map);
    shakti::toc(d_timer, "Stream Compaction");

    shakti::tic(d_timer);
    auto extrema = sc::QuantizedExtrema{};
    const auto& num_extrema = d_indices.size();
    extrema.indices = thrust::host_vector<int>(num_extrema);
    extrema.types = thrust::host_vector<std::int8_t>(num_extrema);
    thrust::copy(d_indices.begin(), d_indices.end(), extrema.indices.begin());
    thrust::copy(d_extremum_sparse_map.begin(), d_extremum_sparse_map.end(),
                 extrema.types.begin());
    shakti::toc(d_timer, "Extrema Copy to Host");

    sara::tic();
    display_queue.enqueue(DisplayTask{frame, std::move(extrema), frame_index});
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
