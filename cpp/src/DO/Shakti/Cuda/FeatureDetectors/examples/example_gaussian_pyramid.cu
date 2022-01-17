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

#include <cstdio>
#include <cstdlib>
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
    static constexpr auto min_extremum_abs_value = 0.04f;
    sc::compute_scale_space_extremum_map(d_dog_octave, d_extremum_flat_map,
                                         min_extremum_abs_value);
    shakti::toc(d_timer, "Extremum Map");

    shakti::tic(d_timer);
    const auto num_extrema = sc::count_extrema(d_extremum_flat_map);
    // storage for the nonzero indices
    auto indices = thrust::device_vector<int>(num_extrema);
    auto d_extremum_sparse_map =
        thrust::device_vector<std::int8_t>(num_extrema);
    // compute indices of nonzero elements
    const auto dev_ptr =
        thrust::device_pointer_cast(d_extremum_flat_map.data());
    const auto indices_end = thrust::copy_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(int(d_extremum_flat_map.size())),
        dev_ptr, indices.begin(), thrust::identity());
    // Recopy extremum types.
    thrust::copy_if(dev_ptr, dev_ptr + d_extremum_flat_map.size(),
                    d_extremum_sparse_map.begin(), IsExtremum{});
    shakti::toc(d_timer, "Stream Compaction");
    SARA_CHECK(num_extrema);

#ifdef DENSE_EXTREMA
    shakti::tic(d_timer);
    d_extremum_flat_map.copy_to_host(h_extremum_map.data());
    shakti::toc(d_timer, "Device To Host");

    sara::tic();
#  pragma omp parallel for
    for (auto s = 1; s < h_extremum_map.depth() - 1; ++s)
      for (auto y = 1; y < h_extremum_map.height() - 1; ++y)
        for (auto x = 1; x < h_extremum_map.width() - 1; ++x)
        {
          if (h_extremum_map(x, y, s) == 1)
            sara::draw_circle(frame, x, y, 4, sara::Red8, 3);
          else if (h_extremum_map(x, y, s) == -1)
            sara::draw_circle(frame, x, y, 4, sara::Blue8, 3);
        }
#else
    shakti::tic(d_timer);
    auto h_indices = thrust::host_vector<int>(num_extrema);
    auto h_extremum = thrust::host_vector<std::int8_t>(num_extrema);
    thrust::copy(indices.begin(), indices.end(), h_indices.begin());
    thrust::copy(d_extremum_sparse_map.begin(), d_extremum_sparse_map.end(),
                 h_extremum.begin());
    shakti::toc(d_timer, "Extrema Copy");

#  pragma omp parallel for
    for (auto k = 0; k < num_extrema; ++k)
    {
      const auto& i = h_indices[k];

      const auto s = i / (w * h);
      const auto y = (i - s * w * h) / w;
      const auto x = i - s * w * h - y * w;

      if (h_extremum[k] == 1)
        sara::draw_circle(frame, x, y, 4, sara::Red8, 3);
      else if (h_extremum[k] == -1)
        sara::draw_circle(frame, x, y, 4, sara::Blue8, 3);
    }
#endif
    sara::display(frame);
    sara::toc("Display");

    if (do_shutdown)
    {
      SARA_DEBUG << "CTRL+C triggered: quitting cleanly..." << std::endl;
      break;
    }
  }

  return 0;
}

// shakti::tic(d_timer);
// d_dog_octave.array().copy_to(h_dog_octave);
// shakti::toc(d_timer, "Device To Host");

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
