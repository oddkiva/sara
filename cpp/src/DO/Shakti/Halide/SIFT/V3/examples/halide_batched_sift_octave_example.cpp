// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <omp.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/SIFT/Draw.hpp>
#include <DO/Shakti/Halide/SIFT/V3/Pipeline.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


auto debug_sift_octave(halide::v3::SiftOctavePipeline& sift_octave)
{
  sara::tic();
  sift_octave.y_convolved.copy_to_host();
  sara::toc("Copy gaussians to host");

  for (auto s = 0; s < sift_octave.params.scale_count + 3; ++s)
  {
    sara::display(sift_octave.gaussian(s, 0));
    sara::draw_text(20, 20,
                      sara::format("Gaussian: scale[%d] = %f", s,
                                   sift_octave.params.scales[s]),
                      sara::Blue8);
    sara::get_key();
  }

  sara::tic();
  sift_octave.gradient_mag.copy_to_host();
  sift_octave.gradient_ori.copy_to_host();
  sara::toc("Copy gradients to host");

  for (auto s = 0; s < sift_octave.params.scale_count + 3; ++s)
  {
    sara::display(sara::color_rescale(sift_octave.gradient_magnitude(s, 0)));
    sara::draw_text(20, 20,
                      sara::format("Gradient magnitude: scale[%d] = %f", s,
                                   sift_octave.params.scales[s]),
                      sara::Blue8);
    sara::get_key();

    sara::display(sara::color_rescale(sift_octave.gradient_orientation(s, 0)));
    sara::draw_text(20, 20,
                      sara::format("Gradient orientation: scale[%d] = %f", s,
                                   sift_octave.params.scales[s]),
                      sara::Blue8);
    sara::get_key();
  }

  sara::tic();
  sift_octave.dog.copy_to_host();
  sara::toc("Copy dog to host");

  for (auto s = 0; s < sift_octave.params.scale_count + 2; ++s)
  {
    sara::display(
        sara::color_rescale(sift_octave.difference_of_gaussians(s, 0)));
    sara::draw_text(
        20, 20,
        sara::format("DoG: scale[%d] = %f", s, sift_octave.params.scales[s]),
        sara::Blue8);
    sara::get_key();
  }
  sara::toc("Display");

  SARA_CHECK(sift_octave.extrema_quantized.size());
  SARA_CHECK(sift_octave.extrema.size());
  SARA_CHECK(sift_octave.extrema_oriented.size());
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // const auto video_filepath = "/home/david/Desktop/GOPR0542.MP4"s;
#endif


  // ===========================================================================
  // SARA PIPELINE
  //
  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray = sara::Image<float>{frame.sizes()};
  auto frame_gray_tensor =
      tensor_view(frame_gray)
          .reshape(
              Eigen::Vector4i{1, 1, frame_gray.height(), frame_gray.width()});


  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);

  auto sift_octave = halide::v3::SiftOctavePipeline{};
  sift_octave.params.initialize_kernels();
  sift_octave.initialize(frame.width(), frame.height());


  // Show the local extrema.
  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;

  auto timer = sara::Timer{};
  auto elapsed_ms = double{};

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    SARA_CHECK(frames_read);

    sara::tic();
    sara::from_rgb8_to_gray32f(frame, frame_gray);
    sara::toc("CPU rgb to grayscale");

    sara::tic();
    buffer_gray_4d.set_host_dirty();
    sara::toc("Set host dirty");

    timer.restart();
    sift_octave.feed(buffer_gray_4d);
    elapsed_ms = timer.elapsed_ms();
    SARA_DEBUG << "[Frame: " << frames_read << "] "
               << "total computation time = " << elapsed_ms << " ms"
               << std::endl;
    sara::toc("Octave computation");

    sara::tic();
    draw_oriented_extrema(frame, sift_octave.extrema_oriented, 1, 2);
    sara::display(frame);
    sara::toc("Display");

    // debug_sift_octave(sift_octave);
  }
}

GRAPHICS_MAIN()
{
  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);

  // test_on_image();
  test_on_video();
  return 0;
}
