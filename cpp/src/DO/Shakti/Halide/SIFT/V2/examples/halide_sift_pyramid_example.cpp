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

#include <algorithm>
#include <cmath>

#include <omp.h>

#include <boost/program_options.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/SIFT/Draw.hpp>
#include <DO/Shakti/Halide/SIFT/V2/Pipeline.hpp>

#ifdef USE_SHAKTI_CUDA_VIDEOIO
#  include <DO/Shakti/Cuda/VideoIO.hpp>
#endif

#include "shakti_bgra8u_to_gray32f_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  auto image = sara::imread<float>(image_filepath);
  auto image_tensor = tensor_view(image).reshape(
      Eigen::Vector4i{1, 1, image.height(), image.width()});
  auto buffer_4d = halide::as_runtime_buffer(image_tensor);

  auto sift_pipeline = halide::v2::SiftPyramidPipeline{};

  sift_pipeline.initialize(-1, 3,  //
                           image.width(), image.height());

  auto timer = sara::Timer{};

  timer.restart();
  {
    buffer_4d.set_host_dirty();
    sift_pipeline.feed(buffer_4d);
  }
  const auto elapsed_ms = timer.elapsed_ms();
  SARA_DEBUG << "SIFT pipeline: " << elapsed_ms << " ms" << std::endl;


#ifdef CHECK_INPUT_UPSCALED
  if (sift_pipeline.start_octave_index < 0)
  {
    auto input_upscaled = sift_pipeline.input_upscaled_view();
    sara::create_window(input_upscaled.sizes());
    sara::display(input_upscaled);
    sara::get_key();
    sara::resize_window(image.sizes());
  }
#endif

  if (!sara::active_window())
    sara::create_window(image.sizes());

// #define CHECK_PYRAMIDS
#ifdef CHECK_PYRAMIDS
  for (auto& octave : sift_pipeline.octaves)
    for (auto s = 0; s < octave.params.num_scales + 3; ++s)
      sara::display(octave.gaussian_view(s));
  sara::get_key();

  for (auto& octave : sift_pipeline.octaves)
    for (auto s = 0; s < octave.params.num_scales + 2; ++s)
    {
      sara::display(sara::color_rescale(octave.dog_view(s)));
      sara::get_key();
    }
  sara::get_key();
#endif

  sara::set_antialiasing();
  sara::display(image);
  for (auto o = 0u; o < sift_pipeline.octaves.size(); ++o)
  {
    auto& octave = sift_pipeline.octaves[o];
    for (auto s = 0u; s < octave.extrema_oriented.size(); ++s)
    {
      SARA_DEBUG << sara::format("[o = %d, s = %d] Num extrema = %d",
                                 sift_pipeline.start_octave_index + o, s,
                                 octave.extrema_oriented[s].size())
                 << std::endl;
      draw_oriented_extrema(octave.extrema_oriented[s],
                            sift_pipeline.octave_scaling_factor(
                                sift_pipeline.start_octave_index + o));
    }
  }

  while (sara::get_key() != sara::KEY_ESCAPE)
    ;
}

auto test_on_video(int argc, char **argv)
{
  namespace po = boost::program_options;

  using namespace std::string_literals;

  // Parameter parsing.
  auto video_filepath = std::string{};
  auto downscale_factor = int{};
  auto skip = int{};
  auto show_features = false;
  auto num_scales_per_octave = int{};
  po::options_description desc("Halide SIFT extractor");

  desc.add_options()     //
      ("help", "Usage")  //
      ("video,v", po::value<std::string>(&video_filepath),
       "input video file")  //
      ("downscale-factor,d",
       po::value<int>(&downscale_factor)->default_value(2),
       "downscale factor")  //
      ("num_scales_per_octave,s", po::value<int>(&num_scales_per_octave)->default_value(1),
       "number of scales per octave")  //
      ("skip", po::value<int>(&skip)->default_value(0),
       "number of frames to skip")  //
      ("show_features,f", po::bool_switch(&show_features),
       "show features")  //
      ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  if (!vm.count("video"))
  {
    std::cout << "The video file must be specified!\n" << desc << "\n";
    return 1;
  }


  // ===========================================================================
  // SARA PIPELINE
  //
  // Input and output from Sara.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  // Initialize CUDA driver.
  DriverApi::init();

  // Create a CUDA context so that we can use the GPU device.
  const auto gpu_id = 0;
  auto cuda_context = DriverApi::CudaContext{gpu_id};
  cuda_context.make_current();

  // nVidia's hardware accelerated video decoder.
  shakti::VideoStream video_stream{video_filepath, cuda_context};
  auto frame =
      sara::Image<sara::Bgra8>{video_stream.width(), video_stream.height()};
  auto device_bgra_buffer =
      DriverApi::DeviceBgraBuffer{video_stream.width(), video_stream.height()};
#else
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
#endif

  auto frame_gray = sara::Image<float>{frame.sizes()};
  auto frame_gray_tensor =
      tensor_view(frame_gray)
          .reshape(
              Eigen::Vector4i{1, 1, frame_gray.height(), frame_gray.width()});

  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  auto buffer_bgra = halide::as_interleaved_runtime_buffer(frame);
#else
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
#endif
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);

  auto sift_pipeline = halide::v2::SiftPyramidPipeline{};

  const auto start_octave_index = 0;
  sift_pipeline.initialize(start_octave_index, num_scales_per_octave,
                           frame.width(), frame.height());


  // Show the local extrema.
  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;

  auto timer = sara::Timer{};
  auto elapsed_ms = double{};

  while (true)
  {
    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    const auto has_frame = video_stream.read(device_bgra_buffer);
#else
    const auto has_frame = video_stream.read();
#endif
    sara::toc("Read frame");
    if (!has_frame)
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }

#ifdef USE_SHAKTI_CUDA_VIDEOIO
    sara::tic();
    device_bgra_buffer.to_host(frame);
    sara::toc("Copy to host");
#endif

    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    timer.restart();
    {
      sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
      shakti_bgra8u_to_gray32f_cpu(buffer_bgra, buffer_gray);
#else
      shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray);
#endif
      sara::toc("CPU RGB to grayscale");

      sara::tic();
      buffer_gray_4d.set_host_dirty();
      sift_pipeline.feed(buffer_gray_4d);
      sara::toc("SIFT");
    }
    elapsed_ms = timer.elapsed_ms();

    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    auto frame_rgb = frame.cwise_transform(  //
        [](const sara::Bgra8& color) -> sara::Rgb8 {
          using namespace sara;
      return {color.channel<R>(), color.channel<G>(), color.channel<B>()};
        });
#endif

    if (show_features)
    {
      for (auto o = 0u; o < sift_pipeline.octaves.size(); ++o)
      {
        auto& octave = sift_pipeline.octaves[o];
#pragma omp parallel for
        for (auto s = 0; s < static_cast<int>(octave.extrema_oriented.size());
             ++s)
          draw_oriented_extrema(
#ifdef USE_SHAKTI_CUDA_VIDEOIO
              frame_rgb,
#else
              frame,
#endif
              octave.extrema_oriented[s],
              sift_pipeline.octave_scaling_factor(
                  sift_pipeline.start_octave_index + o));
      }
    }

#ifdef USE_SHAKTI_CUDA_VIDEOIO
    sara::display(frame_rgb);
#else
    sara::display(frame);
#endif
    const auto text = sara::format("[Frame: %d] Oriented DoG: %0.3f ms", frames_read, elapsed_ms);
    sara::draw_text(100, 100, text, sara::White8, 40, 0, false, true, false);
    sara::toc("Display");
  }
}


int __main(int argc, char** argv)
{
  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);

  // test_on_image();
  test_on_video(argc, argv);
  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
