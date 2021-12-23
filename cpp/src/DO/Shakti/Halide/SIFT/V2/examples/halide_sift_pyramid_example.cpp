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
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

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

  // Video.
  auto video_filepath = std::string{};
  // Video processing parameters.
  auto skip = int{};
  auto start_octave_index = int{};
  // SIFT parameters.
  auto num_scales_per_octave = int{};
  auto nearest_neighbor_ratio = float{};
  // Performance profiling
  auto profile = bool{};
  // Display.
  auto show_features = false;

  po::options_description desc("Halide SIFT extractor");

  desc.add_options()     //
      ("help", "Usage")  //
      ("video,v", po::value<std::string>(&video_filepath),
       "input video file")  //
      ("start-octave,o", po::value<int>(&start_octave_index)->default_value(0),
       "image scale power")  //
      ("num_scales_per_octave,s",
       po::value<int>(&num_scales_per_octave)->default_value(1),
       "number of scales per octave")  //
      ("nearest_neighbor_ratio,m",
       po::value<float>(&nearest_neighbor_ratio)->default_value(0.6f),
       "number of scales per octave")  //
      ("skip", po::value<int>(&skip)->default_value(0),
       "number of frames to skip")  //
      ("profile,p", po::bool_switch(&profile),
       "profile code")  //
      ("show_features,f", po::bool_switch(&show_features),
       "show features")  //
      ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return;
  }

  if (!vm.count("video"))
  {
    std::cout << "The video file must be specified!\n" << desc << "\n";
    return;
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

  sift_pipeline.profile = profile;
  sift_pipeline.initialize(start_octave_index, num_scales_per_octave,
                           frame.width(), frame.height());

  // Show the local extrema.
  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;

  auto feature_timer = sara::Timer{};
  auto matching_timer = sara::Timer{};
  auto feature_time = double{};
  auto matching_time = double{};

  auto keys_prev = sara::KeypointList<sara::OERegion, float>{};
  auto keys_curr = sara::KeypointList<sara::OERegion, float>{};

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

    feature_timer.restart();
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

      sara::tic();
      keys_prev.swap(keys_curr);
      sift_pipeline.get_keypoints(keys_curr);
      sara::toc("Feature Reformatting");
    }
    feature_time = feature_timer.elapsed_ms();

    sara::tic();
    matching_timer.restart();
    auto matches = std::vector<sara::Match>{};
    const auto& fprev = std::get<0>(keys_prev);
    if (!fprev.empty())
    {
      sara::AnnMatcher matcher{keys_prev, keys_curr, nearest_neighbor_ratio};
      matches = matcher.compute_matches();
    }
    matching_time = matching_timer.elapsed_ms();
    sara::toc("Matching");

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
      for (size_t i = 0; i < matches.size(); ++i)
      {
        if (show_features)
        {
          draw(frame, matches[i].x(), sara::Blue8);
          draw(frame, matches[i].y(), sara::Cyan8);
        }
        const Eigen::Vector2f a = matches[i].x_pos();
        const Eigen::Vector2f b = matches[i].y_pos();
        sara::draw_arrow(frame, a, b, sara::Yellow8, 4);
      }
    }

    draw_text(frame, 100, 50,               //
              sara::format("SIFT: %0.f ms", feature_time),  //
              sara::White8, 40, 0, false, true, false);
    draw_text(frame, 100, 100,
              sara::format("Matching: %0.3f ms", matching_time),  //
              sara::White8, 40, 0, false, true, false);
    draw_text(frame, 100, 150,              //
              sara::format("Tracks: %u", matches.size()),  //
              sara::White8, 40, 0, false, true, false);

#ifdef USE_SHAKTI_CUDA_VIDEOIO
    sara::display(frame_rgb);
#else
    sara::display(frame);
#endif

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
