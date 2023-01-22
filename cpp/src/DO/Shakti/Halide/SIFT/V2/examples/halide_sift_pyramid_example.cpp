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

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <boost/program_options.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include <DO/Shakti/Halide/SIFT/Draw.hpp>
#include <DO/Shakti/Halide/SIFT/V2/Pipeline.hpp>

#ifdef USE_SHAKTI_CUDA_VIDEOIO
#  include <DO/Shakti/Cuda/VideoIO.hpp>
#endif


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


int __main(int argc, char** argv)
{
  // Optimization.
#ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
#endif
  std::ios_base::sync_with_stdio(false);

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
  auto match_keypoints = false;
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
      ("match_keypoints", po::bool_switch(&match_keypoints),
       "match SIFT keypoints frame by frame")  //
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
  auto matches = std::vector<sara::Match>{};
  static constexpr auto match_count_max_estimate = 20000;
  matches.reserve(match_count_max_estimate);

  auto F = sara::FundamentalMatrix{};
  auto inliers = sara::Tensor_<bool, 1>{};
  auto sample_best = sara::Tensor_<int, 1>{};

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
      from_bgra8_to_gray32f(frame, frame_gray);
#else
      from_rgb8_to_gray32f(frame, frame_gray);
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

    if (match_keypoints)
    {
      sara::tic();
      matching_timer.restart();
      matches.clear();
      const auto& fprev = sara::features(keys_prev);
      if (!fprev.empty())
      {
        sara::AnnMatcher matcher{keys_prev, keys_curr, nearest_neighbor_ratio};
        matches = matcher.compute_matches();
      }
      matching_time = matching_timer.elapsed_ms();
      sara::toc("Matching");


      sara::tic();
      if (!fprev.empty())
      {
        static constexpr auto num_samples = 200.;
        static constexpr auto f_err_thres = 4.;
        std::tie(F, inliers, sample_best) = sara::estimate_fundamental_matrix(
            matches, keys_prev, keys_curr, num_samples, f_err_thres);
      }
      sara::toc("Estimating the fundamental matrix");
    }


    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    auto frame_rgb = frame.cwise_transform(  //
        [](const sara::Bgra8& color) -> sara::Rgb8 {
          using namespace sara;
          return {color.channel<R>(), color.channel<G>(), color.channel<B>()};
        });
#else
    auto& frame_rgb = frame;
#endif

    if (!match_keypoints)
    {
      if (show_features)
      {
        const auto& fcurr = sara::features(keys_curr);
        const auto num_features = static_cast<int>(fcurr.size());
#pragma omp parallel for
        for (auto i = 0; i < num_features; ++i)
          sara::draw(frame_rgb, fcurr[i], sara::Cyan8);
      }
    }
    else
    {
      const auto num_matches = static_cast<int>(matches.size());
#pragma omp parallel for
      for (auto i = 0; i < num_matches; ++i)
      {
        if (!inliers(i))
          continue;
        if (show_features)
        {
          draw(frame_rgb, matches[i].x(), sara::Blue8);
          draw(frame_rgb, matches[i].y(), sara::Cyan8);
        }
        const Eigen::Vector2f a = matches[i].x_pos();
        const Eigen::Vector2f b = matches[i].y_pos();
        sara::draw_arrow(frame_rgb, a, b, sara::Yellow8, 4);
      }
    }

    draw_text(frame_rgb, 100, 50,                           //
              sara::format("SIFT: %0.f ms", feature_time),  //
              sara::White8, 40, 0, false, true, false);
    draw_text(frame_rgb, 100, 100,
              sara::format("Matching: %0.3f ms", matching_time),  //
              sara::White8, 40, 0, false, true, false);
    draw_text(frame_rgb, 100, 200,                                          //
              sara::format("F-Inliers: %d", inliers.flat_array().count()),  //
              sara::White8, 40, 0, false, true, false);

    sara::display(frame_rgb);

    sara::toc("Display");
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
