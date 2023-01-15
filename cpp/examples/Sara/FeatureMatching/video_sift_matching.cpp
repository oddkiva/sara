// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace std;
using namespace DO::Sara;


auto initialize_crop_region(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = {0.2 * sizes.x(), 0.2 * sizes.y()};
  const Eigen::Vector2i& p2 = {0.8 * sizes.x(), 0.75 * sizes.y()};
  return std::make_pair(p1, p2);
}

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Parameter parsing.
  auto video_filepath = std::string{};
  auto downscale_factor = int{};
  auto skip = int{};
  auto hide_tracks = false;
  auto show_features = false;
  auto save_video = false;
  auto num_scales_per_octave = int{};

  po::options_description desc("video_sift_matching");
  desc.add_options()     //
      ("help", "Usage")  //
      ("video,v", po::value<std::string>(&video_filepath),
       "input video file")  //
      ("downscale-factor,d",
       po::value<int>(&downscale_factor)->default_value(2),
       "downscale factor")  //
      ("num_scales_per_octave,s",
       po::value<int>(&num_scales_per_octave)->default_value(1),
       "number of scales per octave")  //
      ("skip", po::value<int>(&skip)->default_value(0),
       "number of frames to skip")  //
      ("hide_tracks,h", po::bool_switch(&hide_tracks),
       "hide feature tracking")  //
      ("show_features,f", po::bool_switch(&show_features),
       "show features")  //
      ("save_video", po::bool_switch(&save_video),
       "save video")  //
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

  // SIFT extraction parameters.
  // The following to apply correctly for SIFT.
  static constexpr auto scale_camera = 1.f;
  const auto first_octave =
      static_cast<int>(std::round(std::log(downscale_factor) / std::log(2)));
  const auto scale_geometric_factor =
      std::pow(2.f, 1.f / num_scales_per_octave);
  const auto image_pyr_params = ImagePyramidParams(
      first_octave, num_scales_per_octave + 3, scale_geometric_factor,
      /* image_padding_size */ 8, scale_camera,
      /* scale_initial */ 1.2f);

  // OpenMP.
#ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{};
  const Eigen::Vector2i downscaled_sizes = frame.sizes() / downscale_factor;
  auto frame_gray32f_downscaled = Image<float>{downscaled_sizes};

  // Output save.
  const auto basename = fs::basename(video_filepath);
  auto video_writer = std::unique_ptr<VideoWriter>{};

  if (save_video)
    video_writer = std::make_unique<VideoWriter>(
#ifdef __APPLE__
        "/Users/david/Desktop/" + basename + ".sift-matching.mp4",
#else
        "/home/david/Desktop/" + basename + ".sift-matching.mp4",
#endif
        frame.sizes()  //
    );

  // Show the local extrema.
  auto w = create_window(frame.sizes(), "SIFT matching " + basename);
  set_antialiasing();

// #define CROP
#ifdef CROP
  const auto [p1, p2] = initialize_crop_region(frame.sizes());
#else
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
#endif

  auto image_prev = Image<float>{};
  auto image_curr = Image<float>{};

  auto keys_prev = KeypointList<OERegion, float>{};
  auto keys_curr = KeypointList<OERegion, float>{};

  auto feature_timer = Timer{};
  auto matching_timer = Timer{};

  auto frames_read = 0;
  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;
    SARA_DEBUG << "Processing frame " << frames_read << std::endl;

#ifdef CROP
    // Reduce our attention to the central part of the image.
    tic();
    const auto frame_cropped = crop(frame, p1, p2);
    toc("Crop");

    tic();
    frame_gray32f = DO::Sara::from_rgb8_to_gray32f(frame_cropped);
    toc("Grayscale");
#else
    tic();
    frame_gray32f = DO::Sara::from_rgb8_to_gray32f(frame);
    toc("Grayscale");
#endif

    tic();
    feature_timer.restart();
    {
      image_prev.swap(image_curr);
      keys_prev.swap(keys_curr);

      image_curr = frame_gray32f;

      keys_curr = compute_sift_keypoints(frame_gray32f, image_pyr_params);
    }
    const auto feature_time = feature_timer.elapsed_ms();

    matching_timer.restart();
    auto matches = std::vector<Match>{};
    const auto& fprev = std::get<0>(keys_prev);
    if (!fprev.empty())
    {
      AnnMatcher matcher{keys_prev, keys_curr, 0.6f};
      matches = matcher.compute_matches();
    }
    const auto matching_time = matching_timer.elapsed_ms();
    toc("Matching");

    tic();
    auto frame_annotated = Image<Rgb8>{frame};
    if (!hide_tracks)
    {
      for (size_t i = 0; i < matches.size(); ++i)
      {
        if (show_features)
        {
          draw(frame_annotated, matches[i].x(), Blue8, 1, p1.cast<float>());
          draw(frame_annotated, matches[i].y(), Cyan8, 1, p1.cast<float>());
        }
        const Eigen::Vector2f a = p1.cast<float>() + matches[i].x_pos();
        const Eigen::Vector2f b = p1.cast<float>() + matches[i].y_pos();
        draw_arrow(frame_annotated, a, b, Yellow8, 4);
      }
    }
    draw_text(frame_annotated, 100, 50,               //
              format("SIFT: %0.f ms", feature_time),  //
              White8, 40, 0, false, true, false);
    draw_text(frame_annotated, 100, 100,
              format("Matching: %0.3f ms", matching_time),  //
              White8, 40, 0, false, true, false);
    draw_text(frame_annotated, 100, 150,             //
              format("Tracks: %u", matches.size()),  //
              White8, 40, 0, false, true, false);
    set_active_window(w);
    display(frame_annotated);
    toc("Display");

    if (save_video)
    {
      tic();
      video_writer->write(frame_annotated);
      toc("Video-Write");
    }
  }

  return 0;
}
