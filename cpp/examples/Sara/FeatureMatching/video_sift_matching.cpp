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
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <omp.h>


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

auto imshow(const std::string& window_name, const ImageView<Rgb8>& image)
{
  static auto window_names = std::map<std::string, Window>{};
  auto w_it = window_names.find(window_name);

  auto w = Window{};
  if (w_it == window_names.end())
  {
    w = create_window(image.sizes(), window_name);
    window_names[window_name] = w;
  }
  else
    w = w_it->second;

  set_active_window(w);
  display(image);
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Parameter parsing.
  auto video_filepath = std::string{};
  auto downscale_factor = int{};
  auto skip = int{};
  auto show_tracks = bool{};

  po::options_description desc("video_sift_matching");
  desc.add_options()     //
      ("help", "Usage")  //
      ("video,v", po::value<std::string>(&video_filepath),
       "input video file")  //
      ("downscale-factor,d",
       po::value<int>(&downscale_factor)->default_value(2),
       "downscale factor")  //
      ("skip,s", po::value<int>(&skip)->default_value(0),
       "number of frames to skip")  //
      ("show_tracks,t", po::value<bool>(&show_tracks)->default_value(true),
       "show feature tracking")  //
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

  // OpenMP.
  omp_set_num_threads(omp_get_max_threads());

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{};
  auto screen_contents = Image<Rgb8>{frame.sizes()};

  // Output save.
  const auto basename = fs::basename(video_filepath);
  VideoWriter video_writer{
#ifdef __APPLE__
      "/Users/david/Desktop/" + basename + ".sift-matching.mp4",
#else
      "/home/david/Desktop/" + basename + ".sift-matching.mp4",
#endif
      frame.sizes()  //
  };


  // Show the local extrema.
  auto w = create_window(frame.sizes(), "SIFT matching " + basename);
  set_antialiasing();

// #define CROP
#ifdef CROP
  const auto [p1, p2] = initialize_crop_region(frame.sizes());
#else
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();
#endif

  auto image_prev = Image<float>{};
  auto image_curr = Image<float>{};

  auto keys_prev = KeypointList<OERegion, float>{};
  auto keys_curr = KeypointList<OERegion, float>{};

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

    // Reduce our attention to the central part of the image.
    tic();
    const auto frame_cropped = crop(frame, p1, p2);
    toc("Crop");

    tic();
    frame_gray32f = frame_cropped.convert<float>();
    toc("Grayscale");

    if (downscale_factor > 1)
    {
      tic();
      frame_gray32f = downscale(frame_gray32f, downscale_factor);
      toc("Downscale");
    }

    image_prev.swap(image_curr);
    keys_prev.swap(keys_curr);

    image_curr = frame_gray32f;
    const auto image_pyr_params = ImagePyramidParams(0);
    keys_curr = compute_sift_keypoints(frame_gray32f, image_pyr_params);

    // Compute/read matches
    auto matches = std::vector<Match>{};

    const auto& fprev = std::get<0>(keys_prev);
    if (!fprev.empty())
    {
      SARA_DEBUG << "Computing Matches" << endl;
      AnnMatcher matcher{keys_prev, keys_curr, 0.6f};
      matches = matcher.compute_matches();
      SARA_CHECK(matches.size());
    }

    set_active_window(w);
    display(frame);
    if (show_tracks)
    {
      const auto s = 1 / float(downscale_factor);
      for (size_t i = 0; i < matches.size(); ++i)
      {
        draw(matches[i].x(), Blue8, downscale_factor, s * p1.cast<float>());
        draw(matches[i].y(), Cyan8, downscale_factor, s * p1.cast<float>());
        const Eigen::Vector2f a =
            p1.cast<float>() + downscale_factor * matches[i].x_pos();
        const Eigen::Vector2f b =
            p1.cast<float>() + downscale_factor * matches[i].y_pos();
        draw_arrow(a, b, Yellow8, 2);
      }
    }
    draw_text(100, 100, "SIFT matches = " + std::to_string(matches.size()),
              White8, 20, 0, false, true, false);

    grab_screen_contents(screen_contents, w);
    video_writer.write(screen_contents);
  }

  return 0;
}
