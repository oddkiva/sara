// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                 const Image<Rgb8>& image)
{
  auto colors = std::map<int, Rgb8>{};
  for (const auto& [label, points] : regions)
  {
    const auto num_points = points.size();
    Eigen::Vector3f color = Vector3f::Zero();
    for (const auto& p: points)
      color += image(p).cast<float>();
    color /= num_points;

    colors[label] = color.cast<std::uint8_t>();
  }
  return colors;
}

GRAPHICS_MAIN()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  // const auto video_filepath =
  // "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath =
      //"/Users/david/Desktop/Datasets/humanising-autonomy/field.mp4"s;
      //"/Users/david/Desktop/Datasets/humanising-autonomy/turn_bikes.mp4"s;
      //"/Users/david/Desktop/Datasets/humanising-autonomy/toyota.mp4"s;
      //"/Users/david/Desktop/Datasets/humanising-autonomy/barberX.mp4"s;
      "/Users/david/Desktop/Datasets/humanising-autonomy/pass_family.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_downsampled = Image<Rgb8>{frame.sizes() / 4};

  // Show the local extrema.
  create_window(frame_downsampled.sizes());
  set_antialiasing();

  const auto color_threshold = std::sqrt(std::pow(4, 2) * 3);

  auto frames_read = 0;
  constexpr auto skip = 2;

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

    DO::Sara::reduce(frame, frame_downsampled);

    // Watershed.
    const auto regions = color_watershed(frame_downsampled, color_threshold);
    SARA_CHECK(regions.size());

    // Display the good regions.
    const auto colors = mean_colors(regions, frame_downsampled);
    auto partitioning = Image<Rgb8>{frame_downsampled.sizes()};
    for (const auto& [label, points] : regions)
    {
      // Show big segments only.
      for (const auto& p : points)
        partitioning(p) = points.size() < 100 ? Black8 : colors.at(label);
    }
    display(partitioning);
  }

  return 0;
}
