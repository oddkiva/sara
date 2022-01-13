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

#include <omp.h>

#include <set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/WatershedV2.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


auto mean_colors(const std::vector<std::vector<Eigen::Vector2i>>& regions,
                 const Image<Rgb8>& image)
{
  auto colors = std::vector<Rgb8>(regions.size());
#pragma omp parallel for
  for (auto i = 0u; i < regions.size(); ++i)
  {
    const auto& region = regions[i];
    const auto num_points = static_cast<float>(region.size());
    Eigen::Vector3f color = Vector3f::Zero();
    for (const auto& p : region)
      color += image(p).cast<float>();
    if (num_points != 0)
      color /= num_points;

    colors[i] = color.cast<std::uint8_t>();
  }
  return colors;
}

GRAPHICS_MAIN()
{
  omp_set_num_threads(omp_get_max_threads());

  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      //     "/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
      //     //"/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
#define DOWNSAMPLE
#ifdef DOWNSAMPLE
  auto frame_downsampled = Image<Rgb8>{frame.sizes() / 2};
#else
  auto& frame_downsampled = frame;
#endif

  // Show the local extrema.
  create_window(frame_downsampled.sizes());
  set_antialiasing();

  const auto color_threshold = std::sqrt(square(4.f) * 3);

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

#ifdef DOWNSAMPLE
    reduce(frame, frame_downsampled);
#endif

    // Watershed.
    const auto regions =
        v2::color_watershed(frame_downsampled, color_threshold);

    // Display the good regions.
    tic();
    const auto colors = mean_colors(regions, frame_downsampled);
    toc("Mean Color");

    tic();
    auto partitioning = Image<Rgb8>{frame_downsampled.sizes()};
#pragma omp parallel for
    for (auto r = 0u; r < regions.size(); ++r)
    {
      const auto& region = regions[r];
      if (region.empty())
        continue;

      const auto& color = colors[r];
      for (auto p = 0u; p < region.size(); ++p)
        partitioning(region[p]) = region.size() < 100 ? Black8 : color;
    }
    toc("Filling image");

    display(partitioning);
  }

  return 0;
}
