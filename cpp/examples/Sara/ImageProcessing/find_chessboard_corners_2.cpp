// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <omp.h>

#include <map>
#include <unordered_map>

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/DisjointSets/TwoPassConnectedComponents.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Watershed.hpp>
#include <DO/Sara/VideoIO.hpp>


#include "Chessboard/Erode.hpp"


namespace sara = DO::Sara;


auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                 const sara::ImageView<sara::Rgb8>& image)
{
  auto colors = std::unordered_map<int, sara::Rgb8>{};
  for (const auto& [label, points] : regions)
  {
    const auto num_points = static_cast<float>(points.size());
    Eigen::Vector3f color = Eigen::Vector3f::Zero();
    for (const auto& p : points)
      color += image(p).cast<float>();
    color /= num_points;

    colors[label] = color.cast<std::uint8_t>();
  }
  return colors;
}


auto __main(int argc, char** argv) -> int
{
  omp_set_num_threads(omp_get_max_threads());

#ifdef _WIN32
  const auto video_file = sara::select_video_file_from_dialog_box();
  if (video_file.empty())
    return 1;
#else
  if (argc < 2)
    return 1;
  const auto video_file = std::string{argv[1]};

#endif

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_conv = sara::Image<float>{video_frame.sizes()};

#define ADAPTIVE_THRESHOLDING
#ifdef ADAPTIVE_THRESHOLDING
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};
  static constexpr auto tolerance_parameter = 0.f;
#else
  static const auto color_threshold = std::sqrt(sara::square(2.f) * 3);
#endif


  while (video_stream.read())
  {
    ++frame_number;
    if (frame_number % 3 != 0)
      continue;

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }
    SARA_CHECK(frame_number);

#ifdef ADAPTIVE_THRESHOLDING
    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, f);
    sara::toc("Grayscale conversion");

    sara::tic();
    sara::gaussian_adaptive_threshold(f, 32.f, 3.f, tolerance_parameter,
                                      segmentation_map);
    sara::toc("Adaptive thresholding");

    sara::tic();
    auto segmentation_map_eroded = segmentation_map;
    for (auto i = 0; i < 1; ++i)
    {
      sara::binary_erode_3x3(segmentation_map, segmentation_map_eroded);
      segmentation_map.swap(segmentation_map_eroded);
    }
    sara::toc("Erosion 3x3");

    sara::tic();
    const auto labels = sara::two_pass_connected_components(  //
        segmentation_map.convert<int>());
    auto regions = std::map<int, std::vector<Eigen::Vector2i>>{};
    for (auto y = 0; y < labels.height(); ++y)
    {
      for (auto x = 0; x < labels.width(); ++x)
      {
        const auto label = labels(x, y);
        regions[label].emplace_back(x, y);
      }
    }
    sara::toc("Connected components");
#else
    // Watershed.
    sara::tic();
    const auto regions = sara::color_watershed(video_frame, color_threshold);
    sara::toc("Watershed");
#endif

    // Display the good regions.
    const auto colors = mean_colors(regions, video_frame);
    auto partitioning = sara::Image<sara::Rgb8>{video_frame.sizes()};
    partitioning.flat_array().fill(sara::Red8);
    sara::display(partitioning);
    for (const auto& [label, points] : regions)
    {
      auto good = false;
      auto ch = std::vector<Eigen::Vector2d>{};

      if (points.size() > 50)
      {
        auto points_2d = std::vector<Eigen::Vector2d>{};
        points_2d.resize(points.size());
        std::transform(points.begin(), points.end(), points_2d.begin(),
                       [](const auto& p) { return p.template cast<double>(); });
        ch = sara::graham_scan_convex_hull(points_2d);
        // ch = sara::ramer_douglas_peucker(ch, 1.);
        if (!ch.empty())
        {
          const auto area_1 = static_cast<double>(points.size());
          const auto area_2 = sara::area(ch);
          const auto diff = std::abs(area_1 - area_2) / area_1;
          good = diff < 0.2;
        }
      }

      // Show big segments only.
      if (good)
      {
        const auto color = colors.at(label);
        // for (const auto& p : points)
        //   partitioning(p) = color;

        // sara::display(partitioning);
        for (auto i = 0u; i < ch.size(); ++i)
        {
          const auto& a = ch[i];
          const auto& b = ch[(i + 1) % ch.size()];
          sara::draw_line(a, b, color, 3);
        }
        // sara::get_key();
      }
    }

    // sara::display(partitioning);
    sara::draw_text(80, 80, std::to_string(frame_number), sara::White8, 60, 0,
                    false, true);
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
