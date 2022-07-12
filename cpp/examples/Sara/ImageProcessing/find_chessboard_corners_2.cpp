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
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Watershed.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/Erode.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"


namespace sara = DO::Sara;


struct Corner
{
  Eigen::Vector2i coords;
  float score;
  auto position() const -> const Eigen::Vector2i&
  {
    return coords;
  }
  auto operator<(const Corner& other) const -> bool
  {
    return score < other.score;
  }
};

// Select the local maxima of the cornerness functions.
auto select(const sara::ImageView<float>& cornerness,
            const float cornerness_adaptive_thres) -> std::vector<Corner>
{
  const auto extrema = sara::local_maxima(cornerness);

  const auto cornerness_max = cornerness.flat_array().maxCoeff();
  const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

  auto extrema_filtered = std::vector<Corner>{};
  extrema_filtered.reserve(extrema.size());
  for (const auto& p : extrema)
    if (cornerness(p) > cornerness_thres)
      extrema_filtered.push_back({p, cornerness(p)});
  return extrema_filtered;
};


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

  // Harris cornerness parameters.
  //
  // Blur parameter before gradient calculation.
  const auto sigma_D =
      argc < 3 ? std::sqrt(std::pow(1.6f, 2.f) - 1) : std::stof(argv[2]);
  // Integration domain of the second moment.
  const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
  // Threshold parameter.
  const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
  const auto cornerness_adaptive_thres = argc < 6 ? 1e-5f : std::stof(argv[5]);
  // Corner filtering.
  const auto nms_radius = argc < 7 ? 10 : std::stoi(argv[6]);
  static constexpr auto grad_adaptive_thres = 2e-2f;
  static constexpr auto downscale_factor = 2;

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_conv = sara::Image<float>{video_frame.sizes()};
  auto f_ds = sara::Image<float>{video_frame.sizes() / downscale_factor};

#define ADAPTIVE_THRESHOLDING
#ifdef ADAPTIVE_THRESHOLDING
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};
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
    static constexpr auto tolerance_parameter = 0.f;
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
    static const auto color_threshold = std::sqrt(sara::square(2.f) * 3);
    const auto regions = sara::color_watershed(video_frame, color_threshold);
    sara::toc("Watershed");
#endif

    // Calculate Harris cornerness functions.
    sara::tic();
    sara::scale(f, f_ds);
    const auto cornerness = sara::scale_adapted_harris_cornerness(  //
        f_ds,                                                       //
        sigma_I, sigma_D,                                           //
        kappa                                                       //
    );
    auto corners = select(cornerness, cornerness_adaptive_thres);
    sara::nms(corners, cornerness.sizes(), nms_radius);
    sara::toc("Corner detection");

    // Display the good regions.
    const auto colors = mean_colors(regions, video_frame);
    auto partitioning = sara::Image<sara::Rgb8>{video_frame.sizes()};
    partitioning.flat_array().fill(sara::Red8);

    for (const auto& [label, points] : regions)
    {
      auto good = false;
#ifdef ADAPTIVE_THRESHOLDING
      if (segmentation_map(points.front()) != 0)
        continue;
#endif

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
        for (const auto& p : points)
          partitioning(p) = color;
      }
    }

    for (const auto& p : corners)
      sara::fill_circle(partitioning, downscale_factor * p.coords.x(),
                        downscale_factor * p.coords.y(), 4, sara::Magenta8);

    SARA_CHECK(corners.size());
    sara::display(partitioning);
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
