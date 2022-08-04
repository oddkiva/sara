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
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Geometry/Algorithms/BorderFollowing.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/Erode.hpp"


namespace sara = DO::Sara;


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

  const auto erosion_iterations = argc < 3 ? 0 : std::stoi(argv[2]);
  const auto simplification_eps =
      argc < 4 ? 0.5 : static_cast<double>(std::stof(argv[3]));


  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_conv = sara::Image<float>{video_frame.sizes()};

  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};


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

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, f);
    sara::toc("Grayscale conversion");

    sara::tic();
    static constexpr auto tolerance_parameter = 0.f;
    sara::gaussian_adaptive_threshold(f, 32.f, 2.f, tolerance_parameter,
                                      segmentation_map);
    sara::toc("Adaptive thresholding");

    if (erosion_iterations > 0)
    {
      sara::tic();
      for (auto i = 0; i < erosion_iterations; ++i)
      {
        auto segmentation_map_eroded = segmentation_map;
        sara::binary_erode_3x3(segmentation_map, segmentation_map_eroded);
        segmentation_map.swap(segmentation_map_eroded);
      }
      sara::toc("Erosion 3x3");
    }

    sara::tic();
    auto segmentation_map_inverted =
        sara::Image<std::uint8_t>{segmentation_map.sizes()};
    std::transform(segmentation_map.begin(), segmentation_map.end(),
                   segmentation_map_inverted.begin(),
                   [](const auto& v) { return v == 0 ? 255 : 0; });
    const auto border_curves =
        sara::suzuki_abe_follow_border(segmentation_map_inverted);
    sara::toc("Border Following");

    sara::tic();
    auto border_curves_d =
        std::unordered_map<int, std::vector<Eigen::Vector2d>>{};
    for (const auto& [border_id, border] : border_curves)
    {
      auto curve = std::vector<Eigen::Vector2d>{};
      std::transform(border.curve.begin(), border.curve.end(),
                     std::back_inserter(curve),
                     [](const auto& p) { return p.template cast<double>(); });
      if (curve.size() > 3)
      {
        curve = sara::graham_scan_convex_hull(curve);
        curve = sara::ramer_douglas_peucker(curve, simplification_eps);
        // Little hack.
        curve.push_back(curve.front());
      }
      border_curves_d[border_id] = curve;
    }
    sara::toc("Border Simplification");

    auto display = sara::Image<sara::Rgb8>{segmentation_map.sizes()};
    display.flat_array().fill(sara::Black8);
    for (const auto& b : border_curves)
    {
      // const auto& curve = b.second.curve;
      if (b.second.type == sara::Border::Type::HoleBorder)
        continue;
      // if (curve.size() < 5 * 4)
      //   continue;
      const auto& curve = border_curves_d.at(b.first);
      if (curve.size() < 2 || sara::length(curve) < 10)
        continue;
      const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
      sara::draw_polyline(display, curve, color);
    }
    sara::display(display);
    sara::draw_text(80, 80, std::to_string(frame_number), sara::White8, 60, 0,
                    false, true);
    sara::get_key();
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
