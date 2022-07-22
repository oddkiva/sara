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

#include <unordered_set>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/CartesianToPolarCoordinates.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/Corner.hpp"
#include "Chessboard/OrientationHistogram.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
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
  const auto sigma_D = argc < 3 ? 1.f : std::stof(argv[2]);
  // Integration domain of the second moment.
  const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
  // Threshold parameter.
  const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
  const auto cornerness_adaptive_thres = argc < 6 ? 1e-5f : std::stof(argv[5]);

  // Corner filtering.
  static constexpr auto downscale_factor = 2;

#define EDGE_DETECTION
#ifdef EDGE_DETECTION
  // Edge detection.
  static constexpr auto high_threshold_ratio = static_cast<float>(4._percent);
  static constexpr auto low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  using sara::operator""_deg;
  static constexpr auto angular_threshold = static_cast<float>((20._deg).value);

  auto ed = sara::EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};
#endif

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto frame_gray = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_blurred = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_ds =
      sara::Image<float>{video_frame.sizes() / downscale_factor};
#ifndef EDGE_DETECTION
  auto grad_norm = sara::Image<float>{video_frame.sizes() / downscale_factor};
  auto grad_ori = sara::Image<float>{video_frame.sizes() / downscale_factor};
#endif
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};
  auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};

  while (video_stream.read())
  {
    ++frame_number;
    if (frame_number % 3 != 0)
      continue;
    SARA_CHECK(frame_number);

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, frame_gray);
    sara::toc("Grayscale conversion");

    sara::tic();
    sara::apply_gaussian_filter(frame_gray, frame_gray_blurred, 1.f);
    sara::scale(frame_gray_blurred, frame_gray_ds);
    sara::toc("Downscale");

    sara::tic();
    const auto f_ds_blurred = frame_gray_ds.compute<sara::Gaussian>(sigma_D);
    sara::toc("Blur");

#ifdef EDGE_DETECTION
    sara::tic();
    ed(f_ds_blurred);
    sara::toc("Curve detection");
#endif

    sara::tic();
    auto grad_x = sara::Image<float>{f_ds_blurred.sizes()};
    auto grad_y = sara::Image<float>{f_ds_blurred.sizes()};
    sara::gradient(f_ds_blurred, grad_x, grad_y);
    const auto cornerness =
        sara::harris_cornerness(grad_x, grad_y, sigma_I, kappa);
    static const auto border = static_cast<int>(std::round(sigma_I));
    auto corners_int = select(cornerness, cornerness_adaptive_thres, border);
    sara::toc("Corner detection");

    sara::tic();
    auto corners = std::vector<Corner<float>>{};
    std::transform(
        corners_int.begin(), corners_int.end(), std::back_inserter(corners),
        [&grad_x, &grad_y, sigma_I](const Corner<int>& c) -> Corner<float> {
          static const auto radius = static_cast<int>(std::round(sigma_I));
          const auto p = sara::refine_junction_location_unsafe(
              grad_x, grad_y, c.coords, radius);
          return {p, c.score};
        });
    sara::toc("Corner refinement");

    sara::tic();
#ifdef EDGE_DETECTION
    const auto& grad_norm = ed.pipeline.gradient_magnitude;
    const auto& grad_ori = ed.pipeline.gradient_orientation;
#else
    sara::cartesian_to_polar_coordinates(grad_x, grad_y, grad_norm, grad_ori);
#endif

    static constexpr auto N = 72;
    auto hists = std::vector<Eigen::Array<float, N, 1>>{};
    hists.resize(corners.size());
    const auto num_corners = static_cast<int>(corners.size());
#pragma omp parallel for
    for (auto i = 0; i < num_corners; ++i)
    {
      compute_orientation_histogram<N>(hists[i], grad_norm, grad_ori,
                                       corners[i].coords.x(),
                                       corners[i].coords.y(), sigma_D, 4, 5.0f);
      sara::lowe_smooth_histogram(hists[i]);
      hists[i].matrix().normalize();
    };
    sara::toc("Gradient histograms");

    sara::tic();
    auto gradient_peaks = std::vector<std::vector<int>>{};
    gradient_peaks.resize(hists.size());
    std::transform(hists.begin(), hists.end(), gradient_peaks.begin(),
                   [](const auto& h) { return sara::find_peaks(h, 0.3f); });
    auto gradient_peaks_refined = std::vector<std::vector<float>>{};
    gradient_peaks_refined.resize(gradient_peaks.size());
    std::transform(gradient_peaks.begin(), gradient_peaks.end(), hists.begin(),
                   gradient_peaks_refined.begin(),
                   [](const auto& peaks, const auto& hist) {
                     auto peaks_ref = std::vector<float>{};
                     std::transform(peaks.begin(), peaks.end(),
                                    std::back_inserter(peaks_ref),
                                    [&hist](const auto& i) {
                                      return sara::refine_peak(hist, i);
                                    });
                     return peaks_ref;
                   });
    sara::toc("Gradient Dominant Orientations");


#ifdef EDGE_DETECTION
    sara::tic();
    auto edge_label_map = sara::Image<int>{ed.pipeline.edge_map.sizes()};
    edge_label_map.flat_array().fill(-1);
    const auto& curves = ed.pipeline.edges_simplified;
    for (auto label = 0u; label < curves.size(); ++label)
    {
      const auto& curve = curves[label];
      if (curve.size() < 2)
        continue;
      edge_label_map(curve.front().array().round().matrix().cast<int>()) =
          label;
      edge_label_map(curve.back().array().round().matrix().cast<int>()) = label;
    }
    auto adjacent_edges = std::vector<std::unordered_set<int>>{};
    adjacent_edges.resize(corners.size());
    std::transform(  //
        corners.begin(), corners.end(), adjacent_edges.begin(),
        [&edge_label_map](const Corner<float>& c) {
          auto edges = std::unordered_set<int>{};

          static constexpr auto r = 4;
          for (auto v = -r; v <= r; ++v)
          {
            for (auto u = -r; u <= r; ++u)
            {
              const Eigen::Vector2i p =
                  c.coords.cast<int>() + Eigen::Vector2i{u, v};

              const auto in_image_domain =
                  0 <= p.x() && p.x() < edge_label_map.width() &&  //
                  0 <= p.y() && p.y() < edge_label_map.height();
              if (!in_image_domain)
                continue;

              const auto label = edge_label_map(p);
              if (label != -1)
                edges.insert(label);
            }
          }
          return edges;
        });
    sara::toc("X-junction filter");
#endif

    sara::tic();
#ifdef EDGE_DETECTION
    const auto display_u8 = sara::upscale(ed.pipeline.edge_map, 2);
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    std::transform(display_u8.begin(), display_u8.end(), display.begin(),
                   [](const auto& v) {
                     return v != 0 ? sara::Rgb8(64, 64, 64) : sara::Black8;
                   });
#else
    auto display = frame_gray.convert<sara::Rgb8>();
#endif

    for (auto c = 0u; c < corners.size(); ++c)
    {
      const auto& p = corners[c];
      const auto& gradient_peaks = gradient_peaks_refined[c];

      // A chessboard corners should have 4 gradient orientation peaks.
      const auto four_contrast_changes = gradient_peaks_refined[c].size() == 4;

      // The 4 peaks are due to 2 lines crossing each other.
      static constexpr auto angle_degree_thres = 20.f;
      const auto two_crossing_lines =
          std::abs((gradient_peaks[2] - gradient_peaks[0]) * 360.f / N -
                   180.f) < angle_degree_thres &&
          std::abs((gradient_peaks[3] - gradient_peaks[1]) * 360.f / N -
                   180.f) < angle_degree_thres;

      const auto good = four_contrast_changes && two_crossing_lines;

      if (good)
      {
#ifdef EDGE_DETECTION
        const auto& edges = adjacent_edges[c];
        for (const auto& curve_index : edges)
        {
          const auto color =
              sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
          // const auto color = sara::Cyan8;
          const auto& curve = ed.pipeline.edges_as_list[curve_index];
          for (const auto& p : curve)
            display(p * downscale_factor) = color;
        }
#endif
      }

      sara::fill_circle(
          display,
          static_cast<int>(std::round(downscale_factor * p.coords.x())),
          static_cast<int>(std::round(downscale_factor * p.coords.y())), 1,
          sara::Yellow8);
      sara::draw_circle(
          display,
          static_cast<int>(std::round(downscale_factor * p.coords.x())),
          static_cast<int>(std::round(downscale_factor * p.coords.y())), 4,
          good ? sara::Red8 : sara::Blue8, 2);
    }
    sara::draw_text(display, 80, 80, std::to_string(frame_number), sara::White8,
                    60, 0, false, true);
    sara::display(display);
    sara::toc("Display");
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
