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
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/NonMaximumSuppression.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


template <typename T>
struct Corner
{
  Eigen::Vector2<T> coords;
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
            const float cornerness_adaptive_thres) -> std::vector<Corner<int>>
{
  const auto extrema = sara::local_maxima(cornerness);

  const auto cornerness_max = cornerness.flat_array().maxCoeff();
  const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

  auto extrema_filtered = std::vector<Corner<int>>{};
  extrema_filtered.reserve(extrema.size());
  for (const auto& p : extrema)
    if (cornerness(p) > cornerness_thres)
      extrema_filtered.push_back({p, cornerness(p)});
  return extrema_filtered;
};


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
  const auto sigma_D = argc < 3 ? 0.8f : std::stof(argv[2]);
  // Integration domain of the second moment.
  const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
  // Threshold parameter.
  const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
  const auto cornerness_adaptive_thres = argc < 6 ? 1e-5f : std::stof(argv[5]);

  // Corner filtering.
  static constexpr auto downscale_factor = 2;

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


  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto frame_gray = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_blurred = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_ds =
      sara::Image<float>{video_frame.sizes() / downscale_factor};
  auto grad_f_norm = sara::Image<float>{video_frame.sizes()};
  auto grad_f_ori = sara::Image<float>{video_frame.sizes()};
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
    ed(frame_gray_ds);
    sara::toc("Curve detection");

    sara::tic();
    const auto cornerness = sara::scale_adapted_harris_cornerness(  //
        frame_gray_ds,                                              //
        sigma_I, sigma_D,                                           //
        kappa                                                       //
    );
    const auto grad_f =
        frame_gray_ds.compute<sara::Gaussian>(0.5f).compute<sara::Gradient>();
    auto corners_int = select(cornerness, cornerness_adaptive_thres);
    sara::toc("Corner detection");

    sara::tic();
    auto corners = std::vector<Corner<float>>{};
    std::transform(
        corners_int.begin(), corners_int.end(), std::back_inserter(corners),
        [&grad_f, sigma_I](const Corner<int>& c) -> Corner<float> {
          const auto p =
              sara::refine_junction_location_unsafe(grad_f, c.coords, sigma_I);
          return {p, c.score};
        });
    sara::toc("Corner refinement");

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

    sara::tic();
    auto display = frame_gray.convert<sara::Rgb8>();
    for (auto c = 0u; c < corners.size(); ++c)
    {
      const auto& p = corners[c];
      const auto& edges = adjacent_edges[c];
      if (edges.size() != 4)
        continue;

      for (const auto& curve_index : edges)
      {
        const auto& curve_simplified =
            ed.pipeline.edges_simplified[curve_index];

        // const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() %
        // 255);
        const auto color = sara::Cyan8;
        for (auto i = 0u; i < curve_simplified.size() - 1; ++i)
        {
          const auto a = curve_simplified[i].cast<float>() * downscale_factor;
          const auto b =
              curve_simplified[i + 1].cast<float>() * downscale_factor;
          sara::draw_line(display, a, b, color, 2);
        }
      }

      sara::fill_circle(display, std::round(downscale_factor * p.coords.x()),
                        std::round(downscale_factor * p.coords.y()), 1,
                        sara::Yellow8);
      sara::draw_circle(display, std::round(downscale_factor * p.coords.x()),
                        std::round(downscale_factor * p.coords.y()), 4,
                        sara::Red8, 2);
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
