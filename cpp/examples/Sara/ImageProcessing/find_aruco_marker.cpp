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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/VideoIO.hpp>

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

  const auto grad_adaptive_thres = argc < 3 ? 2e-2f : std::stof(argv[2]);

  // Corner filtering.
  const auto kappa = argc < 4 ? 0.04f : std::stof(argv[3]);
  const auto cornerness_adaptive_thres = argc < 5 ? 1e-4f : std::stof(argv[4]);
  const auto nms_radius = argc < 6 ? 2 : std::stoi(argv[5]);
  static constexpr auto sigma_D = 0.8f;
  static constexpr auto sigma_I = 2.0f;

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto video_frame_copy = sara::Image<sara::Rgb8>{};
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_blurred = sara::Image<float>{video_frame.sizes()};
  auto grad_f_norm = sara::Image<float>{video_frame.sizes()};
  auto grad_f_ori = sara::Image<float>{video_frame.sizes()};
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
    sara::apply_gaussian_filter(f, f_blurred, sigma_D);
    sara::gradient_in_polar_coordinates(f_blurred, grad_f_norm, grad_f_ori);
    const auto grad_max = grad_f_norm.flat_array().maxCoeff();
    const auto grad_thres = grad_adaptive_thres * grad_max;
    auto edge_map = sara::suppress_non_maximum_edgels(
        grad_f_norm, grad_f_ori, 2 * grad_thres, grad_thres);
    for (auto e = edge_map.begin(); e != edge_map.end(); ++e)
      if (*e == 127)
        *e = 0;
    sara::toc("Feature maps");

    sara::tic();
    const auto edges = sara::connected_components(edge_map);
    auto edge_label = sara::Image<std::int32_t>{edge_map.sizes()};
    edge_label.flat_array().fill(-1);
    for (const auto& [label, e] : edges)
      for (const auto& p : e)
        edge_label(p) = label;
    sara::toc("Edge grouping");

    sara::tic();
    const auto M = f_blurred.compute<sara::Gradient>()
                       .compute<sara::SecondMomentMatrix>()
                       .compute<sara::Gaussian>(sigma_I);
    auto cornerness = sara::Image<float>{f_blurred.sizes()};
    std::transform(M.begin(), M.end(), cornerness.begin(),
                   [kappa](const auto& m) {
                     return m.determinant() - kappa * pow(m.trace(), 2);
                   });
    auto corners = select(cornerness, cornerness_adaptive_thres);
    sara::nms(corners, cornerness.sizes(), nms_radius);
    sara::toc("Corner detection");

    auto corners_per_curve = std::map<std::int32_t, std::vector<Corner>>{};
    auto visited = sara::Image<std::uint8_t>{f.sizes()};
    visited.flat_array().fill(0);
    for (const auto& corner : corners)
    {
      const auto x = corner.coords.x();
      const auto y = corner.coords.y();
      const auto r = nms_radius;

      for (auto v = y - r; v <= y + r; ++v)
      {
        for (auto u = x - r; u <= x + r; ++u)
        {
          const auto in_image_domain = 0 <= u && u < f.width() &&  //
                                       0 <= v && v < f.height();
          if (!in_image_domain)
            continue;

          const auto& label = edge_label(u, v);
          if (label != -1 && visited(corner.coords) == 0)
          {
            corners_per_curve[label].push_back(corner);
            visited(corner.coords) = 1;
          }
        }
      }
    }

    auto disp = video_frame;// edge_map.convert<sara::Rgb8>();
#ifdef DEBUG
    for (const auto& p : corners)
      sara::fill_circle(disp, p.coords.x(), p.coords.y(), nms_radius,
                        sara::Red8);
#endif
    for (const auto& [label, edge] : edges)
    {
      if (edge.size() < 10)
        continue;

      // Quadrangle filtering.
      // 4 corners.
      auto cs = corners_per_curve.find(label);
      if (cs == corners_per_curve.end())
        continue;
      if (cs->second.size() != 3 && cs->second.size() != 4)
        continue;

      // The convex hull of the point set.
      auto point_set = std::vector<Eigen::Vector2d>{};
      std::transform(edge.begin(), edge.end(), std::back_inserter(point_set),
                     [](const auto& p) { return p.template cast<double>(); });
      const auto ch = sara::graham_scan_convex_hull(point_set);
      const auto area_ch = sara::area(ch);

      // The convex hull from the possibly imcomplete quadrangle.
      auto q = std::vector<Eigen::Vector2d>{};
      std::transform(
          cs->second.begin(), cs->second.end(), std::back_inserter(q),
          [](const auto& c) { return c.coords.template cast<double>(); });
      auto quad = sara::graham_scan_convex_hull(q);

#define METHOD1
#ifdef METHOD1
      auto good = false;

      // If the quadrangle is complete.
      if (quad.size() == 4)
      {
        // Convex hull based filtering
        const auto inter = sara::sutherland_hodgman(ch, quad);
        const auto area_inter = sara::area(inter);
        const auto area_q = sara::area(quad);
        const auto iou = area_inter / (area_ch + area_q - area_inter);
        good = iou > 0.5;
      }

      // If the quadrangle is incomplete.
      else if (quad.size() == 3)  // CAVEAT: The convex hull algorithm can
                                  // collapse collinear points.
      {
        // Calculate the parallelogram area.
        const auto& a = quad[0];
        const auto& b = quad[1];
        const auto& c = quad[2];
        const Eigen::Vector2d ba = a - b;
        const Eigen::Vector2d bc = c - b;
        const auto area_parallelogram = std::sqrt(
            ba.squaredNorm() * bc.squaredNorm() - sara::square(ba.dot(bc)));
        const auto error = std::abs(area_parallelogram - area_ch) / std::max(area_ch, area_parallelogram);
        good = error < 0.3;
      }
#else
      if (quad.size() == 3)
      {
        const auto& a = quad[0];
        const auto& b = quad[1];
        const auto& c = quad[2];
        const auto d = a + c - b;
        quad.push_back(d);
      }

      // Convex hull based filtering
      const auto inter = sara::sutherland_hodgman(ch, quad);
      const auto area_inter = sara::area(inter);
      const auto area_q = sara::area(quad);
      const auto iou = area_inter / (area_ch + area_q - area_inter);
      const auto good = iou > 0.4;
#endif
      if (!good)
        continue;

      std::for_each(edge.begin(), edge.end(),
                    [&disp](const auto& p) { disp(p) = sara::Cyan8; });

      for (const auto& q : quad)
        sara::fill_circle(disp, q.x(), q.y(), 2, sara::Magenta8);
      if (quad.size() == 3)
      {
        const auto& a = quad[0];
        const auto& b = quad[1];
        const auto& c = quad[2];
        const auto d = a + c - b;
        sara::fill_circle(disp, d.x(), d.y(), 2, sara::Red8);
      }
    }
    sara::display(disp);
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
