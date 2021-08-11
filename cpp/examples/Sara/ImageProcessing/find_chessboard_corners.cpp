// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include "Chessboard/EdgeFusion.hpp"
#include "Chessboard/SaddleDetection.hpp"


namespace sara = DO::Sara;
using sara::operator""_deg;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


auto edge_signature(const sara::ImageView<sara::Rgb8>& color,
                    const sara::ImageView<Eigen::Vector2f>& gradients,
                    const std::vector<Eigen::Vector2i>& edge,  //
                    float delta = 1, int width = 3)
{
  auto darks = std::vector<sara::Rgb64f>{};
  auto brights = std::vector<sara::Rgb64f>{};
  for (auto s = 1; s <= width; ++s)
  {
    for (const auto& e : edge)
    {
      const Eigen::Vector2d n = gradients(e).cast<double>().normalized();

      const Eigen::Vector2d b = e.cast<double>() + s * delta * n;
      const Eigen::Vector2d d = e.cast<double>() - s * delta * n;

      if (0 <= d.x() && d.x() < color.width() &&  //
          0 <= d.y() && d.y() < color.height())
        darks.push_back(sara::interpolate(color, d));

      if (0 <= b.x() && b.x() < color.width() &&  //
          0 <= b.y() && b.y() < color.height())
        brights.push_back(sara::interpolate(color, b));
    }
  }

  Eigen::Vector2f mean_gradient = Eigen::Vector2f::Zero();
  mean_gradient = std::accumulate(
      edge.begin(), edge.end(), mean_gradient,
      [&gradients](const auto& g, const auto& e) -> Eigen::Vector2f {
        if (gradients(e).squaredNorm() < 1e-6f)
          return g;
        return g + gradients(e).normalized();
      });
  mean_gradient /= static_cast<float>(edge.size());

  return std::make_tuple(darks, brights, mean_gradient);
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  if (argc < 2)
    return 1;
  const auto folder = std::string{argv[1]};

  constexpr auto sigma = 1.6f;

  constexpr auto nms_radius = 10;
  constexpr float high_threshold_ratio = static_cast<float>(10._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((5._deg).value);

  auto ed = sara::EdgeDetector{{high_threshold_ratio,  //
                                low_threshold_ratio,   //
                                angular_threshold,     //
                                true}};

  for (auto i = 0; i <= 1790; i += 10)
  {
    const auto image_filepath = folder + "/" + std::to_string(i) + ".png";

    auto image = sara::imread<sara::Rgb8>(image_filepath);
    const auto image_gray = image.convert<float>();
    const auto image_blurred = image_gray.compute<sara::Gaussian>(sigma);

    if (sara::active_window() == nullptr)
    {
      sara::create_window(image.sizes());
      sara::set_antialiasing();
    }

    const auto saddle_points = sara::detect_saddle_points(image_blurred,
                                                          nms_radius);

    // Detect edges.
    ed(image_blurred);
    const auto& edges = ed.pipeline.edges_as_list;
    const auto& edges_simplified = ed.pipeline.edges_simplified;

    auto darks = std::vector<std::vector<sara::Rgb64f>>{};
    auto brights = std::vector<std::vector<sara::Rgb64f>>{};
    auto mean_gradients = std::vector<Eigen::Vector2f>{};
    for (const auto& edge : edges)
    {
      auto [dark, bright, g] = edge_signature(  //
          image,                                //
          ed.pipeline.gradient_cartesian,       //
          edge);
      darks.push_back(std::move(dark));
      brights.push_back(std::move(bright));
      mean_gradients.push_back(g);
    }

    // Calculate edge statistics.
    sara::tic();
    const auto edge_stats = sara::CurveStatistics{edges_simplified};
    sara::toc("Edge Shape Statistics");

    sara::tic();
    const auto edge_attributes = sara::EdgeAttributes{
        edges_simplified,    //
        edge_stats.centers,  //
        edge_stats.axes,     //
        edge_stats.lengths   //
    };

    sara::check_edge_grouping(image,                    //
                              edges_simplified,         //
                              edges,                    //
                              mean_gradients,           //
                              edge_attributes.centers,  //
                              edge_attributes.axes,     //
                              edge_attributes.lengths);
    sara::toc("Edge Grouping");

    for (const auto& s: saddle_points)
      sara::fill_circle(s.p.x(), s.p.y(), 10, sara::Red8);

    sara::get_key();
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
