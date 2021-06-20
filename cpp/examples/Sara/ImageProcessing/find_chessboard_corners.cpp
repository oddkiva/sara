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
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/EdgeGrouping.hpp>


namespace sara = DO::Sara;
using sara::operator""_deg;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


auto edge_signature(const sara::Image<sara::Rgb8>& color,
                    const sara::Image<Eigen::Vector2f>& gradients,
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
  mean_gradient = std::accumulate(edge.begin(), edge.end(), mean_gradient,
                                  [&gradients](const auto& g, const auto& e) {
                                    return g + gradients(e).normalized();  //
                                  });
  mean_gradient /= edge.size();

  return std::make_tuple(darks, brights, mean_gradient);
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  const auto folder = std::string{argv[1]};

  constexpr auto sigma = 1.6f;
  constexpr auto nms_radius = 5;

  constexpr float high_threshold_ratio = static_cast<float>(10._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((10._deg).value);

  const auto color_threshold = std::sqrt(std::pow(2, 2) * 3);
  const auto segment_min_size = 50;

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
    auto endpoint_graph = sara::EndPointGraph{edge_attributes};
    endpoint_graph.mark_plausible_alignments();
    sara::toc("Alignment Computation");

    // Draw alignment-based connections.
    const auto& score = endpoint_graph.score;
    for (auto i = 0; i < score.rows(); ++i)
    {
      for (auto j = i + 1; j < score.cols(); ++j)
      {
        const auto& pi = endpoint_graph.endpoints[i];
        const auto& pj = endpoint_graph.endpoints[j];

        if (score(i, j) != std::numeric_limits<double>::infinity())
        {
          sara::draw_line(image, pi.x(), pi.y(), pj.x(), pj.y(), sara::Yellow8,
                          2);
          sara::draw_circle(image, pi.x(), pi.y(), 3, sara::Yellow8, 3);
          sara::draw_circle(image, pj.x(), pj.y(), 3, sara::Yellow8, 3);
        }
      }
    }


    sara::display(image);
    for (const auto& e : edges)
    {
      const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
      for (const auto& p : e)
        sara::fill_circle(p.x(), p.y(), 2, color);
    }

    for (auto i = 0u; i < edges.size(); ++i)
    {
      const auto& e = edges[i];

      // Discard small edges.
      if (e.size() < 2)
        continue;

      const auto& g = mean_gradients[i];
      const Eigen::Vector2f a = std::accumulate(            //
                                    e.begin(), e.end(),     //
                                    Eigen::Vector2f{0, 0},  //
                                    [](const auto& a, const auto& b) {
                                      return a + b.template cast<float>();
                                    }) /
                                e.size();
      const Eigen::Vector2f b = a + 20 * g;

      sara::draw_arrow(a, b, sara::Magenta8, 2);
    }

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
