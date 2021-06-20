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


namespace DO::Sara {

  struct EdgeGraph
  {
    const EdgeAttributes& attributes;
    std::vector<std::size_t> edge_ids;

    sara::Tensor_<std::uint8_t, 2> A;

    EdgeGraph(const EdgeAttributes& attrs)
      : attributes{attrs}
    {
      initialize();
    }

    auto initialize() -> void
    {
      const auto n = static_cast<std::int32_t>(attributes.edges.size());
      std::iota(edge_ids.begin(), edge_ids.end(), 0u);

      A.resize(n, n);
      A.flat_array().fill(0);
      for (auto i = 0; i < n; ++i)
      {
        for (auto j = i; j < n; ++j)
        {
          if (i == j)
            A(i, j) = 1;
          else if (edge(i).size() >= 2 && edge(j).size() >= 2)
            A(i, j) = is_aligned(i, j);
          A(j, i) = A(i, j);
        }
      }
    }

    auto edge(std::size_t i) const -> const Edge&
    {
      return attributes.edges[i];
    }

    auto center(std::size_t i) const -> const Eigen::Vector2d&
    {
      return attributes.centers[i];
    }

    auto axes(std::size_t i) const -> const Eigen::Matrix2d&
    {
      return attributes.axes[i];
    }

    auto lengths(std::size_t i) const -> const Eigen::Vector2d&
    {
      return attributes.lengths[i];
    }

    auto rect(std::size_t i) const -> OrientedBox
    {
      return {center(i), axes(i), lengths(i)};
    }

    auto is_aligned(std::size_t i, std::size_t j) const -> bool
    {
      const auto& ri = rect(i);
      const auto& rj = rect(j);
      const auto& di = ri.axes.col(0);
      const auto& dj = rj.axes.col(0);

      const auto& ai = attributes.edges[i].front();
      const auto& bi = attributes.edges[i].back();

      const auto& aj = attributes.edges[j].front();
      const auto& bj = attributes.edges[j].back();

      constexpr auto dist_thres = 10.f;
      constexpr auto dist_thres_2 = dist_thres * dist_thres;
      const auto is_adjacent = (ai - aj).squaredNorm() < dist_thres_2 ||
                               (ai - bj).squaredNorm() < dist_thres_2 ||
                               (bi - aj).squaredNorm() < dist_thres_2 ||
                               (bi - bj).squaredNorm() < dist_thres_2;

      return is_adjacent && std::abs(di.dot(dj)) > std::cos(30._deg);
    }

    auto group_by_alignment() const
    {
      const auto n = A.rows();

      auto ds = sara::DisjointSets(n);

      for (auto i = 0; i < n; ++i)
        ds.make_set(i);

      for (auto i = 0; i < n; ++i)
        for (auto j = i; j < n; ++j)
          if (A(i, j) == 1)
            ds.join(ds.node(i), ds.node(j));

      auto groups = std::map<std::size_t, std::vector<std::size_t>>{};
      for (auto i = 0; i < n; ++i)
      {
        const auto c = ds.component(i);
        groups[c].push_back(i);
      }

      return groups;
    }
  };

  auto check_edge_grouping(const sara::ImageView<sara::Rgb8>& frame,     //
                           const std::vector<Edge>& edges_refined,       //
                           const std::vector<Eigen::Vector2d>& centers,  //
                           const std::vector<Eigen::Matrix2d>& axes,     //
                           const std::vector<Eigen::Vector2d>& lengths,  //
                           const sara::Point2i& p1,                      //
                           double downscale_factor)                      //
      -> void
  {
    sara::tic();
    const auto edge_attributes = EdgeAttributes{.edges = edges_refined,
                                                .centers = centers,
                                                .axes = axes,
                                                .lengths = lengths};
    const auto edge_graph = EdgeGraph{edge_attributes};
    const auto edge_groups = edge_graph.group_by_alignment();
    SARA_CHECK(edges_refined.size());
    SARA_CHECK(edge_groups.size());
    sara::toc("Edge Grouping By Alignment");


    // Display the quasi-straight edges.
    auto draw_task = [=]() {
      auto edge_group_colors = std::map<std::size_t, Rgb8>{};
      for (const auto& g : edge_groups)
        edge_group_colors[g.first] << rand() % 255, rand() % 255, rand() % 255;

      auto edge_colors = std::vector<Rgb8>(edges_refined.size(), Red8);
      // for (auto& c : edge_colors)
      //   c << rand() % 255, rand() % 255, rand() % 255;
      for (const auto& g : edge_groups)
        for (const auto& e : g.second)
          edge_colors[e] = edge_group_colors[g.first];

      const Eigen::Vector2d p1d = p1.cast<double>();
      const auto& s = downscale_factor;

      auto detection = Image<Rgb8>{frame};
      detection.flat_array().fill(Black8);
      for (const auto& g : edge_groups)
      {
        for (const auto& e : g.second)
        {
          const auto& edge_refined = edges_refined[e];
          if (edge_refined.size() < 2)
            continue;

          const auto& color = edge_colors[e];
          draw_polyline(detection, edge_refined, color, p1d, s);

// #define DEBUG_SHAPE_STATISTICS
#ifdef DEBUG_SHAPE_STATISTICS
          const auto& rect = OrientedBox{.center = c,      //
                                         .axes = axes[e],  //
                                         .lengths = lengths[e]};
          rect.draw(detection, White8, p1d, s);
#endif
        }

      }

      display(detection);
      // get_key();
    };

    tic();
    draw_task();
    toc("Draw");
  }


}  // namespace DO::Sara


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
  mean_gradient /= edge.size();

  return std::make_tuple(darks, brights, mean_gradient);
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  if (argc < 2)
    return 1;
  const auto folder = std::string{argv[1]};

  constexpr auto sigma = 1.6f;

  constexpr float high_threshold_ratio = static_cast<float>(10._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((10._deg).value);

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

    sara::check_edge_grouping(image, edges_simplified, edge_attributes.centers,
                              edge_attributes.axes, edge_attributes.lengths,
                              Eigen::Vector2i::Zero(), 1);

    // sara::display(image);

    for (auto i = 0u; i < edges.size(); ++i)
    {
      const auto& e = edges[i];

      // Discard small edges.
      if (e.size() < 2)
        continue;

      const auto& g = mean_gradients[i];
      const Eigen::Vector2f a = std::accumulate(            //
                                    e.begin(), e.end(),     //
                                    Eigen::Vector2f(0, 0),  //
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
