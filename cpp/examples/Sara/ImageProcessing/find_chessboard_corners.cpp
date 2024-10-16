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

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include <DO/Sara/Chessboard/CircularProfileExtractor.hpp>
#include <DO/Sara/Chessboard/JunctionDetection.hpp>
#include <DO/Sara/Chessboard/NonMaximumSuppression.hpp>


namespace sara = DO::Sara;


inline auto dir(const float angle) -> Eigen::Vector2f
{
  return Eigen::Vector2f{std::cos(angle), std::sin(angle)};
};

auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
    -> std::vector<float>
{
  auto zero_crossings = std::vector<float>{};
  for (auto n = Eigen::Index{}; n < profile.size(); ++n)
  {
    const auto ia = n;
    const auto ib = (n + Eigen::Index{1}) % profile.size();

    const auto& a = profile[ia];
    const auto& b = profile[ib];

    static constexpr auto pi = static_cast<float>(M_PI);
    const auto angle_a = ia * 2.f * M_PI / num_bins;
    const auto angle_b = ib * 2.f * M_PI / num_bins;

    const auto ea = Eigen::Vector2d{std::cos(angle_a),  //
                                    std::sin(angle_a)};
    const auto eb = Eigen::Vector2d{std::cos(angle_b),  //
                                    std::sin(angle_b)};

    // TODO: this all could have been simplified.
    const Eigen::Vector2d dir = (ea + eb) * 0.5;
    auto angle = std::atan2(dir.y(), dir.x());
    if (angle < 0)
      angle += 2 * pi;

    // A zero-crossing is characterized by a negative sign between
    // consecutive intensity values.
    if (a * b < 0)
      zero_crossings.push_back(static_cast<float>(angle));
  }

  return zero_crossings;
}


auto filter_junctions(std::vector<sara::Junction<int>>& junctions,
                      Eigen::MatrixXf& circular_profiles,
                      const sara::ImageView<float>& f,
                      const sara::ImageView<float>& grad_f_norm,
                      const float grad_thres, const int radius)
{
  auto profile_extractor = CircularProfileExtractor{};
  profile_extractor.circle_radius = radius;

  circular_profiles.resize(4, junctions.size());

  auto junctions_filtered = std::vector<sara::Junction<int>>{};
  junctions_filtered.reserve(junctions.size());
  for (auto u = 0u; u < junctions.size(); ++u)
  {
    const auto& j = junctions[u];

    // Local gradient average.
    auto grad_norm = 0.f;
    for (auto v = -radius; v <= radius; ++v)
    {
      for (auto u = -radius; u <= radius; ++u)
      {
        const Eigen::Vector2i q = j.p + Eigen::Vector2i{u, v};
        grad_norm += grad_f_norm(q);
      }
    }
    grad_norm /= sara::square(2 * radius + 1);
    if (grad_norm < grad_thres)
      continue;

    const auto profile = profile_extractor(f, j.p.cast<double>());
    const auto zero_crossings = localize_zero_crossings(  //
        profile,                                          //
        profile_extractor.num_circle_sample_points        //
    );

    // Count the number of zero-crossings: there must be 4 zero-crossings
    // because of the chessboard pattern.
    if (zero_crossings.size() != 4u)
      continue;

    for (auto i = 0; i < 4; ++i)
      circular_profiles(i, junctions_filtered.size()) = zero_crossings[i];

    junctions_filtered.emplace_back(j);
  }

  junctions_filtered.swap(junctions);
  circular_profiles.conservativeResize(4, junctions.size());
}

template <typename T>
auto k_nearest_neighbors(const std::vector<sara::Junction<T>>& points,
                         const int k, const Eigen::MatrixXf& circular_profiles)
    -> std::pair<Eigen::MatrixXi, Eigen::MatrixXf>
{
  const auto n = points.size();

  auto neighbors = Eigen::MatrixXi{k, n};
  auto distances = Eigen::MatrixXf{k, n};
  neighbors.setConstant(-1);
  distances.setConstant(std::numeric_limits<float>::infinity());

  for (auto u = 0u; u < n; ++u)
  {
    const auto& pu = points[u].position();

    // Calculate the direction of the edge gradients.
    auto gradient_dirs = Eigen::MatrixXf{2, circular_profiles.rows()};
    for (auto i = 0; i < gradient_dirs.cols(); ++i)
    {
      const auto& angle = circular_profiles(i, u);
      gradient_dirs.col(i) << std::cos(angle), std::sin(angle);
    }

    for (auto v = 0u; v < n; ++v)
    {
      if (v == u)
        continue;

      const auto& pv = points[v].position();
      const auto d_uv = static_cast<float>((pu - pv).squaredNorm());

      // A good neighbor is valid if it is consistent with the image gradient.
      const Eigen::Vector2f dir_uv =
          (pv - pu).template cast<float>().normalized();
      auto dot_uv = Eigen::VectorXf{circular_profiles.rows()};
      for (auto i = 0; i < gradient_dirs.cols(); ++i)
        dot_uv(i) = std::abs(gradient_dirs.col(i).dot(dir_uv));
      const auto max_dot_uv = dot_uv.maxCoeff();
      static const auto dot_thres = std::cos(10.f / 180.f * M_PI);
      if (max_dot_uv < dot_thres)
        continue;

      for (auto a = 0; a < k; ++a)
      {
        if (d_uv < distances(a, u))
        {
          // Shift the neighbors and distances.
          auto neighbor = static_cast<int>(v);
          auto dist = d_uv;
          for (auto b = a; b < k; ++b)
          {
            std::swap(neighbor, neighbors(b, u));
            std::swap(dist, distances(b, u));
          }

          break;
        }
      }
    }
  }

  return std::make_pair(neighbors, distances);
}


auto find_edge_path(const Eigen::Vector2i& a, const Eigen::Vector2i& b,
                    sara::ImageView<std::uint8_t>& edge_map,
                    const int dilation_radius) -> std::vector<Eigen::Vector2i>
{
  const auto w = edge_map.width();
  const auto to_index = [w](const Eigen::Vector2i& p) {
    return p.x() + p.y() * w;
  };
  const auto to_coords = [w](std::int32_t i) -> Eigen::Vector2i {
    const auto y = i / w;
    const auto x = i - y * w;
    return {x, y};
  };

  const auto is_strong_edgel = [&edge_map](const Eigen::Vector2i& p) {
    return edge_map(p) == 255;
  };

  const auto dilate_on_point = [&edge_map](const Eigen::Vector2i& a, int r) {
    for (auto y = a.y() - r; y <= a.y() + r; ++y)
      for (auto x = a.x() - r; x <= a.x() + r; ++x)
        if (0 <= x && x < edge_map.width() && 0 <= y && y < edge_map.height())
          edge_map(x, y) = 255;
  };
  dilate_on_point(a, dilation_radius);
  dilate_on_point(b, dilation_radius);

  auto predecessor_map = sara::Image<std::int32_t>{edge_map.sizes()};
  static constexpr auto undefined_predecessor = -1;
  predecessor_map.flat_array().fill(undefined_predecessor);

  auto distance_map = sara::Image<std::int32_t>{edge_map.sizes()};
  static constexpr auto infinity = std::numeric_limits<std::int32_t>::max();
  distance_map.flat_array().fill(infinity);

  // Neighborhood defined by 8-connectivity.
  static const auto dir = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{1, 0},    //
      Eigen::Vector2i{1, 1},    //
      Eigen::Vector2i{0, 1},    //
      Eigen::Vector2i{-1, 1},   //
      Eigen::Vector2i{-1, 0},   //
      Eigen::Vector2i{-1, -1},  //
      Eigen::Vector2i{0, -1},   //
      Eigen::Vector2i{1, -1}    //
  };

  struct PointDistance
  {
    Eigen::Vector2i coords;
    std::int32_t distance;
    auto operator<(const PointDistance& other) const -> bool
    {
      return distance > other.distance;
    }
  };

  auto q = std::priority_queue<PointDistance>{};

  // Initialize Dijkstra.
  q.push({a, 0});
  distance_map(a) = 0;
  auto joined = false;

  // Run Dijkstra's algorithm from a.
  while (!q.empty())
  {
    const auto p = q.top();
    q.pop();
    if (!is_strong_edgel(p.coords))
      throw std::runtime_error{"NOT AN EDGEL!"};
    if (p.coords == b)
    {
      joined = true;
      break;
    }

    // Add nonvisited edgel.
    for (const auto& d : dir)
    {
      const Eigen::Vector2i n = p.coords + d;
      // Boundary conditions.
      if (n.x() < 0 || n.x() >= edge_map.width() ||  //
          n.y() < 0 || n.y() >= edge_map.height())
        continue;
      const auto distance = p.distance + 1;

      // Make sure that the neighbor is an edgel.
      if (!is_strong_edgel(n))
        continue;

      // Enqueue the neighbor n if it is not already enqueued
      if (distance < distance_map(n))
      {
        distance_map(n) = distance;
        predecessor_map(n) = to_index(p.coords);
        q.push({n, distance});
      }
    }
  }

  if (!joined)
    return {};

  // Reconstruct the path.
  auto path = std::vector<Eigen::Vector2i>{};
  path.push_back(b);
  auto p = b;
  while (predecessor_map(p) != a.y() * w + a.x())
  {
    path.push_back(p);
    p = to_coords(predecessor_map(p));
  }

  std::reverse(path.begin(), path.end());

  return path;
}

template <typename V>
struct KnnGraph
{
  int _k;
  std::vector<V> _vertices;
  Eigen::MatrixXi _neighbors;
  Eigen::MatrixXf _distances;
  Eigen::MatrixXf _affinities;
  Eigen::MatrixXf _circular_profiles;
  Eigen::MatrixXf _affinity_scores;
  Eigen::VectorXf _unary_scores;

  struct VertexScore
  {
    std::size_t vertex;
    float score;
    inline auto operator<(const VertexScore& other) const
    {
      return score < other.score;
    }
  };


  inline auto vertex(int v) const -> const V&
  {
    return _vertices[v];
  }

  inline auto nearest_neighbor(const int v, const int k) const -> const V&
  {
    return _vertices[_neighbors(k, v)];
  };

  inline auto compute_affinity_scores() -> void
  {
    const auto n = _vertices.size();
    const auto k = _neighbors.rows();

    _affinity_scores.resize(k, n);
    _unary_scores.resize(n);

    for (auto u = 0u; u < n; ++u)
    {
      const auto fu = _circular_profiles.col(u);

      for (auto nn = 0; nn < k; ++nn)
      {
        const auto v = _neighbors(nn, u);
        static constexpr auto undefined_neighbor = -1;
        if (v == undefined_neighbor)
        {
          _affinity_scores(nn, u) = -std::numeric_limits<float>::max();
          continue;
        }
        const auto fv = _circular_profiles.col(v);

        auto affinities = Eigen::Matrix4f{};
        for (auto i = 0; i < fu.size(); ++i)
          for (auto j = 0; j < fv.size(); ++j)
            affinities(i, j) = std::abs(dir(fu(i)).dot(dir(fv(j))));

        const Eigen::RowVector4f best_affinities =
            affinities.colwise().maxCoeff();
        _affinity_scores(nn, u) = best_affinities.sum();
      }

      _unary_scores(u) = _affinity_scores.col(u).sum();
    }
  }

  inline auto grow(sara::ImageView<std::uint8_t>& edge_map,
                   const Eigen::Vector2i& corner_count,
                   [[maybe_unused]] const int downscale_factor,
                   const int dilation_radius) -> bool
  {
    if (_vertices.empty())
    {
      SARA_DEBUG << "No corners found!" << std::endl;
      return false;
    }

    const auto k = _neighbors.rows();
#define DEBUG_GROW
#ifdef DEBUG_GROW
    const auto s = downscale_factor;
#endif

    auto vs = std::vector<VertexScore>{};
    vs.reserve(_vertices.size());
    for (auto u = 0u; u < _vertices.size(); ++u)
      vs.push_back({u, _unary_scores(u)});

    auto q = std::priority_queue<VertexScore>{};
    q.emplace(*std::max_element(vs.begin(), vs.end()));

    auto visited = std::vector<std::uint8_t>(_vertices.size(), 0);

    auto num_corners_added = 0;

    while (!q.empty())
    {
      const auto best = q.top();
      const auto& u = best.vertex;
      const auto& pu = _vertices[u].position();
      visited[u] = 1;

      q.pop();
      ++num_corners_added;

#ifdef DEBUG_GROW
      sara::fill_circle(s * pu.x(), s * pu.y(), 8, sara::Green8);
#endif

      for (auto nn = 0; nn < k; ++nn)
      {
        const auto& v = _neighbors(nn, best.vertex);
        static constexpr auto undefined_neighbor = -1;
        if (v == undefined_neighbor)
        {
          SARA_DEBUG << "SKIPPING INVALID NEIGHBOR..." << std::endl;
          continue;
        }
        const auto& pv = _vertices[v].position();

        const auto edge_path =
            find_edge_path(pu, pv, edge_map, dilation_radius);
        if (edge_path.empty())
          continue;

#ifdef DEBUG_GROW
        for (const auto& p : edge_path)
          sara::fill_circle(s * p.x(), s * p.y(), 2, sara::Blue8);
#endif

        if (!visited[v])
        {
          visited[v] = 1;
          q.push(vs[v]);

#ifdef DEBUG_GROW
          sara::fill_circle(s * pv.x(), s * pv.y(), 8, sara::Red8);
#endif
        }
      }
    }

    const auto found = num_corners_added == corner_count(0) * corner_count(1);
    if (!found)
      sara::draw_text(400, 400, "NO!!!!" + std::to_string(num_corners_added),
                      sara::White8, 60, 0, false, true);
    else
      sara::draw_text(400, 400, "YES!!!", sara::White8, 60, 0, false, true);

    return found;
  }
};


auto __main(int argc, char** argv) -> int
{
#ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
#endif

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
  auto video_frame_copy = sara::Image<sara::Rgb8>{};
  auto frame_number = -1;

  auto corner_count = Eigen::Vector2i{};
  if (argc < 4)
    corner_count << 7, 12;
  // corner_count << 5, 7;
  else
    corner_count << std::atoi(argv[2]), std::atoi(argv[3]);

  const auto downscale_factor = argc < 5 ? 2 : std::stoi(argv[4]);
  static constexpr auto sigma_D = 1.6f;
  static const auto sigma_I =
      argc < 6 ? 6.f / downscale_factor : std::stof(argv[5]);
  static constexpr auto k = 6;
  static constexpr auto grad_adaptive_thres = 2e-2f;

  auto found_count = 0;
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
    auto f = video_frame.convert<float>().compute<sara::Gaussian>(sigma_D);
    if (downscale_factor > 1)
      f = sara::downscale(f, downscale_factor);
    const auto grad_f = f.compute<sara::Gradient>();
    const auto junction_map = sara::junction_map(f, grad_f, sigma_I);
    auto grad_f_norm = sara::Image<float>{f.sizes()};
    auto grad_f_ori = sara::Image<float>{f.sizes()};
    sara::gradient_in_polar_coordinates(f, grad_f_norm, grad_f_ori);
    const auto grad_max = grad_f_norm.flat_array().maxCoeff();
    const auto grad_thres = grad_adaptive_thres * grad_max;
    auto edge_map = sara::suppress_non_maximum_edgels(
        grad_f_norm, grad_f_ori, 2 * grad_thres, grad_thres);
    for (auto e = edge_map.begin(); e != edge_map.end(); ++e)
      if (*e == 127)
        *e = 0;
    sara::toc("Feature maps");

    auto graph = KnnGraph<sara::Junction<int>>{};
    graph._k = k;
    auto& junctions = graph._vertices;
    auto& circular_profiles = graph._circular_profiles;

    // Detect the junctions.
    sara::tic();
    {
      junctions = sara::extract_junctions(
          junction_map, static_cast<int>(std::round(sigma_I)));
      sara::nms(junctions, f.sizes(),
                static_cast<int>(std::round(sigma_I * 2)));
      filter_junctions(junctions, circular_profiles, f, grad_f_norm, grad_thres,
                       static_cast<int>(std::round(sigma_I)));
    }
    sara::toc("junction");

    // Link the junctions together.
    sara::tic();
    {
      auto [nn, dists] = k_nearest_neighbors(junctions, k, circular_profiles);
      graph._neighbors = std::move(nn);
      graph._distances = std::move(dists);
    }
    sara::toc("knn-graph");

    sara::tic();
    graph.compute_affinity_scores();
    sara::toc("affinity scores");

    // TODO: calculate the k-nn graph on the refined junctions.
    sara::tic();
    auto junctions_refined = std::vector<sara::Junction<float>>{};
    junctions_refined.reserve(junctions.size());
    std::transform(junctions.begin(), junctions.end(),
                   std::back_inserter(junctions_refined),
                   [&grad_f](const auto& j) -> sara::Junction<float> {
                     const auto p = sara::refine_junction_location_unsafe(
                         grad_f, j.position(), sigma_I);
                     return {p, j.score};
                   });
    sara::toc("refine junction");

    sara::tic();
    video_frame_copy = edge_map.convert<sara::Rgb8>();
    video_frame_copy.flat_array() /= sara::Rgb8{8, 8, 8};
    if (downscale_factor > 1)
      video_frame_copy = sara::upscale(video_frame_copy, downscale_factor);
    for (auto u = 0u; u < junctions.size(); ++u)
    {
      const auto& jr = junctions_refined[u];

      const Eigen::Vector2f jri = jr.p * downscale_factor;

      sara::draw_circle(video_frame_copy, jri,
                        static_cast<int>(std::round(sigma_D)), sara::Magenta8,
                        3);
      sara::fill_circle(     //
          video_frame_copy,  //
          static_cast<int>(std::round(jri.x())),
          static_cast<int>(std::round(jri.y())),  //
          1, sara::Red8);
    }
    sara::display(video_frame_copy);
    sara::draw_text(80, 80, std::to_string(frame_number), sara::White8, 60, 0,
                    false, true);
    sara::toc("display junctions");

    sara::tic();
    const auto found = graph.grow(edge_map,                        //
                                  corner_count, downscale_factor,  //
                                  static_cast<int>(std::round(sigma_I)));
    sara::toc("grow");
    if (found)
      ++found_count;

    const auto detection_rate_text = std::to_string(found_count) + "/" +
                                     std::to_string(frame_number / 3 + 1);
    sara::draw_text(80, 200, detection_rate_text, sara::White8, 60, 0, false,
                    true);
    SARA_DEBUG << "detection rate = " << detection_rate_text << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
