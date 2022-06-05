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

#include <omp.h>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include "Chessboard/JunctionDetection.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"
#include "Chessboard/SaddlePointDetection.hpp"


namespace sara = DO::Sara;

auto dir(const float angle) -> Eigen::Vector2f
{
  return Eigen::Vector2f{std::cos(angle), std::sin(angle)};
};


struct CircularProfileExtractor
{
  inline CircularProfileExtractor()
  {
    initialize_circle_sample_points();
  }

  // Sample a unit circle centered at the origin.
  inline auto initialize_circle_sample_points() -> void
  {
    static constexpr auto pi = static_cast<double>(M_PI);

    const auto& n = num_circle_sample_points;
    circle_sample_points = std::vector<Eigen::Vector2d>(n);

    for (auto i = 0; i < n; ++i)
    {
      const auto angle = i * 2 * pi / n;
      circle_sample_points[i] << std::cos(angle), std::sin(angle);
    }
  }

  inline auto operator()(const sara::ImageView<float>& image,
                         const Eigen::Vector2d& center) const -> Eigen::ArrayXf
  {
    auto intensity_profile = Eigen::ArrayXf(num_circle_sample_points);

    for (auto n = 0; n < num_circle_sample_points; ++n)
    {
      const Eigen::Vector2d pn =
          center + circle_radius * circle_sample_points[n];

      // Get the interpolated intensity value.
      intensity_profile(n) = interpolate(image, pn);
    }

    // // Collect all the intensity values in the disk for more robustness.
    // const auto image_patch =
    //     sara::safe_crop(image, center.cast<int>(), circle_radius)
    //         .compute<sara::Gaussian>(1.6f);
    // sara::display(image_patch);
    // sara::get_key();

    // // Normalize the intensities.
    // const auto min_intensity = image_patch.flat_array().minCoeff();
    // const auto max_intensity = image_patch.flat_array().maxCoeff();

    // Normalize the intensities.
    const auto min_intensity = intensity_profile.minCoeff();
    const auto max_intensity = intensity_profile.maxCoeff();

    // The intensity treshold is the mid-point value.
    const auto intensity_threshold = (max_intensity + min_intensity) * 0.5f;
    intensity_profile -= intensity_threshold;

    return intensity_profile;
  }

  int num_circle_sample_points = 36;
  double circle_radius = 10.;
  std::vector<Eigen::Vector2d> circle_sample_points;
};


auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
    -> std::vector<float>
{
  auto zero_crossings = std::vector<float>{};
  for (auto n = 0; n < profile.size(); ++n)
  {
    const auto ia = n;
    const auto ib = (n + 1) % profile.size();

    const auto& a = profile[ia];
    const auto& b = profile[ib];

    const auto angle_a = ia * 2 * M_PI / num_bins;
    const auto angle_b = ib * 2 * M_PI / num_bins;

    const auto ea = Eigen::Vector2d{std::cos(angle_a),  //
                                    std::sin(angle_a)};
    const auto eb = Eigen::Vector2d{std::cos(angle_b),  //
                                    std::sin(angle_b)};

    // TODO: this all could have been simplified.
    const Eigen::Vector2d dir = (ea + eb) * 0.5;
    auto angle = std::atan2(dir.y(), dir.x());
    if (angle < 0)
      angle += 2 * M_PI;

    // A zero-crossing is characterized by a negative sign between
    // consecutive intensity values.
    if (a * b < 0)
      zero_crossings.push_back(angle);
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

    static const auto angle_threshold = std::cos(M_PI / 180.f * 20.f);
    auto good = true;
    for (auto i = 0; i < 4; ++i)
    {
      const auto ia = i == 0 ? 3 : i - 1;
      const auto ib = i;
      const auto ic = i == 3 ? 0 : i + 1;
      const auto a = dir(zero_crossings[ia]);
      const auto b = dir(zero_crossings[ib]);
      const auto c = dir(zero_crossings[ic]);
      const auto good_i = std::abs(a.dot(b)) < angle_threshold &&
                          std::abs(b.dot(c)) < angle_threshold;
      if (!good_i)
      {
        good = false;
        break;
      }
    }
    if (!good)
      continue;

    junctions_filtered.emplace_back(j);
  }

  junctions_filtered.swap(junctions);
  circular_profiles.conservativeResize(4, junctions_filtered.size());
}

template <typename Feature>
auto k_nearest_neighbors(const std::vector<Feature>& points, const int k)
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
    for (auto v = 0u; v < n; ++v)
    {
      if (v == u)
        continue;

      const auto& pv = points[v].position();
      const auto d_uv = static_cast<float>((pu - pv).squaredNorm());

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

auto sample_edge_gradient(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const float sigma, const sara::ImageView<float>& image)
  -> Eigen::MatrixXf
{
  const Eigen::Vector2f t = (b - a).normalized();
  const Eigen::Vector2f n = Eigen::Vector2f{-t.y(), t.x()};
  const auto num_samples = static_cast<int>(std::floor((b - a).norm() / sigma)) - 1;
  if (num_samples <= 0)
    return {};
  auto intensity_samples = Eigen::MatrixXf{2, num_samples};

  for (auto i = 0; i < num_samples; ++i)
  {
    const Eigen::Vector2d ai = (a + ((i + 1) * t + 2.5f * n) * sigma).cast<double>();
    const Eigen::Vector2d bi = (a + ((i + 1) * t - 2.5f * n) * sigma).cast<double>();
    const auto intensity_ai = static_cast<float>(sara::interpolate(image, ai));
    const auto intensity_bi = static_cast<float>(sara::interpolate(image, bi));

    intensity_samples.col(i) << intensity_ai, intensity_bi;
  }

  // TODO: calculate statistics: mean and std-dev for a robust inference.

  return intensity_samples;
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

  inline auto compute_affinity_scores(const sara::ImageView<float>& image, const float sigma) -> void
  {
    const auto n = _vertices.size();
    const auto k = _neighbors.rows();

    _affinity_scores.resize(k, n);
    _unary_scores.resize(n);

    for (auto u = 0u; u < n; ++u)
    {
      const auto fu = _circular_profiles.col(u);
      const auto pu = _vertices[u].position();
#if 0
      std::cout << "[" << u << "]" << std::endl;
      std::cout << "fu = " << fu.transpose() << std::endl;
#endif

      for (auto nn = 0; nn < k; ++nn)
      {
        const auto v = _neighbors(nn, u);
        const auto fv = _circular_profiles.col(v);
        const auto pv = _vertices[v].position();
#if 0
        std::cout << "[" << u << "] -> [" << v << "]" << std::endl;
        std::cout << "fv = " << fv.transpose() << std::endl;
#endif

        auto affinities = Eigen::Matrix4f{};
        for (auto i = 0; i < fu.size(); ++i)
          for (auto j = 0; j < fv.size(); ++j)
            affinities(i, j) = std::abs(dir(fu(i)).dot(dir(fv(j))));

        const Eigen::RowVector4f best_affinities =
            affinities.colwise().maxCoeff();
        _affinity_scores(nn, u) = best_affinities.sum();

#if 0
        std::cout << "affinities =\n" << affinities << std::endl;
        std::cout << "best_affinities =\n" << best_affinities << std::endl;
        std::cout << "best_affinities =\n"
                  << best_affinities.array().acos().abs() * 180.f / float(M_PI)
                  << std::endl;
        sara::get_key();
#endif
      }

#if 0
      Eigen::Vector2f laplacian = Eigen::Vector2f::Zero();
      auto mean_sq_dist = float{};
      const Eigen::Vector2f puf = pu.template cast<float>();
      for (auto nn = 0; nn < k; ++nn)
      {
        const auto v = _neighbors(nn, u);
        const Eigen::Vector2f pv =
            _vertices[v].position().template cast<float>();
        laplacian += pv - puf;
        mean_sq_dist = (pv - puf).squaredNorm();
      }

      const auto equidistance_score =
          std::exp(-0.5f * laplacian.squaredNorm() / mean_sq_dist);

      _unary_scores(u) = _affinity_scores.col(u).sum() * equidistance_score;
#endif

      _unary_scores(u) = _affinity_scores.col(u).sum();
    }
  }

  inline auto sort(sara::ImageView<sara::Rgb8>& image) -> void
  {
    auto vs = std::vector<VertexScore>{};
    vs.resize(_vertices.size());
    for (auto v = 0u; v < vs.size(); ++v)
      vs[v] = {v, _unary_scores(v)};
    std::sort(vs.rbegin(), vs.rend());

    for (const auto& [v, score] : vs)
    {
      const auto& pv = _vertices[v].position();
      std::cout << pv.transpose() << std::endl;
      sara::fill_circle(image, pv.x(), pv.y(), 8, sara::Green8);
      sara::display(image);
      sara::get_key();
    }
  }

  inline auto grow(const sara::ImageView<float>& image, const float sigma,
                   const int downscale_factor, const Eigen::Vector2i& corner_count) -> void
  {
    const auto k = _neighbors.rows();
    const auto s = downscale_factor;

    auto vs = std::vector<VertexScore>{};
    vs.resize(_vertices.size());
    for (auto v = 0u; v < vs.size(); ++v)
      vs[v] = {v, _unary_scores(v)};

#if 0
    for (const auto& [v, score]: vs)
      SARA_DEBUG << "v = " << v << " score = " << score << std::endl;
#endif

    auto q = std::priority_queue<VertexScore>{};
    q.emplace(*std::max_element(vs.begin(), vs.end()));

    auto visited = std::vector<std::uint8_t>(_vertices.size(), 0);

    auto num_corners_added = 0;
    auto thres = 1e-2f;
    auto thres_set_from_data = false;

    while (!q.empty())
    {
      const auto best = q.top();
      const auto& u = best.vertex;
      const auto& pu = _vertices[u].position();
      visited[u] = 1;
#if 0
      SARA_DEBUG << "current best " << u << " score = " << best.score << std::endl;
#endif

      q.pop();
      ++num_corners_added;

      // Debug.
      sara::fill_circle(s * pu.x(), s * pu.y(), 4, sara::Green8);

      for (auto nn = 0; nn < k; ++nn)
      {
        const auto& v = _neighbors(nn, best.vertex);
        const auto& pv = _vertices[v].position();

        const auto edge_gradients = sample_edge_gradient(
            pu.template cast<float>(), pv.template cast<float>(), sigma, image);

        const auto N = edge_gradients.cols();

        const auto amean = edge_gradients.row(0).sum() / N;
        const auto adev = std::sqrt(
            (edge_gradients.row(0).array() - amean).square().sum() / N);

        const auto bmean = edge_gradients.row(1).sum() / edge_gradients.cols();
        const auto bdev = std::sqrt(
            (edge_gradients.row(1).array() - bmean).square().sum() / N);

        auto good_edge = false;
        auto diff =  float{};
        static constexpr auto lambda = 1.0f;
        if (amean < bmean)
        {
          diff = bmean - lambda * bdev - amean - lambda * adev;
          good_edge = diff > thres;
        }
        else
        {
          diff = amean - lambda * adev - bmean - lambda * bdev;
          good_edge = diff > thres;
        }
        if (!good_edge)
          continue;

        if (diff > 1e-3f)
        {
          // thres_set_from_data = true;
          thres = diff * 0.5f;
        }

        sara::draw_line(s * pu.x(), s * pu.y(), s * pv.x(), s * pv.y(),
                        sara::Green8, 8);

        if (!visited[v])
        {
          visited[v] = 1;
          q.push(vs[v]);
#if 0
          SARA_DEBUG << "Adding vertex = " << vs[v].vertex << " score " << vs[v].score << std::endl;
#endif

          // Debug.
          sara::fill_circle(s * pv.x(), s * pv.y(), 4, sara::Red8);
        }
      }
    }
    if (num_corners_added != corner_count(0) * corner_count(1))
      sara::get_key();
  }
};


auto __main(int argc, char** argv) -> int
{
  if (argc < 2)
    return 1;

  omp_set_num_threads(omp_get_max_threads());

  const auto video_file = std::string{argv[1]};
  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto video_frame_copy = sara::Image<sara::Rgb8>{};
  auto frame_number = -1;

  auto corner_count = Eigen::Vector2i{};
  if (argc < 4)
    corner_count << 7, 12;
  else
    corner_count << std::atoi(argv[2]), std::atoi(argv[3]);

  static constexpr auto downscale_factor = 1;
  static constexpr auto sigma = 1.6f;
  static constexpr auto k = 6;
  static constexpr auto radius = 5;
  static constexpr auto grad_adaptive_thres = 2e-2f;

  // static constexpr auto tolerance_parameter = 0.0f;
  static const auto kernel_2d = sara::make_gaussian_kernel_2d(16.f);

  while (video_stream.read())
  {
    ++frame_number;
    if (frame_number % 3 != 0)
      continue;

    // if (frame_number < 1000)
    //   continue;

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }

    sara::tic();
    const auto f = sara::downscale(
        video_frame.convert<float>().compute<sara::Gaussian>(sigma),
        downscale_factor);
    const auto grad_f = f.compute<sara::Gradient>();
    const auto junction_map = sara::junction_map(f, grad_f, radius);
    auto grad_f_norm = grad_f.compute<sara::SquaredNorm>();
    grad_f_norm.flat_array() = grad_f_norm.flat_array().sqrt();
    const auto grad_max = grad_f_norm.flat_array().maxCoeff();
    const auto grad_thres = grad_adaptive_thres * grad_max;
    sara::toc("Feature maps");

    sara::tic();
    auto f_pyr = std::vector<sara::Image<float>>{};
    f_pyr.push_back(f);
    // f_pyr.push_back(sara::downscale(f_pyr.back(), 2));
    // f_pyr.push_back(sara::downscale(f_pyr.back(), 2));
    sara::toc("Gaussian pyramid");

#if 0
    sara::tic();
    auto binary_mask = sara::Image<std::uint8_t>{f_pyr.back().sizes()};
    sara::adaptive_thresholding(f_pyr.back(), kernel_2d, binary_mask,
                                tolerance_parameter);
    binary_mask.flat_array() *= 255;
    sara::toc("Adaptive thresholding");
#endif

    auto graph = KnnGraph<sara::Junction<int>>{};
    graph._k = k;
    auto& junctions = graph._vertices;
    auto& circular_profiles = graph._circular_profiles;

    // Detect the junctions.
    sara::tic();
    {
      junctions = sara::extract_junctions(junction_map, radius);
      sara::nms(junctions, f.sizes(), radius);
      filter_junctions(junctions, circular_profiles, f, grad_f_norm, grad_thres,
                       radius);
    }
    sara::toc("junction");

    // Link the junctions together.
    sara::tic();
    {
      auto [nn, dists] = k_nearest_neighbors(junctions, k);
      graph._neighbors = std::move(nn);
      graph._distances = std::move(dists);
    }
    sara::toc("knn-graph");

    graph.compute_affinity_scores(f, sigma);

    // TODO: calculate the k-nn graph on the refined junctions.
    auto junctions_refined = std::vector<sara::Junction<float>>{};
    junctions_refined.reserve(junctions.size());
    std::transform(
        junctions.begin(), junctions.end(),
        std::back_inserter(junctions_refined),
        [&grad_f /*, &junction_map */](const auto& j) -> sara::Junction<float> {
          const auto w = grad_f.width();
          const auto h = grad_f.height();
          const auto in_image_domain =
              radius <= j.p.x() && j.p.x() < w - radius &&  //
              radius <= j.p.y() && j.p.y() < h - radius;
          if (!in_image_domain)
          {
            throw std::runtime_error{"That can't be!!!"};
            return {j.p.template cast<float>(), j.score};
          }

          const auto p = sara::refine_junction_location_unsafe(
              grad_f, j.position(), radius);
          return {p, j.score};
        });


    video_frame_copy = video_frame;
    // video_frame_copy = sara::upscale(binary_mask, 2).convert<sara::Rgb8>();
    for (auto u = 0u; u < junctions.size(); ++u)
    {
      const auto& jr = junctions_refined[u];

      const Eigen::Vector2f jri = jr.p * downscale_factor;

      sara::draw_circle(video_frame_copy, jri, radius, sara::Magenta8, 3);
      sara::fill_circle(video_frame_copy, jri.x(), jri.y(), 1, sara::Red8);

      // for (auto v = 0; v < k; ++v)
      // {
      //   const auto& pv = graph.nearest_neighbor(u, v).p;
      //   sara::draw_arrow(video_frame_copy, j.p.cast<float>(),
      //   pv.cast<float>(),
      //                    sara::Green8, 2);
      // }
    }

    sara::display(video_frame_copy);

    graph.grow(f, sigma, downscale_factor, corner_count);
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
