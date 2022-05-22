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

#include <unordered_map>

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization.hpp>

#include "Chessboard/SaddlePointDetection.hpp"


namespace sara = DO::Sara;


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


auto knn_graph(const std::vector<sara::SaddlePoint>& points, const int k)
    -> std::pair<Eigen::MatrixXi, Eigen::MatrixXf>
{
  const auto n = points.size();

  auto neighbors = Eigen::MatrixXi{k, n};
  auto distances = Eigen::MatrixXf{k, n};
  neighbors.setConstant(-1);
  distances.setConstant(std::numeric_limits<float>::infinity());

  for (auto u = 0u; u < n; ++u)
  {
    const auto& pu = points[u].p;
    for (auto v = 0u; v < n; ++v)
    {
      if (v == u)
        continue;

      const auto& pv = points[v].p;
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


template <typename V>
struct KnnGraph
{
  int _k;
  std::vector<V> _vertices;
  Eigen::MatrixXi _neighbors;
  Eigen::MatrixXf _distances;

  inline auto vertex(int v) const -> const V&
  {
    return _vertices[v];
  }

  inline auto nearest_neighbor(const int v, const int k) const -> const V&
  {
    return _vertices[_neighbors(k, v)];
  };
};


auto __main(int argc, char** argv) -> int
{
  if (argc < 2)
    return 1;

  static constexpr auto adaptive_thres = 0.05f;
  static constexpr auto sigma = 1.6f;

  const auto video_file = std::string{argv[1]};
  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto video_frame_copy = sara::Image<sara::Rgb8>{};
  auto frame_number = -1;

  auto profile_extractor = CircularProfileExtractor{};

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

    const auto image_gray = video_frame.convert<float>();
    const auto image_blurred = image_gray.compute<sara::Gaussian>(sigma);
    video_frame_copy = image_blurred.convert<sara::Rgb8>();

    // Calculate the hessian matrix.
    const auto hessian = image_blurred.compute<sara::Hessian>();

    // Chessboard corners are saddle points of the image, which are
    // characterized by the property det(H(x, y)) < 0.
    const auto det_of_hessian = hessian.compute<sara::Determinant>();

    static constexpr auto k = 4;
    auto graph = KnnGraph<sara::SaddlePoint>{};
    graph._k = k;
    auto& saddle_points = graph._vertices;

    // Adaptive thresholding.
    const auto thres = det_of_hessian.flat_array().minCoeff() * adaptive_thres;
    saddle_points = extract_saddle_points(det_of_hessian, hessian, thres);

    // Non-maxima suppression.
    nms(saddle_points, video_frame.sizes(), profile_extractor.circle_radius);

    auto [nn, dists] = knn_graph(saddle_points, k);
    graph._neighbors = std::move(nn);
    graph._distances = std::move(dists);

    for (auto u = 0u; u < saddle_points.size(); ++u)
    {
      const auto& s = saddle_points[u];

      const auto r = profile_extractor.circle_radius;
      if (s.p.x() < r || s.p.x() >= video_frame.width() - r ||  //
          s.p.y() < r || s.p.y() >= video_frame.height() - r)
        continue;

      const auto profile = profile_extractor(image_blurred, s.p.cast<double>());
      const auto zero_crossings = localize_zero_crossings(  //
          profile,                                          //
          profile_extractor.num_circle_sample_points        //
      );

      // Count the number of zero-crossings: there must be 4 zero-crossings
      // because of the chessboard pattern.
      if (zero_crossings.size() != 4u)
        continue;

      sara::draw_circle(video_frame_copy, s.p.x(), s.p.y(),
                        profile_extractor.circle_radius, sara::Red8);
      sara::draw(video_frame, s);

      for (const auto& angle : zero_crossings)
      {
        const auto& r = profile_extractor.circle_radius;
        const auto e = Eigen::Vector2d{std::cos(angle), std::sin(angle)};
        const auto p = s.p.cast<double>() + r * e;
        sara::fill_circle(video_frame_copy, p.x(), p.y(), 2, sara::Yellow8);
      }

      for (auto v = 0; v < k; ++v)
      {
        const auto& pv = graph.nearest_neighbor(u, v).p;
        sara::draw_arrow(video_frame_copy, s.p.cast<float>(), pv.cast<float>(),
                         sara::Blue8, 2);
      }
    }

    sara::display(video_frame_copy);
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
