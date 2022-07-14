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

#if __has_include(<execution>) && !defined(__APPLE__)
#  include <execution>
#endif
#include <exception>

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
#include <DO/Sara/ImageProcessing/Otsu.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/NonMaximumSuppression.hpp"


namespace sara = DO::Sara;


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

#define INSPECT_PATCH
#ifdef INSPECT_PATCH
static constexpr auto square_size = 20;
#else
static constexpr auto square_size = 5;
#endif
static constexpr auto num_squares = 4;
static constexpr auto l = (num_squares + 2) * square_size;
static constexpr auto ld = static_cast<double>(l);
static constexpr auto aruco_perimeter = num_squares * 4 + 4;

auto normalize_quad(const sara::ImageView<float>& image,
                    const std::vector<Eigen::Vector2d>& quad)
    -> sara::Image<float>
{
  auto patch = sara::Image<float>{l, l};
  const auto H = sara::homography(quad[0], quad[1], quad[2], quad[3], {0, 0},
                                  {ld, 0}, {ld, ld}, {0, ld});
  const Eigen::Matrix3d H_inv = H.inverse();

  for (auto y = 0; y < l; ++y)
  {
    for (auto x = 0; x < l; ++x)
    {
      const auto p = Eigen::Vector2i(x, y);
      const Eigen::Vector2d Hp =
          (H_inv * p.cast<double>().homogeneous()).hnormalized();

      const auto in_image_domain =                      //
          0 <= Hp.x() && Hp.x() < image.width() - 1 &&  //
          0 <= Hp.y() && Hp.y() < image.height() - 1;
      if (!in_image_domain)
      {
        patch(p) = 0.f;
        continue;
      }

      const auto intensity = static_cast<float>(sara::interpolate(image, Hp));
      patch(p) = intensity;
    }
  }

  return patch;
}


auto __main(int argc, char** argv) -> int
{
  try
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

    const auto grad_adaptive_thres = argc < 3 ? 1e-1f : std::stof(argv[2]);

    // Corner filtering.
    const auto kappa = argc < 4 ? 0.04f : std::stof(argv[3]);
    const auto cornerness_adaptive_thres =
        argc < 5 ? 1e-5f : std::stof(argv[4]);
    static constexpr auto sigma_D = 0.8f;
    static constexpr auto sigma_I = 1.5f;

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
      const auto grad_f = f_blurred.compute<sara::Gradient>();
      const auto M =
          grad_f.compute<sara::SecondMomentMatrix>().compute<sara::Gaussian>(
              sigma_I);
      auto cornerness = sara::Image<float>{M.sizes()};
      std::transform(
#if __has_include(<execution>) && !defined(__APPLE__)
          std::execution::par_unseq,
#endif
          M.begin(), M.end(), cornerness.begin(), [kappa](const auto& m) {
            return m.determinant() - kappa * pow(m.trace(), 2);
          });
      sara::gradient_in_polar_coordinates(f_blurred, grad_f_norm, grad_f_ori);
#if __has_include(<execution>) && !defined(__APPLE__)
      const auto grad_max = *std::max_element(
          std::execution::par_unseq, grad_f_norm.begin(), grad_f_norm.end());
#else
      const auto grad_max = grad_f_norm.flat_array().maxCoeff();
#endif
      const auto grad_thres = grad_adaptive_thres * grad_max;
      auto edge_map = sara::suppress_non_maximum_edgels(
          grad_f_norm, grad_f_ori, 2 * grad_thres, grad_thres);
      std::for_each(
#if __has_include(<execution>) && !defined(__APPLE__)
          std::execution::par_unseq,
#endif
          edge_map.begin(), edge_map.end(), [](auto& e) {
            if (e == 127)
              e = 0;
          });
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
      auto candidate_quads = std::vector<std::vector<Eigen::Vector2d>>{};
      for (const auto& [label, edge_curve] : edges)
      {
        if (edge_curve.size() < 10)
          continue;

        // The convex hull of the point set.
        auto curve_points = std::vector<Eigen::Vector2d>{};
        std::transform(edge_curve.begin(), edge_curve.end(),
                       std::back_inserter(curve_points),
                       [](const auto& p) { return p.template cast<double>(); });
        const auto ch = sara::graham_scan_convex_hull(curve_points);
        if (sara::length(ch) < 10 * 4 || sara::area(ch) < 100)
          continue;

        // Collect the dominant points, they must have some good cornerness
        // measure.
        auto dominant_points = std::vector<Corner<double>>{};
        const auto n = ch.size();
        for (auto i = 0u; i < n; ++i)
        {
          const Eigen::Vector2i p = ch[i].array().round().cast<int>();
          const auto score = cornerness(p);
          if (score > cornerness_adaptive_thres)
            dominant_points.push_back({ch[i], score});
        }
        // No point continuing at this point.
        if (dominant_points.size() < 4)
          continue;

        // There must be 4 clusters of dominant points.
        static constexpr auto min_sep_length = 4.;
        static constexpr auto min_sep_length_2 = sara::square(min_sep_length);
        auto cluster_cuts = std::vector<std::size_t>{};
        for (auto i = 0u; i < dominant_points.size(); ++i)
        {
          const auto& p = dominant_points[i];
          const auto& q = i == dominant_points.size() - 1
                              ? dominant_points[0]
                              : dominant_points[i + 1];
          if ((p.coords - q.coords).squaredNorm() > min_sep_length_2)
            cluster_cuts.push_back(i == dominant_points.size() - 1 ? 0 : i + 1);
        }
        if (cluster_cuts.size() != 4)
          continue;

        // Form the quad by finding the best dominant point in each cluster.
        const auto num_cuts = cluster_cuts.size();
        const auto num_points = dominant_points.size();
        auto quad = std::vector<Eigen::Vector2d>{};
        quad.reserve(4);
        for (auto i = 0u; i < cluster_cuts.size(); ++i)
        {
          // Form the cluster open interval [a, b).
          auto a = cluster_cuts[i];
          const auto& b =
              i == num_cuts - 1 ? cluster_cuts[0] : cluster_cuts[i + 1];

          // Find the best corner.
          auto best_corner = dominant_points[a];
          while (a != b)
          {
            if (best_corner.score < dominant_points[a].score)
              best_corner = dominant_points[a];
            a = (a == num_points - 1) ? 0 : a + 1;
          }
          quad.push_back(best_corner.coords);
        }

        // Refine the corner location.
        std::transform(
            quad.begin(), quad.end(), quad.begin(),
            [&grad_f](const Eigen::Vector2d& c) -> Eigen::Vector2d {
              static const auto radius = static_cast<int>(std::round(sigma_I));
              const Eigen::Vector2i ci = c.array().round().cast<int>();
              const auto p =
                  sara::refine_junction_location_unsafe(grad_f, ci, radius);
              return p.cast<double>();
            });

        candidate_quads.push_back(quad);
      }
      sara::toc("Candidate Quads");
      SARA_CHECK(candidate_quads.size());

      sara::tic();
      auto patches = std::vector<sara::Image<std::uint8_t>>{};
      for (const auto& q : candidate_quads)
      {
        const auto patch = normalize_quad(f, q);
        auto patch_binarized = sara::otsu_adaptive_binarization(patch);
        patch_binarized.flat_array() *= 255;
        patches.emplace_back(std::move(patch_binarized));
      }
      sara::toc("Binarized patch");

      sara::tic();
      using Code = Eigen::Matrix<int, num_squares + 2, num_squares + 2>;
      auto codes = std::vector<Code>{};
      for (auto& patch : patches)
      {
        auto code = Code{};
        for (auto i = 0; i < num_squares + 2; ++i)
          for (auto j = 0; j < num_squares + 2; ++j)
          {
            auto count = 0;
            for (auto v = 0; v < square_size; ++v)
            {
              for (auto u = 0; u < square_size; ++u)
              {
                const auto p = Eigen::Vector2i{square_size * j + u,  //
                                               square_size * i + v};
                if (patch(p) != 0)
                  ++count;
              }
            }

            static constexpr auto half_area = square_size * square_size / 2;
            code(i, j) = count > half_area ? 1 : 0;

            for (auto v = 0; v < square_size; ++v)
            {
              for (auto u = 0; u < square_size; ++u)
              {
                const auto p = Eigen::Vector2i{square_size * j + u,  //
                                               square_size * i + v};
                patch(p) = code(i, j) != 0 ? 255 : 0;
              }
            }
          }
        codes.push_back(code);
      }
      sara::toc("ARUCO generation");

      sara::tic();
      auto plausible_codes = std::vector<std::uint8_t>(codes.size());
      for (auto k = 0u; k < codes.size(); ++k)
      {
        const auto& code = codes[k];
        auto black_border_count = 0;
        for (auto i = 0; i < num_squares + 2; ++i)
          for (auto j = 0; j < num_squares + 2; ++j)
            if (i == 0 || i == num_squares + 1 ||  //
                j == 0 || j == num_squares + 1)
              black_border_count += static_cast<int>(code(i, j) == 0);
        plausible_codes[k] = (black_border_count >= (aruco_perimeter) -2);
      }
      const auto plausible_code_count =
          std::accumulate(plausible_codes.begin(), plausible_codes.end(), int{},
                          [](int a, std::uint8_t b) -> int { return a + b; });
      sara::toc("ARUCO validation");
      SARA_CHECK(plausible_code_count);

      sara::tic();
      auto disp = f.convert<sara::Rgb8>();
      for (auto k = 0u; k < codes.size(); ++k)
      {
        if (!plausible_codes[k])
          continue;

        const auto& q = candidate_quads[k];
        const auto n = q.size();
        for (auto i = 0u; i < n; ++i)
        {
          const Eigen::Vector2i a = q[i].array().round().cast<int>();
          const Eigen::Vector2i b = q[(i + 1) % n].array().round().cast<int>();
          sara::draw_line(disp, a.x(), a.y(), b.x(), b.y(), sara::Magenta8, 1);
        }
        for (auto i = 0u; i < n; ++i)
        {
          const Eigen::Vector2i a = q[i].array().round().cast<int>();
          sara::fill_circle(disp, a.x(), a.y(), 2, sara::Red8);
        }
      }
      sara::display(disp);

#ifdef INSPECT_PATCH
      for (auto k = 0u; k < codes.size(); ++k)
      {
        if (!plausible_codes[k])
          continue;

        const auto& patch = patches[k];
        sara::display(patch);
        for (auto y = 0; y <= num_squares; ++y)
          sara::draw_line(0, (y + 1) * square_size, (l - 1),
                          (y + 1) * square_size, sara::Red8, 1);

        for (auto x = 0; x <= num_squares; ++x)
          sara::draw_line((x + 1) * square_size, 0, (x + 1) * square_size,
                          (l - 1), sara::Red8, 1);
        sara::get_key();
      }
#endif
      sara::toc("Display");
    }
  }
  catch (std::exception& e)
  {
    SARA_DEBUG << e.what() << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
