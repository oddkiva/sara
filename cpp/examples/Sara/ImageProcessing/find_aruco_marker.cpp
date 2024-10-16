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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/FeatureDetectors/Harris.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/CartesianToPolarCoordinates.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/Otsu.hpp>

#if defined(_OPENMP)
#  include <omp.h>
#endif

#if __has_include(<execution>) && !defined(__APPLE__)
#  include <execution>
#endif
#include <exception>
#include <filesystem>


namespace sara = DO::Sara;
namespace fs = std::filesystem;


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

// #define INSPECT_PATCH
#if defined(INSPECT_PATCH)
static constexpr auto square_size = 20;
static constexpr auto square_padding = 4;
#else
static constexpr auto square_size = 5;
static constexpr auto square_padding = 1;
#endif
static constexpr auto half_area =
    sara::square(square_size - 2 * square_padding) / 2;
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
#if defined(_OPENMP)
    omp_set_num_threads(omp_get_max_threads());
#endif


#if defined(_WIN32)
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
    const auto sigma_D = argc < 4 ? 0.5f : std::stof(argv[3]);
    const auto sigma_I = argc < 5 ? 1.2f : std::stof(argv[4]);
    const auto kappa = argc < 6 ? 0.04f : std::stof(argv[5]);
    const auto cornerness_adaptive_thres =
        argc < 7 ? 1e-5f : std::stof(argv[6]);
    const auto downscale_factor = argc < 8 ? 1 : std::stoi(argv[7]);
    SARA_CHECK(grad_adaptive_thres);
    SARA_CHECK(sigma_D);
    SARA_CHECK(sigma_I);
    SARA_CHECK(kappa);
    SARA_CHECK(cornerness_adaptive_thres);

    auto video_stream = sara::VideoStream{video_file};
    auto video_frame = video_stream.frame();
    auto video_frame_copy = sara::Image<sara::Rgb8>{};
    auto frame_number = -1;

    const auto video_path = fs::path{video_file};
    const auto video_filename = video_path.filename().string();

    auto video_writer = sara::VideoWriter
    {
#if defined(__APPLE__)
      (fs::path{"/Users/oddkiva/Desktop"} / video_filename).string(),  //
#else
      (fs::path{"/home/david/Desktop"} / video_filename).string(),  //
#endif
          video_stream.sizes(),  //
          30                     //
    };

    auto f = sara::Image<float>{video_frame.sizes()};
    auto f_ds = sara::Image<float>{video_frame.sizes() / downscale_factor};
    auto f_blurred = sara::Image<float>{f_ds.sizes()};

    auto grad_f = std::array{sara::Image<float>{f_ds.sizes()},
                             sara::Image<float>{f_ds.sizes()}};
    auto grad_f_norm = sara::Image<float>{f_ds.sizes()};
    auto grad_f_ori = sara::Image<float>{f_ds.sizes()};

    auto cornerness = sara::Image<float>{f.sizes()};

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
      if (downscale_factor > 1)
      {
        const auto tmp = f.compute<sara::Gaussian>(1.f);
        sara::scale(tmp, f_ds);
      }
      else
        f_ds = f;
      sara::toc("downscale");

      sara::tic();
      sara::apply_gaussian_filter(f_ds, f_blurred, sigma_D);

#if defined(SLOW_IMPL)
      grad_f = f_blurred.compute<sara::Gradient>();
      const auto M =
          grad_f.compute<sara::SecondMomentMatrix>().compute<sara::Gaussian>(
              sigma_I);
      cornerness = sara::Image<float>{M.sizes()};
      std::transform(
#  if __has_include(<execution>) && !defined(__APPLE__)
          std::execution::par_unseq,
#  endif
          M.begin(), M.end(), cornerness.begin(), [kappa](const auto& m) {
            return m.determinant() - kappa * sara::square(m.trace());
          });
      sara::gradient_in_polar_coordinates(f_blurred, grad_f_norm, grad_f_ori);
#else
      sara::gradient(f_blurred, grad_f[0], grad_f[1]);
      sara::cartesian_to_polar_coordinates(grad_f[0], grad_f[1],  //
                                           grad_f_norm, grad_f_ori);

      cornerness = sara::harris_cornerness(grad_f[0], grad_f[1],  //
                                           sigma_I, kappa);
#endif

      const auto cornerness_thres =
          cornerness.flat_array().maxCoeff() * cornerness_adaptive_thres;

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
        static constexpr auto eps = 1.;
        const auto ch = sara::ramer_douglas_peucker(
            sara::graham_scan_convex_hull(curve_points), eps);
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
          if (score > cornerness_thres)
            dominant_points.push_back({ch[i], score});
        }
#if defined(DEBUG_ME)
        SARA_CHECK(dominant_points.size());
#endif
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
            [&grad_f, sigma_I](const Eigen::Vector2d& c) -> Eigen::Vector2d {
              static const auto radius = static_cast<int>(std::round(sigma_I));
              const auto w = grad_f[0].width();
              const auto h = grad_f[0].height();
              const Eigen::Vector2i ci = c.array().round().cast<int>();
              const auto in_domain =                          //
                  radius <= ci.x() && ci.x() < w - radius &&  //
                  radius <= ci.y() && ci.y() < h - radius;
              if (!in_domain)
                return ci.cast<double>();

#if defined(SLOW_IMPL)
              const auto p = sara::refine_junction_location_unsafe(  //
                  grad_f, ci, radius);
#else
              const auto p = sara::refine_junction_location_unsafe(  //
                  grad_f[0], grad_f[1], ci, radius);
#endif
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
        const auto patch = normalize_quad(f_ds, q);
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
            static constexpr auto& b = square_padding;
            for (auto v = b; v < square_size - b; ++v)
            {
              for (auto u = b; u < square_size - b; ++u)
              {
                const auto p = Eigen::Vector2i{square_size * j + u,  //
                                               square_size * i + v};
                if (patch(p) != 0)
                  ++count;
              }
            }
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
#if INSPECT_EDGE_MAP
      auto disp = sara::upscale(edge_map, downscale_factor)  //
                      .convert<sara::Rgb8>();
#else
      auto disp = f.convert<sara::Rgb8>();
#endif
      for (auto k = 0u; k < codes.size(); ++k)
      {
        if (!plausible_codes[k])
          continue;

        const auto& q = candidate_quads[k];
        const auto n = q.size();
        for (auto i = 0u; i < n; ++i)
        {
          const Eigen::Vector2i a =
              (downscale_factor * q[i].array()).round().cast<int>();
          const Eigen::Vector2i b =
              (downscale_factor * q[(i + 1) % n].array()).round().cast<int>();
          sara::draw_line(disp, a.x(), a.y(), b.x(), b.y(), sara::Magenta8, 2);
        }
        for (auto i = 0u; i < n; ++i)
        {
          const Eigen::Vector2i a =
              (downscale_factor * q[i].array()).round().cast<int>();
          sara::fill_circle(disp, a.x(), a.y(), 3, sara::Red8);
        }
      }
      sara::display(disp);

#if defined(INSPECT_PATCH)
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

#if defined(DEBUG_ME)
      sara::get_key();
#endif

      sara::tic();
      video_writer.write(disp);
      sara::toc("Video write");
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
