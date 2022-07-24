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

#include <unordered_map>
#include <unordered_set>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Chessboard/CircularProfileExtractor.hpp"
#include "Chessboard/Corner.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"
#include "Chessboard/OrientationHistogram.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

auto get_curve_shape_statistics(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
    -> sara::CurveStatistics
{
  auto curves_64f = std::vector<std::vector<Eigen::Vector2d>>{};
  curves_64f.resize(curve_pts.size());
  std::transform(curve_pts.begin(), curve_pts.end(), curves_64f.begin(),
                 [](const auto& points) {
                   auto points_2d = std::vector<Eigen::Vector2d>{};
                   points_2d.resize(points.size());
                   std::transform(
                       points.begin(), points.end(), points_2d.begin(),
                       [](const auto& p) { return p.template cast<double>(); });
                   return points_2d;
                 });

  return sara::CurveStatistics{curves_64f};
}

auto mean_gradient(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
    const sara::ImageView<float>& Ix,                            //
    const sara::ImageView<float>& Iy)                            //
    -> std::vector<Eigen::Vector2f>
{
  auto mean_gradients = std::vector<Eigen::Vector2f>(curve_pts.size());

  std::transform(
      curve_pts.begin(), curve_pts.end(), mean_gradients.begin(),
      [&Ix, &Iy](const std::vector<Eigen::Vector2i>& points) {
        static const Eigen::Vector2f zero2f = Eigen::Vector2f::Zero();
        Eigen::Vector2f g = std::accumulate(
            points.begin(), points.end(), zero2f,
            [&Ix, &Iy](const Eigen::Vector2f& gradient,
                       const Eigen::Vector2i& point) -> Eigen::Vector2f {
              const Eigen::Vector2f g =
                  gradient + Eigen::Vector2f{Ix(point), Iy(point)};
              return g;
            });
        const Eigen::Vector2f mean = g / points.size();
        return mean;
      });

  return mean_gradients;
}

auto kl_divergence(const Eigen::Matrix2f& A, const Eigen::Matrix2f& B)
{
  return 0.5f * ((A.inverse() * B).trace() - std::log(B.norm() / A.norm()));
}

enum class Direction : std::uint8_t
{
  Up = 0,
  Right = 1,
  Down = 2,
  Left = 3
};

inline auto direction_type(const float angle)
{
  static constexpr auto pi = static_cast<float>(M_PI);
  static constexpr auto pi_over_4 = static_cast<float>(M_PI / 4.);
  static constexpr auto three_pi_over_4 = static_cast<float>(3. * M_PI / 4.);
  static constexpr auto five_pi_over_4 = static_cast<float>(5. * M_PI / 4.);
  static constexpr auto seven_pi_over_4 = static_cast<float>(7. * M_PI / 4.);
  if (-pi <= angle && angle < -three_pi_over_4)
    return Direction::Left;
  if (-three_pi_over_4 <= angle && angle < -pi_over_4)
    return Direction::Up;
  if (-pi_over_4 <= angle && angle < pi_over_4)
    return Direction::Right;
  if (pi_over_4 <= angle && angle < three_pi_over_4)
    return Direction::Down;
  if (three_pi_over_4 <= angle && angle < five_pi_over_4)
    return Direction::Left;
  if (five_pi_over_4 <= angle && angle < seven_pi_over_4)
    return Direction::Up;
  else  // seven_pi_over_4 -> two_pi
    return Direction::Right;
}

inline auto direction_type(const Eigen::Vector2f& d)
{
  const auto angle = std::atan2(d.y(), d.x());
  return direction_type(angle);
}

inline auto is_good_x_corner(  //
    const std::unordered_set<int>& adjacent_edges,
    const std::vector<float>& gradient_peaks,  //
    const std::vector<float>& zero_crossings,  //
    int N) -> bool
{
  // Topological constraints from the image.
  const auto four_adjacent_edges = adjacent_edges.size() == 4;
  if (!four_adjacent_edges)
    return false;

  const auto four_zero_crossings = zero_crossings.size() == 4;
  if (four_zero_crossings)
    return true;

  // A chessboard corner should have 4 gradient orientation peaks.
  const auto four_contrast_changes = gradient_peaks.size() == 4;
  if (!four_contrast_changes)
    return false;

  // The 4 peaks are due to 2 lines crossing each other.
  static constexpr auto angle_degree_thres = 20.f;
  const auto two_crossing_lines =
      std::abs((gradient_peaks[2] - gradient_peaks[0]) * 360.f / N - 180.f) <
          angle_degree_thres &&
      std::abs((gradient_peaks[3] - gradient_peaks[1]) * 360.f / N - 180.f) <
          angle_degree_thres;
  return two_crossing_lines;
}

auto reconstruct_black_square_from_corner(
    int c,  //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::unordered_set<int>& best_corners,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>
{
  auto find_edge = [&](const std::unordered_set<int>& edges, Direction what) {
    for (const auto& e : edges)
      if (direction_type(edge_grads[e]) == what)
        return e;
    return -1;
  };

  auto find_next_square_vertex = [&](int edge, Direction what,
                                     int current_vertex) -> int {
    struct Vertex {
      int id;
      float score;
      auto operator<(const Vertex& other) const
      {
        return score < other.score;
      }
    };

    const auto& cs = corners_adjacent_to_edge[edge];
    auto vs = std::vector<Vertex>(corners_adjacent_to_edge[edge].size());
    std::transform(cs.begin(), cs.end(), vs.begin(),
                   [&corners](const auto& v) -> Vertex {
                     return {v, corners[v].score};
                   });
    std::sort(vs.rbegin(), vs.rend());

    for (const auto& v : vs)
    {
      if (v.id != current_vertex)
      {
        const auto& a = corners[current_vertex].coords;
        const auto& b = corners[v.id].coords;
        const Eigen::Vector2f dir = (b - a).normalized();
        using sara::operator""_deg;
        if (what == Direction::Up && dir.x() > 0)
          return v.id;
        if (what == Direction::Right && dir.y() > 0)
          return v.id;
        if (what == Direction::Down && dir.x() < 0)
          return v.id;
        if (what == Direction::Left && dir.y() < 0)
          return v.id;
      }
    }
    return -1;
  };

  static const auto dirs = std::array{
      Direction::Up,
      Direction::Right,
      Direction::Down,
      Direction::Left,
  };

  auto square = std::array<int, 4>{};
  std::fill(square.begin(), square.end(), -1);

  square[0] = c;

  for (auto i = 1; i <= 4; ++i)
  {
    const auto edge = find_edge(edges_adjacent_to_corner[square[i - 1]],  //
                                dirs[i - 1]);
    if (edge == -1)
      return std::nullopt;

    square[i % 4] = find_next_square_vertex(edge, dirs[i - 1], square[i - 1]);
    if (square[i % 4] == -1)
      return std::nullopt;
  }

  // I want unambiguously good square.
  if (square[0] != c)
    return std::nullopt;  // Ambiguity.

  return square;
}

auto __main(int argc, char** argv) -> int
{
  try
  {
    using sara::operator""_deg;

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
    const auto sigma_D = argc < 3 ? 1.f : std::stof(argv[2]);
    // Integration domain of the second moment.
    const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
    // Threshold parameter.
    const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
    const auto cornerness_adaptive_thres =
        argc < 6 ? 1e-5f : std::stof(argv[5]);

    // Corner filtering.
    static constexpr auto downscale_factor = 2;

    // Edge detection.
    static constexpr auto high_threshold_ratio = static_cast<float>(4._percent);
    static constexpr auto low_threshold_ratio =
        static_cast<float>(high_threshold_ratio / 2.);
    static constexpr auto angular_threshold =
        static_cast<float>((10._deg).value);
    auto ed = sara::EdgeDetector{{
        high_threshold_ratio,  //
        low_threshold_ratio,   //
        angular_threshold      //
    }};

    // Circular profile extractor.
    auto profile_extractor = CircularProfileExtractor{};
    profile_extractor.circle_radius = 2 * sigma_I;

    auto video_stream = sara::VideoStream{video_file};
    auto video_frame = video_stream.frame();
    auto frame_number = -1;

    auto frame_gray = sara::Image<float>{video_frame.sizes()};
    auto frame_gray_blurred = sara::Image<float>{video_frame.sizes()};
    auto frame_gray_ds =
        sara::Image<float>{video_frame.sizes() / downscale_factor};
    auto grad_norm = sara::Image<float>{video_frame.sizes() / downscale_factor};
    auto grad_ori = sara::Image<float>{video_frame.sizes() / downscale_factor};
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
      const auto f_ds_blurred = frame_gray_ds.compute<sara::Gaussian>(sigma_D);
      sara::toc("Blur");

      sara::tic();
      ed(f_ds_blurred);
      sara::toc("Curve detection");

      sara::tic();
      auto grad_x = sara::Image<float>{f_ds_blurred.sizes()};
      auto grad_y = sara::Image<float>{f_ds_blurred.sizes()};
      sara::gradient(f_ds_blurred, grad_x, grad_y);
      const auto cornerness = sara::harris_cornerness(grad_x, grad_y,  //
                                                      sigma_I, kappa);
      static const auto border = 2 * static_cast<int>(std::round(sigma_I));
      auto corners_int = select(cornerness, cornerness_adaptive_thres, border);
      sara::toc("Corner detection");

      sara::tic();
      auto corners = std::vector<Corner<float>>{};
      std::transform(
          corners_int.begin(), corners_int.end(), std::back_inserter(corners),
          [&grad_x, &grad_y, sigma_I](const Corner<int>& c) -> Corner<float> {
            static const auto radius =
                2 * static_cast<int>(std::round(sigma_I));
            const auto p = sara::refine_junction_location_unsafe(
                grad_x, grad_y, c.coords, radius);
            return {p, c.score};
          });
      sara::nms(corners, cornerness.sizes(), border);
      sara::toc("Corner refinement");

      sara::tic();
      auto profiles = std::vector<Eigen::ArrayXf>{};
      profiles.resize(corners.size());
      auto zero_crossings = std::vector<std::vector<float>>{};
      zero_crossings.resize(corners.size());
      for (auto c = 0u; c < corners.size(); ++c)
      {
        const auto& p = corners[c].coords;
        const auto& r = profile_extractor.circle_radius;
        const auto w = f_ds_blurred.width();
        const auto h = f_ds_blurred.height();
        if (!(r + 1 <= p.x() && p.x() < w - r - 1 &&  //
              r + 1 <= p.y() && p.y() < h - r - 1))
          continue;
        profiles[c] = profile_extractor(f_ds_blurred,  //
                                        corners[c].coords.cast<double>());
        zero_crossings[c] = localize_zero_crossings(
            profiles[c], profile_extractor.num_circle_sample_points);
      }
      sara::toc("Circular profile");

      sara::tic();
      const auto& grad_norm = ed.pipeline.gradient_magnitude;
      const auto& grad_ori = ed.pipeline.gradient_orientation;

      static constexpr auto N = 72;
      auto hists = std::vector<Eigen::Array<float, N, 1>>{};
      hists.resize(corners.size());
      const auto num_corners = static_cast<int>(corners.size());
#pragma omp parallel for
      for (auto i = 0; i < num_corners; ++i)
      {
        compute_orientation_histogram<N>(hists[i], grad_norm, grad_ori,
                                         corners[i].coords.x(),
                                         corners[i].coords.y(),  //
                                         sigma_D, 4, 5.0f);
        sara::lowe_smooth_histogram(hists[i]);
        hists[i].matrix().normalize();
      };
      sara::toc("Gradient histograms");


      sara::tic();
      auto gradient_peaks = std::vector<std::vector<int>>{};
      gradient_peaks.resize(hists.size());
      std::transform(hists.begin(), hists.end(), gradient_peaks.begin(),
                     [](const auto& h) { return sara::find_peaks(h, 0.3f); });
      auto gradient_peaks_refined = std::vector<std::vector<float>>{};
      gradient_peaks_refined.resize(gradient_peaks.size());
      std::transform(gradient_peaks.begin(), gradient_peaks.end(),
                     hists.begin(), gradient_peaks_refined.begin(),
                     [](const auto& peaks, const auto& hist) {
                       auto peaks_ref = std::vector<float>{};
                       std::transform(peaks.begin(), peaks.end(),
                                      std::back_inserter(peaks_ref),
                                      [&hist](const auto& i) {
                                        return sara::refine_peak(hist, i);
                                      });
                       return peaks_ref;
                     });
      sara::toc("Gradient Dominant Orientations");

      sara::tic();
      auto edge_label_map = sara::Image<int>{ed.pipeline.edge_map.sizes()};
      edge_label_map.flat_array().fill(-1);
      const auto& edges = ed.pipeline.edges_simplified;
      for (auto edge_id = 0u; edge_id < edges.size(); ++edge_id)
      {
        const auto& curve = edges[edge_id];
        if (curve.size() < 2)
          continue;
        edge_label_map(curve.front().array().round().matrix().cast<int>()) =
            edge_id;
        edge_label_map(curve.back().array().round().matrix().cast<int>()) =
            edge_id;
      }

      auto edges_adjacent_to_corner = std::vector<std::unordered_set<int>>{};
      edges_adjacent_to_corner.resize(corners.size());
      std::transform(                        //
          corners.begin(), corners.end(),    //
          edges_adjacent_to_corner.begin(),  //
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

                const auto edge_id = edge_label_map(p);
                if (edge_id != -1)
                  edges.insert(edge_id);
              }
            }
            return edges;
          });

      auto corners_adjacent_to_edge = std::vector<std::unordered_set<int>>{};
      corners_adjacent_to_edge.resize(edges.size());
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto& corner = corners[c];

        static constexpr auto r = 4;
        for (auto v = -r; v <= r; ++v)
        {
          for (auto u = -r; u <= r; ++u)
          {
            const Eigen::Vector2i p =
                corner.coords.cast<int>() + Eigen::Vector2i{u, v};

            const auto in_image_domain =
                0 <= p.x() && p.x() < edge_label_map.width() &&  //
                0 <= p.y() && p.y() < edge_label_map.height();
            if (!in_image_domain)
              continue;

            const auto edge_id = edge_label_map(p);
            if (edge_id != -1)
              corners_adjacent_to_edge[edge_id].insert(c);
          }
        }
      }

      auto edges_adjacent_to_edge = std::vector<std::unordered_set<int>>{};
      edges_adjacent_to_edge.resize(edges.size());
      for (const auto& edge_ids : edges_adjacent_to_corner)
        for (const auto& ei : edge_ids)
          for (const auto& ej : edge_ids)
            if (ei != ej)
              edges_adjacent_to_edge[ei].insert(ej);
      sara::toc("Topological Linking");

      sara::tic();
      const auto edge_stats = get_curve_shape_statistics(  //
          ed.pipeline.edges_as_list);
      const auto edge_grads = mean_gradient(  //
          ed.pipeline.edges_as_list,          //
          grad_x, grad_y);
      sara::toc("Edge Shape Stats");

      sara::tic();
      auto edge_dirs = std::vector<std::unordered_map<int, Direction>>(  //
          edge_grads.size());
      for (auto c = 0u; c < corners.size(); ++c)
      {
        const auto& edge_ids = edges_adjacent_to_corner[c];
        for (const auto& edge_id : edge_ids)
        {
          const auto& g = edge_grads[edge_id];
          edge_dirs[c][edge_id] = direction_type(g);
        }
      }
      sara::toc("Edge Direction");


      sara::tic();
      auto best_corners = std::unordered_set<int>{};
      for (auto c = 0; c < num_corners; ++c)
        if (is_good_x_corner(edges_adjacent_to_corner[c],
                             gradient_peaks_refined[c], zero_crossings[c], N))
          best_corners.insert(c);
      sara::toc("Best corner selection");

      sara::tic();
#if 0
      auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
      display.flat_array().fill(sara::Black8);
#else
      auto display = frame_gray.convert<sara::Rgb8>();
#endif

#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto& p = corners[c];
        const auto good = is_good_x_corner(edges_adjacent_to_corner[c],  //
                                           gradient_peaks_refined[c],    //
                                           zero_crossings[c],            //
                                           N);

        if (good)
        {
          const auto& edges = edges_adjacent_to_corner[c];
          for (const auto& edge_id : edges)
          {
            const auto color = sara::Cyan8;
            const auto& curve = ed.pipeline.edges_as_list[edge_id];
            const auto& box = edge_stats.oriented_box(edge_id);
            const auto thickness = box.lengths(1) / box.lengths(0);
            const auto is_thick = thickness > 0.1 && box.lengths(1) > 2.;

            for (const auto& p : curve)
            {
              const Eigen::Vector2i q = p * downscale_factor;
              display(q.x(), q.y()) = color;
              display(q.x() + 1, q.y()) = color;
              display(q.x() + 1, q.y() + 1) = color;
              display(q.x(), q.y() + 1) = color;
            }

            const auto& dir_color = sara::Magenta8;
            box.draw(display, is_thick ? sara::Red8 : dir_color,
                     Eigen::Vector2d::Zero(), downscale_factor);
          }
        }

        sara::fill_circle(
            display,
            static_cast<int>(std::round(downscale_factor * p.coords.x())),
            static_cast<int>(std::round(downscale_factor * p.coords.y())), 1,
            sara::Yellow8);
        sara::draw_circle(
            display,
            static_cast<int>(std::round(downscale_factor * p.coords.x())),
            static_cast<int>(std::round(downscale_factor * p.coords.y())), 4,
            good ? sara::Red8 : sara::Blue8, 2);
      }
      sara::draw_text(display, 80, 80, std::to_string(frame_number),
                      sara::White8, 60, 0, false, true);
      sara::display(display);
      sara::toc("Display");

      sara::tic();
      for (const auto& c : best_corners)
      {
        const auto square = reconstruct_black_square_from_corner(
            c, corners, edge_grads, best_corners, edges_adjacent_to_corner,
            corners_adjacent_to_edge);
        if (square == std::nullopt)
          continue;

        for (auto i = 0; i < 4; ++i)
        {
          const Eigen::Vector2f a =
              corners[(*square)[i]].coords * downscale_factor;
          const Eigen::Vector2f b =
              corners[(*square)[(i + 1) % 4]].coords * downscale_factor;
          sara::draw_line(display, a, b, sara::Green8, 6);
        }
      }
      sara::display(display);
      sara::toc("Black Square Reconstruction");

      sara::get_key();
    }
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
