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

#include <set>
#include <unordered_map>
#include <unordered_set>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include "Utilities/ImageOrVideoReader.hpp"

#include "Chessboard/CircularProfileExtractor.hpp"
#include "Chessboard/Corner.hpp"
#include "Chessboard/EdgeStatistics.hpp"
#include "Chessboard/LineReconstruction.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"
#include "Chessboard/OrientationHistogram.hpp"
#include "Chessboard/SquareReconstruction.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

//  Seed corner selection.
inline auto is_good_x_corner(const std::vector<float>& zero_crossings) -> bool
{
  const auto four_zero_crossings = zero_crossings.size() == 4;
  return four_zero_crossings;
#if 0
  if (!four_zero_crossings)
    return false;

  auto dirs = Eigen::Matrix<float, 2, 4>{};
  for (auto i = 0; i < 4; ++i)
    dirs.col(i) = dir(zero_crossings[i]);

  // The 4 peaks are due to 2 lines crossing each other.
  using sara::operator""_deg;
  static constexpr auto angle_thres = static_cast<float>((160._deg).value);

  const auto two_crossing_lines =
      dirs.col(0).dot(dirs.col(2)) < std::cos(angle_thres) &&
      dirs.col(1).dot(dirs.col(3)) < std::cos(angle_thres);

  return two_crossing_lines;
#endif
}


//  Seed corner selection.
inline auto is_seed_corner(  //
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
    const auto downscale_factor = argc < 7 ? 2.f : std::stof(argv[6]);

    // Edge detection.
    const auto high_threshold_ratio = argc < 8
                                          ? static_cast<float>(4._percent)  //
                                          : std::stof(argv[7]);
    const auto low_threshold_ratio =
        static_cast<float>(high_threshold_ratio / 2.);
    static constexpr auto angular_threshold =
        static_cast<float>((10._deg).value);
    auto ed = sara::EdgeDetector{{
        high_threshold_ratio,  //
        low_threshold_ratio,   //
        angular_threshold      //
    }};

    // Visual inspection option
    const auto pause = argc < 9 ? false : static_cast<bool>(std::stoi(argv[8]));
    const auto check_edge_map = argc < 10
                                    ? false  //
                                    : static_cast<bool>(std::stoi(argv[9]));

    // 4 is kind of a magic number, but the idea is that we assume that
    // x-corners on the chessboard are separated by 10 pixels at the minimum.
    //
    // This is important when we analyze the circular intensity profile for each
    // corner because this will decide whether we filter a corner out.
    const auto image_border =
        argc < 11 ? static_cast<int>(std::round(4 * sigma_I / downscale_factor))
                  : std::stoi(argv[10]);
    const auto& radius = image_border;

    // Circular profile extractor.
    //
    // This is simple and works really well.
    auto profile_extractor = CircularProfileExtractor{};
    profile_extractor.circle_radius = radius;

    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto frame_number = -1;

    auto frame_gray = sara::Image<float>{video_frame.sizes()};
    auto frame_gray_blurred = sara::Image<float>{video_frame.sizes()};

    const Eigen::Vector2i image_ds_sizes =
        (frame_gray.sizes().cast<float>() / downscale_factor)
            .array()
            .round()
            .matrix()
            .cast<int>();
    auto frame_gray_ds = sara::Image<float>{image_ds_sizes};
    auto grad_norm = sara::Image<float>{image_ds_sizes};
    auto grad_ori = sara::Image<float>{image_ds_sizes};
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};

    auto timer = sara::Timer{};

    while (video_stream.read())
    {
      ++frame_number;
      if (frame_number % 3 != 0)
        continue;
      SARA_DEBUG << "Frame #" << frame_number << std::endl;

      if (sara::active_window() == nullptr)
      {
        sara::create_window(video_frame.sizes(), video_file);
        sara::set_antialiasing();
      }

      timer.restart();

      sara::tic();
      sara::from_rgb8_to_gray32f(video_frame, frame_gray);
      sara::apply_gaussian_filter(frame_gray, frame_gray_blurred,
                                  downscale_factor);
      sara::toc("Grayscale conversion");

      sara::tic();
      sara::resize_v2(frame_gray_blurred, frame_gray_ds);
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
      auto corners_int = select(cornerness, cornerness_adaptive_thres,  //
                                image_border);
      sara::toc("Corner detection");

      sara::tic();
      auto corners = std::vector<Corner<float>>{};
      std::transform(
          corners_int.begin(), corners_int.end(), std::back_inserter(corners),
          [&grad_x, &grad_y, radius](const Corner<int>& c) -> Corner<float> {
            const auto p = sara::refine_junction_location_unsafe(
                grad_x, grad_y, c.coords, radius);
            return {p, c.score};
          });
      sara::nms(corners, cornerness.sizes(), radius);
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
      sara::toc("Circular intensity profile");

      sara::tic();
      {
        auto corners_filtered = std::vector<Corner<float>>{};
        auto profiles_filtered = std::vector<Eigen::ArrayXf>{};
        auto zero_crossings_filtered = std::vector<std::vector<float>>{};

        for (auto c = 0u; c < corners.size(); ++c)
        {
          if (is_good_x_corner(zero_crossings[c]))
          {
            corners_filtered.emplace_back(corners[c]);
            profiles_filtered.emplace_back(profiles[c]);
            zero_crossings_filtered.emplace_back(zero_crossings[c]);
          }
        }

        corners_filtered.swap(corners);
        profiles_filtered.swap(profiles);
        zero_crossings_filtered.swap(zero_crossings);
      }
      sara::toc("Filtering from intensity profile");

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
      const auto edge_stats = get_curve_shape_statistics(  //
          ed.pipeline.edges_as_list);
      const auto edge_grad_means = gradient_mean(  //
          ed.pipeline.edges_as_list,               //
          grad_x, grad_y);
      const auto edge_grad_covs = gradient_covariance(  //
          ed.pipeline.edges_as_list,                    //
          grad_x, grad_y);
      sara::toc("Edge Shape Stats");


      sara::tic();
      auto edge_label_map = sara::Image<int>{ed.pipeline.edge_map.sizes()};
      edge_label_map.flat_array().fill(-1);
#if 0
      const auto& edges = ed.pipeline.edges_simplified;
#else
      const auto& edges = ed.pipeline.edges_as_list;
#endif
      for (auto edge_id = 0u; edge_id < edges.size(); ++edge_id)
      {
        const auto& curvei = edges[edge_id];
        const auto& edgei = sara::reorder_and_extract_longest_curve(curvei);
        auto curve = std::vector<Eigen::Vector2d>(edgei.size());
        std::transform(edgei.begin(), edgei.end(), curve.begin(),
                       [](const auto& p) { return p.template cast<double>(); });
        if (curve.size() < 2)
          continue;

        const Eigen::Vector2i s =
            curve.front().array().round().matrix().cast<int>();
        const Eigen::Vector2i e =
            curve.back().array().round().matrix().cast<int>();

        // Ignore weak edges, they make the edge map less interpretable.
        if (ed.pipeline.edge_map(s) == 127 || ed.pipeline.edge_map(e) == 127)
          continue;

        // Ignore small edges.
        const auto& curve_simplified = ed.pipeline.edges_simplified[edge_id];
        if (curve_simplified.size() < 2 ||
            sara::length(curve_simplified) < radius)
          continue;

        // Edge gradient distribution similar to cornerness measure.
        const auto& grad_cov = edge_grad_covs[edge_id];
        const auto grad_dist_param = 0.2f;
        const auto cornerness =
            grad_cov.determinant() -  //
            grad_dist_param * sara::square(grad_cov.trace());
        if (cornerness > 0)
          continue;

        edge_label_map(s) = edge_id;
        edge_label_map(e) = edge_id;
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
      sara::toc("Topological Linking");


      sara::tic();
      auto best_corners = std::unordered_set<int>{};
      for (auto c = 0; c < num_corners; ++c)
        if (is_seed_corner(edges_adjacent_to_corner[c],
                           gradient_peaks_refined[c], zero_crossings[c], N))
          best_corners.insert(c);
      sara::toc("Best corner selection");

      sara::tic();
      using Square = std::array<int, 4>;
      const auto compare_square = [](const Square& a, const Square& b) {
        return std::lexicographical_compare(a.begin(), a.end(),  //
                                            b.begin(), b.end());
      };
      using SquareSet = std::set<Square, decltype(compare_square)>;

      auto black_squares = SquareSet{compare_square};
      for (const auto& c : best_corners)
      {
        const auto square = reconstruct_black_square_from_corner(
            c, corners, edge_grad_means, edges_adjacent_to_corner,
            corners_adjacent_to_edge);
        if (square == std::nullopt)
          continue;
        black_squares.insert(*square);
      }
      sara::toc("Black square reconstruction");

      sara::tic();
      auto white_squares = SquareSet{compare_square};
      for (const auto& c : best_corners)
      {
        const auto square = reconstruct_white_square_from_corner(
            c, corners, edge_grad_means, edges_adjacent_to_corner,
            corners_adjacent_to_edge);
        if (square == std::nullopt)
          continue;

        white_squares.insert(*square);
      }
      sara::toc("White square reconstruction");

      sara::tic();
      auto lines = std::vector<std::vector<int>>{};
      for (const auto& square : black_squares)
      {
        const auto new_lines = grow_lines_from_square(
            square, corners, edge_stats, edge_grad_means, edge_grad_covs,
            edges_adjacent_to_corner, corners_adjacent_to_edge);

        lines.insert(lines.end(), new_lines.begin(), new_lines.end());
      }
      for (const auto& square : white_squares)
      {
        const auto new_lines = grow_lines_from_square(
            square, corners, edge_stats, edge_grad_means, edge_grad_covs,
            edges_adjacent_to_corner, corners_adjacent_to_edge);

        lines.insert(lines.end(), new_lines.begin(), new_lines.end());
      }
      sara::toc("Line Reconstruction");


      const auto pipeline_time = timer.elapsed_ms();
      SARA_DEBUG << "Processing time = " << pipeline_time << "ms" << std::endl;


      sara::tic();
      auto display = sara::Image<sara::Rgb8>{};
      if (check_edge_map)
      {
        // Resize
        auto display_32f_ds = ed.pipeline.edge_map.convert<float>();
        auto display_32f = sara::Image<float>{video_frame.sizes()};
        sara::scale(display_32f_ds, display_32f);

        display = display_32f.convert<sara::Rgb8>();
      }
      else
        display = frame_gray.convert<sara::Rgb8>();

// #define INVESTIGATE_X_CORNER_HISTOGRAMS
#ifndef INVESTIGATE_X_CORNER_HISTOGRAMS
#  pragma omp parallel for
#endif
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto& p = corners[c];
        const auto good = is_seed_corner(edges_adjacent_to_corner[c],  //
                                         gradient_peaks_refined[c],    //
                                         zero_crossings[c],            //
                                         N);

        // Remove noisy corners to understand better the behaviour of the
        // algorithm.
        if (edges_adjacent_to_corner[c].empty())
          continue;

#ifdef INVESTIGATE_X_CORNER_HISTOGRAMS
        if (good)
        {
          SARA_DEBUG << "[GOOD] gradient ori peaks[" << c << "]\n"
                     << Eigen::Map<const Eigen::ArrayXf>(
                            gradient_peaks_refined[c].data(),
                            gradient_peaks_refined[c].size()) *
                            360.f / N
                     << std::endl;
          SARA_DEBUG << "[GOOD] zero crossings[" << c << "]\n"
                     << Eigen::Map<const Eigen::ArrayXf>(
                            zero_crossings[c].data(),
                            zero_crossings[c].size()) *
                            180.f / static_cast<float>(M_PI)
                     << std::endl;
        }
        else
        {
          SARA_DEBUG << "[BAD] gradient ori peaks[" << c << "]\n"
                     << Eigen::Map<const Eigen::ArrayXf>(
                            gradient_peaks_refined[c].data(),
                            gradient_peaks_refined[c].size()) *
                            360.f / N
                     << std::endl;
          SARA_DEBUG << "[BAD] zero crossings[" << c << "]\n"
                     << Eigen::Map<const Eigen::ArrayXf>(
                            zero_crossings[c].data(),
                            zero_crossings[c].size()) *
                            180.f / static_cast<float>(M_PI)
                     << std::endl;
        }
#endif

#ifdef INSPECT_EDGE_GEOMETRY
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
#endif

        sara::fill_circle(
            display,
            static_cast<int>(std::round(downscale_factor * p.coords.x())),
            static_cast<int>(std::round(downscale_factor * p.coords.y())), 1,
            sara::Yellow8);
        sara::draw_circle(
            display,
            static_cast<int>(std::round(downscale_factor * p.coords.x())),
            static_cast<int>(std::round(downscale_factor * p.coords.y())),
            radius * downscale_factor, good ? sara::Red8 : sara::Cyan8, 2);

#ifdef INVESTIGATE_X_CORNER_HISTOGRAMS
        sara::display(display);
        sara::get_key();
#endif
      }
      sara::draw_text(display, 80, 80, std::to_string(frame_number),
                      sara::White8, 60, 0, false, true);

      for (const auto& line : lines)
      {
        for (auto i = 0u; i < line.size() - 1; ++i)
        {
          const Eigen::Vector2f a = corners[line[i]].coords * downscale_factor;
          const Eigen::Vector2f b =
              corners[line[i + 1]].coords * downscale_factor;
          sara::draw_line(display, a, b, sara::Cyan8, 1);
        }
      }

      const auto draw_square = [&corners, downscale_factor,
                                &display](const auto& square,  //
                                          const auto& color,   //
                                          const int thickness) {
        for (auto i = 0; i < 4; ++i)
        {
          const Eigen::Vector2f a =
              corners[square[i]].coords * downscale_factor;
          const Eigen::Vector2f b =
              corners[square[(i + 1) % 4]].coords * downscale_factor;
          sara::draw_line(display, a, b, color, thickness);
        }
      };

      for (const auto& square : white_squares)
        draw_square(square, sara::Red8, 3);

      for (const auto& square : black_squares)
        draw_square(square, sara::Green8, 2);

      sara::display(display);
      sara::toc("Display");

      if (pause)
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
