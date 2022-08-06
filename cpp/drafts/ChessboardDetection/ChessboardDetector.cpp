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

#include "ChessboardDetector.hpp"
#include "EdgeStatistics.hpp"
#include "LineReconstruction.hpp"
#include "NonMaximumSuppression.hpp"
#include "OrientationHistogram.hpp"
#include "SquareReconstruction.hpp"

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/FeatureDescriptors/Orientation.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/Harris.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>


namespace DO::Sara {

  // Strong edge filter.
  static inline auto is_strong_edge(const ImageView<float>& grad_mag,
                                    const std::vector<Eigen::Vector2i>& edge,
                                    const float grad_thres) -> float
  {
    if (edge.empty())
      return 0.f;
    const auto mean_edge_gradient =
        std::accumulate(
            edge.begin(), edge.end(), 0.f,
            [&grad_mag](const float& grad, const Eigen::Vector2i& p) -> float {
              return grad + grad_mag(p);
            }) /
        edge.size();
    return mean_edge_gradient > grad_thres;
  }

  // Seed corner selection.
  static inline auto is_good_x_corner(const std::vector<float>& zero_crossings)
      -> bool
  {
    const auto four_zero_crossings = zero_crossings.size() == 4;
#if 0
    if (!four_zero_crossings)
      return false;

    auto dirs = Eigen::Matrix<float, 2, 4>{};
    for (auto i = 0; i < 4; ++i)
      dirs.col(i) = dir(zero_crossings[i]);

    // The 4 peaks are due to 2 lines crossing each other.
    using operator""_deg;
    static constexpr auto angle_thres = static_cast<float>((160._deg).value);

    const auto two_crossing_lines =
        dirs.col(0).dot(dirs.col(2)) < std::cos(angle_thres) &&
        dirs.col(1).dot(dirs.col(3)) < std::cos(angle_thres);

    return two_crossing_lines;
#else
    return four_zero_crossings;
#endif
  }

  // Seed corner selection.
  auto is_seed_corner(  //
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

#if 0
    auto dirs = Eigen::Matrix<float, 2, 4>{};
    for (auto i = 0; i < 4; ++i)
      dirs.col(i) = dir(zero_crossings[i]);

    // The 4 peaks are due to 2 lines crossing each other.
    static constexpr auto angle_thres = static_cast<float>((160._deg).value);

    const auto two_crossing_lines =
        dirs.col(0).dot(dirs.col(2)) < std::cos(angle_thres) &&
        dirs.col(1).dot(dirs.col(3)) < std::cos(angle_thres);

    return two_crossing_lines;
#else
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
#endif
  }


  auto ChessboardDetector::operator()(const ImageView<float>& image)
      -> const std::vector<OrderedChessboardCorners>&
  {
    preprocess_image(image);
    filter_edges();

    detect_corners();
    calculate_circular_intensity_profiles();
    filter_corners_with_intensity_profiles();

    calculate_orientation_histograms();
    calculate_edge_shape_statistics();

    link_edges_and_corners();

    select_seed_corners();
    parse_squares();
    grow_chessboards();
    extract_chessboard_corners_from_chessboard_squares();

    // TODO:
    // parse_lines();

    return _cb_corners;
  }

  auto ChessboardDetector::preprocess_image(const ImageView<float>& image)
      -> void
  {
    tic();

    // First blur the original image to reduce anti-aliasing.
    if (_f_blurred.sizes() != image.sizes())
      _f_blurred.resize(image.sizes());
    apply_gaussian_filter(image, _f_blurred, _params.downscale_factor);

    // Now we can downscale the image.
    const Eigen::Vector2i image_ds_sizes =
        (image.sizes().cast<float>() / _params.downscale_factor)
            .array()
            .round()
            .matrix()
            .cast<int>();
    if (_f_ds.sizes() != image_ds_sizes)
      _f_ds.resize(image_ds_sizes);
    resize_v2(_f_blurred, _f_ds);

    toc("Rescale");
  }

  auto ChessboardDetector::filter_edges() -> void
  {
    const auto& image_ds_sizes = _f_ds.sizes();

    // Blur the image again before calculating the gradient.
    tic();
    if (_f_ds_blurred.sizes() != image_ds_sizes)
      _f_ds_blurred.resize(image_ds_sizes);
    apply_gaussian_filter(_f_ds, _f_ds_blurred, _params.sigma_D);
    toc("Blur");

    tic();
    if (_grad_x.sizes() != image_ds_sizes)
      _grad_x.resize(image_ds_sizes);
    if (_grad_y.sizes() != image_ds_sizes)
      _grad_y.resize(image_ds_sizes);
    gradient(_f_ds_blurred, _grad_x, _grad_y);
    toc("Gradient Cartesian Coordinate");

    // No need to profile this, it is done internally...
    _ed.operator()(_grad_x, _grad_y);
  }

  auto ChessboardDetector::detect_corners() -> void
  {
    const auto& image_ds_sizes = _f_ds.sizes();
    tic();
    if (_cornerness.sizes() != image_ds_sizes)
      _cornerness.resize(image_ds_sizes);
    _cornerness = harris_cornerness(_grad_x, _grad_y,  //
                                    _params.sigma_I, _params.kappa);
    toc("Cornerness map");

    tic();
    _corners_int = select(_cornerness, _params.cornerness_adaptive_thres,  //
                          _params.corner_filtering_radius);
    toc("Corner selection");

    tic();
    _corners.clear();
    std::transform(
        _corners_int.begin(), _corners_int.end(), std::back_inserter(_corners),
        [this](const Corner<int>& c) -> Corner<float> {
          const auto p = refine_junction_location_unsafe(
              _grad_x, _grad_y, c.coords, _params.corner_filtering_radius);
          return {p, c.score};
        });
    nms(_corners, _cornerness.sizes(), _params.corner_filtering_radius);
    toc("Corner refinement");
  }

  auto ChessboardDetector::calculate_circular_intensity_profiles() -> void
  {
    tic();
    _profiles.clear();
    _zero_crossings.clear();
    _profiles.resize(_corners.size());
    _zero_crossings.resize(_corners.size());
    auto num_corners = static_cast<int>(_corners.size());
#pragma omp parallel for
    for (auto c = 0; c < num_corners; ++c)
    {
      const auto& p = _corners[c].coords;
      const auto& r = _profile_extractor.circle_radius;
      const auto w = _f_ds_blurred.width();
      const auto h = _f_ds_blurred.height();
      if (!(r + 1 <= p.x() && p.x() < w - r - 1 &&  //
            r + 1 <= p.y() && p.y() < h - r - 1))
        continue;
      _profiles[c] = _profile_extractor(_f_ds_blurred,  //
                                        _corners[c].coords.cast<double>());
      _zero_crossings[c] = localize_zero_crossings(
          _profiles[c], _profile_extractor.num_circle_sample_points);
    }
    toc("Circular intensity profile");
  }

  auto ChessboardDetector::filter_corners_with_intensity_profiles() -> void
  {
    tic();
    {
      auto corners_filtered = std::vector<Corner<float>>{};
      auto profiles_filtered = std::vector<Eigen::ArrayXf>{};
      auto zero_crossings_filtered = std::vector<std::vector<float>>{};

      for (auto c = 0u; c < _corners.size(); ++c)
      {
        if (is_good_x_corner(_zero_crossings[c]))
        {
          corners_filtered.emplace_back(_corners[c]);
          profiles_filtered.emplace_back(_profiles[c]);
          zero_crossings_filtered.emplace_back(_zero_crossings[c]);
        }
      }

      corners_filtered.swap(_corners);
      profiles_filtered.swap(_profiles);
      zero_crossings_filtered.swap(_zero_crossings);
    }
    toc("Corner filtering from intensity profile");
  }

  auto ChessboardDetector::calculate_orientation_histograms() -> void
  {
    tic();
    const auto& _grad_norm = _ed.pipeline.gradient_magnitude;
    const auto& _grad_ori = _ed.pipeline.gradient_orientation;

    _hists.resize(_corners.size());

    const auto num_corners = static_cast<int>(_corners.size());
#pragma omp parallel for
    for (auto i = 0; i < num_corners; ++i)
    {
      compute_orientation_histogram<N>(_hists[i], _grad_norm, _grad_ori,
                                       _corners[i].coords.x(),
                                       _corners[i].coords.y(),  //
                                       _params.sigma_D, 4, 5.0f);
      lowe_smooth_histogram(_hists[i]);
      _hists[i].matrix().normalize();
    };
    toc("Gradient histograms");
  }

  auto ChessboardDetector::calculate_edge_shape_statistics() -> void
  {
    tic();
    _edge_stats = get_curve_shape_statistics(_ed.pipeline.edges_as_list);
    _edge_grad_means = gradient_mean(_ed.pipeline.edges_as_list,  //
                                     _grad_x, _grad_y);
    _edge_grad_covs = gradient_covariance(_ed.pipeline.edges_as_list,  //
                                          _grad_x, _grad_y);
    toc("Edge shape statistics");
  }

  auto ChessboardDetector::link_edges_and_corners() -> void
  {
    tic();

    _edge_label_map.resize(_ed.pipeline.edge_map.sizes());
    _edge_label_map.flat_array().fill(-1);

#ifdef DEBUG_FILTERED_EDGES
    auto edge_map_filtered =
        sara::Image<std::uint8_t>{ed.pipeline.edge_map.sizes()};
    edge_map_filtered.flat_array().fill(0);
#endif

    const auto& edges = _ed.pipeline.edges_as_list;
    for (auto edge_id = 0u; edge_id < edges.size(); ++edge_id)
    {
      const auto& curvei = edges[edge_id];
      const auto& edgei = reorder_and_extract_longest_curve(curvei);
      auto curve = std::vector<Eigen::Vector2d>(edgei.size());
      std::transform(edgei.begin(), edgei.end(), curve.begin(),
                     [](const auto& p) { return p.template cast<double>(); });
      if (curve.size() < 2)
        continue;

      const Eigen::Vector2i s =
          curve.front().array().round().matrix().cast<int>();
      const Eigen::Vector2i e =
          curve.back().array().round().matrix().cast<int>();

      static constexpr auto strong_edge_thres = 2.f / 255.f;
      if (!is_strong_edge(_ed.pipeline.gradient_magnitude, curvei,
                          strong_edge_thres))
        continue;

#ifdef DEBUG_FILTERED_EDGES
      for (const auto& p : curvei)
        edge_map_filtered(p) = 255;
#endif

      // Edge gradient distribution similar to cornerness measure.
      const auto& grad_cov = _edge_grad_covs[edge_id];
      const auto grad_dist_param = 0.2f;
      const auto cornerness = grad_cov.determinant() -  //
                              grad_dist_param * square(grad_cov.trace());
      if (cornerness > 0)
        continue;

      _edge_label_map(s) = edge_id;
      _edge_label_map(e) = edge_id;
    }

#ifdef DEBUG_FILTERED_EDGES
    sara::display(edge_map_filtered);
    sara::get_key();
#endif

    _edges_adjacent_to_corner.clear();
    _edges_adjacent_to_corner.resize(_corners.size());
    std::transform(                         //
        _corners.begin(), _corners.end(),   //
        _edges_adjacent_to_corner.begin(),  //
        [this](const Corner<float>& c) {
          auto edges = std::unordered_set<int>{};

          const auto& r = _params.corner_edge_linking_radius;
          for (auto v = -r; v <= r; ++v)
          {
            for (auto u = -r; u <= r; ++u)
            {
              const Eigen::Vector2i p =
                  c.coords.cast<int>() + Eigen::Vector2i{u, v};

              const auto in_image_domain =
                  0 <= p.x() && p.x() < _edge_label_map.width() &&  //
                  0 <= p.y() && p.y() < _edge_label_map.height();
              if (!in_image_domain)
                continue;

              const auto edge_id = _edge_label_map(p);
              if (edge_id != -1)
                edges.insert(edge_id);
            }
          }
          return edges;
        });

    const auto num_corners = static_cast<int>(_corners.size());

    const auto& _edges = _ed.pipeline.edges_as_list;
    _corners_adjacent_to_edge.clear();
    _corners_adjacent_to_edge.resize(_edges.size());
    for (auto c = 0; c < num_corners; ++c)
    {
      const auto& corner = _corners[c];

      const auto& r = _params.corner_edge_linking_radius;
      for (auto v = -r; v <= r; ++v)
      {
        for (auto u = -r; u <= r; ++u)
        {
          const Eigen::Vector2i p =
              corner.coords.cast<int>() + Eigen::Vector2i{u, v};

          const auto in_image_domain =
              0 <= p.x() && p.x() < _edge_label_map.width() &&  //
              0 <= p.y() && p.y() < _edge_label_map.height();
          if (!in_image_domain)
            continue;

          const auto edge_id = _edge_label_map(p);
          if (edge_id != -1)
            _corners_adjacent_to_edge[edge_id].insert(c);
        }
      }
    }
    toc("Topological linking");
  }

  auto ChessboardDetector::select_seed_corners() -> void
  {
    tic();
    _best_corners.clear();
    for (auto c = 0u; c < _corners.size(); ++c)
      if (is_seed_corner(_edges_adjacent_to_corner[c],
                         _gradient_peaks_refined[c], _zero_crossings[c], N))
        _best_corners.insert(c);
    toc("Best corner selection");
  }

  auto ChessboardDetector::parse_squares() -> void
  {
    tic();
    _black_squares.clear();
    for (const auto& c : _best_corners)
    {
      const auto square = reconstruct_black_square_from_corner(
          c, _corners, _edge_grad_means, _edge_grad_covs,
          _edges_adjacent_to_corner, _corners_adjacent_to_edge);
      if (square == std::nullopt)
        continue;
      _black_squares.insert({*square, Square::Type::Black});
    }
    toc("Black square reconstruction");

    tic();
    _white_squares.clear();
    for (const auto& c : _best_corners)
    {
      const auto square = reconstruct_white_square_from_corner(
          c, _corners, _edge_grad_means, _edge_grad_covs,
          _edges_adjacent_to_corner, _corners_adjacent_to_edge);
      if (square == std::nullopt)
        continue;

      _white_squares.insert({*square, Square::Type::White});
    }
    toc("White square reconstruction");
  }

  auto ChessboardDetector::parse_lines() -> void
  {
    tic();

    _lines.clear();

    for (const auto& square : _black_squares)
    {
      const auto new_lines = grow_lines_from_square(
          square.v, _corners, _edge_stats, _edge_grad_means, _edge_grad_covs,
          _edges_adjacent_to_corner, _corners_adjacent_to_edge);

      _lines.insert(_lines.end(), new_lines.begin(), new_lines.end());
    }

    for (const auto& square : _white_squares)
    {
      const auto new_lines = grow_lines_from_square(
          square.v, _corners, _edge_stats, _edge_grad_means, _edge_grad_covs,
          _edges_adjacent_to_corner, _corners_adjacent_to_edge);

      _lines.insert(_lines.end(), new_lines.begin(), new_lines.end());
    }

    toc("Line reconstruction");
  }

  auto ChessboardDetector::grow_chessboards() -> void
  {
    tic();

    _squares = to_list(_black_squares, _white_squares);

    // Create IDs for each square edge.
    const auto edge_ids = populate_edge_ids(_squares);
    // Populate the list of squares adjacent to each edge.
    const auto squares_adj_to_edge =
        populate_squares_adj_to_edge(edge_ids, _squares);

    // The state of the region growing procedure..
    auto is_square_visited = std::vector<std::uint8_t>(_squares.size(), 0);

    // Initialize the list of squares from which we will grow chessboards.
    auto square_ids = std::queue<int>{};
    for (auto s = 0u; s < _squares.size(); ++s)
      square_ids.push(s);

#ifdef DEBUG_REGION_GROWING
    // For debugging purposes only, otherwise this will slow down the
    // algorithm...
    auto display = _f_blurred.convert<Rgb8>();
#else
    auto display = Image<Rgb8>{};
#endif

    // Recover the chessboards.
    _chessboards.clear();
    while (!square_ids.empty())
    {
      // The seed square.
      const auto square_id = square_ids.front();
      square_ids.pop();
      if (is_square_visited[square_id])
        continue;

      // Grow the chessboard from the seed square.
      auto cb = grow_chessboard(                    //
          square_id,                                // Seed square
          _corners,                                 // Corner locations
          _squares, edge_ids, squares_adj_to_edge,  // Square graph
          is_square_visited,                        // Region growing state
          _params.downscale_factor, display);

      _chessboards.emplace_back(std::move(cb));
    }

    toc("Chessboard growing");
  }

  auto ChessboardDetector::extract_chessboard_corners_from_chessboard_squares()
      -> void
  {
    tic();
    // Each grown chessboard consists of an ordered list of squares.
    // We want to retrieve the ordered list of corners.
    _cb_corners.clear();
    _cb_corners.reserve(_chessboards.size());
    for (const auto& chessboard : _chessboards)
    {
      const auto m = rows(chessboard) + 1;
      const auto n = cols(chessboard) + 1;

      auto corners = OrderedChessboardCorners{};
      // Preallocate and initialize the list of ordered corners.
      corners.resize(m);
      std::for_each(corners.begin(), corners.end(), [n](auto& row) {
        row.resize(n);
        static const Eigen::Vector2f nan2d = Eigen::Vector2f::Constant(  //
            std::numeric_limits<float>::quiet_NaN());
        std::fill(row.begin(), row.end(), nan2d);
      });

      // Get the chessboard x-corners.
      for (auto i = 0; i < m - 1; ++i)
      {
        for (auto j = 0; j < n - 1; ++j)
        {
          const auto is_square_undefined = chessboard[i][j].id == -1;
          if (is_square_undefined)
            continue;

          const auto& square = _squares[chessboard[i][j].id];

          // top-left
          const auto& a = square.v[0];
          // top-right
          const auto& b = square.v[1];
          // bottom-right
          const auto& c = square.v[2];
          // bottom-left
          const auto& d = square.v[3];

          // Update the list of coordinates.
          //
          // N.B.: it does not matter if we rewrite the coordinates, they are
          // guaranteed to be the same.
          corners[i][j] = _corners[a].coords;
          corners[i][j + 1] = _corners[b].coords;
          corners[i + 1][j + 1] = _corners[c].coords;
          corners[i + 1][j] = _corners[d].coords;
        }
      }

      // Important rescale the coordinates back to the original image sizes.
      for (auto i = 0; i < m; ++i)
        for (auto j = 0; j < n; ++j)
          corners[i][j] *= _params.downscale_factor;

      _cb_corners.emplace_back(std::move(corners));
    }

    toc("Chessboard ordered corners");
  }

  auto ChessboardDetector::extract_chessboard_vertices_from_chessboard_squares()
      -> void
  {
    tic();
    // Each grown chessboard consists of an ordered list of squares.
    // We want to retrieve the ordered list of corners.
    _cb_vertices.clear();
    _cb_vertices.reserve(_chessboards.size());
    for (const auto& chessboard : _chessboards)
    {
      const auto m = rows(chessboard) + 1;
      const auto n = cols(chessboard) + 1;

      auto vertices = OrderedChessboardVertices{};
      // Preallocate and initialize the list of ordered corners.
      vertices.resize(m);
      std::for_each(vertices.begin(), vertices.end(), [n](auto& row) {
        row.resize(n);
        std::fill(row.begin(), row.end(), -1);
      });

      // Get the chessboard x-corners.
      for (auto i = 0; i < m - 1; ++i)
      {
        for (auto j = 0; j < n - 1; ++j)
        {
          const auto is_square_undefined = chessboard[i][j].id == -1;
          if (is_square_undefined)
            continue;

          const auto& square = _squares[chessboard[i][j].id];

          // top-left
          const auto& a = square.v[0];
          // top-right
          const auto& b = square.v[1];
          // bottom-right
          const auto& c = square.v[2];
          // bottom-left
          const auto& d = square.v[3];

          // Update the list of coordinates.
          //
          // N.B.: it does not matter if we rewrite the coordinates, they are
          // guaranteed to be the same.
          vertices[i][j] = a;
          vertices[i][j + 1] = b;
          vertices[i + 1][j + 1] = c;
          vertices[i + 1][j] = d;
        }
      }

      _cb_vertices.emplace_back(std::move(vertices));
    }

    toc("Chessboard ordered corners");
  }
}  // namespace DO::Sara
