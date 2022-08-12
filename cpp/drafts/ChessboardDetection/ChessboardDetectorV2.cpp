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

#include <drafts/ChessboardDetection/ChessboardDetectorV2.hpp>
#include <drafts/ChessboardDetection/NonMaximumSuppression.hpp>
#include <drafts/ChessboardDetection/OrientationHistogram.hpp>
#include <drafts/ChessboardDetection/SquareReconstruction.hpp>

#include <DO/Sara/FeatureDescriptors/Orientation.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/Harris.hpp>

#include <boost/log/trivial.hpp>


namespace DO::Sara {

  auto ChessboardDetectorV2::operator()(const ImageView<float>& image)
      -> const std::vector<OrderedChessboardCorners>&
  {
    calculate_feature_pyramids(image);
    extract_corners();
    detect_edges();
    filter_edges();
    group_and_filter_corners();
    link_corners_to_edge_endpoints_topologically();
    filter_corners_topologically();
    calculate_circular_intensity_profiles();
    filter_corners_with_intensity_profiles();

    link_corners_to_edges();
    calculate_edge_adjacent_to_corners();
    calculate_edge_shape_statistics();
    calculate_orientation_histograms();
    select_seed_corners();

    parse_squares();
    grow_chessboards();

    return _cb_corners;
  }

  auto ChessboardDetectorV2::calculate_feature_pyramids(
      const ImageView<float>& image) -> void
  {
    tic();
    _gauss_pyr = gaussian_pyramid(image, gaussian_pyramid_params);
    toc("gaussian_pyramid pyramid");

    tic();
    if (static_cast<int>(_grad_x_pyr.size()) != _gauss_pyr.octave_count())
      _grad_x_pyr.resize(_gauss_pyr.octave_count());
    if (static_cast<int>(_grad_y_pyr.size()) != _gauss_pyr.octave_count())
      _grad_y_pyr.resize(_gauss_pyr.octave_count());
    for (auto o = 0; o < _gauss_pyr.octave_count(); ++o)
    {
      const auto& g = _gauss_pyr(0, o);
      if (_grad_x_pyr[o].sizes() != g.sizes())
        _grad_x_pyr[o].resize(g.sizes());
      if (_grad_y_pyr[o].sizes() != g.sizes())
        _grad_y_pyr[o].resize(g.sizes());

      gradient(_gauss_pyr(0, o), _grad_x_pyr[o], _grad_y_pyr[o]);
    }
    toc("Gradient pyramid");

    tic();
    if (static_cast<int>(_cornerness_pyr.size()) != _gauss_pyr.octave_count())
      _cornerness_pyr.resize(_gauss_pyr.octave_count());
    for (auto o = 0; o < _gauss_pyr.octave_count(); ++o)
    {
      _cornerness_pyr[o] = harris_cornerness(  //
          _grad_x_pyr[o], _grad_y_pyr[o],      //
          corner_detection_params.sigma_I,     //
          corner_detection_params.kappa);
    }
    toc("Cornerness pyramid");
  }

  auto ChessboardDetectorV2::extract_corners() -> void
  {
    if (static_cast<int>(_corners_per_scale.size()) !=
        _gauss_pyr.octave_count())
      _corners_per_scale.resize(_gauss_pyr.octave_count());

    for (auto o = 0u; o < _corners_per_scale.size(); ++o)
    {
      _corners_per_scale[o] = detect_corners(       //
          _cornerness_pyr[o],                       //
          _grad_x_pyr[o], _grad_y_pyr[o],           //
          gaussian_pyramid_params.scale_initial(),  //
          corner_detection_params.sigma_I,          //
          static_cast<int>(o), radius_factor);
    }
  }

  auto ChessboardDetectorV2::detect_edges() -> void
  {
    tic();

    const auto octave = upscale ? 1 : 0;
    const auto& g = _gauss_pyr(0, octave);

    // Downscale the image to combat against aliasing.
#if 0
    const auto scale_inter_delta = std::sqrt(
        square(scale_aa) - square(gaussian_pyramid_params.scale_initial()));
#else
    const auto scale_inter_delta = scale_aa;
#endif
    const auto frame_blurred = g.compute<Gaussian>(scale_inter_delta);

    const Eigen::Vector2i sizes_inter =
        (g.sizes().cast<float>() / scale_aa).array().round().cast<int>();
    auto frame_aa = Image<float>{sizes_inter};
    resize_v2(frame_blurred, frame_aa);

    // VERY IMPORTANT DETAIL: BLUR AGAIN.
    frame_aa = frame_aa.compute<Gaussian>(corner_detection_params.sigma_D);

    // Calculate the gradient at the antialias scale.
    _grad_x_scale_aa.resize(sizes_inter);
    _grad_y_scale_aa.resize(sizes_inter);
    gradient(frame_aa, _grad_x_scale_aa, _grad_y_scale_aa);

    // Filter the gradient map.
    _ed.operator()(_grad_x_scale_aa, _grad_y_scale_aa);
    toc("Edge detection");
  }

  auto ChessboardDetectorV2::filter_edges() -> void
  {
    tic();
    _is_strong_edge.clear();
    const auto& edges = _ed.pipeline.edges_as_list;
    std::transform(edges.begin(), edges.end(),
                   std::back_inserter(_is_strong_edge),
                   [this](const auto& edge) -> std::uint8_t {
                     static constexpr auto strong_edge_thres = 4.f / 255.f;
                     return is_strong_edge(_ed.pipeline.gradient_magnitude,
                                           edge, strong_edge_thres);
                   });

    _edge_map.resize(_ed.pipeline.gradient_magnitude.sizes());
    _edge_map.flat_array().fill(0);

    _endpoint_map.resize(_ed.pipeline.gradient_magnitude.sizes());
    _endpoint_map.flat_array().fill(-1);

    const auto num_edges = static_cast<int>(edges.size());
#pragma omp parallel for
    for (auto e = 0; e < num_edges; ++e)
    {
      if (!_is_strong_edge[e])
        continue;

      const auto& edge = edges[e];
      const auto& edge_ordered = reorder_and_extract_longest_curve(edge);

      auto curve = std::vector<Eigen::Vector2d>(edge_ordered.size());
      std::transform(edge_ordered.begin(), edge_ordered.end(), curve.begin(),
                     [](const auto& p) { return p.template cast<double>(); });
      if (curve.size() < 2)
        continue;

      const Eigen::Vector2i a =
          curve.front().array().round().matrix().cast<int>();
      const Eigen::Vector2i b =
          curve.back().array().round().matrix().cast<int>();

      for (const auto& p : edge)
        _edge_map(p) = 255;

      _endpoint_map(a) = 2 * e;
      _endpoint_map(b) = 2 * e + 1;
    }
    toc("Edge filtering");
  }

  auto ChessboardDetectorV2::group_and_filter_corners() -> void
  {
    tic();
    _corners.clear();
    for (auto o = 0u; o < _corners_per_scale.size(); ++o)
    {
      const auto scale_factor = _gauss_pyr.octave_scaling_factor(o);
      std::transform(_corners_per_scale[o].begin(), _corners_per_scale[o].end(),
                     std::back_inserter(_corners),
                     [scale_factor](const Corner<float>& corner) {
                       auto c = corner;
                       c.coords *= scale_factor;
                       c.score /= scale_factor;
                       c.scale *= scale_factor;
                       return c;
                     });
    }

    const auto octave = upscale ? 1 : 0;
    const auto& g = _gauss_pyr(0, octave);
    scale_aware_nms(_corners, g.sizes(), radius_factor);
    toc("Corner grouping and NMS");
  }

  auto ChessboardDetectorV2::link_corners_to_edge_endpoints_topologically()
      -> void
  {
    tic();
    const auto& edges = _ed.pipeline.edges_as_list;
    _corners_adjacent_to_endpoints.clear();
    _corners_adjacent_to_endpoints.resize(2 * edges.size());

    const auto w = _endpoint_map.width();
    const auto h = _endpoint_map.height();

    const auto num_corners = static_cast<int>(_corners.size());
    const auto& r = corner_endpoint_linking_radius;
    for (auto c = 0; c < num_corners; ++c)
    {
      const auto& corner = _corners[c];

      const Eigen::Vector2i p = (corner.coords / scale_aa)  //
                                    .array()
                                    .round()
                                    .matrix()
                                    .cast<int>();

      const auto umin = std::clamp(p.x() - r, 0, w);
      const auto umax = std::clamp(p.x() + r, 0, w);
      const auto vmin = std::clamp(p.y() - r, 0, h);
      const auto vmax = std::clamp(p.y() + r, 0, h);
      for (auto v = vmin; v < vmax; ++v)
      {
        for (auto u = umin; u < umax; ++u)
        {
          const auto endpoint_id = _endpoint_map(u, v);
          if (endpoint_id != -1)
            _corners_adjacent_to_endpoints[endpoint_id].insert(
                CornerRef{c, corner.score});
        }
      }
    }
    toc("Corner-edge endpoint topological linking");
  }

  auto ChessboardDetectorV2::filter_corners_topologically() -> void
  {
    tic();
    auto best_corner_ids = std::unordered_set<int>{};
    for (const auto& corners_adj_to_endpoint : _corners_adjacent_to_endpoints)
    {
      if (corners_adj_to_endpoint.empty())
        continue;
      const auto best_corner = corners_adj_to_endpoint.rbegin();
      best_corner_ids.insert(best_corner->id);
    }
    auto corners_filtered = std::vector<Corner<float>>{};
    std::transform(best_corner_ids.begin(), best_corner_ids.end(),
                   std::back_inserter(corners_filtered),
                   [this](const auto& id) { return _corners[id]; });
    corners_filtered.swap(_corners);
    toc("Corner topological filtering");
  }

  auto ChessboardDetectorV2::calculate_circular_intensity_profiles() -> void
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
      const auto& corner = _corners[c];

      // Retrieve the image where the corner was detected.
      const auto& frame = _gauss_pyr(0, corner.octave);
      const auto w = frame.width();
      const auto h = frame.height();

      // Rescale the coordinates.
      const Eigen::Vector2d p =
          (corner.coords / _gauss_pyr.octave_scaling_factor(corner.octave))
              .cast<double>();

      // Readapt the radius of the circular profile to the image scale.
      const auto r = M_SQRT2 * gaussian_pyramid_params.scale_initial() *
                     corner_detection_params.sigma_I * radius_factor;

      // Boundary check.
      if (!(r + 1 <= p.x() && p.x() < w - r - 1 &&  //
            r + 1 <= p.y() && p.y() < h - r - 1))
        continue;

      _profiles[c] = _profile_extractor(frame, p, r);
      _zero_crossings[c] = localize_zero_crossings(
          _profiles[c], _profile_extractor.num_circle_sample_points);
    }
    toc("Circular intensity profile");
  }

  auto ChessboardDetectorV2::filter_corners_with_intensity_profiles() -> void
  {
    tic();
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
    toc("Corner filtering from intensity profile");
  }

  auto ChessboardDetectorV2::link_corners_to_edges() -> void
  {
    tic();

    const auto& edges = _ed.pipeline.edges_as_list;
    _corners_adjacent_to_edge.clear();
    _corners_adjacent_to_edge.resize(edges.size());

    const auto w = _endpoint_map.width();
    const auto h = _endpoint_map.height();

    const auto num_corners = static_cast<int>(_corners.size());
    const auto& r = corner_endpoint_linking_radius;
    for (auto c = 0; c < num_corners; ++c)
    {
      const auto& corner = _corners[c];

      const Eigen::Vector2i p = (corner.coords / scale_aa)  //
                                    .array()
                                    .round()
                                    .matrix()
                                    .cast<int>();

      const auto umin = std::clamp(p.x() - r, 0, w);
      const auto umax = std::clamp(p.x() + r, 0, w);
      const auto vmin = std::clamp(p.y() - r, 0, h);
      const auto vmax = std::clamp(p.y() + r, 0, h);
      for (auto v = vmin; v < vmax; ++v)
      {
        for (auto u = umin; u < umax; ++u)
        {
          const auto endpoint_id = _endpoint_map(u, v);
          if (endpoint_id == -1)
            continue;

          const auto edge_id = endpoint_id / 2;
          _corners_adjacent_to_edge[edge_id].insert(c);
        }
      }
    }

    toc("Corners-to-edge topological linking");
  }

  auto ChessboardDetectorV2::calculate_orientation_histograms() -> void
  {
    tic();
    const auto& _grad_norm = _ed.pipeline.gradient_magnitude;
    const auto& _grad_ori = _ed.pipeline.gradient_orientation;

    _hists.resize(_corners.size());
    _gradient_peaks.resize(_corners.size());
    _gradient_peaks_refined.resize(_corners.size());

    const auto num_corners = static_cast<int>(_corners.size());
#pragma omp parallel for
    for (auto i = 0; i < num_corners; ++i)
    {
      const Eigen::Vector2f p = _corners[i].coords / scale_aa;
      const auto radius = _corners[i].radius() / scale_aa;
      compute_orientation_histogram<N>(_hists[i], _grad_norm, _grad_ori, p.x(),
                                       p.y(), radius, radius_factor);
      lowe_smooth_histogram(_hists[i]);
      _hists[i].matrix().normalize();

      _gradient_peaks[i] = find_peaks(_hists[i], 0.5f);
      _gradient_peaks_refined[i] = refine_peaks(_hists[i], _gradient_peaks[i]);
      std::for_each(_gradient_peaks_refined[i].begin(),
                    _gradient_peaks_refined[i].end(), [](auto& v) {
                      v *= static_cast<float>(2 * M_PI / N);
                      v -= static_cast<float>(M_PI * 0.5);
                      if (v < 0)
                        v += static_cast<float>(2 * M_PI);
                    });
    }
    toc("Gradient histograms");
  }

  auto ChessboardDetectorV2::calculate_edge_adjacent_to_corners() -> void
  {
    _edges_adjacent_to_corner.clear();
    _edges_adjacent_to_corner.resize(_corners.size());
    std::transform(                         //
        _corners.begin(), _corners.end(),   //
        _edges_adjacent_to_corner.begin(),  //
        [this](const Corner<float>& c) {
          auto edges = std::unordered_set<int>{};

          const auto& r = corner_endpoint_linking_radius;
          for (auto v = -r; v <= r; ++v)
          {
            for (auto u = -r; u <= r; ++u)
            {
              const Eigen::Vector2i p =
                  (c.coords / scale_aa).cast<int>() + Eigen::Vector2i{u, v};

              const auto in_image_domain =
                  0 <= p.x() && p.x() < _endpoint_map.width() &&  //
                  0 <= p.y() && p.y() < _endpoint_map.height();
              if (!in_image_domain)
                continue;

              const auto endpoint_id = _endpoint_map(p);
              if (endpoint_id == -1)
                continue;
              const auto edge_id = endpoint_id / 2;
              edges.insert(edge_id);
            }
          }
          return edges;
        });
  }

  auto ChessboardDetectorV2::calculate_edge_shape_statistics() -> void
  {
    tic();
    _edge_stats = get_curve_shape_statistics(_ed.pipeline.edges_as_list);
    _edge_grad_means = gradient_mean(_ed.pipeline.edges_as_list,  //
                                     _grad_x_scale_aa, _grad_y_scale_aa);
    _edge_grad_covs = gradient_covariance(_ed.pipeline.edges_as_list,  //
                                          _grad_x_scale_aa, _grad_y_scale_aa);
    toc("Edge shape statistics");
  }

  auto ChessboardDetectorV2::select_seed_corners() -> void
  {
    tic();
    _best_corners.clear();
    for (auto c = 0u; c < _corners.size(); ++c)
      if (is_seed_corner(_edges_adjacent_to_corner[c], _zero_crossings[c]))
        _best_corners.insert(c);
    toc("Best corner selection");
  }

  auto ChessboardDetectorV2::parse_squares() -> void
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

  auto ChessboardDetectorV2::grow_chessboards() -> void
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
          1.f, display);

      _chessboards.emplace_back(std::move(cb));
    }

    toc("Chessboard growing");
  }

}  // namespace DO::Sara
