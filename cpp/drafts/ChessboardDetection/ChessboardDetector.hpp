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

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>

#include "CircularProfileExtractor.hpp"
#include "Corner.hpp"
#include "SquareGraph.hpp"

#include <set>
#include <unordered_set>


namespace DO::Sara {

  inline constexpr auto operator"" _percent(long double x) -> long double
  {
    return x / 100;
  }

  auto is_seed_corner(  //
      const std::unordered_set<int>& adjacent_edges,
      const std::vector<float>& gradient_peaks,  //
      const std::vector<float>& zero_crossings,  //
      int N) -> bool;


  class ChessboardDetector
  {
  public:
    using OrderedChessboardCorners = std::vector<std::vector<Eigen::Vector2f>>;
    using OrderedChessboardVertices = std::vector<std::vector<int>>;

    struct Parameters
    {
      // The Harris's cornerness parameters.
      //
      // @brief Blur parameter before the gradient calculation.
      // This is a good default parameter and there is little reason to change
      // it in my experience.
      float sigma_D = 0.8f;
      // @brief Integration domain of the second moment.
      // This is a good default parameter and there is little reason to change
      // it in my experience.
      float sigma_I = 2.4f;
      // @brief The threshold parameter.
      // This is a good default parameter and there is little reason to change
      // it in my experience.
      float kappa = 0.04f;
      // @brief Extra parameter, which might be unnecessary.
      float cornerness_adaptive_thres = 1e-5f;

      //! @brief Corner radius (used for corner refinement and non-maximum
      //! suppression).
      //! @{
      int corner_filtering_radius;
      //! @}

      // Edge filtering.
      //
      // @brief These are good default parameters and there is little reason to
      // change them in my experience.
      // @{
      float high_threshold_ratio = static_cast<float>(4._percent);
      float low_threshold_ratio = static_cast<float>(2._percent);
      float angular_threshold = static_cast<float>((10._deg).value);
      // @}
      //
      // @brief Good values are:
      // { 0.5f, 1.f, 1.2f, 1.25f, 1.5f, 2.f }.
      //
      // This really depends on the image resolution. These days, we work on HD
      // resolution, i.e., 1920x1080.
      //
      // For small images, we may need to upscale image, so the downscale factor
      // should be 0.5f.
      float downscale_factor = 1.25f;
      // This is a good magic constant that connects edge to corners.
      int corner_edge_linking_radius = 4;

      // This is a good policy that sets the corner radius, with little reason
      // to change this in my experience.
      inline auto set_corner_nms_radius() -> void
      {
        corner_filtering_radius = static_cast<int>(  //
            std::round(4 * sigma_I / downscale_factor));
        SARA_CHECK(corner_filtering_radius);
      }

      // This may behave better or worse, this is a moving part.
      inline auto set_corner_edge_linking_radius_to_corner_filtering_radius()
          -> void
      {
        corner_edge_linking_radius = corner_filtering_radius;
      }
    };

    inline ChessboardDetector() = default;

    inline explicit ChessboardDetector(const Parameters& params)
      : _params{params}
    {
      _ed.parameters = {
          params.high_threshold_ratio,  //
          params.low_threshold_ratio,   //
          params.angular_threshold      //
      };
      _profile_extractor.circle_radius = params.corner_filtering_radius;
    }

    DO_SARA_EXPORT
    auto operator()(const ImageView<float>& image)
        -> const std::vector<OrderedChessboardCorners>&;

    auto preprocess_image(const ImageView<float>& image) -> void;
    auto filter_edges() -> void;
    auto detect_corners() -> void;
    auto calculate_circular_intensity_profiles() -> void;
    auto filter_corners_with_intensity_profiles() -> void;
    auto calculate_orientation_histograms() -> void;
    auto calculate_edge_shape_statistics() -> void;
    auto link_edges_and_corners() -> void;
    auto select_seed_corners() -> void;

    auto parse_squares() -> void;
    auto grow_chessboards() -> void;
    auto extract_chessboard_corners_from_chessboard_squares() -> void;
    auto extract_chessboard_vertices_from_chessboard_squares() -> void;

    // TODO: redo this but this time with a smarter geometric modelling of
    // the distorted lines.
    auto parse_lines() -> void;

    //! @brief Preprocessed images.
    //! @{
    Image<float> _f_blurred;
    Image<float> _f_ds;
    Image<float> _f_ds_blurred;
    //! @}

    //! @brief Intermediate feature maps used for the edge filtering and for the
    //! corner detection.
    //! @{
    Image<float> _grad_x, _grad_y;
    Image<float> _grad_mag, _grad_ori;
    //! @}

    //! @brief Cornerness map.
    Image<float> _cornerness;
    //! @brief The quantized locations of corners
    //! They are extracted from the cornerness map.
    std::vector<Corner<int>> _corners_int;
    //! @brief The refined locations of corners.
    std::vector<Corner<float>> _corners;

    //! @brief The edge detector
    EdgeDetector _ed;

    //! @brief Data structures to analyze the topological structures
    std::vector<std::unordered_set<int>> _edges_adjacent_to_corner;
    std::vector<std::unordered_set<int>> _corners_adjacent_to_edge;

    //! @brief Descriptors for the corners
    //! @{
    CircularProfileExtractor _profile_extractor;
    std::vector<Eigen::ArrayXf> _profiles;
    std::vector<std::vector<float>> _zero_crossings;

    static constexpr auto N = 72;
    std::vector<Eigen::Array<float, N, 1>> _hists;
    std::vector<std::vector<int>> _gradient_peaks;
    std::vector<std::vector<float>> _gradient_peaks_refined;
    //! @}

    //! @brief Edge shape statistics.
    //! @{
    Image<int> _edge_label_map;
    CurveStatistics _edge_stats;
    std::vector<Eigen::Vector2f> _edge_grad_means;
    std::vector<Eigen::Matrix2f> _edge_grad_covs;
    //! @}

    //! @brief The best corners.
    std::unordered_set<int> _best_corners;

    //! @brief The list of seed squares constructed in a greedy manner.
    //! @{
    SquareSet _black_squares;
    SquareSet _white_squares;
    //! @}

    //! @brief Chessboards grown from the list of squares.
    //! @{
    std::vector<Square> _squares;
    std::vector<Chessboard> _chessboards;
    //! @}

    //! @brief TODO: Recover missed corners.
    std::vector<std::vector<int>> _lines;

    //! @brief The final output.
    std::vector<OrderedChessboardCorners> _cb_corners;
    std::vector<OrderedChessboardVertices> _cb_vertices;

    //! @brief The list of parameters.
    Parameters _params;
  };


  inline auto transpose(const ChessboardDetector::OrderedChessboardCorners& in)
      -> ChessboardDetector::OrderedChessboardCorners
  {
    const auto m = in.size();
    const auto n = in.front().size();

    auto out = ChessboardDetector::OrderedChessboardCorners{};
    out.resize(n);
    for (auto i = 0u; i < n; ++i)
      out[i].resize(m);

    for (auto i = 0u; i < m; ++i)
      for (auto j = 0u; j < n; ++j)
        out[j][i] = in[i][j];

    return out;
  }


}  // namespace DO::Sara
