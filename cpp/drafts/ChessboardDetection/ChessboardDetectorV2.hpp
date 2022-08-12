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

#include <drafts/ChessboardDetection/CircularProfileExtractor.hpp>
#include <drafts/ChessboardDetection/Corner.hpp>
#include <drafts/ChessboardDetection/EdgeStatistics.hpp>
#include <drafts/ChessboardDetection/SquareGraph.hpp>


namespace DO::Sara {

  class ChessboardDetectorV2
  {
  public:
    using OrderedChessboardCorners = std::vector<std::vector<Eigen::Vector2f>>;
    using OrderedChessboardVertices = std::vector<std::vector<int>>;

    //! @brief Harris's corner detection parameters.
    struct HarrisCornerDetectionParameters
    {
      //! @brief Blur parameter before the gradient calculation.
      //! This is a good default parameter and there is little reason to change
      //! it in my experience.
      float sigma_D = 0.8f;

      //! @brief Integration domain of the second moment.
      //! This is a good default parameter and there is little reason to change
      //! it in my experience.
      //!
      //! A good rule to sigma_I is set it as sigma_I = 3 sigma_D.
      float sigma_I = 2.4f;

      //! @brief The threshold parameter.
      //! This is a good default parameter and there is little reason to change
      //! it in my experience.
      float kappa = 0.04f;
    };

    //! @brief Parameters used to construct the Gaussian pyramid.
    bool upscale = false;
    ImagePyramidParams gaussian_pyramid_params = ImagePyramidParams(  //
        0,     // first octave index
        2,     // 2 scales per octave
        2.f,   // scale geometric factor
        1,     // image border
        1.f,   // image scale of the camera
        1.2f,  // start image scale of the gaussian pyramid
        2      // maximum number of scales
    );

    //! @brief Harris's cornerness detection parameters for each scale of the
    //! Gaussian pyramid.
    HarrisCornerDetectionParameters corner_detection_params;

    //! @brief Good values are:
    //! { 0.5f, 1.f, 1.2f, 1.25f, 1.5f, 2.f }.
    //!
    //! This really depends on the image resolution. These days, we work on HD
    //! resolution, i.e., 1920x1080.
    //!
    //! For small images, we may need to upscale image, so the downscale factor
    //! should be 0.5f.
    float scale_aa = std::sqrt(2.f);

    //! @brief Edge filtering parameters.
    //!
    //! These are good default parameters and there is little reason to
    //! change them in my experience.
    EdgeDetector::Parameters edge_detection_params = EdgeDetector::Parameters{
        .high_threshold_ratio = static_cast<float>(4._percent),
        .low_threshold_ratio = static_cast<float>(2._percent),
        .angular_threshold = static_cast<float>((10._deg).value),
        .eps = 0.5,
        .collapse_threshold = 0.5,
        .collapse_adaptive = false,
    };

    //! The default value seems to be a very good value for the analysis
    //! of the patch centered in each corner in my experience.
    int corner_endpoint_linking_radius;
    static constexpr auto radius_factor = 2.f;

    inline ChessboardDetectorV2() = default;

    inline auto initialize_multiscale_harris_corner_detection_params(
        const bool upscale_image = false,  //
        const int num_scales = 2,          //
        const float sigma_D = 0.8f,        //
        const float sigma_I = 3 * 0.8f,    //
        const float kappa = 0.04f) -> void
    {
      // First set up the Gaussian pyramid parameters.
      upscale = upscale_image;
      const auto scale_initial = std::sqrt(1.f + square(sigma_D));
      gaussian_pyramid_params = ImagePyramidParams{
          upscale ? -1 : 0,      // First octave index
          2,                     // 2 scales per octave, i.e.:
                                 // - 1 used for the corner detection,
                                 // - 1 used to downsample the image.
          2.f,                   // The scale geometric factor (don't change)
          1,                     // The image padding size (don't change)
          upscale ? 0.5f : 1.f,  // The original image scale
          scale_initial,  // The image scale of the Gaussian pyramid at each
                          // octave
          num_scales      // The maximum number of octaves
      };
      SARA_CHECK(gaussian_pyramid_params.scale_camera());
      SARA_CHECK(gaussian_pyramid_params.scale_initial());
      SARA_CHECK(gaussian_pyramid_params.num_octaves_max());

      // Now set the Harris corner detection parameters that will be applied at
      // each scale of the pyramid.
      corner_detection_params.sigma_I = sigma_I;
      corner_detection_params.kappa = kappa;
    }

    inline auto initialize_filter_radius_according_to_scale() -> void
    {
      const auto& scale_initial = gaussian_pyramid_params.scale_initial();
      const auto& sigma_I = corner_detection_params.sigma_I;
      corner_endpoint_linking_radius = static_cast<int>(std::round(
          radius_factor * M_SQRT2 * scale_initial * sigma_I / scale_aa));

      SARA_CHECK(corner_endpoint_linking_radius);
    }

    inline auto initialize_edge_detector() -> void
    {
      _ed = EdgeDetector{edge_detection_params};
    }

    DO_SARA_EXPORT
    auto operator()(const ImageView<float>& image)
        -> const std::vector<OrderedChessboardCorners>&;

    //! @brief Corner detection and filtering.
    //! @{
    auto calculate_feature_pyramids(const ImageView<float>& image) -> void;
    auto extract_corners() -> void;
    auto detect_edges() -> void;
    auto filter_edges() -> void;
    auto group_and_filter_corners() -> void;
    auto link_corners_to_edge_endpoints_topologically() -> void;
    auto filter_corners_topologically() -> void;
    auto calculate_circular_intensity_profiles() -> void;
    auto filter_corners_with_intensity_profiles() -> void;
    //! @}

    //! @brief The features for the grid structure recovery.
    //! @{
    auto link_corners_to_edges() -> void;
    auto calculate_orientation_histograms() -> void;
    auto calculate_edge_adjacent_to_corners() -> void;
    auto calculate_edge_shape_statistics() -> void;
    auto select_seed_corners() -> void;
    //! @}

    //! @brief Grid structure recovery.
    auto parse_squares() -> void;
    auto grow_chessboards() -> void;
    auto extract_chessboard_corners_from_chessboard_squares() -> void;
    auto extract_chessboard_vertices_from_chessboard_squares() -> void;

    // TODO: redo this but this time with a smarter geometric modelling of
    // the distorted lines.
    auto parse_lines() -> void;

    //! @brief The edge detector.
    EdgeDetector _ed;
    //! @brief Circular intensity profile extractor.
    CircularProfileExtractor _profile_extractor;

    //! @brief The feature pyramids.
    //! @{
    ImagePyramid<float> _gauss_pyr;
    std::vector<Image<float>> _grad_x_pyr;
    std::vector<Image<float>> _grad_y_pyr;
    std::vector<Image<float>> _cornerness_pyr;
    //! @}

    //! @brief The list of corners at each scale.
    std::vector<std::vector<Corner<float>>> _corners_per_scale;
    //! @brief The list of corners filtered and merged together.
    std::vector<Corner<float>> _corners;

    //! @brief Circular intensity descriptors.
    std::vector<Eigen::ArrayXf> _profiles;
    std::vector<std::vector<float>> _zero_crossings;

    //! @brief Data structures to analyze the topological structures
    Image<float> _grad_x_scale_aa;
    Image<float> _grad_y_scale_aa;
    Image<std::uint8_t> _edge_map;
    Image<std::int32_t> _endpoint_map;
    std::vector<std::uint8_t> _is_strong_edge;

    //! @brief These are used for the topological filtering of corners.
    //! @{
    struct CornerRef
    {
      std::int32_t id;
      float score;
      inline auto operator<(const CornerRef& other) const -> bool
      {
        return score < other.score;
      }
    };
    std::vector<std::set<CornerRef>> _corners_adjacent_to_endpoints;
    //! @}

    //! @brief Gradient histograms.
    static constexpr auto N = 72;
    std::vector<Eigen::Array<float, N, 1>> _hists;
    std::vector<std::vector<int>> _gradient_peaks;
    std::vector<std::vector<float>> _gradient_peaks_refined;

    //! @brief The best corners.
    std::unordered_set<int> _best_corners;

    //! @brief Features for the square reconstruction and the recovery of the
    //! grid structure.
    std::vector<std::unordered_set<int>> _edges_adjacent_to_corner;
    std::vector<std::unordered_set<int>> _corners_adjacent_to_edge;
    CurveStatistics _edge_stats;
    std::vector<Eigen::Vector2f> _edge_grad_means;
    std::vector<Eigen::Matrix2f> _edge_grad_covs;

    //! @brief For the square reconstruction.
    //! @{
    SquareSet _black_squares;
    SquareSet _white_squares;
    //! @}

    //! @brief TODO: Recover missed corners.
    std::vector<std::vector<int>> _lines;

    //! @brief Chessboards grown from the list of squares.
    //! @{
    std::vector<Square> _squares;
    std::vector<Chessboard> _chessboards;
    //! @}

    std::vector<OrderedChessboardCorners> _cb_corners;
    std::vector<OrderedChessboardVertices> _cb_vertices;
  };


  inline auto
  transpose(const ChessboardDetectorV2::OrderedChessboardCorners& in)
      -> ChessboardDetectorV2::OrderedChessboardCorners
  {
    const auto m = in.size();
    const auto n = in.front().size();

    auto out = ChessboardDetectorV2::OrderedChessboardCorners{};
    out.resize(n);
    for (auto i = 0u; i < n; ++i)
      out[i].resize(m);

    for (auto i = 0u; i < m; ++i)
      for (auto j = 0u; j < n; ++j)
        out[j][i] = in[i][j];

    return out;
  }


}  // namespace DO::Sara
