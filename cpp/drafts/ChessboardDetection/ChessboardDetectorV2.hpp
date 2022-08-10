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
#include "EdgeStatistics.hpp"
#include "SquareGraph.hpp"

#include <set>
#include <unordered_set>


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
    float edge_detection_downscale_factor = std::sqrt(2.f);

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
    float corner_edge_linking_radius;
    static constexpr auto radius_factor = 2.f;

    inline ChessboardDetectorV2() = default;

    inline auto initialize_multiscale_harris_corner_detection_params(
        const float sigma_D = 0.8f,      //
        const float sigma_I = 3 * 0.8f,  //
        const float kappa = 0.04f,       //
        const int num_scales = 2,        //
        bool upscale_image = false) -> void
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

      // Now set the Harris corner detection parameters that will be applied at
      // each scale of the pyramid.
      corner_detection_params.sigma_I = sigma_I;
      corner_detection_params.kappa = kappa;
    }

    inline auto initialize_filter_radius_according_to_scale() -> void
    {
      corner_edge_linking_radius = static_cast<int>(  //
          std::round(radius_factor * gaussian_pyramid_params.scale_initial() *
                     corner_detection_params.sigma_I /
                     edge_detection_downscale_factor));
      SARA_CHECK(corner_edge_linking_radius);
    }

    inline auto initialize_edge_detector() -> void
    {
      _ed = EdgeDetector{edge_detection_params};
    }

    DO_SARA_EXPORT auto operator()(const ImageView<float>& image)
        -> const std::vector<OrderedChessboardCorners>&;

    auto calculate_feature_pyramids(const ImageView<float>& image) -> void;
    auto extract_corners() -> void;
    auto detect_edges() -> void;
    auto filter_edges() -> void;
    auto group_and_filter_corners() -> void;
    auto filter_corners_topologically() -> void;

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

    //! @brief Corner local region descriptors.
    std::vector<Eigen::ArrayXf> _profiles;
    std::vector<std::vector<float>> _zero_crossings;

    //! @brief Data structures to analyze the topological structures
    Image<std::uint8_t> _edge_map;
    Image<std::int32_t> _endpoint_map;
    std::vector<std::uint8_t> _is_strong_edge;

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

    //! @brief For the square reconstruction.
    std::vector<std::unordered_set<int>> _edges_adjacent_to_corner;
  };


}  // namespace DO::Sara
