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

#include "Corner.hpp"
#include "NonMaximumSuppression.hpp"

#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>


namespace DO::Sara {

  auto select(const DO::Sara::ImageView<float>& cornerness,
              const float image_scale, const float sigma_I, const int octave,
              const float cornerness_adaptive_thres, const int border)
      -> std::vector<Corner<int>>
  {
    tic();
    const auto extrema = local_maxima(cornerness);
    toc("local maxima");

    tic();
    const auto cornerness_max = cornerness.flat_array().maxCoeff();
    const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;
    toc("cornerness thres");

    tic();
    auto extrema_filtered = std::vector<Corner<int>>{};
    extrema_filtered.reserve(extrema.size());
    for (const auto& p : extrema)
    {
      const auto in_image_domain =
          border <= p.x() && p.x() < cornerness.width() - border &&  //
          border <= p.y() && p.y() < cornerness.height() - border;
      if (in_image_domain && cornerness(p) > cornerness_thres)
        extrema_filtered.push_back(
            {p, cornerness(p), image_scale * sigma_I, octave});
    }
    toc("corner population");

    return extrema_filtered;
  };

  auto detect_corners(const ImageView<float>& cornerness,
                      const ImageView<float>& grad_x,
                      const ImageView<float>& grad_y,  //
                      const float image_scale,         //
                      const float sigma_I,             //
                      const int octave,                //
                      const float radius_factor) -> std::vector<Corner<float>>
  {
    static constexpr auto cornerness_adaptive_thres = 0.f;
    static const auto corner_filtering_radius =
        static_cast<int>(std::round(M_SQRT2 * image_scale * sigma_I));

    tic();
    const auto corners_quantized = select(  //
        cornerness,                         //
        image_scale, sigma_I, octave,       //
        cornerness_adaptive_thres,          //
        corner_filtering_radius);
    toc("Corner selection");

    tic();
    auto corners = std::vector<Corner<float>>(corners_quantized.size());
    const auto num_corners = static_cast<int>(corners.size());
#pragma omp parallel for
    for (auto c = 0; c < num_corners; ++c)
    {
      const auto& cq = corners_quantized[c];
      const auto p = refine_junction_location_unsafe(grad_x, grad_y, cq.coords,
                                                     corner_filtering_radius);
      // TODO: interpolate the cornerness.
      corners[c] = {p, cq.score, image_scale * sigma_I, octave};
    }
    toc("Corner refinement");

    tic();
    scale_aware_nms(corners, cornerness.sizes(), radius_factor);
    toc("Corner NMS");

    return corners;
  }


  // Seed corner selection.
  auto is_good_x_corner(const std::vector<float>& zero_crossings) -> bool
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


}  // namespace DO::Sara
