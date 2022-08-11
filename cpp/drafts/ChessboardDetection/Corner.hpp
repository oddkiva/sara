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

#include <DO/Sara/ImageProcessing.hpp>

#include <unordered_set>


namespace DO::Sara {

  template <typename T>
  struct Corner
  {
    Eigen::Vector2<T> coords;
    float score;
    float scale;
    int octave;

    inline auto position() const -> const Eigen::Vector2<T>&
    {
      return coords;
    }

    inline auto radius() const -> float
    {
      return static_cast<float>(M_SQRT2 * scale);
    }

    inline auto operator<(const Corner& other) const -> bool
    {
      return score < other.score;
    }
  };

  // Select the local maxima of the cornerness functions.
  auto select(const ImageView<float>& cornerness,  //
              const float image_scale, const float sigma_I, int octave,
              const float cornerness_adaptive_thres, const int border)
      -> std::vector<Corner<int>>;

  auto detect_corners(const ImageView<float>& cornerness,
                      const ImageView<float>& grad_x,
                      const ImageView<float>& grad_y,  //
                      const float image_scale,         //
                      const float sigma_I,             //
                      const int octave,                //
                      const float radius_factor) -> std::vector<Corner<float>>;

  auto is_good_x_corner(const std::vector<float>& zero_crossings) -> bool;

  auto is_seed_corner(const std::unordered_set<int>& adjacent_edges,
                      const std::vector<float>& zero_crossings) -> bool;

}  // namespace DO::Sara
