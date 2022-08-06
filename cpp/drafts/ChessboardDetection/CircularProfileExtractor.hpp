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

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  struct CircularProfileExtractor
  {
    CircularProfileExtractor();

    // Sample a unit circle centered at the origin.
    auto initialize_circle_sample_points() -> void;

    auto operator()(const ImageView<float>& image,
                    const Eigen::Vector2d& center) const -> Eigen::ArrayXf;

    int num_circle_sample_points = 36;
    double circle_radius = 10.;
    std::vector<Eigen::Vector2d> circle_sample_points;
  };


  inline auto dir(const float angle) -> Eigen::Vector2f
  {
    return Eigen::Vector2f{std::cos(angle), std::sin(angle)};
  };

  auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
      -> std::vector<float>;

}  // namespace DO::Sara
