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

#include "CircularProfileExtractor.hpp"

#include <DO/Sara/ImageProcessing/Interpolation.hpp>


namespace DO::Sara {

  CircularProfileExtractor ::CircularProfileExtractor()
  {
    initialize_circle_sample_points();
  }

  // Sample a unit circle centered at the origin.
  auto CircularProfileExtractor::initialize_circle_sample_points() -> void
  {
    static constexpr auto pi = static_cast<double>(M_PI);

    const auto& n = num_circle_sample_points;
    circle_sample_points = std::vector<Eigen::Vector2d>(n);

    for (auto i = 0; i < n; ++i)
    {
      const auto angle = i * 2. * pi / n;
      circle_sample_points[i] << std::cos(angle), std::sin(angle);
    }
  }

  auto CircularProfileExtractor::operator()(const ImageView<float>& image,
                                            const Eigen::Vector2d& center) const
      -> Eigen::ArrayXf
  {
    auto intensity_profile = Eigen::ArrayXf(num_circle_sample_points);

    for (auto n = 0; n < num_circle_sample_points; ++n)
    {
      // Use Abeles' spoke pattern, which really helps to filter out non
      // chessboard x-corners.
      auto mean_intensity = 0.f;
      for (auto r = 0; r < static_cast<int>(circle_radius); ++r)
      {
        const Eigen::Vector2d pn = center +  //
                                   circle_radius * circle_sample_points[n];
        mean_intensity += static_cast<float>(interpolate(image, pn));
      }
      mean_intensity /= static_cast<int>(circle_radius);
      // Get the interpolated intensity value.
      intensity_profile(n) = mean_intensity;
    }

    // Normalize the intensities.
    const auto min_intensity = intensity_profile.minCoeff();
    const auto max_intensity = intensity_profile.maxCoeff();

    // The intensity treshold is the mid-point value.
    const auto intensity_threshold = (max_intensity + min_intensity) * 0.5f;
    intensity_profile -= intensity_threshold;

    return intensity_profile;
  }

  auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
      -> std::vector<float>
  {
    auto zero_crossings = std::vector<float>{};
    for (auto n = Eigen::Index{}; n < profile.size(); ++n)
    {
      const auto ia = n;
      const auto ib = (n + Eigen::Index{1}) % profile.size();

      const auto& a = profile[ia];
      const auto& b = profile[ib];

      static constexpr auto pi = static_cast<float>(M_PI);
      const auto angle_a = ia * 2.f * M_PI / num_bins;
      const auto angle_b = ib * 2.f * M_PI / num_bins;

      const auto ea = Eigen::Vector2d{std::cos(angle_a),  //
                                      std::sin(angle_a)};
      const auto eb = Eigen::Vector2d{std::cos(angle_b),  //
                                      std::sin(angle_b)};

      // TODO: this all could have been simplified.
      const Eigen::Vector2d dir = (ea + eb) * 0.5;
      auto angle = std::atan2(dir.y(), dir.x());
      if (angle < 0)
        angle += 2 * pi;

      // A zero-crossing is characterized by a negative sign between
      // consecutive intensity values.
      if (a * b < 0)
        zero_crossings.push_back(static_cast<float>(angle));
    }

    return zero_crossings;
  }

}  // namespace DO::Sara
