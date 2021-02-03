// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>
#include <DO/Sara/Graphics/ImageDraw.hpp>


namespace DO::Sara {

  // ==========================================================================
  // Shape Statistics of the Edge.
  // ==========================================================================
  struct OrientedBox
  {
    const Eigen::Vector2d& center;
    const Eigen::Matrix2d& axes;
    const Eigen::Vector2d& lengths;

    auto length_ratio() const
    {
      return lengths(0) / lengths(1);
    }

    auto line() const
    {
      return Projective::line(center.homogeneous().eval(),
                              (center + axes.col(0)).homogeneous().eval());
    }

    auto draw(ImageView<Rgb8>& detection, const Rgb8& color,  //
              const Point2d& c1, const double s) const -> void
    {
      const Vector2d u = axes.col(0);
      const Vector2d v = axes.col(1);
      const auto p = std::array<Vector2d, 4>{
          c1 + s * (center + (lengths(0) + 0) * u + (lengths(1) + 0) * v),
          c1 + s * (center - (lengths(0) + 0) * u + (lengths(1) + 0) * v),
          c1 + s * (center - (lengths(0) + 0) * u - (lengths(1) + 0) * v),
          c1 + s * (center + (lengths(0) + 0) * u - (lengths(1) + 0) * v),
      };
      auto pi = std::array<Vector2i, 4>{};
      std::transform(p.begin(), p.end(), pi.begin(),
                     [](const Vector2d& v) { return v.cast<int>(); });

      draw_line(detection, pi[0].x(), pi[0].y(), pi[1].x(), pi[1].y(), color,
                2);
      draw_line(detection, pi[1].x(), pi[1].y(), pi[2].x(), pi[2].y(), color,
                2);
      draw_line(detection, pi[2].x(), pi[2].y(), pi[3].x(), pi[3].y(), color,
                2);
      draw_line(detection, pi[3].x(), pi[3].y(), pi[0].x(), pi[0].y(), color,
                2);
    }
  };

}  // namespace DO::Sara
