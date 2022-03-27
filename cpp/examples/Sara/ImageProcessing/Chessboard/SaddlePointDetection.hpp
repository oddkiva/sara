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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>


namespace DO::Sara {

  struct SaddlePoint
  {
    Eigen::Vector2i p;
    Eigen::Matrix2f hessian;
    float score;

    inline auto operator<(const SaddlePoint& other) const
    {
      return score < other.score;
    }

    inline auto axes() const -> Eigen::Matrix2f
    {
      const auto svd =
          hessian.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Eigen::Vector2f S = svd.singularValues();
      const auto& axes = svd.matrixU();
      return axes;
    }
  };

  inline auto draw(ImageView<Rgb8>& image, const SaddlePoint& s) -> void
  {
    const auto axes = s.axes();

    const auto a = Eigen::Vector2f(s.p.x(), s.p.y());
    static constexpr auto radius = 20.f;
    const Eigen::Vector2f b = a + radius * axes.col(0);
    const Eigen::Vector2f c = a + radius * axes.col(1);

    draw_arrow(image, a, b, Cyan8, 2);
    draw_arrow(image, a, c, Cyan8, 2);
  }

  auto extract_saddle_points(const ImageView<float>& det_of_hessian,
                             const ImageView<Eigen::Matrix2f>& hessian,
                             float thres) -> std::vector<SaddlePoint>;

  auto nms(std::vector<SaddlePoint>& saddle_points,
           const Eigen::Vector2i& image_sizes, int nms_radius) -> void;

}  // namespace DO::Sara
