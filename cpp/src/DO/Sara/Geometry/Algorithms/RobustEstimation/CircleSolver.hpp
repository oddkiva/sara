// ========================================================================== //
// this file is part of sara, a basic set of libraries in c++ for computer
// vision.
//
// copyright (c) 2020-present david ok <david.ok8@gmail.com>
//
// this source code form is subject to the terms of the mozilla public
// license v. 2.0. if a copy of the mpl was not distributed with this file,
// you can obtain one at http://mozilla.org/mpl/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Geometry/CircleFit.hpp>


namespace DO::Sara {

  template <typename T>
  struct CircleSolver2D
  {
    using model_type = Line2<T>;

    static constexpr sample_size = 3;

    template <typename Mat>
    inline auto operator()(const Mat& pts) const
    {
      return fit_circle_2d(pts.transpose());
    }
  };

  template <typename T>
  struct CirclePointDistance2D
  {
    CirclePointDistance2D() = default;

    CirclePointDistance2D(const Circle<T, 2>& c) noexcept
      : circle{c}
    {
    }

    template <typename Mat>
    auto operator()(const Mat& points) -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      return ((points.colwise() - circle.center)
                  .colwise()
                  .squaredNorm()
                  .array() -
              std::pow(circle.radius, 2))
          .abs()
          .matrix();
    }

    Circle<T, 2> circle;
  };

}  // namespace DO::Sara
