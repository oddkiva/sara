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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara {

  template <typename T>
  struct LineSolver2D
  {
    using model_type = Projective::Line2<T>;

    static constexpr auto num_points = 2;

    template <typename Mat>
    inline auto operator()(const Mat& ab) const
    {
      const Eigen::Matrix<T, 3, 2> abT = ab.transpose();
      const auto& a = abT.col(0);
      const auto& b = abT.col(1);
      return Projective::line(a.eval(), b.eval());
    }
  };

  template <typename T>
  struct LinePointDistance2D
  {
    using model_type = Projective::Line2<T>;
    using scalar_type = T;

    LinePointDistance2D() = default;

    LinePointDistance2D(const Projective::Line2<T>& l) noexcept
      : line{l}
    {
    }

    template <typename Mat>
    inline auto operator()(const Mat& points) const
        -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      return (points * line / line.head(2).norm()).cwiseAbs();
    }

    Projective::Line2<T> line;
  };

}  // namespace DO::Sara
