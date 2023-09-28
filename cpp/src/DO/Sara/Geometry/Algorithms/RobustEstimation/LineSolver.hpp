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
#include <DO/Sara/Geometry/Algorithms/RobustEstimation/PointList.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara {

  template <typename T>
  struct LineSolver2D
  {
    using model_type = Projective::Line2<T>;
    using data_point_type = TensorView_<T, 2>;

    static constexpr auto num_points = 2;
    static constexpr auto num_models = 1;

    inline auto operator()(const data_point_type& ab) const
        -> std::array<model_type, num_models>
    {
      const auto abT = ab.colmajor_view().matrix();
      const Eigen::Vector3<T> a = abT.col(0);
      const Eigen::Vector3<T> b = abT.col(1);
      return {Projective::line(a, b)};
    }
  };

  template <typename T>
  struct LinePointDistance2D
  {
    using model_type = Projective::Line2<T>;
    using scalar_type = T;

    LinePointDistance2D() = default;

    inline auto set_model(const Projective::Line2<T>& l) noexcept
    {
      line = l;
    }

    template <typename Derived>
    inline auto operator()(const Eigen::MatrixBase<Derived>& points) const
        -> Eigen::Vector<T, Eigen::Dynamic>
    {
      if (points.rows() == 2)
        return ((line.transpose() * points.colwise().homogeneous()) /
                line.head(2).norm())
            .cwiseAbs();

      if (points.rows() == 3)
        return ((line.transpose() * points) / line.head(2).norm()).cwiseAbs();

      throw std::runtime_error{"The point dimension must be 2 or 3"};
      return {};
    }

    Projective::Line2<T> line;
  };

}  // namespace DO::Sara
