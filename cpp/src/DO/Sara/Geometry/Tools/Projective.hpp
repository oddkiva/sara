// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara::Projective {

  //! @addtogroup GeometryTools
  //! @{

  template <typename T, int N>
  using Point = Eigen::Matrix<T, N + 1, 1>;

  template <typename T>
  using Point2 = Point<T, 2>;

  template <typename T>
  using Point3 = Point<T, 3>;


  template <typename T>
  using Line2 = Eigen::Matrix<T, 3, 1>;


  template <typename T>
  using Plane3 = Eigen::Matrix<T, 4, 1>;


  template <typename T, int N>
  inline auto euclidean(const Point<T, N>& p) -> Point<T, N>
  {
    return p.hnormalized();
  }

  template <typename T>
  inline auto intersection(const Line2<T>& l1, const Line2<T>& l2) -> Point2<T>
  {
    return l1.cross(l2);
  }

  template <typename T>
  inline auto line(const Point2<T>& p, const Point2<T>& q) -> Line2<T>
  {
    return p.cross(q);
  }

  template <typename T>
  inline auto normal(const Line2<T>& l) -> Eigen::Matrix<T, 2, 1>
  {
    return l.head(2);
  }

  template <typename T>
  inline auto tangent(const Line2<T>& l) -> Eigen::Matrix<T, 2, 1>
  {
    const auto n = normal(l);
    return {-n(1), n(0)};
  }

  template <typename T>
  inline auto point_to_line_distance(const Point2<T>& p, const Line2<T>& l)
  {
    return std::abs(l.dot(p) / l.head(2).norm());
  }

  //! @}

}  // namespace DO::Sara::Projective
