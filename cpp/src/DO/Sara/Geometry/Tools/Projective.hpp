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
  using Line = Eigen::Matrix<T, N + 1, 1>;

  template <typename T, int N>
  using Point = Eigen::Matrix<T, N + 1, 1>;

  template <typename T>
  using Line2 = Line<T, 2>;

  template <typename T>
  using Line3 = Line<T, 3>;

  template <typename T>
  using Point2 = Line<T, 2>;

  template <typename T>
  using Point3 = Line<T, 3>;

  template <typename T, int N>
  inline Point<T, N> euclidean(const Point<T, N>& p)
  {
    return p.hnormalized();
  }

  template <typename T>
  inline Point2<T> intersection(const Line2<T>& l1, const Line2<T>& l2)
  {
    return l1.cross(l2);
  }

  template <typename T>
  inline Line2<T> line(const Point2<T>& p, const Point2<T>& q)
  {
    return p.cross(q);
  }

  template <typename T>
  inline double distance(const Point2<T>& p, const Line2<T>& l)
  {
    return std::abs(l.dot(p) / l.head(2).norm());
  }

  //! @}

}  // namespace DO::Sara::Projective
