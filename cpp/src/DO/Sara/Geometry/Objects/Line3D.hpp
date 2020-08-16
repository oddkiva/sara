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

#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara::Projective {

  //! @addtogroup GeometryTools
  //! @{

  template <typename T>
  using PluckerCoordinates = Eigen::Matrix<T, 6, 1>;

  template <typename T>
  struct Line3
  {
    Eigen::Matrix<T, 3, 1> point;
    Eigen::Matrix<T, 3, 1> direction;

    inline auto normalize_direction()
    {
      direction.normalize();
    }

    inline auto moment() const
    {
      return point.cross(direction);
    }

    inline auto moment(const Eigen::Matrix<T, 3, 1>& other_point) const
    {
      return moment() - other_point.cross(direction);
    }

    inline auto plucker_coordinates(const Line3<T>& l) const -> PluckerCoordinates<T>
    {
      auto pc = PluckerCoordinates<T>{};
      pc << direction, moment();
      return pc;
    }

    inline auto point(T t) const -> Eigen::Matrix<T, 3, 1>
    {
      return point + t * direction;
    }

    inline auto reciprocal_product(const Line3& other) const
    {
      return direction.dot(other.point) + other.direction.dot(point);
    }

    inline auto intersect(const Line3& other, T tol) const
    {
      return std::abs(reciprocal_product(other)) < tol;
    }

    inline auto distance(const Line3& other, T tol) const
    {
      const auto& l1 = direction;
      const auto& l2 = other.direction;
      const auto l1_cross_l2_norm = l1.cross(l2).norm();
      if (l1_cross_l2_norm > tol)
        return reciprocal_product(other) / l1_cross_l2_norm;
      else
      {
        const auto s = l1(0) / l2(0);
        return (l1.cross(moment() - s * other.moment())).norm() / l1.squaredNorm();
      }
    }

    // TODO: implement common perpendicular and feet of the common
    // perpendicular.
    //
    // cf. http://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
  };


  //! @}

}  // namespace DO::Sara::Projective
