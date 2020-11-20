// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/StdVectorHelpers.hpp>


namespace DO::Sara {

  //! @addtogroup GeometryAlgorithms
  //! @{

  namespace detail {

    template <typename T>
    auto orthogonal_distance(const Eigen::Matrix<T, 2, 1>& a,
                             const Eigen::Matrix<T, 2, 1>& b,
                             const Eigen::Matrix<T, 2, 1>& x)
    {
      auto M = Matrix2d{};
      M.col(0) = (b - a).normalized();
      M.col(1) = x - a;
      return abs(M.determinant());
    }

    template <typename T>
    auto ramer_douglas_peucker(const Eigen::Matrix<T, 2, 1>* in_first,  //
                               const Eigen::Matrix<T, 2, 1>* in_last,   //
                               T eps)                                   //
        -> std::vector<Eigen::Matrix<T, 2, 1>>
    {
      if (in_first == in_last)
        return {*in_first};

      auto pivot = in_first;
      auto pivot_dist = 0.;

      for (auto p = in_first + 1; p != in_last + 1; ++p)
      {
        auto dist = orthogonal_distance(*in_first, *in_last, *p);
        if (pivot_dist < dist)
        {
          pivot = p;
          pivot_dist = dist;
        }
      }

      auto out = std::vector<Eigen::Matrix<T, 2, 1>>{};
      if (pivot_dist > eps)
      {
        auto v1 = ramer_douglas_peucker(in_first, pivot, eps);
        auto v2 = ramer_douglas_peucker(pivot, in_last, eps);

        out.insert(out.end(), v1.begin(), v1.end());
        if (!v2.empty())
          out.insert(out.end(), v2.begin() + 1, v2.end());
      }
      else
        out = {*in_first, *in_last};

      return out;
    }

  }  // namespace detail

  DO_SARA_EXPORT
  std::vector<Point2d> ramer_douglas_peucker(std::vector<Point2d> contours,
                                             double eps);

  //! @}

}  // namespace DO::Sara
