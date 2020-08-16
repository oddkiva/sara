// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Sara/Geometry/Objects/Circle.hpp>


namespace DO::Sara {

  //! @brief Fit a circle from the set of possibly noisy 2D points.
  /*!
   *  This function implements a direct method based on perpendicular
   *  bisectors.
   *
   *  Reference:
   *  http://www.cs.bsu.edu/homepages/kjones/kjones/circles.pdf
   *
   */
  template <typename T, int N>
  auto fit_circle_2d(const Eigen::Matrix<T, N, 2>& p)
      -> Circle<T, 2>
  {
    auto x = p.col(0);
    auto y = p.col(1);

    // Some convenient notations.
    const auto n = p.rows();
    const auto x2 = x.array().square().matrix();
    const auto x3 = x.array().pow(3).matrix();
    const auto y2 = y.array().square().matrix();
    const auto y3 = y.array().pow(3).matrix();
    const auto xy = (x.array() * y.array()).matrix();

    // Auxiliary variables.
    const auto A = n * x2.sum() - std::pow(x.sum(), 2);
    const auto B = n * xy.sum() - x.sum() * y.sum();
    const auto C = n * y2.sum() - std::pow(y.sum(), 2);
    const auto D = 0.5 * (n * x.dot(y2) - x.sum() * y2.sum() + n * x3.sum() -
                          x.sum() * x2.sum());
    const auto E = 0.5 * (n * y.dot(x2) - y.sum() * x2.sum() + n * y3.sum() -
                          y.sum() * y2.sum());

    // Calculate the center.
    const auto c = Eigen::Matrix<T, 2, 1>{(D * C - B * E) / (A * C - B * B),
                                          (A * E - B * D) / (A * C - B * B)};

    // Calculate the radius.
    const auto r = ((x.array() - c.x()).pow(2) +  //
                    (y.array() - c.y()).pow(2))
                       .sqrt()
                       .sum() /
                   n;

    return {c, r};
  }

}  // namespace DO::Sara
