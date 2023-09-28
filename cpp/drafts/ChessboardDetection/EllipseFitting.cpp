// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>

namespace DO::Sara {

  auto fit_ellipse(const std::vector<Eigen::Vector2f>& line) -> Eigen::Matrix3f
  {
    if (line.size() < 6)
      throw std::runtime_error{
          "Ellipse fitting not feasible: 6 data points minimum!"};

    // Write down the equation.
    const auto n = static_cast<int>(line.size());
    auto A = Eigen::MatrixXf(n, 6);
    for (auto i = 0; i < n; ++i)
    {
      const auto& x = line[i].x();
      const auto& y = line[i].y();
      A.row(i) << x * x, y * y, x * y, 2 * x, 2 * y, 1;
    }

    auto svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector<float, 6> e = svd.matrixV().col(5);

    auto E = Eigen::Matrix3f{};
    // clang-format off
    E << e(0), e(2), e(3),
         e(2), e(1), e(4),
         e(3), e(4), e(5);
    // clang-format on

    return E;
  }

}  // namespace DO::Sara
