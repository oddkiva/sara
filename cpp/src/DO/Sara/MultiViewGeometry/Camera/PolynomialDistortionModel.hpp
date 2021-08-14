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

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T, int K = 3, int P = 2>
  struct PolynomialDistortionModel
  {
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Matrix2 = Eigen::Matrix<T, 2, 2>;

    using RadialCoefficientArray = Eigen::Matrix<T, K, 1>;
    using TangentialCoefficientArray = Eigen::Matrix<T, P, 1>;

    //! @brief Radial distortion coefficients.
    RadialCoefficientArray k;
    //! @brief Tangential distortion coefficients.
    TangentialCoefficientArray p;

    //! @brief Apply only in the normalized coordinates.
    inline auto lens_distortion(const Vector2& xun) const -> Vector2
    {
      // Radial term.
      const auto r2 = xun.squaredNorm();
      auto rpowers = RadialCoefficientArray{};
      rpowers[0] = r2;
      for (auto i = 1; i < K; ++i)
        rpowers[i] = rpowers[i - 1] * r2;
      const auto radial = Vector2{k.dot(rpowers) * xun};

      // Tangential term.
      const Matrix2 Tmat = r2 * Matrix2::Identity() + 2 * xun * xun.transpose();
      const Vector2 tangential = Tmat * p;

      return radial + tangential;
    }

    //! @brief Apply only in the normalized coordinates.
    inline auto distort(const Vector2& xun) const -> Vector2
    {
      return xun + lens_distortion(xun);
    }

    //! @brief Iterative method to remove distortion.
    inline auto correct(const Vector2& xd,
                        int num_iterations = 10,
                        T eps = T(1e-8)) const
        -> Vector2
    {
      auto xu = xd;
      for (auto iter = 0; iter < num_iterations &&
                          (xu + lens_distortion(xu) - xd).norm() > eps;
           ++iter)
        xu = xd - lens_distortion(xu);

      return xu;
    }
  };

  template <typename T, int K = 3, int P = 2>
  struct DecenteredPolynomialDistortionModel : PolynomialDistortionModel<T, K, P>
  {
    using Base = PolynomialDistortionModel<T, K, P>;
    using Vector2 = typename Base::Vector2;
    using Matrix2 = typename Base::Matrix2;
    using RadialCoefficientArray = typename Base::RadialCoefficientArray;
    using TangentialCoefficientArray = typename Base::TangentialCoefficientArray;

    using Base::k;
    using Base::p;
    Vector2 dc = Vector2::Zero();

    //! @brief Apply only in the normalized coordinates.
    inline auto lens_distortion(const Vector2& xdn) const -> Vector2
    {
      // Radial term.
      const auto r2 = (xdn - dc).squaredNorm();
      auto rpowers = RadialCoefficientArray{};
      for (auto i = 0; i < K; ++i)
        rpowers[i] = std::pow(r2, i + 1);
      const auto radial = Vector2{k.dot(rpowers) * xdn};

      // Tangential term.
      const Matrix2 Tmat = r2 * Matrix2::Identity() + 2 * xdn * xdn.transpose();
      const Vector2 tangential = Tmat * p;

      return radial + tangential;
    }

    //! @brief Apply only in the normalized coordinates.
    inline auto distort(const Vector2& xdn) const -> Vector2
    {
      return xdn + lens_distortion(xdn);
    }

    //! @brief Iterative method to remove distortion.
    inline auto correct(const Vector2& xd,
                        int num_iterations = 10,
                        T eps = T(1e-8)) const
        -> Vector2
    {
      auto xu = xd;
      for (auto iter = 0; iter < num_iterations &&
                          (xu + lens_distortion(xu) - xd).norm() > eps;
           ++iter)
        xu = xd - lens_distortion(xu);

      return xu;
    }
  };

}  // namespace DO::Sara
