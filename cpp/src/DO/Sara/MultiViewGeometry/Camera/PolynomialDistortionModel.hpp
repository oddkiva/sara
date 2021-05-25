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
    using vector2 = Eigen::Matrix<T, 2, 1>;
    using matrix2 = Eigen::Matrix<T, 2, 2>;

    using radial_coefficient_array = Eigen::Matrix<T, K, 1>;
    using tangential_coefficient_array = Eigen::Matrix<T, P, 1>;

    //! @brief Radial distortion coefficients.
    radial_coefficient_array k;
    //! @brief Tangential distortion coefficients.
    tangential_coefficient_array p;

    //! @brief Apply only in the normalized coordinates.
    inline auto lens_distortion(const vector2& xdn) const -> vector2
    {
      // Radial term.
      const auto r2 = xdn.squaredNorm();
      auto rpowers = radial_coefficient_array{};
      for (auto i = 0; i < K; ++i)
        rpowers[i] = std::pow(r2, i + 1);
      const auto radial = vector2{k.dot(rpowers) * xdn};

      // Tangential term.
      const matrix2 Tmat = r2 * matrix2::Identity() + 2 * xdn * xdn.transpose();
      const vector2 tangential = Tmat * p;

      return radial + tangential;
    }

    //! @brief Apply only in the normalized coordinates.
    inline auto distort(const vector2& xdn) const -> vector2
    {
      return xdn + lens_distortion(xdn);
    }

    //! @brief Iterative method to remove distortion.
    inline auto correct(const vector2& xd,
                        int num_iterations = 10,
                        T eps = T(1e-8)) const
        -> vector2
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
    using base_type = PolynomialDistortionModel<T, K, P>;
    using vector2 = typename base_type::vector2;
    using matrix2 = typename base_type::matrix2;
    using radial_coefficient_array = typename base_type::radial_coefficient_array;
    using tangential_coefficient_array = typename base_type::tangential_coefficient_array;

    using base_type::k;
    using base_type::p;
    vector2 dc = vector2::Zero();

    //! @brief Apply only in the normalized coordinates.
    inline auto lens_distortion(const vector2& xdn) const -> vector2
    {
      // Radial term.
      const auto r2 = (xdn - dc).squaredNorm();
      auto rpowers = radial_coefficient_array{};
      for (auto i = 0; i < K; ++i)
        rpowers[i] = std::pow(r2, i + 1);
      const auto radial = vector2{k.dot(rpowers) * xdn};

      // Tangential term.
      const matrix2 Tmat = r2 * matrix2::Identity() + 2 * xdn * xdn.transpose();
      const vector2 tangential = Tmat * p;

      return radial + tangential;
    }

    //! @brief Apply only in the normalized coordinates.
    inline auto distort(const vector2& xdn) const -> vector2
    {
      return xdn + lens_distortion(xdn);
    }

    //! @brief Iterative method to remove distortion.
    inline auto correct(const vector2& xd,
                        int num_iterations = 10,
                        T eps = T(1e-8)) const
        -> vector2
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
