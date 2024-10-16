// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>

#include <array>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup MinimalSolvers Minimal Solvers
  //! @{

  //! @brief Optimized Nister 5-point solver for the estimation of the essential
  //! matrix.
  struct NisterFivePointAlgorithm
  {
    using model_type = EssentialMatrix;
    using matrix_type = Eigen::Matrix<double, 3, 5>;
    using matrix_view_type = Eigen::Map<const matrix_type>;
    using data_point_type = std::array<TensorView_<double, 2>, 2>;

    static constexpr auto num_points = 5;
    static constexpr auto num_models = 10;

    DO_SARA_EXPORT
    auto extract_null_space(const Matrix<double, 3, 5>& p_left,
                            const Matrix<double, 3, 5>& p_right) const
        -> Eigen::Matrix<double, 9, 4>;

    DO_SARA_EXPORT
    auto build_essential_matrix_constraints(const double X[9],  //
                                            const double Y[9],  //
                                            const double Z[9],  //
                                            const double W[9]) const
        -> Eigen::Matrix<double, 10, 20>;

    DO_SARA_EXPORT
    auto
    inplace_gauss_jordan_elimination(Matrix<double, 10, 20>&) const -> void;

    DO_SARA_EXPORT
    auto calculate_resultant_determinant(
        const double row_major_resultant_matrix[6 * 10]) const
        -> UnivariatePolynomial<double>;

    DO_SARA_EXPORT
    auto calculate_resultant_determinant_minors(
        const double row_major_resultant_matrix[6 * 10]) const
        -> std::array<UnivariatePolynomial<double>, 3>;

    DO_SARA_EXPORT
    auto solve_reduced_constraint_system(
        const Eigen::Matrix<double, 6, 10, Eigen::RowMajor>&,  //
        const double X[9], const double Y[9],                  //
        const double Z[9],
        const double W[9]) const -> std::vector<EssentialMatrix>;

    DO_SARA_EXPORT
    auto find_essential_matrices(const Matrix<double, 3, 5>& left,
                                 const Matrix<double, 3, 5>& right) const
        -> std::vector<EssentialMatrix>;

    auto operator()(const Matrix<double, 3, 5>& left,
                    const Matrix<double, 3, 5>& right) const
    {
      return find_essential_matrices(left, right);
    }

    auto
    operator()(const data_point_type& X) const -> std::vector<EssentialMatrix>
    {
      const matrix_type left = X[0].colmajor_view().matrix();
      const matrix_type right = X[1].colmajor_view().matrix();
      return this->operator()(left, right);
    }
  };

  //! @}

}  // namespace DO::Sara
