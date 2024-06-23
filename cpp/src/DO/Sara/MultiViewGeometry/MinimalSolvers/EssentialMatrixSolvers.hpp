// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/Polynomial.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>

#include <array>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup MinimalSolvers Minimal Solvers
  //! @{

  //! @brief Matrix aliases.
  //! @{
  using Matrix10d = Eigen::Matrix<double, 10, 10>;
  using Vector10d = Eigen::Matrix<double, 10, 10>;
  using Vector9d = Eigen::Matrix<double, 9, 1>;
  //! @}


  //! @brief Five-point algorithm for the essential matrix.
  //! @{
  struct FivePointAlgorithmBase
  {
    const Monomial x{variable("x")};
    const Monomial y{variable("y")};
    const Monomial z{variable("z")};
    const Monomial one_{one()};

    DO_SARA_EXPORT
    auto extract_null_space(const Matrix<double, 3, 5>& p_left,
                            const Matrix<double, 3, 5>& p_right) const
        -> Matrix<double, 9, 4>;

    DO_SARA_EXPORT
    auto reshape_null_space(const Matrix<double, 9, 4>&) const
        -> std::array<Matrix3d, 4>;

    DO_SARA_EXPORT
    auto essential_matrix_expression(const std::array<Matrix3d, 4>&) const
        -> Polynomial<Matrix3d>;

    DO_SARA_EXPORT
    auto build_essential_matrix_constraints(
        const Polynomial<Matrix3d>&,
        const std::array<Monomial, 20>&) const -> Matrix<double, 10, 20>;
  };


  struct NisterFivePointAlgorithm : FivePointAlgorithmBase
  {
    using model_type = EssentialMatrix;
    using matrix_type = Eigen::Matrix<double, 3, 5>;
    using matrix_view_type = Eigen::Map<const matrix_type>;
    using data_point_type = std::array<TensorView_<double, 2>, 2>;

    static constexpr auto num_points = 5;
    static constexpr auto num_models = 10;

    // clang-format off
    const std::array<Monomial, 20> monomials{
        x.pow(3),
        y.pow(3),
        x.pow(2) * y,
        x * y.pow(2),
        x.pow(2) * z,
        x.pow(2),
        y.pow(2) * z,
        y.pow(2),
        x * y * z,
        x * y,
        //
        x,
        x * z,
        x * z.pow(2),
        //
        y,
        y * z,
        y * z.pow(2),
        //
        one_,
        z,
        z.pow(2),
        z.pow(3)
    };
    // clang-format on

    auto build_essential_matrix_constraints(const Polynomial<Matrix3d>& E) const
        -> Matrix<double, 10, 20>
    {
      return FivePointAlgorithmBase::build_essential_matrix_constraints(
          E, monomials);
    }

    auto build_essential_matrix_fast(const double X[9],  //
                                     const double Y[9],  //
                                     const double Z[9],  //
                                     const double W[9]) const
        -> Eigen::Matrix<double, 10, 20>;


    DO_SARA_EXPORT
    auto
    inplace_gauss_jordan_elimination(Matrix<double, 10, 20>&) const -> void;

    DO_SARA_EXPORT
    auto
    form_resultant_matrix(const Matrix<double, 6, 10>&,
                          UnivariatePolynomial<double, -1>[3][3]) const -> void;

    DO_SARA_EXPORT
    auto solve_essential_matrix_constraints(const std::array<Matrix3d, 4>&,
                                            const Matrix<double, 10, 20>&) const
        -> std::vector<EssentialMatrix>;

    DO_SARA_EXPORT
    auto find_essential_matrices(const Matrix<double, 3, 5>& left,
                                 const Matrix<double, 3, 5>& right) const
        -> std::vector<EssentialMatrix>;

    DO_SARA_EXPORT
    auto find_essential_matrices_fast(const Matrix<double, 3, 5>& left,
                                      const Matrix<double, 3, 5>& right) const
        -> std::vector<EssentialMatrix>;

    auto operator()(const Matrix<double, 3, 5>& left,
                    const Matrix<double, 3, 5>& right) const
    {
      return find_essential_matrices_fast(left, right);
    }

    auto
    operator()(const data_point_type& X) const -> std::vector<EssentialMatrix>
    {
      const matrix_type left = X[0].colmajor_view().matrix();
      const matrix_type right = X[1].colmajor_view().matrix();
      return this->operator()(left, right);
    }
  };

  struct SteweniusFivePointAlgorithm : FivePointAlgorithmBase
  {
    using model_type = EssentialMatrix;
    using matrix_type = Eigen::Matrix<double, 3, 5>;
    using matrix_view_type = Eigen::Map<const matrix_type>;
    using data_point_type = std::array<TensorView_<double, 2>, 2>;

    static constexpr auto num_points = 5;
    static constexpr auto num_models = 10;

    // clang-format off
    const std::array<Monomial, 20> monomials{
        x * x * x,
        x * x * y,
        x * y * y,
        y * y * y,
        x * x * z,
        x * y * z,
        y * y * z,
        x * z * z,
        y * z * z,
        z * z * z,
        x * x,
        x * y,
        y * y,
        x * z,
        y * z,
        z * z,
        //  The solutions of interests
        x,
        y,
        z,
        one_
    };
    // clang-format on

    auto build_essential_matrix_constraints(const Polynomial<Matrix3d>& E) const
        -> Matrix<double, 10, 20>
    {
      return FivePointAlgorithmBase::build_essential_matrix_constraints(
          E, monomials);
    }

    DO_SARA_EXPORT
    auto solve_essential_matrix_constraints(const Matrix<double, 9, 4>&,
                                            const Matrix<double, 10, 20>&) const
        -> std::vector<EssentialMatrix>;

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

  //! @}

} /* namespace DO::Sara */
