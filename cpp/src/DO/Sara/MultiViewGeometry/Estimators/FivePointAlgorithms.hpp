#pragma once

#include <DO/Sara/Core/Math/Polynomial.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <array>


namespace DO::Sara {

//! @{
using Matrix10d = Eigen::Matrix<double, 10, 10>;
using Vector10d = Eigen::Matrix<double, 10, 10>;
using Vector9d = Eigen::Matrix<double, 9, 1>;
//! @}


struct FivePointAlgorithmBase
{
  const Monomial x{variable("x")};
  const Monomial y{variable("y")};
  const Monomial z{variable("z")};
  const Monomial one_{one()};

  auto extract_null_space(const Matrix<double, 3, 5>& p_left,
                          const Matrix<double, 3, 5>& p_right) const
      -> Matrix<double, 9, 4>;

  auto reshape_null_space(const Matrix<double, 9, 4>&) const
      -> std::array<Matrix3d, 4>;

  auto essential_matrix_expression(const std::array<Matrix3d, 4>&) const
      -> Polynomial<Matrix3d>;

  auto build_essential_matrix_constraints(const Polynomial<Matrix3d>&,
                                          const std::array<Monomial, 20>&) const
      -> Matrix<double, 10, 20>;
};


struct NisterFivePointAlgorithm : FivePointAlgorithmBase
{
  const std::array<Monomial, 20> monomials{
      x.pow(3),
      y.pow(3),
      x.pow(2) * y,
      x * y.pow(2),
      x.pow(2) * z,
      x.pow(2),
      y.pow(2) * z,
      y.pow(2),
      x * y* z,
      x * y,
      //
      x, x* z, x* z.pow(2),
      //
      y, y* z, y* z.pow(2),
      //
      one_, z, z.pow(2), z.pow(3)};

  auto build_essential_matrix_constraints(const Polynomial<Matrix3d>& E) const
      -> Matrix<double, 10, 20>
  {
    return FivePointAlgorithmBase::build_essential_matrix_constraints(
        E, monomials);
  }

  auto inplace_gauss_jordan_elimination(Matrix<double, 10, 20>&) const -> void;

  auto form_resultant_matrix(const Matrix<double, 6, 10>&,
                             Univariate::UnivariatePolynomial<double>[3][3]) const
      -> void;

  auto solve_essential_matrix_constraints(const std::array<Matrix3d, 4>&,
                                          const Matrix<double, 10, 20>&) const
      -> std::vector<Matrix3d>;
};


struct SteweniusFivePointAlgorithm : FivePointAlgorithmBase
{
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
      x, y, z, one_
  };

  auto build_essential_matrix_constraints(const Polynomial<Matrix3d>& E) const
      -> Matrix<double, 10, 20>
  {
    return FivePointAlgorithmBase::build_essential_matrix_constraints(
        E, monomials);
  }

  auto solve_essential_matrix_constraints(const Matrix<double, 9, 4>&,
                                          const Matrix<double, 10, 20>&) const
      -> std::vector<Matrix3d>;
};

} /* namespace DO::Sara */
