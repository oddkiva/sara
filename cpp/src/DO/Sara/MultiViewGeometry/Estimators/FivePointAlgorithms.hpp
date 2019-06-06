#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/Math/Polynomial.hpp>


namespace DO { namespace Sara {

  struct NisterFivePointAlgorithm
  {
    const Monomial x{variable("x")};
    const Monomial y{variable("y")};
    const Monomial z{variable("z")};
    const Monomial one_{one()};

    const std::array<Monomial, 20> monomials{
        x.pow(3), y.pow(3), x.pow(2) * y, x* y.pow(2), x.pow(2) * z, x.pow(2),
        y.pow(2) * z, y.pow(2), x* y* z, x* y,
        //
        x, x* z, x* z.pow(2),
        //
        y, y* z, y* z.pow(2),
        //
        one_, z, z.pow(2), z.pow(3)};

    auto extract_null_space(const Matrix<double, 3, 5>& p,
                            const Matrix<double, 3, 5>& q)
        -> std::array<Matrix3d, 4>;

    auto
    essential_matrix_expression(const std::array<Matrix3d, 4>& null_space_bases)
        -> Polynomial<Matrix3d>;

    auto build_epipolar_constraints(const Polynomial<Matrix3d>& E)
        -> Matrix<double, 10, 20>;

    auto solve_epipolar_constraints(const Matrix<double, 10, 20>& A)
        -> std::vector<Vector3d>;

    auto find_essential_matrices(const Matrix<double, 3, 5>& p,
                                 const Matrix<double, 3, 5>& q)
        -> std::vector<Matrix3d>;
  };

} /* namespace Sara */
} /* namespace DO */
