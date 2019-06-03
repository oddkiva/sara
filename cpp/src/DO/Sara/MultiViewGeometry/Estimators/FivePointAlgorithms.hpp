#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/Math/Polynomial.hpp>


namespace DO { namespace Sara {

  inline void
  five_point_essential_matrix_nister(const Matrix<double, 3, 5>& p,  //
                                     const Matrix<double, 3, 5>& q)
  {
      Matrix<double, 5, 9> A;

      for (int i = 0; i < 5; ++i)
        A.row(i) <<                          //
            p(i, 0) * q.col(i).transpose(),  //
            p(i, 1) * q.col(i).transpose(),  //
            q.col(i).transpose();

      // Calculate the bases of the null-space.
      Eigen::JacobiSVD<Matrix<double, 5, 9>> svd(A, Eigen::ComputeFullV);
      Matrix<double, 4, 9> K = svd.matrixV().rightCols(4);  // K as Ker.

      // The essential matrix lives in right null space K.
      // E = x * K[1,:] + y * K[2, :] + z * K3 + 1 * K4.
      // E = K * e
      // e = np.array([x, y, z, 1])
      const Matrix3d X = Map<Matrix<double, 3, 3>>{K.col(0).data()};
      const Matrix3d Y = Map<Matrix<double, 3, 3>>{K.col(1).data()};
      const Matrix3d Z = Map<Matrix<double, 3, 3>>{K.col(2).data()};
      const Matrix3d W = Map<Matrix<double, 3, 3>>{K.col(3).data()};


      const auto x = Monomial{variable("x")};
      const auto y = Monomial{variable("y")};
      const auto z = Monomial{variable("z")};
      const auto one_ = Monomial{one()};

      const auto E = x * X + y * Y + z * Z + one_ * W;

      // cf. Nister's paper
      // In numpy notations, E must satisfy:
      // 2 * E * E.T * E - np.tr(E * E.T) * E == 0
      const auto EEt = E * E.t();

      auto P = det(E);
      auto P1 = EEt * E;
      auto P2 = trace(EEt) * E;
      P2 *= -0.5;
      const auto Q = P1 - P2;

      const Monomial monomials[] = {x.pow(3), y.pow(3), x.pow(2) * y,
                                    x * y.pow(2), x.pow(2) * z, x.pow(2),
                                    y.pow(2) * z, y.pow(2), x * y * z, x * y,
                                    //
                                    x, x * z, x * z.pow(2),
                                    //
                                    y, y * z, y * z.pow(2),
                                    //
                                    one_, z, z.pow(2), z.pow(3)};

      // ===========================================================================
      // As per Nister paper.
      //
      Matrix<double, 10, 20> A;
      A.setZero();

      // Save Q in the matrix.
      for (int j = 0; j < 20; ++j)
      {
        auto coeff = P.coeffs.find(monomials[j]);
        if (coeff == P.coeffs.end())
          continue;
        A(0, j) = coeff->second;
      }

      // Save P in the matrix.
      for (int a = 0; a < 3; ++a)
      {
        for (int b = 0; b < 3; ++b)
        {
          const auto i = 3 * a + b;
          for (int j = 0; j < 20; ++j)
            A(i, j) = Q(a, b).coeffs[monomials[j]];
        }
      }

      // ===========================================================================
      // 1. Perform Gauss-Jordan elimination on A.
      Eigen::FullPivLU<Matrix<double, 10, 20>> lu(A);
      Matrix<double, 10, 20> U = lu.matrixLU().triangularView<Upper>();
      cout << "U =\n" << U << endl;
  }

} /* namespace Sara */
} /* namespace DO */
