#pragma once

#include <DO/Sara/Core.hpp>


namespace DO { namespace Sara {

  inline void
  five_point_essential_matrix_li_hartley(const Matrix<double, 3, 5>& x,  //
                                         const Matrix<double, 3, 5>& y,
                                         Matrix3d& E)
  {
      Matrix<double, 5, 9> A;

      for (int i = 0; i < 5; ++i)
        A.row(i) <<                          //
            x(i, 0) * y.col(i).transpose(),  //
            x(i, 1) * y.col(i).transpose(),  //
            y.col(i).transpose();

      // Calculate the bases of the null-space.
      Eigen::JacobiSVD<Matrix<double, 5, 9>> svd(A, Eigen::ComputeFullV);
      Matrix<double, 4, 9> K = svd.matrixV().rightCols(4);  // K as Ker.

      auto F = std::array<Matrix<double, 3, 3>, 4>{};
      for (int i = 0; i < 4; ++i)
        F[i] = Map<Matrix<double, 3, 3>>{K.col(i).data()};

      // The essential matrix lives in right null space K.
      // E = x * K[1,:] + y * K[2, :] + z * K3 + 1 * K4.
      // E = K * e
      // e = np.array([x, y, z, 1])

      // cf. Nister's paper
      // In numpy notations, E must satisfy:
      // 2 * E * E.T * E - np.tr(E * E.T) * E == 0

      MultiArray<Matrix<double, 3, 3>, 3> c(3, 3, 3);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 3; ++k)
            c(i, j, k) = 2 * F[i] * F[j].transpose() * F[k] -
                         (F[i] * F[j].transpose()).trace() * F[k];

      // Calculate the coefficient for z**3.
      std::array<Matrix<double, 3, 3>, 4> C;
      C[0] = c(2, 2, 2);
      C[1] = c(2, 2, 0) + c(2, 2, 1) + c(2, 2, 3)
           + c(2, 0, 2) + c(2, 1, 2) + c(2, 3, 2)
           + c(0, 2, 2) + c(1, 2, 2) + c(3, 2, 2)
  }

} /* namespace Sara */
} /* namespace DO */
