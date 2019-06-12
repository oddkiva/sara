#pragma once

#include <DO/Sara/Core.hpp>


namespace DO { namespace Sara {

  using Mat34d = Matrix<double, 3, 4>;

  struct Camera
  {
    using matrix_type = Matrix<double, 3, 4>;

    matrix_type& operator matrix_type()
    {
      return P;
    }

    const matrix_type& operator matrix_type() const
    {
      return P;
    }

    matrix_type _P;
  };


  auto linear_triangulation(const Camera& P1, const Camera& P2,
                            const MatrixXd& u1, const MatrixXd& u2, MatrixXd& X)
  {
    MatrixXd A{u1.cols() * 4, 4};

    for (int i = 0; i < u1.cols(); ++i)
    {
      A.row(4 * i + 0) = u1(1, i) * P1.row(2) - P1.row(1);
      A.row(4 * i + 1) = u1(0, i) * P1.row(2) - P1.row(0);
      A.row(4 * i + 2) = u2(1, i) * P2.row(2) - P2.row(1);
      A.row(4 * i + 3) = u2(0, i) * P2.row(2) - P2.row(0);
    }

    JacobiSVD<MatrixXd> svd(A);
    svd.computeFullV();
    VectorXd x = V.col(4);
  }




} /* namespace Sara */
} /* namespace DO */
