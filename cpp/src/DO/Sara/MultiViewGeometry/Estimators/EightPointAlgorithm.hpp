#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  inline void eight_point_alg_fundamental(const MatrixXd& x, const MatrixXd& y,
                                          Matrix3d& F)
  {
    Matrix<double, 8, 9> A(8, 9);

    for (int i = 0; i < 8; ++i)
      A.row(i) <<                          //
          x(i, 0) * y.col(i).transpose(),  //
          x(i, 1) * y.col(0).transpose(),  //
          y.col(i).transpose();

    Eigen::JacobiSVD<MatrixXd> svd(A);
    Matrix<double, 9, 1> S = svd.singularValues();
    Matrix<double, 9, 1> f = svd.matrixV().col(8);

    F.col(0) = f.block(0, 0, 1, 3);
    F.col(1) = f.block(3, 0, 1, 3);
    F.col(2) = f.block(6, 0, 1, 3);
  }

  inline void eight_point_alg_homography(const MatrixXd& x, const MatrixXd& y,
                                         Matrix3d& H)
  {
    const Vector2d p1 = x.col(0);
    const Vector2d p2 = x.col(1);
    const Vector2d p3 = x.col(2);
    const Vector2d p4 = x.col(3);

    const Vector2d q1 = y.col(0);
    const Vector2d q2 = y.col(1);
    const Vector2d q3 = y.col(2);
    const Vector2d q4 = y.col(3);

    using T = double;

    Matrix<double, 8, 8> M;
    M <<
    p1.x(), p1.y(), T(1),   T(0),   T(0), T(0), -p1.x()*q1.x(), -p1.y()*q1.x(),
      T(0),   T(0), T(0), p1.x(), p1.y(), T(1), -p1.x()*q1.y(), -p1.y()*q1.y(),
    p2.x(), p2.y(), T(1),   T(0),   T(0), T(0), -p2.x()*q2.x(), -p2.y()*q2.x(),
      T(0),   T(0), T(0), p2.x(), p2.y(), T(1), -p2.x()*q2.y(), -p2.y()*q2.y(),
    p3.x(), p3.y(), T(1),   T(0),   T(0), T(0), -p3.x()*q3.x(), -p3.y()*q3.x(),
      T(0),   T(0), T(0), p3.x(), p3.y(), T(1), -p3.x()*q3.y(), -p3.y()*q3.y(),
    p4.x(), p4.y(), T(1),   T(0),   T(0), T(0), -p4.x()*q4.x(), -p4.y()*q4.x(),
      T(0),   T(0), T(0), p4.x(), p4.y(), T(1), -p4.x()*q4.y(), -p4.y()*q4.y();

    Matrix<double, 8, 1> b;
    b << q1.x(), q1.y(), q2.x(), q2.y(), q3.x(), q3.y(), q4.x(), q4.y();

    Matrix<T, 8, 1> h(M.colPivHouseholderQr().solve(b));

    H << h[0], h[1], h[2],
         h[3], h[4], h[5],
         h[6], h[7], 1.0;
  }

} /* namespace Sara */
} /* namespace DO */
