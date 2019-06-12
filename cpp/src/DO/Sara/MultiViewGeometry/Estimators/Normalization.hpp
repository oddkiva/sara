#pragma once

#include <DO/Sara/Core.hpp>


namespace DO { namespace Sara {

  MatrixXd homogeneous(const MatrixXd& p)
  {
    MatrixXd hp(p.rows() + 1, p.cols());
    hp.block(0, 0, p.rows(), p.cols()) = p;
    hp.row(p.rows()) = RowVectorXd::Ones(p.cols());
    return hp;
  }

  Vector2d mean(const MatrixXd& p)
  {
    auto mean = p.rowwise().sum();
    mean /= p.cols();
    return mean;
  }

  Matrix3d isotropic_scaling(const MatrixXd& p)
  {
    const auto m = mean(p);
    MatrixXd cov = p;
    cov.array() = cov.array().square();
    cov /= p.cols();
    cov -= m;

    Matrix3d T;
    T.setZero();

    T <<  //
      1/cov(0),        0, -m(0),
             0, 1/cov(1), -m(1),
             0,        0,     1;
    T.topRows(2) /= std::sqrt(2);

    return T;
  }

  double cov_xy(const MatrixXd& p, const Vector2d& m)
  {
    auto x = p.row(0);
    auto y = p.row(1);
    x.array() -= m.x();
    y.array() -= m.y();
    const double c = (x * y.transpose()) / p.cols();
    return c;
  }

  auto var(const MatrixXd& p, const Vector2d& m)
  {
    Vector2d x = p.row(0);
    Vector2d y = p.row(1);
    x.array() = x.array() * x.array();
    y.array() = y.array() * y.array();

    double var_x = x.array().sum() / p.cols();
    double var_y = y.array().sum() / p.cols();

    var_x -= m.x()*m.x();
    var_y -= m.y()*m.y();

    return Vector2d{var_x, var_y};
  }

  Matrix3d normalizer(const MatrixXd& p)
  {
    Matrix2d T;
    Vector2d m = mean(p);
    Vector2d p = var(p);
    T <<  //
      1 / p.x(),         0, -m.x() / p.x(),  //
        0, 1 / p.y(), -m.y() / p.y(),  //
        0,         0,              1;
    return T;
  }


} /* namespace Sara */
} /* namespace DO */
