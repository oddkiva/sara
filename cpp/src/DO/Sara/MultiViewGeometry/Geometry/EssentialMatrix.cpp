#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>


namespace DO::Sara {

auto extract_relative_motion_svd(const Matrix3d& E)
    -> std::vector<Motion>
{
  auto svd =
      Eigen::BDCSVD<Matrix3d>(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto U = svd.matrixU();
  Matrix3d Vt = svd.matrixV().transpose();

  static_assert(std::is_same<decltype(U), Matrix3d>::value);

  if (U.determinant() < 0)
    U.col(2) *= -1;
  if (Vt.determinant() < 0)
    Vt.row(2) *= -1;

  Matrix3d W;
  W << 0, 1, 0,
      -1, 0, 0,
       0, 0, 1;

  Matrix3d Ra, Rb;
  Vector3d ta, tb;

  Ra = U * W * Vt;
  Rb = U * W.transpose() * Vt;

  ta = U.col(2);
  tb = -ta;

  return {{Ra, ta}, {Ra, tb}, {Rb, ta}, {Rb, tb}};
}

auto extract_relative_motion_horn(const Matrix3d& E)
    -> std::vector<Motion>
{
  const Matrix3d EEt = E * E.transpose();
  const Matrix3d cofET = cofactors_transposed(E);
  const RowVector3d norm_cofE = cofET.colwise().norm();

  auto i = int{};
  norm_cofE.maxCoeff(&i);

  const Vector3d ta =
      cofET.col(i) / norm_cofE(i) * std::sqrt(0.5 * EEt.trace());
  const Vector3d tb = -ta;

  const double ta_sq_norm = ta.squaredNorm();
  const Matrix3d Ra = (cofET - skew_symmetric_matrix(ta) * E) / ta_sq_norm;
  const auto F = 2. * (ta * ta.transpose()) / ta_sq_norm - Matrix3d::Identity();

  const Matrix3d Rb = F * Ra;

  return {{Ra, ta}, {Ra, tb}, {Rb, ta}, {Rb, tb}};
}

} /* namespace DO::Sara */
