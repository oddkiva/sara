#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>


namespace DO::Sara {

  auto SevenPointAlgorithmDoublePrecision::extract_nullspace(
      const Eigen::Matrix<double, 4, 7>& X)
      -> std::array<Eigen::Matrix3d, 2>
  {
    return Impl::extract_nullspace(X);
  }

  auto SevenPointAlgorithmDoublePrecision::form_determinant_constraint(
      const Eigen::Matrix3d& F1, const Eigen::Matrix3d& F2)
      -> UnivariatePolynomial<double, 3>
  {
    return Impl::form_determinant_constraint(F1, F2);
  }

  auto SevenPointAlgorithmDoublePrecision::operator()(
      const Eigen::Matrix<double, 4, 7>& X) const
      -> std::array<std::optional<Eigen::Matrix3d>, 3>
  {
    return Impl::solve(X);
  }

}  // namespace DO::Sara
