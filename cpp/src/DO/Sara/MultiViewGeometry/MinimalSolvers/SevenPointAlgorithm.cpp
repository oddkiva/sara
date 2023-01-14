#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>


namespace DO::Sara {

  auto SevenPointAlgorithmDoublePrecision::extract_nullspace(
      const Eigen::Matrix<double, 4, 7>& X) -> std::array<Eigen::Matrix3d, 2>
  {
    return Impl::extract_nullspace(X);
  }

  auto SevenPointAlgorithmDoublePrecision::form_determinant_constraint(
      const Eigen::Matrix3d& F1, const Eigen::Matrix3d& F2)
      -> UnivariatePolynomial<double, 3>
  {
    return Impl::form_determinant_constraint(F1, F2);
  }

  auto SevenPointAlgorithmDoublePrecision::solve_determinant_constraint(
      const std::array<Eigen::Matrix3d, 2>& F) -> std::vector<Eigen::Matrix3d>
  {
    return Impl::solve_determinant_constraint(F);
  }

  auto SevenPointAlgorithmDoublePrecision::operator()(
      const Eigen::Matrix<double, 4, 7>& X) const
      -> std::vector<Eigen::Matrix3d>
  {
    // The fundamental matrix lives in the nullspace of data matrix X, which
    // has rank 2, i.e., Null(X) = Span(F[0], F[1])
    //
    // The fundamental matrix is a linear combination F[0] + α F[1].
    const auto F = extract_nullspace(X);

    // Extract the roots of polynomial in α: det(F[0] + α F[1]).
    //
    // We get three possible solutions at most.
    return solve_determinant_constraint(F);
  }

  auto SevenPointAlgorithmDoublePrecision::operator()(
      const SevenPointAlgorithmDoublePrecision::matrix_type& x,
      const SevenPointAlgorithmDoublePrecision::matrix_type& y) const
      -> std::vector<FundamentalMatrix>
  {
    auto X = Eigen::Matrix<double, 4, 7>{};
    for (auto i = 0; i < 7; ++i)
      X.col(i) << x.col(i).head(2), y.col(i).head(2);
    // X << x.colwise().hnormalized(), y.colwise().hnormalized();

    const auto matrices = (*this)(X);

    auto Fs = std::vector<FundamentalMatrix>(matrices.size());
    std::transform(matrices.begin(), matrices.end(), Fs.begin(),
                   [](const auto& F) -> FundamentalMatrix { return F; });

    return Fs;
  }

}  // namespace DO::Sara
