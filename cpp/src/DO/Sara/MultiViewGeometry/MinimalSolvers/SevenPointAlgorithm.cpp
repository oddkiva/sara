#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>


namespace DO::Sara {

  auto solve_fundamental_matrix(const Eigen::Matrix<double, 4, 7>& X)
      -> std::array<std::optional<Eigen::Matrix3d>, 3>
  {
    // auto solver = SevenPointAlgorithm<double>{};
    // solver.extract_nullspace(X);

    return {};
  }

}  // namespace DO::Sara
