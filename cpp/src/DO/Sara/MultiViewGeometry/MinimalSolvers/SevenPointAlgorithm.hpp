#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/PolynomialRoots.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures2.hpp>


namespace DO::Sara {

  template <typename T = double>
  struct SevenPointAlgorithm
  {
    static constexpr auto num_points = 7;
    static constexpr auto num_models = 3;
    static constexpr auto num_candidate_models = 3;

    using data_point_type = Eigen::Matrix<T, 4, 7>;

    auto extract_nullspace(const data_point_type& X) const
        -> std::array<Eigen::Matrix3<T>, 2>
    {
      auto F = Matrix3d{};

      // 1. solve the linear system from the 8-point correspondences.
      Matrix<double, 7, 9> A;
      for (int i = 0; i < 8; ++i)
      {
        const Eigen::Vector3<T> p_left = X.col(i).head(2).homogeneous();
        const Eigen::Vector3<T> p_right = X.col(i).head(2).homogeneous();
        A.row(i) <<                                     //
            p_right(0, i) * p_left.col(i).transpose(),  //
            p_right(1, i) * p_left.col(i).transpose(),  //
            p_right(2, i) * p_left.col(i).transpose();
      }

      auto svd = Eigen::BDCSVD<Matrix<double, 8, 9>>{A, Eigen::ComputeFullV};
      const Eigen::Matrix<double, 9, 1> f1 = svd.matrixV().col(7).normalized();
      const Eigen::Matrix<double, 9, 1> f2 = svd.matrixV().col(8).normalized();

      const auto to_matrix = [](const auto& f) {
        auto F = Eigen::Matrix3<T>{};
        F.row(0) = f.segment(0, 3).transpose();
        F.row(1) = f.segment(3, 3).transpose();
        F.row(2) = f.segment(6, 3).transpose();
        return F;
      };

      return {to_matrix(f1), to_matrix(f2)};
    }

    auto form_determinant_constraint(const Eigen::Matrix3<T>& F1,
                                     const Eigen::Matrix3<T>& F2) const
    {
      auto P = UnivariatePolynomial<T, 3>{};

      // The coefficients are calculated with SymPy in Python where we solve
      // det(F) = det(F1 + α * F2) = 0.
      //
      // Here is the Python code.
      // import sympy as sp
      //
      // # Solve the cubic polynomial.
      // F1 = sp.MatrixSymbol('F1', 3, 3)
      // F2 = sp.MatrixSymbol('F2', 3, 3)
      // α = sp.symbols('α')
      //
      // # Form the symbolic matrix expression as reported in the paper.
      // F = sp.Matrix(F1 + α * F2)
      //
      // # Form the polynomial in the variable α.
      // det_F, _ = sp.polys.poly_from_expr(F.det(), α)
      //
      // # Collect the coefficients "c[i]" as denoted in the paper.
      // c = det_F.all_coeffs()


      P[3] = F2(0, 0) * F2(1, 1) * F2(2, 2) - F2(0, 0) * F2(1, 2) * F2(2, 1) -
             F2(0, 1) * F2(1, 0) * F2(2, 2) + F2(0, 1) * F2(1, 2) * F2(2, 0) +
             F2(0, 2) * F2(1, 0) * F2(2, 1) - F2(0, 2) * F2(1, 1) * F2(2, 0);

      P[2] = F1(0, 0) * F2(1, 1) * F2(2, 2) - F1(0, 0) * F2(1, 2) * F2(2, 1) -
             F1(0, 1) * F2(1, 0) * F2(2, 2) + F1(0, 1) * F2(1, 2) * F2(2, 0) +
             F1(0, 2) * F2(1, 0) * F2(2, 1) - F1(0, 2) * F2(1, 1) * F2(2, 0) -
             F1(1, 0) * F2(0, 1) * F2(2, 2) + F1(1, 0) * F2(0, 2) * F2(2, 1) +
             F1(1, 1) * F2(0, 0) * F2(2, 2) - F1(1, 1) * F2(0, 2) * F2(2, 0) -
             F1(1, 2) * F2(0, 0) * F2(2, 1) + F1(1, 2) * F2(0, 1) * F2(2, 0) +
             F1(2, 0) * F2(0, 1) * F2(1, 2) - F1(2, 0) * F2(0, 2) * F2(1, 1) -
             F1(2, 1) * F2(0, 0) * F2(1, 2) + F1(2, 1) * F2(0, 2) * F2(1, 0) +
             F1(2, 2) * F2(0, 0) * F2(1, 1) - F1(2, 2) * F2(0, 1) * F2(1, 0);

      P[1] = F1(0, 0) * F1(1, 1) * F2(2, 2) - F1(0, 0) * F1(1, 2) * F2(2, 1) -
             F1(0, 0) * F1(2, 1) * F2(1, 2) + F1(0, 0) * F1(2, 2) * F2(1, 1) -
             F1(0, 1) * F1(1, 0) * F2(2, 2) + F1(0, 1) * F1(1, 2) * F2(2, 0) +
             F1(0, 1) * F1(2, 0) * F2(1, 2) - F1(0, 1) * F1(2, 2) * F2(1, 0) +
             F1(0, 2) * F1(1, 0) * F2(2, 1) - F1(0, 2) * F1(1, 1) * F2(2, 0) -
             F1(0, 2) * F1(2, 0) * F2(1, 1) + F1(0, 2) * F1(2, 1) * F2(1, 0) +
             F1(1, 0) * F1(2, 1) * F2(0, 2) - F1(1, 0) * F1(2, 2) * F2(0, 1) -
             F1(1, 1) * F1(2, 0) * F2(0, 2) + F1(1, 1) * F1(2, 2) * F2(0, 0) +
             F1(1, 2) * F1(2, 0) * F2(0, 1) - F1(1, 2) * F1(2, 1) * F2(0, 0);

      P[0] = F1(0, 0) * F1(1, 1) * F1(2, 2) - F1(0, 0) * F1(1, 2) * F1(2, 1) -
             F1(0, 1) * F1(1, 0) * F1(2, 2) + F1(0, 1) * F1(1, 2) * F1(2, 0) +
             F1(0, 2) * F1(1, 0) * F1(2, 1) - F1(0, 2) * F1(1, 1) * F1(2, 0);
    }

    auto operator()(const data_point_type& X) const
    {
      const auto F = extract_nullspace(X);
      const auto det_F = form_determinant_constraint(F[0], F[1]);
      const auto α = std::array<T, 3>{};
      const auto found = compute_cubic_real_roots(det_F, α[0], α[1], α[2]);
      const auto F0 = std::array<Eigen::Matrix3<T>, 3>{};
      std::transform(α.begin(), α.end(), F0.begin(),
                     [&F](const auto& α_i) { return F[0] + α_i * F[1]; });
      return F0;
    }
  };


  template <typename EpipolarDistance>
  inline auto
  normalized_epipolar_residual(const std::vector<std::size_t>& subset,
                               const Eigen::Matrix3d& F,
                               const std::vector<Eigen::Vector4d>& matches,
                               const EpipolarDistance& distance)
      -> std::vector<double>
  {
    return normalized_residual(subset, F, matches, distance);
  }

}  // namespace DO::Sara
