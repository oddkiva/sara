#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/PolynomialRoots.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures2.hpp>

#include <optional>


namespace DO::Sara {

  template <typename T>
  struct SevenPointAlgorithmImpl
  {
    static constexpr auto num_points = 7;
    static constexpr auto num_models = 3;

    using data_point_type = Eigen::Matrix<T, 4, 7>;

    static auto extract_nullspace(const data_point_type& X)
        -> std::array<Eigen::Matrix3<T>, 2>
    {
      // 1. solve the linear system from the 8-point correspondences.
      auto A = Eigen::Matrix<T, 7, 9>{};
      for (int i = 0; i < X.cols(); ++i)
      {
        const auto p_left = X.col(i).head(2).homogeneous();
        const auto p_right = X.col(i).tail(2);
        // clang-format off
        A.row(i) << p_right(0) * p_left.transpose(),
                    p_right(1) * p_left.transpose(),
                    p_left.transpose();
        // clang-format on
      }

      auto svd = Eigen::BDCSVD<Matrix<T, 7, 9>>{A, Eigen::ComputeFullV};
      const Eigen::Vector<T, 9> f1 = svd.matrixV().col(7);
      const Eigen::Vector<T, 9> f2 = svd.matrixV().col(8);

      const auto to_matrix = [](const auto& f) {
        auto F = Eigen::Matrix3<T>{};
        F.row(0) = f.segment(0, 3).transpose();
        F.row(1) = f.segment(3, 3).transpose();
        F.row(2) = f.segment(6, 3).transpose();
        return F;
      };

      return {to_matrix(f1), to_matrix(f2)};
    }

    static auto form_determinant_constraint(const Eigen::Matrix3<T>& F1,
                                            const Eigen::Matrix3<T>& F2)
    {
      auto P = UnivariatePolynomial<T, 3>{};

      // Lambda-Twist has a nice formula for the determinant. Let's reuse it
      // instead of using SymPy.
      //
      // clang-format off
      P[3] = F2.determinant();

      P[2] = F1.col(0).dot(F2.col(1).cross(F2.col(2))) +
             F1.col(1).dot(F2.col(2).cross(F2.col(0))) +
             F1.col(2).dot(F2.col(0).cross(F2.col(1)));

      P[1] = F2.col(0).dot(F1.col(1).cross(F1.col(2))) +
             F2.col(1).dot(F1.col(2).cross(F1.col(0))) +
             F2.col(2).dot(F1.col(0).cross(F1.col(1)));

      P[0] = F1.determinant();
      // clang-format on

      return P;
    }

    static auto solve(const data_point_type& X)
        -> std::vector<Eigen::Matrix3<T>>
    {
      // The fundamental matrix lives in the nullspace of data matrix X, which
      // has rank 2, i.e., Null(X) = Span(F[0], F[1])
      //
      // The fundamental matrix is a linear combination F[0] + α F[1].
      const auto F = extract_nullspace(X);

      // Because the fundamental matrix is rank 2, the determinant must be 0,
      // i.e.: det(F[0] + α F[1]) = 0
      // This is a cubic polynomial in α.
      const auto det_F = form_determinant_constraint(F[0], F[1]);

      // We determine 3 real roots α_i at most.
      auto α = std::array<T, num_models>{};
      const auto all_real_roots = compute_cubic_real_roots(det_F,  //
                                                           α[0], α[1], α[2]);
      const auto num_real_roots = all_real_roots ? 3 : 1;

      // Form the candidate fundamental matrices.
      auto F0 = std::vector<Eigen::Matrix3<T>>(num_real_roots);
      std::transform(α.begin(), α.end(), F0.begin(),
                     [&F](const auto& α_i) -> Eigen::Matrix3<T> {
                       return F[0] + α_i * F[1];
                     });

      return F0;
    }
  };

  struct DO_SARA_EXPORT SevenPointAlgorithmDoublePrecision
  {
    using Impl = SevenPointAlgorithmImpl<double>;

    static constexpr auto num_points = Impl::num_points;
    static constexpr auto num_models = Impl::num_models;

    using data_point_type = Impl::data_point_type;

    static auto extract_nullspace(const data_point_type& X)
        -> std::array<Eigen::Matrix3d, 2>;

    static auto form_determinant_constraint(const Eigen::Matrix3d& F1,
                                            const Eigen::Matrix3d& F2)
        -> UnivariatePolynomial<double, 3>;

    auto operator()(const data_point_type& X) const
        -> std::vector<Eigen::Matrix3d>;

    Impl _impl;
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
