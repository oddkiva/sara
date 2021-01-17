#pragma once

#include <DO/Sara/ImageProcessing/Differential.hpp>


namespace DO::Sara {

  //! @brief Evaluate the mean curvature at point p of the isosurface u = 0.
  //!
  //! The mean curvature formula is
  //!   (∇u Hu ∇u.T - |∇u|^2 trace(Hu)) / (2 * |∇u|^3).
  template <typename T, int N>
  inline auto mean_curvature(const ImageView<T, N>& u,           //
                             const Eigen::Matrix<int, N, 1>& x,  //
                             T eps = T(1e-6)) -> T
  {
    const auto Du = gradient(u, x);
    const auto Du_norm_2 = Du.squaredNorm();
    if (Du_norm_2 < eps)
      return T{};

    const auto Du_norm_3_inverse = std::pow(Du_norm_2, -T(1.5));
    const auto Hu = hessian(u, x);

    return (Du.transpose() * Hu * Du - Du_norm_2 * Hu.trace()) *  //
           T(0.5) * Du_norm_3_inverse;
  }

  //! Evaluate the mean curvature motion at point p of the isosurface u = 0.
  //! The mean curvature motion is:
  //!   (∇u Hu ∇u.T - |∇u|^2 trace(Hu)) / (2 * |∇u|^2).
  template <typename T, int N>
  inline auto mean_curvature_flow(const ImageView<T, N>& u,    //
                                  const Matrix<int, N, 1>& x,  //
                                  T eps = T(1e-6)) -> T
  {
    const auto Du = gradient(u, x);
    const auto Du_norm_2 = Du.squaredNorm();
    if (Du_norm_2 < eps)
      return 0;

    const auto Du_norm_2_inverse = 1 / Du_norm_2;
    const auto Hu = hessian(u, x);
    return (Du.transpose() * Hu * Du - Du_norm_2 * Hu.trace()) *  //
           T(0.5) * Du_norm_2_inverse;
  }

  //! Evaluate the Gaussian curvature motion at point p of the isosurface u = 0.
  //! phi(x, y, z) = 0 <=> u(x, y) - z = 0
  template <typename T>
  inline auto gaussian_curvature(const ImageView<T, 2>& u, const Vector2i& x)
      -> T
  {
    const auto Du = gradient(u, x);
    const auto Hu = hessian(u, x);
    const auto denominator = 1 / std::pow(Du.squaredNorm() + 1, 2);
    return Hu.determinant() * denominator;
  }

  //! Evaluate the Gaussian curvature motion at point p of the isosurface u = 0.
  template <typename T>
  inline auto gaussian_curvature(const ImageView<T, 3>& u,  //
                                 const Vector3i& x,         //
                                 T eps = T(1e-6)) -> T
  {
    const auto Du = gradient(u, x);
    const auto Hu = hessian(u, x);

    const auto Du_norm_2 = Du.squaredNorm();

    if (Du_norm_2 < eps)
      return 0;

    auto cofactors = [](const Eigen::Matrix<T, 3, 3>& m) {
      auto cof_m = Eigen::Matrix<T, 3, 3>{};
      cof_m.col(0) = m.col(1).cross(m.col(2));
      cof_m.col(1) = m.col(2).cross(m.col(0));
      cof_m.col(2) = m.col(0).cross(m.col(1));
      return cof_m;
    };

    const auto cofactors_Hu = cofactors(Hu);
    const auto Du_norm_2_inverse = 1 / Du_norm_2;

    return Du.transpose() * cofactors_Hu * Du * Du_norm_2_inverse;
  }

}  // namespace DO::Sara
