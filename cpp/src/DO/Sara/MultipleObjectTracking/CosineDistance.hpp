#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

  template <typename Derived>
  inline auto cosine_similarity(const Eigen::MatrixBase<Derived>& a,
                                const Eigen::MatrixBase<Derived>& b,
                                Eigen::MatrixBase<Derived>& c) -> void
  {
    c = a * b.transpose();
  }

  template <typename Derived>
  inline auto cosine_distance(const Eigen::MatrixBase<Derived>& a,
                              const Eigen::MatrixBase<Derived>& b,
                              Eigen::MatrixBase<Derived>& c) -> void
  {
    using T = typename Eigen::MatrixBase<Derived>::Scalar;
    static constexpr auto one = static_cast<T>(1);
    c = cosine_similarity(a, b);
    c.array() = one - c.array();
  }

}  // namespace DO::Sara
