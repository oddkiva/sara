#pragma once

#include <array>
#include <complex>
#include <limits>


namespace DO::Sara {

  //! @brief Calculates
  //! Implemented as described in:
  //! http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
  template <typename T>
  inline auto compute_quadratic_roots(T a, T b, T c)
      -> std::array<std::complex<T>, 2>
  {
    const auto sqrt_delta = std::sqrt(std::complex<double>(b * b - 4 * a * c));

    if (b >= 0)
      return {(-b - sqrt_delta) / (2 * a),  //
              (2 * c) / (-b - sqrt_delta)};
    else
      return {(2 * c) / (-b + sqrt_delta),  //
              (-b + sqrt_delta) / (2 * a)};
  }

  template <typename T>
  inline auto compute_quadratic_real_roots(T a, T b, T c) -> std::array<T, 2>
  {
    const auto delta = b * b - 4 * a * c;
    if (delta < 0)
      return {
          std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::quiet_NaN(),
      };

    const auto sqrt_delta = std::sqrt(delta);
    if (b >= 0)
      return {(-b - sqrt_delta) / (2 * a),  //
              (2 * c) / (-b - sqrt_delta)};
    else
      return {(2 * c) / (-b + sqrt_delta),  //
              (-b + sqrt_delta) / (2 * a)};
  }

}  // namespace DO::Sara
