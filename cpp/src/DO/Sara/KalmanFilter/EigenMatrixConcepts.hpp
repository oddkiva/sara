#pragma once

#include <Eigen/Core>

#include <concepts>


namespace DO::Sara::KalmanFilter {

  template <typename T>
  concept EigenVector = requires(T)
  {
    typename T::Scalar;
    // clang-format off
    { T{}.rows() } -> std::same_as<Eigen::Index>;
    { T{}.cols() } -> std::same_as<Eigen::Index>;
    { T{}(int{}) } -> std::same_as<typename T::Scalar>;
    // clang-format on
  };

  template <typename T>
  concept EigenMatrix = requires(T)
  {
    typename T::Scalar;
    // clang-format off
    { T{}.rows() } -> std::same_as<Eigen::Index>;
    { T{}.cols() } -> std::same_as<Eigen::Index>;
    { T{}(int{}, int{}) } -> std::same_as<typename T::Scalar>;
    { T{}.transpose() };
    // clang-format on
  };

  template <typename T>
  concept EigenSquareMatrix = EigenMatrix<T> && requires
  {
    T::Rows == T::Cols;
  };

  template <typename T>
  concept CompileTimeFixedMatrix = requires
  {
    typename T::scalar_type;
    // clang-format off
    { T::Rows } -> std::same_as<int>;
    { T::Cols } -> std::same_as<int>;
    { T{} } -> std::same_as<Eigen::Matrix<typename T::scalar_type, T::Rows, T::Cols>>;
    // clang-format on
  };

}  // namespace DO::Sara::KalmanFilter
