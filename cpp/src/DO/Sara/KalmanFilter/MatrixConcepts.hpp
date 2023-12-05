#pragma once

#include <Eigen/Core>

#include <concepts>


namespace DO::Sara {


  template <typename T>
  concept MatrixConcept = requires(T)
  {
    // clang-format off
    { T{}.rows() };
    { T{}.cols() };
    { T{}(int{}, int{}) };
    { T{}.transpose() };
    // clang-format on
  };

  template <typename T>
  concept VectorConcept = MatrixConcept<T> && requires(T)
  {
    T::ColsAtCompileTime == 1;
    // clang-format off
    { T{}(int{}) };
    // clang-format on
  };

  template <typename T>
  concept CompileTimeSizedMatrixConcept = MatrixConcept<T> && requires(T)
  {
    std::same_as<T, Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime,
                                  T::ColsAtCompileTime>>;
    // clang-format off
    { T::RowsAtCompileTime };
    { T::ColsAtCompileTime };
    // clang-format on
  };

  template <typename T>
  concept CompileTimeSquareMatrixConcept =
      CompileTimeSizedMatrixConcept<T> && requires
  {
    T::RowsAtCompileTime == T::ColsAtCompileTime;
  };

}  // namespace DO::Sara
