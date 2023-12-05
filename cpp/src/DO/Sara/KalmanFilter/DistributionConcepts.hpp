#pragma once

#include <DO/Sara/KalmanFilter/EigenMatrixConcepts.hpp>

#include <concepts>
#include <limits>


namespace DO::Sara::KalmanFilter {

  template <typename T>
  concept GaussianDistribution = requires(T dist)
  {
    typename T::scalar_type;
    typename T::mean_type;
    typename T::covariance_matrix_type;
    // clang-format off
    // Constructor.
    { T{typename T::mean_type{}, typename T::covariance_matrix_type{}} } -> std::same_as<T>;
    // Methods.
    { dist.mean() } -> std::same_as<const typename T::mean_type&>;
    { dist.covariance_matrix() } -> std::same_as<const typename T::covariance_matrix_type&>;
    // clang-format on
  };

  template <typename T>
  concept ZeroMeanGaussianDistribution = requires(T dist)
  {
    typename T::scalar_type;
    typename T::mean_type;
    typename T::covariance_matrix_type;
    // clang-format off
    { T{typename T::covariance_matrix_type{}} } -> std::same_as<T>;
    { dist.covariance_matrix() } -> std::same_as<const typename T::covariance_matrix_type&>;
    // clang-format on
  };

  template <typename T>
  concept StateDistribution = GaussianDistribution<T>;

  template <typename T>
  concept NoiseDistribution = ZeroMeanGaussianDistribution<T>;

  template <typename T>
  concept FixedSizeStateDistribution =  //
      GaussianDistribution<T> &&        //
      CompileTimeFixedMatrix<decltype(T{}.mean())> &&
      CompileTimeFixedMatrix<decltype(T{}.covariance_matrix())>;

  template <typename T>
  concept FixedSizeNoiseDistribution =  //
      ZeroMeanGaussianDistribution<T> &&
      CompileTimeFixedMatrix<decltype(T{}.mean())> &&
      CompileTimeFixedMatrix<decltype(T{}.covariance_matrix())>;

}  // namespace DO::Sara::KalmanFilter
