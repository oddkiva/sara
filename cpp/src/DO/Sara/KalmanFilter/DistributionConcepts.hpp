#pragma once

#include <DO/Sara/KalmanFilter/MatrixConcepts.hpp>

#include <concepts>
#include <limits>


namespace DO::Sara {

  template <typename T>
  concept ZeroMeanGaussianDistribution = requires(T dist)
  {
    typename T::Scalar;
    typename T::Mean;
    typename T::CovarianceMatrix;
    // clang-format off
    { T{typename T::CovarianceMatrix{}} } -> std::same_as<T>;
    { dist.covariance_matrix() } -> std::same_as<const typename T::CovarianceMatrix&>;
    // clang-format on
  };

  template <typename T>
  concept GaussianDistribution = ZeroMeanGaussianDistribution<T> &&
      requires(T dist)
  {
    // clang-format off
    // Constructor.
    {
      T{typename T::Mean{}, typename T::CovarianceMatrix{}}
    } -> std::same_as<T>;
    // Methods.
    { dist.mean() } -> std::same_as<const typename T::Mean&>;
    // clang-format on
  };

  template <typename T>
  concept StateDistribution = GaussianDistribution<T>;

  template <typename T>
  concept NoiseDistribution = ZeroMeanGaussianDistribution<T>;

  template <typename T>
  concept FixedSizeStateDistribution =  //
      GaussianDistribution<T> &&        //
      CompileTimeSizedMatrixConcept<decltype(T{}.mean())> &&
      CompileTimeSquareMatrixConcept<decltype(T{}.covariance_matrix())>;

  template <typename T>
  concept FixedSizeNoiseDistribution =  //
      ZeroMeanGaussianDistribution<T> &&
      CompileTimeSizedMatrixConcept<decltype(T{}.mean())> &&
      CompileTimeSquareMatrixConcept<decltype(T{}.covariance_matrix())>;

}  // namespace DO::Sara
