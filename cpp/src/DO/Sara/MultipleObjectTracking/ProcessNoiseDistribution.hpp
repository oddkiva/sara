#pragma once

#include <DO/Sara/MultipleObjectTracking/BaseDefinitions.hpp>
#include <DO/Sara/MultipleObjectTracking/ObservationNoiseDistribution.hpp>


namespace DO::Sara::MultipleObjectTracking {

  //! The process noise is the error that the state transition equation makes.
  //!
  //! In the Kalman filter, the process noise follows a zero-mean Gaussian
  //! distribution.
  template <typename T>
  struct ProcessNoiseDistribution
  {
    using Mean = typename StateDistribution<T>::Mean;
    using CovarianceMatrix = typename StateDistribution<T>::CovarianceMatrix;

    static constexpr auto distance_error =
        ObservationNoiseDistribution<T>::distance_error;
    static constexpr auto typical_pedestrian_height =
        ObservationNoiseDistribution<T>::typical_pedestrian_height;

    static constexpr auto speed_error = 5._km / hour;
    static constexpr auto acceleration_error = (3._km / hour) / 0.5_s;

    static constexpr auto car_speed_acceleration = (60.L * mile / hour) / 5._s;
    static constexpr auto car_speed_acceleration_error =
        0.2L * car_speed_acceleration;

    inline static auto default_standard_deviation() -> Mean
    {
      auto σ = Mean{};

      // clang-format off
      σ <<
        ObservationNoiseDistribution<T>::standard_deviation(),
        speed_error,
        speed_error,
        0,
        0,
        acceleration_error,
        acceleration_error,
        0,
        0;
      // clang-format on

      return σ;
    }

    inline static auto default_covariance_matrix() -> CovarianceMatrix
    {
      return default_standard_deviation().array().square().asDiagonal();
    }

    CovarianceMatrix Σ;
  };

}  // namespace DO::Sara::MultipleObjectTracking
