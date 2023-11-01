#pragma once

#include <DO/Sara/MultipleObjectTracking/BaseDefinitions.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>


namespace DO::Sara::MultipleObjectTracking {

  //! The observation noise is the discrepancy between the state predicted by
  //! state-transition equation and the observation i.e. the cylindric bounding
  //! box.
  //!
  //! The observation is in our case obtained from metrologic formulas and a
  //! specified camera calibration parameter.
  //!
  //! The observation noise should also factor in measurement error when we
  //! obtain the observation vector.
  //!
  //! In the Kalman filter, the observation noise follows a zero-mean Gaussian
  //! distribution.
  template <typename T>
  struct ObservationNoiseDistribution
  {
    using Mean = typename ObservationDistribution<T>::Mean;
    using CovarianceMatrix =
        typename ObservationDistribution<T>::CovarianceMatrix;

    static constexpr auto distance_error = 0.5_m;
    static constexpr auto typical_pedestrian_height = 1.75_m;

    static inline auto default_standard_deviation(  //
        const Length σ_x = distance_error,          //
        const Length σ_y = distance_error,          //
        const T σ_a = static_cast<T>(0.1),
        const Length σ_h = 5._percent * typical_pedestrian_height)
        -> CovarianceMatrix
    {
      return {σ_x.as<T>(), σ_y.as<T>(), σ_a, σ_h.as<T>()};
    }

    inline static auto default_covariance_matrix() -> CovarianceMatrix
    {
      return default_standard_deviation().array().square().asDiagonal();
    }

    CovarianceMatrix Σ;
  };

}  // namespace DO::Sara::MultipleObjectTracking
