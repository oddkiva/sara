#pragma once

#include <DO/Sara/KalmanFilter/DistributionConcepts.hpp>


namespace DO::Sara::KalmanFilter {

  template <GaussianDistribution State,                     //
            GaussianDistribution Observation,               //
            ZeroMeanGaussianDistribution ObservationNoise,  //
            MatrixConcept ObservationModelMatrix>
  struct ObservationEquation
  {
    using T = typename ObservationModelMatrix::Scalar;
    using Innovation = Observation;
    using KalmanGain = typename Observation::covariance_matrix_type;

    inline auto innovation(const State& x_a_priori,
                           const Observation& z) const  //
        -> Innovation
    {
      return {
          z - H * x_a_priori,  //
          H * x_a_priori.covariance_matrix() * H.transpose() +
              v.covariance_matrix()  //
      };
    }

    inline auto kalman_gain_matrix(const Observation& x_predicted,
                                   const Innovation& S) const  //
        -> KalmanGain
    {
      return x_predicted.covariance_matrix() * H.transpose() *
             S.covariance_matrix().inverse();
    }

    inline auto update(const State& x_predicted, const Observation& z) -> State
    {
      const auto y = innovation(x_predicted, z);
      const auto K = kalman_gain_matrix(x_predicted, y);

      static const auto I = State::CovarianceMatrix::Identity();
      return {
          x_predicted.mean() + K * y.mean(),             //
          (I - K * H) * x_predicted.covariance_matrix()  //
      };
    }

    inline auto residual(const Observation& z,
                         const State& x) const  //
        -> typename Observation::mean_type
    {
      return z.mean() - H * x.covariance_matrix();
    }

    const ObservationModelMatrix H;
    ObservationNoise v;
  };

}  // namespace DO::Sara::KalmanFilter
