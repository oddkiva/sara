#pragma once

#include <DO/Sara/MultipleObjectTracking/BaseDefinitions.hpp>
#include <DO/Sara/MultipleObjectTracking/ObservationNoiseDistribution.hpp>


namespace DO::Sara::MultipleObjectTracking {

  template <typename T>
  struct ObservationEquation
  {
    using State = StateDistribution<T>;

    using Observation = ObservationDistribution<T>;
    using ObservationNoise = ObservationNoiseDistribution<T>;
    using ObservationModelMatrix = Eigen::Matrix<T, 4, 12>;

    using Innovation = Observation;
    using KalmanGain = typename ObservationDistribution<T>::CovarianceMatrix;

    inline static auto observation_model_matrix() -> ObservationModelMatrix
    {
      auto H = ObservationModelMatrix{};

      static const auto I = Eigen::Matrix4<T>::Identity();
      static const auto O = Eigen::Matrix<T, 4, 8>::Zero();
      H << I, O;

      return H;
    }

    inline auto innovation(const State& x_a_priori,
                           const Observation& z) const  //
        -> Innovation
    {
      return {
          .μ = z - H * x_a_priori,                     //
          .Σ = H * x_a_priori.Σ * H.transpose() + v.Σ  //
      };
    }

    inline auto kalman_gain_matrix(const Observation& x_predicted,
                                   const Innovation& S) const  //
        -> KalmanGain
    {
      return x_predicted.Σ * H.transpose() * S.Σ.inverse();
    }

    inline auto update(const State& x_predicted, const Observation& z) -> State
    {
      const auto y = innovation(x_predicted, z);
      const auto K = kalman_gain_matrix(x_predicted, y);

      static const auto I = State::CovarianceMatrix::Identity();
      return {
          .μ = x_predicted.μ + K * y.μ,     //
          .Σ = (I - K * H) * x_predicted.Σ  //
      };
    }

    inline auto residual(const Observation& z,
                         const State& x) const  //
        -> typename Observation::Mean
    {
      return z.μ - H * x.μ;
    }

    const ObservationModelMatrix H = observation_model_matrix();
    ObservationNoise v;
  };

}  // namespace DO::Sara::MultipleObjectTracking
