#pragma once

#include <DO/Sara/MultipleObjectTracking/ProcessNoiseDistribution.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>


namespace DO::Sara::MultipleObjectTracking {

  //! @brief The state transition equation.
  //! @{
  template <typename T>
  using StateTransitionModelMatrix = Eigen::Matrix<T, 12, 12>;

  template <typename T>
  inline auto state_transition_model_matrix(const Time Δt)
      -> StateTransitionModelMatrix<T>
  {
    auto F = StateTransitionModelMatrix<T>{};

    const auto a = Δt.as<T>();
    const auto b = static_cast<T>(0.5) * (a * a);

    static const auto I = Eigen::Matrix4<T>::Identity();
    static const auto O = Eigen::Matrix4<T>::Zero();

    // clang-format off
    F << I, a * I, b * I,
         O,     I, b * I,
         O,     O,     I;
    // clang-format on

    return F;
  }
  //! @}

  //! The process noise is the error that the state transition equation makes.
  //!
  //! In the Kalman filter, the process noise follows a zero-mean Gaussian
  //! distribution.
  template <typename T>
  struct StateTransitionEquation
  {
    using State = StateDistribution<T>;
    using StateTransitionMatrix = StateTransitionModelMatrix<T>;
    using ProcessNoise = ProcessNoiseDistribution<T>;

    auto predict(const State& x) const -> State
    {
      return {
          .μ = F * x.μ,                       //
          .Σ = F * x.Σ * F.transpose() + w.Σ  //
      };
    }

    StateTransitionMatrix F;
    ProcessNoise w;
  };
}  // namespace DO::Sara::MultipleObjectTracking
