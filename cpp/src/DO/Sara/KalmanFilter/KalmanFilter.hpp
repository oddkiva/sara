#pragma once


namespace DO::Sara::KalmanFilter {

  template <typename StateTransitionEquation, typename ObservationEquation>
  struct KalmanFilter
  {
    using State = typename StateTransitionEquation::State;
    using Observation = typename StateTransitionEquation::Observation;

    auto predict(const State& x) -> State
    {
      return _state_transition_equation.predict(x);
    }

    auto update(const State& x_predicted, const Observation& z) -> State
    {
      return _observation_equation.update(x_predicted, z);
    }

    StateTransitionEquation _state_transition_equation;
    ObservationEquation _observation_equation;
  };

}  // namespace DO::Sara::KalmanFilter
