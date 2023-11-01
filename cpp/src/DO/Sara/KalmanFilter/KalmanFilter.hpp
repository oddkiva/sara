#pragma once


namespace DO::Sara::KalmanFilter {

  //! @brief The step-by-step evolution of the 3D bounding box of the
  //! pedestrian.
  template <typename StateVector,            //
            typename StateTransitionMatrix,  //
            typename ProcessNoiseVector>
  struct StateTransitionEquation
  {
    StateTransitionMatrix F;
    ProcessNoiseVector w;

    inline auto operator()(const StateVector& x) const -> StateVector
    {
      return F * x + w;
    }
  };

  //! @brief The projection of the 3D bounding box into the image plane.
  //! Sort of...
  template <typename StateVector,        //
            typename ObservationVector,  //
            typename ObservationMatrix,  //
            typename MeasurementNoiseVector>
  struct ObservationEquation
  {
    ObservationMatrix H;
    MeasurementNoiseVector v;

    inline auto operator()(const StateVector& x) const -> ObservationVector
    {
      // z = observation vector.
      const auto z = H * x + v;
      return z;
    }
  };


  template <typename StateTransitionEquation, typename ObservationEquation>
  struct KalmanFilter
  {
    template <typename StateVector>
    auto predict(const StateVector& x)
        -> std::pair<APrioriStateMeanVector, APrioriStateCovarianceMatrix>;

    auto update() -> std::pair<APosterioriStateMeanVector,
                               APosterioriStateCovarianceMatrix>;
  };

}  // namespace DO::Sara::KalmanFilter
