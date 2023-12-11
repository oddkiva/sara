#pragma once

#include <DO/Sara/KalmanFilter/DistributionConcepts.hpp>


namespace DO::Sara::KalmanFilter {

  template <MatrixConcept StateTransitionMatrix,
            GaussianDistribution StateDistribution,
            ZeroMeanGaussianDistribution ProcessNoiseDistribution>
  struct StateTransitionEquation
  {
    auto predict(const StateDistribution& x) -> StateDistribution
    {
      return {
          F * x.mean(),                                                      //
          F * x.covariance_matrix() * F.transpose() + w.covariance_matrix()  //
      };
    };

    StateTransitionMatrix F;
    ProcessNoiseDistribution w;
  };

}  // namespace DO::Sara::KalmanFilter
