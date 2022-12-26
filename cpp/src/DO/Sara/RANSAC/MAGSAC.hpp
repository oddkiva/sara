#pragma once

#include <vector>


namespace DO::Sara {

  namespace magsac {

    template <typename Model, typename DataPoint>
    struct LogLikelihood
    {
      using T = typename Model::Scalar;
      auto operator()(const Model& theta, const std::vector<DataPoint>& X) -> T
      {
        return std::log(L(theta, X, sigma);
      }

    };

  }  // namespace magsac

  template <typename QualityFunction>
  struct Magsac
  {
    QualityFunction Q;
  };

}  // namespace DO::Sara
