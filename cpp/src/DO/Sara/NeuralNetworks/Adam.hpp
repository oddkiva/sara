#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  struct Adam
  {
    void step(const TensorView_<float, 4>& grad_theta0,
              TensorView_<float, 4>& theta0, int t0)
    {
      const auto t1 = t0 + 1;
      const auto m1 = beta1 * m0 + (1 - beta1) * grad_theta0;

      auto g1_squared = grad_theta0;
      g1_squared.array() = g1_squared_array().pow(2);

      auto v1 = beta2 * v0 + (1 - beta2) * g1_squared;

      auto m1_hat = m1 / (1 - std::pow(beta1, t1));
      auto v1_hat = v1 / (1 - std::pow(beta2, t1));
      auto v1_hat_sqrt = v1_hat;
      v1_hat_sqrt.array() = v1_hat_sqrt.array().sqrt() + eps;

      auto theta1 = theta0 + alpha * m1_hat / v1_hat;
    }

    float alpha;
    float beta1, beta2;
    float eps;

    Tensor_<float, 4> m0, v0;
    int t{0};
  };

} /* namespace DO::Sara */
