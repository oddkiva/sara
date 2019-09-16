#pragma once

#include <Eigen/Core>


namespace np {

  using namespace Eigen;

  inline
  VectorXd linspace(double a, double b, int num = 50)
  {
    auto linspace = VectorXd{ num + 1 };
    for (int i = 0; i < num + 1; ++i)
      linspace[i] = a + (b-a) * i / num;
    return linspace;
  }

  inline
  VectorXd sin(const VectorXd& x)
  {
    return x.array().sin();
  }

  inline
  VectorXd cos(const VectorXd& x)
  {
    return x.array().cos();
  }

}
