#pragma once

#include <Eigen/Core>


namespace DO::Sara {

  struct Camera
  {
    using matrix_type = Eigen::Matrix<double, 3, 4>;

    operator matrix_type&()
    {
      return P;
    }

    operator const matrix_type&() const
    {
      return P;
    }

    matrix_type P;
  };


} /* namespace DO::Sara */
