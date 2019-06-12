#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  void eight_point_fundamental_matrix(const Matrix<double, 3, 8>& x,
                                      const Matrix<double, 3, 8>& y,
                                      Matrix3d& F);

  void four_point_homography(const Matrix<double, 3, 4>& x,
                             const Matrix<double, 3, 4>& y,  //
                             Matrix3d& H);

} /* namespace Sara */
} /* namespace DO */
