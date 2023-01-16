// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  /*!
   *  @ingroup MultiViewGeometry
   *  @defgroup Utilities Utilities
   *  @{
   */

  //! @brief Elementary transformations.
  //! @{

  template <typename T>
  inline auto homogeneous(const TensorView_<T, 2>& x) -> Tensor_<T, 2>
  {
    auto X = Tensor_<T, 2>(x.size(0), x.size(1) + 1);
    X.matrix().leftCols(x.size(1)) = x.matrix();
    X.matrix().col(x.size(1)).setOnes();
    return X;
  }

  inline auto cofactors_transposed(const Matrix3d& E)
  {
    Matrix3d cofE;
    cofE.col(0) = E.col(1).cross(E.col(2));
    cofE.col(1) = E.col(2).cross(E.col(0));
    cofE.col(2) = E.col(0).cross(E.col(1));
    return cofE;
  }

  template <typename T>
  inline auto skew_symmetric_matrix(const Matrix<T, 3, 1>& a) -> Matrix<T, 3, 3>
  {
    auto A = Matrix<T, 3, 3>{};
    // clang-format off
    A << T(0), -a(2), a(1),
         a(2), T(0), -a(0),
        -a(1), a(0), T(0);
    // clang-format on

    return A;
  }
  //! @}

  //! @}

} /* namespace DO::Sara */
