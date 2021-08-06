// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

  //! @addtogroup MinimalSolvers
  //! @{

  //! @{
  //! @brief Invariant up to affine-transformations.
  DO_SARA_EXPORT
  auto triangulate_single_point_linear_eigen(const Matrix34d& P1,
                                             const Matrix34d& P2,
                                             const Vector3d& u1,
                                             const Vector3d& u2) -> Vector4d;

  DO_SARA_EXPORT
  auto triangulate_single_point_linear_eigen_v2(const Matrix34d& P1,
                                                const Matrix34d& P2,
                                                const Vector3d& ray1,
                                                const Vector3d& ray2)
      -> std::tuple<Vector4d, double, double>;

  DO_SARA_EXPORT
  auto triangulate_linear_eigen(const Matrix34d& P1, const Matrix34d& P2,
                                const MatrixXd& u1, const MatrixXd& u2)
      -> MatrixXd;
  //! @}

  //! @}

} /* namespace DO::Sara */
