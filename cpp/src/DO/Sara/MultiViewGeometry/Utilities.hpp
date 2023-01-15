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

#include <algorithm>
#include <random>


namespace DO::Sara {

  /*!
   *  @ingroup MultiViewGeometry
   *  @defgroup Utilities Utilities
   *  @{
   */

  //! @brief Geometry.
  //! @{
  template <typename T>
  inline auto homogeneous(const TensorView_<T, 2>& x) -> Tensor_<T, 2>
  {
    auto X = Tensor_<T, 2>(x.size(0), x.size(1) + 1);
    X.matrix().leftCols(x.size(1)) = x.matrix();
    X.matrix().col(x.size(1)).setOnes();
    return X;
  }

  template <typename S>
  inline auto compute_normalizer(const TensorView_<S, 2>& X) -> Matrix<S, 3, 3>
  {
    auto slope = Eigen::RowVector2<S>{};
    auto offset = Eigen::RowVector2<S>{};

    if (X.size(1) == 2)
    {
      const Eigen::RowVector2<S> min = X.matrix().colwise().minCoeff();
      const Eigen::RowVector2<S> max = X.matrix().colwise().maxCoeff();

      slope = (max - min).cwiseInverse();
      offset = -min.cwiseQuotient(max - min);
    }
    else if (X.size(1) == 3)
    {
      const Eigen::RowVector3<S> min = X.matrix().colwise().minCoeff();
      const Eigen::RowVector3<S> max = X.matrix().colwise().maxCoeff();

      slope = (max - min).cwiseInverse().head(2);
      offset = -min.cwiseQuotient(max - min).head(2);
    }
    else
      throw std::runtime_error{
          "To compute the normalization matrix the input data "
          "dimension must be 2 or 3!"};


    auto T = Eigen::Matrix<S, 3, 3>{};
    T.setZero();
    T.template topLeftCorner<2, 2>() = slope.asDiagonal();
    T.col(2) << offset.transpose(), S(1);

    return T;
  }

  template <typename S>
  inline auto apply_transform(const Matrix<S, 3, 3>& T,
                              const TensorView_<S, 2>& X) -> Tensor_<S, 2>
  {
    auto TX = Tensor_<S, 2>{X.sizes()};
    auto TX_ = TX.colmajor_view().matrix();

    const auto X_matrix = X.colmajor_view().matrix();

    if (X.size(1) == 2)
      TX_ = (T * X_matrix.colwise().homogeneous()).colwise().hnormalized();
    else if (X.size(1) == 3)
    {
      TX_ = T * X_matrix;
      TX_.array().rowwise() /= TX_.array().row(2);
    }
    else
      throw std::runtime_error{"To apply the transform the input data "
                               "dimension must be 2 or 3!"};

    return TX;
  }
  //! @}


  //! @brief Elementary transformations.
  //! @{
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
