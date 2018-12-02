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
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>


namespace DO { namespace Sara {

  // NumPy-like interface for tensors.
  DO_SARA_EXPORT
  auto range(int n) -> Tensor_<int, 1>;

  DO_SARA_EXPORT
  auto random_samples(int num_samples,      //
                      int sample_size,      //
                      int num_data_points)  //
      -> Tensor_<int, 2>;


  template <typename T>
  inline auto shuffle(const TensorView_<T, 1>& x) -> Tensor_<T, 1>
  {
    auto x_shuffled = x;
    std::random_shuffle(x_shuffled.begin(), x_shuffled.end());
    return x_shuffled;
  }


  // Geometry.
  template <typename T>
  auto homogeneous(const TensorView_<T, 2>& x) -> Tensor_<T, 2>
  {
    auto X = Tensor_<T, 2>(x.size(0), x.size(1) + 1);
    X.matrix().leftCols(x.size(1)) = x.matrix();
    X.matrix().col(x.size(1)).setOnes();
    return X;
  }

  template <typename S>
  auto compute_normalizer(const TensorView_<S, 2>& X) -> Matrix<S, 3, 3>
  {
    const Matrix<S, 1, 3> min = X.matrix().colwise().minCoeff();
    const Matrix<S, 1, 3> max = X.matrix().colwise().maxCoeff();

    const Matrix<S, 2, 2> scale =
        (max - min).cwiseInverse().head(2).asDiagonal();

    auto T = Matrix<S, 3, 3>{};
    T.setZero();
    T.template topLeftCorner<2, 2>() = scale;
    T.col(2) << -min.cwiseQuotient(max - min).transpose().head(2), S(1);

    return T;
  }

  template <typename S>
  auto apply_transform(const Matrix<S, 3, 3>& T, const TensorView_<S, 2>& X)
      -> Tensor_<S, 2>
  {
    auto TX = Tensor_<S, 2>{X.sizes()};
    auto TX_ = TX.colmajor_view().matrix();

    TX_ = T * X.colmajor_view().matrix();
    TX_.array().rowwise() /= TX_.array().row(2);

    return TX;
  }


  // Data transformations.
  auto extract_centers(const std::vector<OERegion>& features)
      -> Tensor_<float, 2>;

  auto to_point_indices(const TensorView_<int, 2>& samples,
                        const TensorView_<int, 2>& matches)  //
      -> Tensor_<int, 3>;

  auto to_coordinates(const TensorView_<int, 3>& point_indices,
                      const TensorView_<float, 2>& p1,
                      const TensorView_<float, 2>& p2)  //
      -> Tensor_<float, 4>;

} /* namespace Sara */
} /* namespace DO */
