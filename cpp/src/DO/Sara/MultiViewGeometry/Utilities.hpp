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
#include <DO/Sara/FeatureMatching.hpp>

#include <algorithm>
#include <random>


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
    std::shuffle(x_shuffled.begin(), x_shuffled.end(), std::mt19937{});
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


  // Elementary transformations.
  template <typename T>
  auto skew_symmetric_matrix(const Matrix<T, 3, 1>& a) -> Matrix<T, 3, 3>
  {
    auto A = Matrix<T, 3, 3>{};
    A <<  T(0), -a(2),  a(1),
          a(2),  T(0), -a(0),
         -a(1),  a(0),  T(0);

    return A;
  }

  template <typename T>
  auto rotation_x(T theta) -> Matrix<T, 3, 3>
  {
    auto R = Matrix<T, 3, 3>{};
    R << T(1), T(0), T(0), T(0), std::cos(theta), -std::sin(theta), T(0),
        std::sin(theta), std::cos(theta);
    return R;
  }

  template <typename T>
  auto rotation_y(T phi) -> Matrix<T, 3, 3>
  {
    auto R = Matrix<T, 3, 3>{};
    R << std::cos(phi), T(0), -std::sin(phi), T(0), T(1), T(0), std::sin(phi),
        T(0), std::cos(phi);
    return R;
  }

  template <typename T>
  auto rotation_z(T kappa) -> Matrix<T, 3, 3>
  {
    auto R = Matrix<T, 3, 3>{};
    R << std::cos(kappa), -std::sin(kappa), T(0), std::sin(kappa),
        std::cos(kappa), T(0), T(0), T(0), T(1);
    return R;
}



  // Data transformations.
  auto extract_centers(const std::vector<OERegion>& features)
      -> Tensor_<float, 2>;

  auto to_point_indices(const TensorView_<int, 2>& samples,
                        const TensorView_<int, 2>& matches)  //
      -> Tensor_<int, 3>;

  template <typename T>
  auto to_coordinates(const TensorView_<int, 3>& point_indices,
                      const TensorView_<T, 2>& p1,
                      const TensorView_<T, 2>& p2)  //
  {
    const auto num_samples = point_indices.size(0);
    const auto sample_size = point_indices.size(1);
    const auto num_points = 2;
    const auto coords_dim = p1.size(1);

    auto p = Tensor_<T, 4>{{num_samples, sample_size, num_points, coords_dim}};

    for (auto s = 0; s < num_samples; ++s)
      for (auto m = 0; m < sample_size; ++m)
      {
        auto p1_idx = point_indices(s, m, 0);
        auto p2_idx = point_indices(s, m, 1);

        p[s][m][0].flat_array() = p1[p1_idx].flat_array();
        p[s][m][1].flat_array() = p2[p2_idx].flat_array();
      }

    return p;
  }

} /* namespace Sara */
} /* namespace DO */
