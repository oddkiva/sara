// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Tensor.hpp>


// On Eigen matrix data structures.
namespace DO::Sara::EigenExt {

  template <typename T>
  inline auto arange(T start, T stop, T step)
      -> Eigen::Matrix<T, Eigen::Dynamic, 1>
  {
    const auto bound = (stop - start) / step;
    const auto num_samples = static_cast<int>(
        bound - std::floor(bound) > 0 ? std::ceil(bound) : std::floor(bound));
    stop = start + (num_samples - 1) * step;
    return Eigen::Matrix<T, Eigen::Dynamic, 1>::LinSpaced(num_samples, start,
                                                          stop);
  }

  template <typename Mat>
  inline auto flatten(const Mat& x)  //
      -> Eigen::Map<
          const Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, 1>>
  {
    return {x.data(), x.size()};
  }

  template <typename Mat>
  inline auto flatten(Mat& x)
      -> Eigen::Map<Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, 1>>
  {
    return {x.data(), x.size()};
  }

  template <typename MatX, typename MatY>
  auto meshgrid(const MatX& x, const MatY& y)  //
      -> std::tuple<
          Eigen::Matrix<typename MatX::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<typename MatY::Scalar, Eigen::Dynamic, Eigen::Dynamic>>
  {
    using OutMatX =
        Eigen::Matrix<typename MatX::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using OutMatY =
        Eigen::Matrix<typename MatY::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    auto xv = OutMatX{x.size(), y.size()};
    auto yv = OutMatY{x.size(), y.size()};

    const auto nx = x.size();  // cols
    const auto ny = y.size();  // rows

    auto flat_x = flatten(x);
    auto flat_y = flatten(y).transpose();

    for (auto c = 0; c < ny; ++c)
      xv.col(c) = flat_x;

    for (auto r = 0; r < nx; ++r)
      yv.row(r) = flat_y;

    return std::make_tuple(xv, yv);
  }

  template <typename Mat>
  auto hstack(const Mat& x, const Mat& y)  //
      -> Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>
  {
    if (x.rows() != y.rows())
      throw std::runtime_error{"The number of rows are not equal for hstack!"};

    auto xy =
        Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>{
            x.rows(), x.cols() + y.cols()};
    xy << x, y;
    return xy;
  }

  template <typename Mat>
  auto vstack(const Mat& x, const Mat& y)  //
      -> Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>
  {
    if (x.cols() != y.cols())
      throw std::runtime_error{
          "The number of columns are not equal for vstack!"};

    auto xy =
        Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>{
            x.rows() + y.rows(), x.cols()};
    xy << x, y;
    return xy;
  }

  template <typename Mat>
  auto vstack(const std::vector<Mat>& x)  //
      -> Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>
  {
    if (x.empty())
      return {};

    if (x.size() == 1)
      return x[0];

    for (const auto& xi : x)
      if (xi.cols() != x.front().cols())
        throw std::runtime_error(
            "The number of columns are not equal for vstack!");

    using index_type = decltype(x.front().rows());
    static_assert(index_type{} == 0);

    auto num_rows = index_type{};
    for (const auto& xi : x)
      num_rows += xi.rows();

    using OutMat =
        Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    OutMat stack_x(num_rows, x.front().cols());

    auto start_row = index_type{};
    for (auto i = 0u; i < x.size(); ++i)
    {
      stack_x.block(start_row, 0, x[i].rows(), x[i].cols()) = x[i];
      start_row += x[i].rows();
    }

    return stack_x;
  }

} /* namespace DO::Sara::EigenExt */


// On the MultiArray class.
namespace DO::Sara {

  template <typename T>
  inline auto range(T n) -> Tensor_<T, 1>
  {
    static_assert(std::numeric_limits<T>::is_integer);
    auto indices = Tensor_<T, 1>{static_cast<int>(n)};
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  template <typename T>
  inline auto arange(T start, T stop, T step) -> Tensor_<T, 1>
  {
    const auto bound = (stop - start) / step;
    const auto num_samples = static_cast<int>(
        bound - std::floor(bound) > 0 ? std::ceil(bound) : std::floor(bound));
    stop = start + (num_samples - 1) * step;
    auto r = range(num_samples).cast<T>();
    std::for_each(std::begin(r), std::end(r),
                  [&](auto& i) { i = start + i * step; });
    return r;
  }

  template <typename T, int O>
  inline auto vstack(const TensorView<T, 2, O>& x, const TensorView<T, 2, O>& y)
      -> Tensor<T, 2, O>
  {
    if (x.cols() != y.cols())
      throw std::runtime_error{
          "The number of columns are not equal for vstack!"};
    auto xy = Tensor<T, 2, O>{{x.rows() + y.rows(), x.cols()}};
    xy.matrix() << x.matrix(), y.matrix();
    return xy;
  }

} /* namespace DO::Sara */
