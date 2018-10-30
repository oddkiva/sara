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


namespace DO { namespace Sara {

  template <typename T>
  inline auto arange(T start, T stop, T step)
      -> Eigen::Matrix<T, Eigen::Dynamic, 1>
  {
    const auto num_samples = int((stop - start) / step);
    Eigen::Matrix<T, Eigen::Dynamic, 1> samples(num_samples);
    for (int i = 0; i < num_samples; ++i)
      samples[i] = start + i * step;
    return samples;
  }

  template <typename Mat>
  inline auto flatten(const Mat& x)  //
      -> Eigen::Map<
          const Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, 1>>
  {
    using Scalar = typename Mat::Scalar;
    return {x.data(), x.size()};
  }

  template <typename Mat>
  inline auto flatten(Mat& x)
      -> Eigen::Map<Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, 1>>
  {
    using Scalar = typename Mat::Scalar;
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
    if (x.rows() != y.rows())
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

    auto num_rows = 0;
    for (const auto& xi : x)
      num_rows += xi.rows();

    using OutMat =
        Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    OutMat stack_x(num_rows, x.front().cols());

    auto start_row = 0;

    for (auto i = 0u; i < x.size(); ++i)
    {
      stack_x.block(start_row, 0, x[i].rows(), x[i].cols()) = x[i];
      start_row += x[i].rows();
    }

    return stack_x;
  }

} /* namespace Sara */
} /* namespace DO */
