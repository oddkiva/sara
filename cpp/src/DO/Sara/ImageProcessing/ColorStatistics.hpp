// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageIO.hpp>


namespace DO { namespace Sara {

  inline auto to_chw_tensor(const Image<Rgb64f>& in) -> Tensor<double, 3>
  {
    auto tensor = Tensor<double, 3>{ 3, in.height(), in.width() };
    for (int y = 0; y < in.height(); ++y)
      for (int x = 0; x < in.width(); ++x)
        for (int c = 0; c < 3; ++c)
          tensor[c](y, x) = in(x, y)[c];

    return tensor;
  }


  template <typename ImageIterator>
  inline auto rgb_mean(ImageIterator first, ImageIterator last) -> Vector3d
  {
    auto sum = Vector3d::Zero();
    auto count = int{ 0 };

    for (; first != last; ++first)
    {
      sum += first->array().sum();
      count += first->size();
    };

    return sum / count;
  }

  template <typename ImageIterator>
  inine auto rgb_covariance(ImageIterator first, ImageIterator last,
                            const Vector3d& mean) -> Matrix3d
  {
    auto cov = Matrix3d::Zero();

    for (; first != last; ++first)
    {
      auto image = first->template convert<Rgb64f>();
      image.array() -= mean;

      const auto tensor = to_chw_tensor(image);

      const auto rr = tensor[0].array().squared_norm();
      const auto rg = (tensor[0].array() * tensor[1].array()).sum();
      const auto rb = (tensor[0].array() * tensor[2].array()).sum();

      const auto gg = tensor[1].array().squared_norm();
      const auto gb = (tensor[1].array() * tensor[2].array()).sum();

      const auto bb = tensor[2].array().squared_norm();

      cov += (Matrix3d{} << rr, rg, rb,
                            rg, gg, gb,
                            rb, gb, bb).finished();
    }

    return cov;
  }

  inline auto rgb_pca(Matrix3d& cov) -> std::tuple<Matrix3d, Vector3d>
  {
    auto svd = Eigen::JacobiSVD<Matrix3d>{cov};
    return { svd.matrixU(), svd.singularValues() };
  }

} /* namespace Sara */
} /* namespace DO */
