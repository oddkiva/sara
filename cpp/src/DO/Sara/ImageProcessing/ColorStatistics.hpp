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

#include <DO/Sara/Core/Tensor.hpp>


namespace DO { namespace Sara {


  //! @brief Image iterator trait class.
  template <typename Iterator>
  struct ImageIteratorTraits
  {
    using iterator = Iterator;
    using value_type = typename Iterator::value_type;
    using pixel_type = typename value_type::pixel_type;
    using channel_type = typename PixelTraits<pixel_type>::channel_type;

    static const int num_channels = PixelTraits<pixel_type>::num_channels;
  };

  //! @brief Compute the sample color mean vector.
  template <typename ImageIterator>
  inline auto color_sample_mean_vector(ImageIterator first, ImageIterator last)
      -> Matrix<double, ImageIteratorTraits<ImageIterator>::num_channels, 1>
  {
    constexpr auto num_channels = ImageIteratorTraits<ImageIterator>::num_channels;
    using out_vector_type = Matrix<double, num_channels, 1>;

    auto sum = out_vector_type::Zero().eval();
    auto count = int{0};

    for (; first != last; ++first)
    {
      const auto tensor = to_cwh_tensor(*first);
      for (auto c = 0; c < num_channels; ++c)
        sum[c] += tensor[c].array().template cast<double>().sum();
      count += first->size();
    };

    return sum / count;
  }

  //! @brief Compute the color covariance matrix.
  /*!
   * Read wikipedia article:
   * https://en.wikipedia.org/wiki/Sample_mean_and_covariance regarding Bessel's
   * correction.
   */
  template <typename ImageIterator>
  inline auto color_sample_covariance_matrix(
      ImageIterator first, ImageIterator last,
      const Matrix<double, ImageIteratorTraits<ImageIterator>::num_channels, 1>&
          mean) -> Matrix3d
  {
    constexpr auto num_channels = ImageIteratorTraits<ImageIterator>::num_channels;
    using out_matrix_type = Matrix<double, num_channels, num_channels>;

    auto cov = out_matrix_type::Zero().eval();
    auto count = int{0};

    for (; first != last; ++first)
    {
      const auto tensor = to_cwh_tensor(*first);
      count += first->size();

      for (int i = 0; i < num_channels; ++i)
        for (int j = i; j < num_channels; ++j)
        {
          auto t_i = tensor[i].array().template cast<double>() - mean(i);
          auto t_j = tensor[j].array().template cast<double>() - mean(j);
          cov(i, j) += (t_i * t_j).sum();
        }
    }

    for (int i = 1; i < num_channels; ++i)
      for (int j = 0; j < i; ++j)
        cov(i, j) = cov(j, i);

    // Divide (N-1) which is Bessel's correction.
    return cov / (count - 1);
  }

  //! @brief Perform PCA on the color data.
  template <typename T, int N>
  inline auto color_pca(const Matrix<T, N, N>& covariance_matrix)
      -> std::pair<Matrix<T, N, N>, Matrix<T, N, 1>>
  {
    auto svd = Eigen::JacobiSVD<Matrix<T, N, N>>{covariance_matrix, ComputeFullU};
    return std::make_pair(svd.matrixU(), svd.singularValues());
  }

} /* namespace Sara */
} /* namespace DO */
