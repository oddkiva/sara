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

    using color_mean_type = Matrix<double, num_channels, 1>;
    using color_covariance_matrix_type =
        Matrix<double, num_channels, num_channels>;
  };

  //! @brief Compute the sample color mean vector.
  template <typename ImageIterator>
  inline auto color_sample_mean_vector(ImageIterator first, ImageIterator last)
      -> Matrix<double, ImageIteratorTraits<ImageIterator>::num_channels, 1>
  {
    constexpr auto num_channels = ImageIteratorTraits<ImageIterator>::num_channels;
    using out_vector_type = Matrix<double, num_channels, 1>;

    auto sum = out_vector_type::Zero().eval();
    auto count = 0ull;

    for (; first != last; ++first)
    {
      const auto tensor = to_cwh_tensor(*first);
      for (auto c = 0; c < num_channels; ++c)
        sum[c] += tensor[c].flat_array().template cast<double>().sum();
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
    auto count = 0ull;

    for (; first != last; ++first)
    {
      const auto tensor = to_cwh_tensor(*first);
      count += first->size();

      for (int i = 0; i < num_channels; ++i)
        for (int j = i; j < num_channels; ++j)
        {
          auto t_i = tensor[i].flat_array().template cast<double>() - mean(i);
          auto t_j = tensor[j].flat_array().template cast<double>() - mean(j);
          cov(i, j) += (t_i * t_j).sum();
        }
    }

    for (int i = 1; i < num_channels; ++i)
      for (int j = 0; j < i; ++j)
        cov(i, j) = cov(j, i);

    // Divide by (N-1), cf. Bessel's correction.
    return cov / (count - 1);
  }

  //! @brief Perform PCA on the color data.
  template <typename _Matrix>
  inline auto color_pca(const _Matrix& covariance_matrix)
      -> std::pair<typename Eigen::JacobiSVD<_Matrix>::MatrixUType,
                   typename Eigen::JacobiSVD<_Matrix>::SingularValuesType>
  {
    auto svd = Eigen::JacobiSVD<_Matrix>{covariance_matrix, ComputeFullU};
    return std::make_pair(svd.matrixU(), svd.singularValues());
  }

  template <typename ImageIterator>
  inline auto
  online_color_covariance(ImageIterator first, ImageIterator last) -> std::pair<
      typename ImageIteratorTraits<ImageIterator>::color_mean_type,
      typename ImageIteratorTraits<ImageIterator>::color_covariance_matrix_type>
  {
    constexpr auto num_channels =
        ImageIteratorTraits<ImageIterator>::num_channels;
    using mean_type = ImageIteratorTraits<ImageIterator>::color_mean_type;
    using covariance_matrix_type =
        ImageIteratorTraits<ImageIterator>::color_covariance_matrix_type;

    auto s = mean_type::Zero().eval();
    auto c = covariance_matrix_type::Zero().eval();
    auto n = 0ull;

    for (; first != last; ++first)
    {
      const auto si = first->array().template cast<double>().sum();
      const auto ci = out_matrix_type::Zero().eval();

      const auto ni = first->size();

      const auto mi = (si / ni).eval();

      const auto tensor = to_cwh_tensor(*first);
      for (auto i = 0; i < num_channels; ++i)
        for (auto j = i; j < num_channels; ++j)
        {
          const auto t_i =
              (tensor[i].flat_array().template cast<double>() - mi(i)).sum();
          const auto t_j =
              (tensor[j].flat_array().template cast<double>() - mi(j)).sum();

          // Use Bjorck's trick to increase numerical stability (cf. Chan,
          // Golub and LeVeque).
          const auto t_i_res = (tensor[i].flat_array().template cast<double> - mi(i))/ni;
          const auto t_j_res = (tensor[j].flat_array().template cast<double> - mi(j))/ni;
          ci(i, j) = (t_i * t_j).sum() - t_i_res * t_j_res;
        }

      for (int i = 1; i < num_channels; ++i)
        for (int j = 0; j < i; ++j)
          ci(i, j) = ci(j, i);

      c = c + ci +
          (ni * s / n - si) * (ni * s / n - si).transpose() * m / (n * (m + n));

      n += ni;
    }

    return make_pair((s/n).eval(), (c/(n-1)).eval());
  }


} /* namespace Sara */
} /* namespace DO */
