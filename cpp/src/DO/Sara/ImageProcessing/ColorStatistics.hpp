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


  inline namespace naive {

    //! @brief Compute the sample color mean vector.
    //! The implementation is the naive textbook algorithm.
    template <typename ImageIterator>
    inline auto color_sample_mean_vector(ImageIterator first,
                                         ImageIterator last)
        -> typename ImageIteratorTraits<ImageIterator>::color_mean_type
    {
      constexpr auto num_channels =
          ImageIteratorTraits<ImageIterator>::num_channels;

      using out_vector_type =
          typename ImageIteratorTraits<ImageIterator>::color_mean_type;

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

    //! @brief Compute the color sample covariance matrix.
    /*!
     * Read wikipedia article:
     * https://en.wikipedia.org/wiki/Sample_mean_and_covariance regarding
     * Bessel's
     * correction.
     *
     * This function is implemented with the naive two-pass algorithm.
     */
    template <typename ImageIterator>
    inline auto color_sample_covariance_matrix(
        ImageIterator first, ImageIterator last,
        const typename ImageIteratorTraits<ImageIterator>::color_mean_type&
            mean) ->
        typename ImageIteratorTraits<
            ImageIterator>::color_covariance_matrix_type
    {
      constexpr auto num_channels =
          ImageIteratorTraits<ImageIterator>::num_channels;

      using out_matrix_type = typename ImageIteratorTraits<
          ImageIterator>::color_covariance_matrix_type;

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

  } /* namespace naive */


  inline namespace chan_golub_leveque {

    //! @brief Perform online covariance matrix update
    /*!
     * This function implements the method presented in:
     *
     * Updating Formulae and a Pairwise Algorithm for Computing Sample Variances
     * Tony F. Chan, Gene H. Golub, Randall J. Leveque, November 1979
     * STAN-CS-79-773
     *
     * N.B. we don't use the pairwise algorithm scheme yet.
     */
    template <typename ImageIterator>
    inline auto online_color_covariance(ImageIterator first, ImageIterator last)
        -> std::pair<
            typename ImageIteratorTraits<ImageIterator>::color_mean_type,
            typename ImageIteratorTraits<
                ImageIterator>::color_covariance_matrix_type>
    {
      using vector_type =
          typename ImageIteratorTraits<ImageIterator>::color_mean_type;
      using matrix_type = typename ImageIteratorTraits<
          ImageIterator>::color_covariance_matrix_type;

      constexpr auto num_channels =
          ImageIteratorTraits<ImageIterator>::num_channels;

      auto count = 0ull;
      auto sum = vector_type{};
      auto cov = matrix_type{};
      auto correction = matrix_type{};

      // Initialize the variables with the first image.
      {
        // Read the image as a tensor.
        const auto tensor = to_cwh_tensor(*first);

        // Initialize the number of colors count so far.
        count = first->size();

        // Initialize the sum.
        for (auto i = 0; i < num_channels; ++i)
          sum[i] = tensor[i].flat_array().template cast<double>().sum();

        // Initialize the covariance matrix

        // 1. Calculate the mean expression which is used in the covariance
        //    matrix.
        const auto mean = sum / count;

        // 2. For each entry of the covariance matrix, compute the necessary
        //    algebraic expressions used for the sum of products.
        for (auto i = 0; i < num_channels; ++i)
        {
          for (auto j = i; j < num_channels; ++j)
          {
            // 2a. Compute the expressions.
            const auto t_i =
                (tensor[i].flat_array().template cast<double>() - mean[i]);
            const auto t_j =
                (tensor[j].flat_array().template cast<double>() - mean[j]);

            // 2b. Compute the two residual expressions, which are zero in exact
            //     arithmetic computation but they increase the numerical
            //     stability.
            //
            //     This is Professor Ake Bjorck's trick (cf. paper for details).
            const auto t_i_res =
                (tensor[i].flat_array().template cast<double>() - mean[i])
                    .sum() /
                count;
            const auto t_j_res =
                (tensor[j].flat_array().template cast<double>() - mean[j])
                    .sum() /
                count;

            cov(i, j) = (t_i * t_j).sum() - t_i_res * t_j_res;
          }
        }

        for (int i = 1; i < num_channels; ++i)
          for (int j = 0; j < i; ++j)
            cov(i, j) = cov(j, i);
      }

      for (++first; first != last; ++first)
      {
        const auto tensor = to_cwh_tensor(*first);

        const auto count_i = first->size();

        auto sum_i = vector_type{};
        for (auto i = 0; i < num_channels; ++i)
          sum_i[i] = tensor[i].flat_array().template cast<double>().sum();

        const auto mean_i = sum_i / count_i;

        auto cov_i = matrix_type{};
        for (auto i = 0; i < num_channels; ++i)
        {
          for (auto j = i; j < num_channels; ++j)
          {
            const auto t_i =
                (tensor[i].flat_array().template cast<double>() - mean_i[i]);
            const auto t_j =
                (tensor[j].flat_array().template cast<double>() - mean_i[j]);

            // Bjorck's trick again.
            const auto t_i_res =
                (tensor[i].flat_array().template cast<double>() - mean_i[i])
                    .sum() /
                count_i;
            const auto t_j_res =
                (tensor[j].flat_array().template cast<double>() - mean_i[j])
                    .sum() /
                count_i;

            cov_i(i, j) = (t_i * t_j).sum() - t_i_res * t_j_res;

            correction(i, j) = (sum(i) * count_i / count - sum_i(i)) *
                               (sum(j) * count_i / count - sum_i(j)) * count /
                               count_i / (count + count_i);
          }
        }

        for (int i = 1; i < num_channels; ++i)
        {
          for (int j = 0; j < i; ++j)
          {
            cov_i(i, j) = cov_i(j, i);
            correction(i, j) = correction(j, i);
          }
        }

        count += count_i;
        sum += sum_i;
        cov += cov_i + correction;
      }

      return std::make_pair((sum / count).eval(), (cov / (count - 1)).eval());
    }

  } /* namespace chan_golub_leveque */

  //! @brief Perform PCA on the color data.
  template <typename _Matrix>
  inline auto color_pca(const _Matrix& covariance_matrix)
      -> std::pair<typename Eigen::JacobiSVD<_Matrix>::MatrixUType,
                   typename Eigen::JacobiSVD<_Matrix>::SingularValuesType>
  {
    auto svd = Eigen::JacobiSVD<_Matrix>{covariance_matrix, ComputeFullU};
    return std::make_pair(svd.matrixU(), svd.singularValues());
  }


} /* namespace Sara */
} /* namespace DO */
