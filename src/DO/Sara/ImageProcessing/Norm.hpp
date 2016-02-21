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

#ifndef DO_SARA_IMAGEPROCESSING_NORM_HPP
#define DO_SARA_IMAGEPROCESSING_NORM_HPP


#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Differential
    @{
   */

  /*!
    @brief Squared norm computation
    @param[in] src scalar field.
    @param[in, out] dst scalar field of squared norms
   */
  template <typename T, int M, int N, int D>
  void squared_norm(const ImageView<Matrix<T,M,N>, D>& src, ImageView<T, D>& dst)
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    auto src_it = src.begin();
    auto src_it_end = src.end();
    auto dst_it = dst.begin();
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->squaredNorm();
  }

  /*!
    @brief Squared norm computation
    @param[in] src scalar field.
    @return scalar field of squared norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> squared_norm(const ImageView<Matrix<T,M,N>, D>& src)
  {
    auto squared_norm_image = Image<T, D>{ src.sizes() };
    squared_norm(src, squared_norm_image);
    return squared_norm_image;
  }

  /*!
    @brief Blue norm computation
    @param[in] src scalar field.
    @param[in, out] scalar field of norms
   */
  template <typename T, int M, int N, int D>
  void blue_norm(const ImageView<Matrix<T,M,N>, D>& src, ImageView<T, D>& dst)
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    auto src_it = src.begin();
    auto src_it_end = src.end();
    auto dst_it = dst.begin();
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->blueNorm();
  }

  /*!
    @brief Blue norm computation
    @param[in] src scalar field.
    @return scalar field of norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> blue_norm(const ImageView<Matrix<T,M,N>, D>& src)
  {
    auto blue_norm_image = Image<T, D>{ src.sizes() };
    blue_norm(src, blue_norm_image);
    return blue_norm_image;
  }

  /*!
    @brief Stable norm computation
    @param[in] src scalar field.
    @param[in, out] scalar field of norms
   */
  template <typename T, int M, int N, int D>
  void stable_norm(const ImageView<Matrix<T,M,N>, D>& src, ImageView<T, D>& dst)
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    auto src_it = src.begin();
    auto src_it_end = src.end();
    auto dst_it = dst.begin();
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->stableNorm();
  }

  /*!
    @brief Stable norm computation
    @param[in] src scalar field.
    @return scalar field of norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> stable_norm(const ImageView<Matrix<T,M,N>, D>& src)
  {
    auto stable_norm_image = Image<T, D>{ src.sizes() };
    stable_norm(src, stable_norm_image);
    return stable_norm_image;
  }

#define CREATE_NORM_FUNCTOR(Function, function)                       \
   /*! @brief Helper class to use Image<T,N>::compute<Function>() */  \
  struct Function                                                     \
  {                                                                   \
    template <typename MatrixField>                                   \
    struct Dimension {                                                \
      enum { value = MatrixField::Dimension };                        \
    };                                                                \
                                                                      \
    template <typename MatrixField>                                   \
    using Scalar = typename MatrixField::pixel_type::Scalar;          \
                                                                      \
    template <typename MatrixField>                                   \
    using OutPixel = typename MatrixField::pixel_type::Scalar;        \
                                                                      \
    template <typename SrcField, typename DstField>                   \
    void operator()(const SrcField& src, DstField& dst) const         \
    {                                                                 \
      return function(src, dst);                                      \
    }                                                                 \
  }

  CREATE_NORM_FUNCTOR(SquaredNorm, squared_norm);
  CREATE_NORM_FUNCTOR(BlueNorm, blue_norm);
  CREATE_NORM_FUNCTOR(StableNorm, stable_norm);

#undef CREATE_NORM_FUNCTOR

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_NORM_HPP */
