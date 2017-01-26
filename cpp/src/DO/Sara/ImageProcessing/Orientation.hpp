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

#ifndef DO_SARA_IMAGEPROCESSING_ORIENTATION_HPP
#define DO_SARA_IMAGEPROCESSING_ORIENTATION_HPP


#include <DO/Sara/Core/Image/Image.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Differential
    @{
   */

  /*!
    @brief Computes an orientation field from a 2D vector field.
    @param[in]
      src
      an image in which each element **src(x,y)** are 2D vectors, i.e., a discretized
      2D vector field, e.g. image gradients.
    @param[in,out]
      dst
      an image where each pixel **dst(x,y)** contains the orientation of the 2D
      vector 'src(x,y)'
   */
  template <typename T>
  void orientation(const ImageView<Matrix<T,2,1> >& src, ImageView<T>& dst)
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
        "Source and destination image sizes are not equal!"
      };

    auto src_it = src.begin();
    auto src_it_end = src.end();
    auto  dst_it = dst.begin();
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = std::atan2(src_it->y(), src_it->x());
  }

  /*!
    @brief Computes an orientation field from a 2D vector field.
    @param[in]
      src
      an image in which each element **src(x,y)** are 2D vectors, i.e., a discretized
      2D vector field, e.g. image gradients.
    @param[out]
      dst
      an image where each pixel **dst(x,y)** contains the orientation of the 2D
      vector **src(x,y)**
   */
  template <typename T>
  Image<T> orientation(const ImageView<Matrix<T,2,1> >& src)
  {
    auto ori = Image<T>{ src.sizes() };
    orientation(src, ori);
    return ori;
  }

  //! @brief Specialized class to use Image<T,N>::compute<Orientation>()
  struct Orientation
  {
    template <typename SrcImageView>
    using OutPixel = typename SrcImageView::pixel_type::Scalar;

    template <typename SrcImageView, typename DstImageView>
    inline void operator()(const SrcImageView& src, DstImageView& dst) const
    {
      return orientation(src, dst);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_ORIENTATION_HPP */
