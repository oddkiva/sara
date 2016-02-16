// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
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
  void orientation(const Image<Matrix<T,2,1> >& src, Image<T>& dst)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    typedef typename Image<Matrix<T,2,1> >::const_iterator InputIterator;
    typedef typename Image<T>::iterator OutputIterator;

    InputIterator src_it(src.begin());
    InputIterator src_it_end(src.end());
    OutputIterator dst_it(dst.begin());
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
  Image<T> orientation(const Image<Matrix<T,2,1> >& src)
  {
    Image<T> ori;
    orientation(src, ori);
    return ori;
  }

  //! @brief Specialized class to use Image<T,N>::compute<Orientation>()
  struct Orientation
  {
    template <typename VectorField>
    using Scalar = typename VectorField::pixel_type::Scalar;

    template <typename VectorField>
    using ReturnType = Image<Scalar<VectorField>>;

    template <typename VectorField>
    inline Image<Scalar<VectorField>> operator()(const VectorField& in) const
    {
      return orientation(in);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_ORIENTATION_HPP */
