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

#ifndef DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP
#define DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP


#include <DO/Sara/Core/Image/Image.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Differential
    @{
   */

  //! @brief Helper class to use Image<T,N>::compute<Determinant>().
  struct Determinant
  {
    template <typename ImageView>
    using Matrix = typename ImageView::pixel_type;

    template <typename ImageView>
    using Scalar = typename Matrix<ImageView>::Scalar;

    template <typename ImageView>
    inline auto operator()(const ImageView& src) const
        -> Image<Scalar<ImageView>, ImageView::Dimension>
    {
      auto dst = Image<Scalar<ImageView>, ImageView::Dimension>{ src.sizes() };

      auto dst_i = dst.begin();
      auto src_i = src.begin();
      for ( ; src_i != src.end(); ++src_i, ++dst_i)
        *dst_i = src_i->determinant();

      return dst;
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP */
