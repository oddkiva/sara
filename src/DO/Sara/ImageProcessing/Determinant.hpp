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
    template <typename SrcImageView>
    using Matrix = typename SrcImageView::pixel_type;

    template <typename SrcImageView>
    using OutPixel =
        decltype(std::declval<Matrix<SrcImageView>>().determinant());

    template <typename SrcImageView, typename DstImageView>
    inline void operator()(const SrcImageView& src, DstImageView& dst) const
    {
      if (src.sizes() != dst.sizes())
        throw std::domain_error{
          "Source and destination image sizes are not equal!"
        };

      auto dst_i = dst.begin();
      auto src_i = src.begin();
      for ( ; src_i != src.end(); ++src_i, ++dst_i)
        *dst_i = src_i->determinant();
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP */
