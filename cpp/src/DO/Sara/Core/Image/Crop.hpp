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

#ifndef DO_SARA_CORE_IMAGE_SUBIMAGE_HPP
#define DO_SARA_CORE_IMAGE_SUBIMAGE_HPP


#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  //! @{
  //! @brief Crop an image.
  template <typename T, int N>
  Image<T, N> crop(const ImageView<T, N>& src,
                   const typename ImageView<T, N>::vector_type& begin_coords,
                   const typename ImageView<T, N>::vector_type& end_coords)
  {
    auto dst = Image<T, N>{ end_coords - begin_coords };

    auto src_it = src.begin_subarray(begin_coords, end_coords);
    for (auto dst_it = dst.begin() ; dst_it != dst.end(); ++dst_it, ++src_it)
    {
      // If a and b are coordinates out bounds.
      if (src_it.position().minCoeff() < 0 ||
          (src_it.position() - src.sizes()).minCoeff() >= 0)
        *dst_it = PixelTraits<T>::min();
      else
        *dst_it = *src_it;
    }

    return dst;
  }

  template <typename T>
  inline Image<T> crop(const ImageView<T>& src, int top_left_x, int top_left_y,
                       int width, int height)
  {
    auto begin_coords = Vector2i{ top_left_x, top_left_y };
    auto end_coords = Vector2i{ top_left_x + width, top_left_y + height };
    return crop(src, begin_coords, end_coords);
  }

  template <typename T>
  inline Image<T> crop(const ImageView<T>& src, int center_x, int center_y,
                       int radius)
  {
    return crop(src, center_x - radius, center_y - radius, 2 * radius + 1,
                2 * radius + 1);
  }


  struct Crop
  {
    template <typename SrcImageView>
    using OutPixel = typename SrcImageView::pixel_type;

    template <typename ImageView_>
    using Coords = typename ImageView_::coord_type;

    template <typename Pixel, int N>
    void operator()(const ImageView<Pixel, N>& src, ImageView<Pixel, N>& dst,
                    const Coords<ImageView<Pixel, N>>& begin_coords,
                    const Coords<ImageView<Pixel, N>>& end_coords) const
    {
      dst = crop(src, begin_coords, end_coords);
    }
  };
  //! @}

} /* namespace Sara */
} /* namespace DO */



#endif /* DO_SARA_CORE_IMAGE_SUBIMAGE_HPP */
