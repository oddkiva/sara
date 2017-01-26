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

#ifndef DO_SARA_IMAGEPROCESSING_INTERPOLATION_HPP
#define DO_SARA_IMAGEPROCESSING_INTERPOLATION_HPP


#include <stdexcept>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup ImageProcessing
    @defgroup Interpolation Interpolation
    @{
   */

  // ====================================================================== //
  // Interpolation
  //! @brief Interpolation function
  template <typename T, int N>
  auto interpolate(const ImageView<T, N>& image,
                   const Matrix<double, N, 1>& pos)

      -> typename PixelTraits<T>::template Cast<double>::pixel_type
  {
    // Typedefs.
    using DoublePixel =
        typename PixelTraits<T>::template Cast<double>::pixel_type;

    // Find the smallest integral bounding box that encloses the position.
    auto start = Matrix<int, N, 1>{};
    auto end = Matrix<int, N, 1>{};
    auto frac = Matrix<double, N, 1>{};
    for (int i = 0; i < N; ++i)
    {
      if (pos[i] < 0 || pos[i] >= image.size(i))
        throw std::out_of_range{
          "Cannot interpolate: position is out of image domain"
        };

      auto ith_int_part = double{};
      frac[i] = std::modf(pos[i], &ith_int_part);
      start[i] = static_cast<int>(ith_int_part);
    }
    end.array() = start.array() + 2;

    // Compute the weighted sum.
    auto it = image.begin_subarray(start, end);
    auto interpolated_value = PixelTraits<DoublePixel>::min();
    auto offset = Matrix<int, N, 1>{};
    for ( ; !it.end(); ++it)
    {
      auto weight = 1.;
      for (auto i = 0; i < N; ++i)
      {
        weight *= (it.position()[i] == start[i]) ? (1. - frac[i]) : frac[i];
        offset[i] = it.position()[i] < image.size(i) ? 0 : -1;
      }

      auto dst_color = PixelTraits<T>::template Cast<double>::apply(it(offset));
      interpolated_value += weight * dst_color;
    }

    return interpolated_value;
  }

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_INTERPOLATION_HPP */
