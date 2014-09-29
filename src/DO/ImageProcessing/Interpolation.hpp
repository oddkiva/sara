// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_IMAGEPROCESSING_INTERPOLATION_HPP
#define DO_IMAGEPROCESSING_INTERPOLATION_HPP


#include <stdexcept>

#include <DO/Core/Image.hpp>
#include <DO/Core/Pixel/PixelTraits.hpp>


namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup Interpolation Interpolation
    @{
   */

  // ====================================================================== //
  // Interpolation
  //! \brief Interpolation function
  template <typename T, int N>
  typename PixelTraits<T>::template Cast<double>::pixel_type
  interpolate(const Image<T, N>& image, const Matrix<double, N, 1>& pos)
  {
    typedef typename PixelTraits<T>::template Cast<double>::pixel_type
      pixel_type;
    typedef typename Image<T, N>::const_subarray_iterator
      const_subarray_iterator;

    Matrix<double, N, 1> a, b;
    a.setZero();
    b = image.sizes().template cast<double>() - Matrix<double, N, 1>::Ones();

    if ((pos - a).minCoeff() < 0 || (b - pos).minCoeff() <= 0)
      throw std::range_error("Cannot interpolate: position is out of range");

    Matrix<int, N, 1> start, end;
    Matrix<double, N, 1> frac;
    for (int i = 0; i < N; ++i)
    {
      double ith_int_part;
      frac[i] = std::modf(pos[i], &ith_int_part);
      start[i] = static_cast<int>(ith_int_part);
    }
    end.array() = start.array() + 2;

    pixel_type value(color_min_value<pixel_type>());
    const_subarray_iterator it(image.begin_subrange(start, end));
    const_subarray_iterator it_end(image.end_subrange());
    for ( ; it != it_end; ++it)
    {
      double weight = 1.;
      for (int i = 0; i < N; ++i)
        weight *= (it.position()[i] == start[i]) ? (1.-frac[i]) : frac[i];
      pixel_type color;
      convert_channel(*it, color);
      value += weight*color;
    }
    return value;
  }

}


#endif /* DO_IMAGEPROCESSING_INTERPOLATION_HPP */