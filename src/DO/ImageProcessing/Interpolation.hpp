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

namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup Interpolation Interpolation
    @{
   */

  // ====================================================================== //
  // Interpolation
  //! \brief Interpolation function
  template <typename T, int N, typename F> 
  typename ColorTraits<T>::Color64f interpolate(const Image<T,N>& image,
                                                const Matrix<F, N, 1>& pos)
  {
    DO_STATIC_ASSERT(
      !std::numeric_limits<F>::is_integer,
      INTERPOLATION_NOT_ALLOWED_FROM_VECTOR_WITH_INTEGRAL_SCALAR_TYPE);
    
    Matrix<F, N, 1> a, b;
    a.setZero();
    b = image.sizes().template cast<F>() - Matrix<F, N, 1>::Ones();
    if ((pos - a).minCoeff() < 0 || (b - pos).minCoeff() <= 0)
      throw std::range_error("Cannot interpolate: position is out of range");

    Matrix<int, N, 1> start, end;
    Matrix<F, N, 1> frac;
    for (int i = 0; i < N; ++i)
    {
      double ith_int_part;
      frac[i] = std::modf(pos[i], &ith_int_part);
      start[i] = static_cast<int>(ith_int_part);
    }
    end.array() = start.array() + 2;

    typedef typename ColorTraits<T>::Color64f Col64f;
    typedef typename Image<T, N>::const_subrange_iterator CSubrangeIterator;

    Col64f val(ColorTraits<Col64f>::zero());
    CSubrangeIterator it(image.begin_subrange(start, end));
    static const int num_times = 1 << N;

    for (int i = 0; i < num_times; ++i, ++it)
    {
      double weight = 1.;
      for (int i = 0; i < N; ++i)
        weight *= (it.position()[i] == start[i]) ? (1.-frac[i]) : frac[i];
      Col64f color;
      convertColor(color, *it);
      val += color*weight;
    }
    return val;
  }

}

#endif /* DO_IMAGEPROCESSING_INTERPOLATION_HPP */