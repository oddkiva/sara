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
  typename ColorTraits<T>::Color64f interpolate(const Image<T,N> &I,
                                                const Matrix<F, N, 1>& xx)
  {
    DO_STATIC_ASSERT(
      !std::numeric_limits<F>::is_integer,
      INTERPOLATION_NOT_ALLOWED_FROM_VECTOR_WITH_INTEGRAL_SCALAR_TYPE);
    Matrix<F, N, 1> x(xx);
    for (int i=0; i < N; ++i)
    {
      if (x[i]<0)
        x[i] = 0.;
      else if (x[i] > I.size(i)-1)
        x[i] = static_cast<F>(I.size(i)-1);
    }

    Matrix<F, N, 1> a;
    Matrix<int, N, 1> c;
    for (int i = 0; i < N; ++i) {
      c[i] = static_cast<int>(x[i]);
      a[i] = x[i]-static_cast<F>(c[i]);
    }

    typedef typename ColorTraits<T>::Color64f Col64f;
    Col64f val(ColorTraits<Col64f>::zero());
    CoordsIterator<N> p(c, (c.array()+1).matrix()), end;
    for ( ; p != end; ++p)
    {
      double f = 1.;
      for (int i = 0; i < N; ++i)
        f *= ((*p)[i] == c[i]) ? (1.-a[i]) : a[i];
      if (f==0)
        continue;
      Col64f pix;
      convertColor(pix, I(*p));
      val += pix*f;
    }
    return val;
  }

}

#endif /* DO_IMAGEPROCESSING_INTERPOLATION_HPP */