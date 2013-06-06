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

#ifndef DO_IMAGEPROCESSING_SCALING_HPP
#define DO_IMAGEPROCESSING_SCALING_HPP

namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup Scaling Reduce, enlarge, warp image...
    @{
   */

  // ====================================================================== //
  // Naive upsampling and downsampling functions.
  template <typename T, int N>
  Image<T, N> upscale(const Image<T, N>& I, int fact)
  {
    Image<T, N> I1(I.sizes()*fact);
    CoordsIterator<N> c(I1.begin_coords()), end;
    for ( ; c != end; ++c)
      I1(*c) = I((*c)/fact);
    return I1;
  }

  template <typename T, int N>
  Image<T, N> downscale(const Image<T, N>& I, int fact)
  {
    Image<T, N> I1(I.sizes()/fact);
    CoordsIterator<N> c(I1.begin_coords()), end;
    for ( ; c != end; ++c)
      I1(*c) = I((*c)*fact);
    return I1;
  }

  template <typename T, int N>
  inline std::pair<T, T> range(const Matrix<T, N, 1>& v)
  {
    return std::make_pair(
      *std::min_element(v.data(), v.data()+N),
      *std::max_element(v.data(), v.data()+N)
    );
  }

  template <typename T, int N>
  Image<T, N> reduce(const Image<T,N>& I, Matrix<int, N, 1> newSizes,
                     bool keepRatio=false)  
  {
    typedef Matrix<double, N, 1> Vectord;

    // Convert in floating type scalar.
    typedef typename ColorTraits<T>::Color64f Color64f;
    Image<Color64f, N> oriI;
    convert(oriI, I);

    // Determine the right blurring factor.
    Vectord oriSizes(I.sizes().template cast<double>());
    Vectord f = oriSizes.cwiseQuotient(newSizes.template cast<double>());
    std::pair<double, double> mM = range(f);
    assert(mM.first >= 1.);
    if (keepRatio)
    {
      f.fill(mM.second);
      newSizes = (oriSizes/mM.second).template cast<int>();
    }

    Vectord sigmas = 1.5*((f.array().sqrt()-.99).matrix());

    // Blur with the Deriche filter.
    inPlaceDericheBlur(oriI, sigmas);

    // Create the new image by interpolating pixel values.
    Image<T, N> nI(newSizes);
    CoordsIterator<N> r(nI.begin_coords()), end;
    for ( ; r != end; ++r)
    {
      Vectord x( r->template cast<double>().cwiseProduct(f) );
      convertColor(nI(*r), interpolate(oriI, x));
    }

    return nI;
  }
  
  template <typename T>
  inline  Image<T,2> reduce(const Image<T,2>& I, int w, int h, bool keepRatio=false)
  { return reduce(I, Vector2i(w,h), keepRatio); }
  
  template <typename T>
  inline Image<T,3> reduce(const Image<T,3>& I, int w, int h, int d, bool keepRatio=false)
  { return reduce(I, Vector3i(w,h,d), keepRatio); }
  
  template <typename T,int N>
  inline Image<T,N> reduce(const Image<T,N>& I,double fact)  
  {
    Matrix<int, N, 1> nd( (I.sizes().template cast<double>()/fact).
                         template cast<int>() );
    return reduce(I,nd);
  }

  template <typename T, int N>
  inline Image<T,N> enlarge(const Image<T,N>& I, Matrix<int, N, 1> newSizes,
                            bool keepRatio=false)  
  {
    typedef Matrix<double, N, 1> Vectord;

    // Determine the right blurring factor.
    Vectord oriSizes(I.sizes().template cast<double>());
    Vectord f = oriSizes.cwiseQuotient(newSizes.template cast<double>());
    std::pair<double, double> mM = range(f);
    assert(mM.second <= 1.);
    if (keepRatio)
    {
      f.fill(mM.second);
      newSizes = (oriSizes/mM.second).template cast<int>();
    }

    // Create the new image by interpolation.
    Image<T,N> nI(newSizes);
    CoordsIterator<N> r(nI.begin_coords()), end;
    for ( ; r != end; ++r)
    {
      Vectord x( r->template cast<double>().cwiseProduct(f) );
      convertColor(nI(*r), interpolate(I, x));
    }
    return nI;
  }

  template <typename T>
  inline Image<T,2> enlarge(const Image<T,2>& I, int w, int h,
                            bool keepRatio = false)
  { return enlarge(I, Point2i(w,h), keepRatio); }

  template <typename T>
  inline Image<T,3> enlarge(const Image<T,3>& I, int w, int h, int d,
                            bool keepRatio = false)
  { return enlarge(I, Vector3i(w,h,d), keepRatio); }

  template <typename T,int N>
  inline Image<T,N> enlarge(const Image<T,N>&I,double fact)  
  {
    Matrix<int, N, 1> nd( (I.sizes().template cast<double>()*fact).
                         template cast<int>() );
    return enlarge(I,nd);
  }

  //! @} file
}

#endif /* DO_IMAGEPROCESSING_SCALING_HPP */