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

#ifndef DO_IMAGEPROCESSING_DERICHE_HPP
#define DO_IMAGEPROCESSING_DERICHE_HPP

namespace DO {
  /*!
    \ingroup ImageProcessing
    \defgroup Deriche Deriche Filter
    Deriche Infinite Impulse Response (IIR) filters which approximates
    - Gaussian smoothing
    - Gaussian smoothed first-order derivative
    - Gaussian smoothed second-order derivative
    @{
   */

  //! \brief Applies Deriche filter with specified order $o$ to dimension $d$.
  template <typename T, int N>
  void inPlaceDeriche(Image<T, N>& I, 
                      typename ColorTraits<T>::ChannelType sigma,
                      int order, int d, bool neumann = true)
  {
    typedef typename ColorTraits<T>::ChannelType S;

    // Checks parameter values
    assert(sigma>0 && order>=0 && order<3 && d>=0 && d<N);

    // Computes coefficients of the recursive filter
    const S alpha = 1.695f/sigma;
    const S ea = std::exp(alpha);
    const S ema = std::exp(-alpha);
    const S em2a = ema*ema;
    const S b1 = 2*ema;
    const S b2 = -em2a;

    S ek,ekn,parity,a1,a2,a3,a4,g0,sumg1,sumg0;

    switch(order) {

      // first-order derivative
    case 1:                 
      ek = -(1-ema)*(1-ema)*(1-ema)/(2*(ema+1)*ema);
      a1 = a4 = 0;
      a2 = ek*ema;
      a3 = -ek*ema;
      parity = -1;
      if (neumann) {
        sumg1 = (ek*ea) / ((ea-1)*(ea-1));
        g0 = 0;
        sumg0 = g0 + sumg1;
      }
      else
        g0 = sumg0 = sumg1 = 0;
      break;

      // second-order derivative
    case 2:               
      ekn = ( -2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea) );
      ek = -(em2a-1)/(2*alpha*ema);
      a1 = ekn;
      a2 = -ekn*(1+ek*alpha)*ema;
      a3 = ekn*(1-ek*alpha)*ema;
      a4 = -ekn*em2a;
      parity = 1;
      if (neumann) {
        sumg1 = ekn/2;
        g0 = ekn;
        sumg0 = g0 + sumg1;
      }
      else
        g0 = sumg0 = sumg1 = 0;
      break;

      // smoothing
    default:
      ek = (1-ema)*(1-ema) / (1+2*alpha*ema - em2a);
      a1 = ek;
      a2 = ek*ema*(alpha-1);
      a3 = ek*ema*(alpha+1);
      a4 = -ek*em2a;
      parity = 1;
      if (neumann) {
        sumg1 = ek*(alpha*ea+ea-1) / ((ea-1)*(ea-1));
        g0 = ek;
        sumg0 = g0 + sumg1;
      }
      else
        g0 = sumg0 = sumg1 = 0;
      break;
    }

    // filter init
    T *Y = new T[I.size(d)];
    const size_t offset = I.stride(d);
    const size_t nb = I.size(d);

    typedef typename Image<T, N>::coords_type Coords;
    Coords beg; beg.fill(0);
    Coords end((I.sizes().array() - 1).matrix());
    end[d]=0;

    // Do not change the code! It works optimally with interleaved color!
    for (CoordsIterator<N> p(beg,end); p != CoordsIterator<N>(); ++p) {
      T *ima = &(I(*p));
      T I2(*ima); ima += offset;
      T I1(*ima); ima += offset;
      T Y2(sumg0*I2);
      *Y = Y2; ++Y;
      T Y1(g0*I1 + sumg1*I2);
      *Y = Y1; ++Y;
      for (size_t i=2; i<nb; i++) {
        I1 = *ima; ima+=offset;
        T Y0(a1*I1 + a2*I2 + b1*Y1 + b2*Y2);
        *Y = Y0; ++Y;
        I2=I1; Y2=Y1; Y1=Y0;
      }
      ima -= offset;
      I2 = *ima;
      Y2 = Y1 = (parity*sumg1)*I2;
      *ima = *(--Y)+Y2;
      ima-=offset;
      I1 = *ima;
      *ima = *(--Y)+Y1;
      for (size_t i=nb-3; ; i--) {
        T Y0(a3*I1+a4*I2+b1*Y1+b2*Y2);
        ima-=offset;
        I2=I1;
        I1=*ima;
        *ima=*(--Y)+Y0;
        Y2=Y1;
        Y1=Y0;
        if (i==0)
          break;
      } 
    }
    delete [] Y;
  }
  //! \brief Applies Deriche filter-based blurring.
  template <typename T, int N>
  void inPlaceDericheBlur(Image<T,N>&I,
                          const Matrix<typename ColorTraits<T>::ChannelType, N, 1>& sigmas,
                          bool neumann = true)
  {
    for (int i=0;i<N;i++)
      inPlaceDeriche(I,sigmas[i], 0, i, neumann);
  }
  //! \brief Applies Deriche filter-based blurring.
  template <typename T, int N>
  void inPlaceDericheBlur(Image<T,N>& I,
                          typename ColorTraits<T>::ChannelType sigma,
                          bool neumann = true)
  {
    typedef typename ColorTraits<T>::ChannelType S;
    Matrix<S, N, 1> Sigma; Sigma.fill(sigma);
    inPlaceDericheBlur(I, Sigma, neumann);
  }
  //! \brief Returns the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T,N> dericheBlur(const Image<T,N>& I,
                         typename ColorTraits<T>::ChannelType sigma,
                         bool neumann = true)
  {
    Image<T,N> J(I);
    inPlaceDericheBlur(J,sigma,neumann);
    return J;
  }
  //! \brief Returns the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T,N> dericheBlur(const Image<T,N>& I,
                         const Matrix<typename ColorTraits<T>::ChannelType, N, 1>& sigmas,
                         bool neumann = true)
  {
    Image<T,N> J=I.clone();
    inPlaceDericheBlur(J,sigmas,neumann);
    return J;
  }

  //! \brief Helper class to use Image<T,N>::compute<DericheBlur>(T sigma)
  template <typename T, int N>
  struct DericheBlur
  {
    typedef Image<T, N> ReturnType;
    typedef typename ColorTraits<T>::ChannelType ParamType;
    DericheBlur(const Image<T, N>& src) : src_(src) {}
    ReturnType operator()(ParamType sigma) const { return dericheBlur(src_, sigma); }
    const Image<T, N>& src_;
  };

  //! @}
} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DERICHE_HPP */