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


#include "../Core/Image.hpp"


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

  //! \brief Apply Deriche filter with specified order $o$ to dimension $d$.
  template <typename T, int N>
  void inplace_deriche(Image<T, N>& signal, 
                       typename PixelTraits<T>::channel_type sigma,
                       int derivative_order, int axis, bool neumann = true)
  {
    typedef typename PixelTraits<T>::channel_type S;

    // Sanity check.
    if (sigma <= 0)
      throw std::runtime_error("sigma must be positive");
    if (derivative_order < 0 || derivative_order >= 3)
      throw std::runtime_error("derivative order must be between 0 and 2");
    if (axis < 0 || axis >= N)
      throw std::runtime_error("axis of derivative must be between 0 and N-1");

    // Compute the coefficients of the recursive filter
    const S alpha = static_cast<S>(1.695)/sigma;
    const S ea = std::exp(alpha);
    const S ema = std::exp(-alpha);
    const S em2a = ema*ema;
    const S b1 = 2*ema;
    const S b2 = -em2a;

    S ek,ekn,parity,a1,a2,a3,a4,g0,sumg1,sumg0;

    switch(derivative_order)
    {
    // first-order derivative
    case 1:                 
      ek = -(1-ema)*(1-ema)*(1-ema)/(2*(ema+1)*ema);
      a1 = a4 = 0;
      a2 = ek*ema;
      a3 = -ek*ema;
      parity = -1;
      if (neumann)
      {
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
      if (neumann)
      {
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
    std::vector<T> output_signal(signal.size(axis));
    typename std::vector<T>::iterator Y = output_signal.begin();
    const size_t offset = signal.stride(axis);
    const size_t nb = signal.size(axis);

    typename Image<T, N>::iterator it;

    for (it = signal.begin(); it != signal.end(); ++it)
    {
      T *ima = it;
      T I2(*ima); ima += offset;
      T I1(*ima); ima += offset;
      T Y2(sumg0*I2);
      *Y = Y2; ++Y;
      T Y1(g0*I1 + sumg1*I2);
      *Y = Y1; ++Y;

      for (size_t i=2; i<nb; i++)
      {
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

      for (size_t i=nb-3; ; i--)
      {
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
  }

  //! \brief Apply Deriche blurring.
  template <typename T, int N>
  void inplace_deriche_blur(
    Image<T, N>&I,
    const Matrix<typename PixelTraits<T>::channel_type, N, 1>& sigmas,
    bool neumann = true)
  {
    for (int i=0;i<N;i++)
      inplace_deriche(I,sigmas[i], 0, i, neumann);
  }

  //! \brief Apply Deriche blurring.
  template <typename T, int N>
  void inplace_deriche_blur(
    Image<T,N>& I,
    typename PixelTraits<T>::channel_type sigma,
    bool neumann = true)
  {
    typedef typename PixelTraits<T>::channel_type S;
    Matrix<S, N, 1> Sigma; Sigma.fill(sigma);
    inplace_deriche_blur(I, Sigma, neumann);
  }

  //! \brief Return the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T,N> deriche_blur(const Image<T,N>& I,
                         typename PixelTraits<T>::channel_type sigma,
                         bool neumann = true)
  {
    Image<T,N> J(I);
    inplace_deriche_blur(J,sigma,neumann);
    return J;
  }

  //! \brief Return the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T,N> deriche_blur(
    const Image<T,N>& I,
    const Matrix<typename PixelTraits<T>::channel_type, N, 1>& sigmas,
    bool neumann = true)
  {
    Image<T,N> J=I.clone();
    inplace_deriche_blur(J,sigmas,neumann);
    return J;
  }

  //! \brief Helper class to use: Image<T,N>::compute<DericheBlur>(T sigma)
  template <typename T, int N>
  struct DericheBlur
  {
    typedef Image<T, N> ReturnType;
    typedef typename PixelTraits<T>::ChannelType ParamType;
    DericheBlur(const Image<T, N>& src) : src_(src) {}
    ReturnType operator()(ParamType sigma) const { return deriche_blur(src_, sigma); }
    const Image<T, N>& src_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DERICHE_HPP */