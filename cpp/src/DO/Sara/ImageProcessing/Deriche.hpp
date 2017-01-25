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

#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup ImageProcessing
    @defgroup Deriche Deriche Filter
    Deriche Infinite Impulse Response (IIR) filters which approximates
    - Gaussian smoothing
    - Gaussian smoothed first-order derivative
    - Gaussian smoothed second-order derivative
    @{
   */

  //! @brief Apply Deriche filter with specified order $o$ to dimension $d$.
  template <typename T, int N>
  void inplace_deriche(ImageView<T, N>& inout_signal,
                       typename PixelTraits<T>::channel_type sigma,
                       int derivative_order, int axis, bool neumann = true)
  {
    using S = typename PixelTraits<T>::channel_type;
    using Vector = typename Image<T, N>::vector_type;

    // Sanity check.
    if (sigma <= 0)
      throw std::runtime_error("sigma must be positive");
    if (derivative_order < 0 || derivative_order >= 3)
      throw std::runtime_error("derivative order must be between 0 and 2");
    if (axis < 0 || axis >= N)
      throw std::runtime_error("axis of derivative must be between 0 and N-1");

    // Compute the coefficients of the recursive filter.
    //
    // The constant 1.695 is mysterious... Also found in CImg library.
    // TODO: ask where this constant comes from.
    const auto alpha = static_cast<S>(1.695)/sigma;
    const auto ea = std::exp(alpha);
    const auto ema = std::exp(-alpha);
    const auto em2a = ema*ema;
    const auto b1 = 2*ema;
    const auto b2 = -em2a;

    S ek, ekn;
    S parity;
    S a1, a2, a3, a4;
    S g0, sumg1, sumg0;

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

    // Initialize two temporary arrays.
    const auto size = inout_signal.size(axis);
    const auto step = inout_signal.stride(axis);
    auto y_causal = std::vector<T>(size);
    auto y_anticausal = std::vector<T>(size);

    auto start = Vector::Zero();
    auto end = inout_signal.sizes();
    end[axis] = 1;

    // In 2D, we scan the beginning of each row/columns.
    for (auto it = inout_signal.begin_subarray(start, end); !it.end(); ++it)
    {
      auto ptr = &(*it);

      // Causal signal: i == 0.
      T *forward_x[2] =  { ptr, ptr - step };
      y_causal[0] = sumg0* *forward_x[0];

      // Causal signal: i == 1.
      for (auto k = 0; k < 2; ++k)
        forward_x[k] += step;
      y_causal[1] = g0 * *forward_x[0] + sumg1 * *forward_x[1];

      // Causal signal: i = 2 .. size-1
      for (auto i = 2; i < size; ++i)
      {
        for (auto k = 0; k < 2; ++k)
          forward_x[k] += step;
        y_causal[i] = a1 * *forward_x[0] + a2 * *forward_x[1]
                    + b1 * y_causal[i-1] + b2 * y_causal[i-2];
      }

      // Anti-causal signal: i == size-1
      T *backward_x[2] =  { ptr + (size-1)*step, ptr + size*step };
      y_anticausal[size-1] = parity * sumg1 * *backward_x[0];

      // Anti-causal signal: i == size-2
      for (auto k = 0; k < 2; ++k)
        forward_x[k] += step;
      y_anticausal[size-2] = y_anticausal[size-1];

      // Anti-causal signal: i == size-3 .. 0
      for (auto i = size - 3; i >= 0; --i)
      {
        for (auto k = 0; k < 2; ++k)
          backward_x[k] -= step;
        y_anticausal[i] = a3 * *backward_x[0] + a4 * *backward_x[1]
                        + b1 * y_anticausal[i+1] + b2 * y_anticausal[i+2];
      }

      // Store the sum of the two signals.
      for (auto i = 0; i < size; ++i)
      {
        *ptr = y_causal[i] + y_anticausal[i];
        ptr += step;
      }
    }
  }

  //! @brief Apply Deriche blurring.
  template <typename T, int N>
  void inplace_deriche_blur(
      ImageView<T, N>& inout_signal,
      const Matrix<typename PixelTraits<T>::channel_type, N, 1>& sigmas,
      bool neumann = true)
  {
    for (auto i = 0; i < N; ++i)
      inplace_deriche(inout_signal,sigmas[i], 0, i, neumann);
  }

  //! @brief Apply Deriche blurring.
  template <typename T, int N>
  void inplace_deriche_blur(ImageView<T, N>& inout_signal,
                            typename PixelTraits<T>::channel_type sigma,
                            bool neumann = true)
  {
    using S = typename PixelTraits<T>::channel_type;
    auto sigmas = Matrix<S, N, 1>{};
    sigmas.fill(sigma);
    inplace_deriche_blur(inout_signal, sigmas, neumann);
  }

  //! @brief Return the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T, N> deriche_blur(const ImageView<T, N>& in_signal,
                           typename PixelTraits<T>::channel_type sigma,
                           bool neumann = true)
  {
    auto out_signal = Image<T, N>{ in_signal };
    inplace_deriche_blur(out_signal, sigma, neumann);
    return out_signal;
  }

  //! @brief Return the blurred image using Deriche filter.
  template <typename T, int N>
  Image<T, N> deriche_blur(
      const ImageView<T, N>& I,
      const Matrix<typename PixelTraits<T>::channel_type, N, 1>& sigmas,
      bool neumann = true)

  {
    auto J = I;
    inplace_deriche_blur(J, sigmas, neumann);
    return J;
  }

  //! @brief Wrapper class to use: Image<T,N>::compute<DericheBlur>(T sigma)
  struct DericheBlur
  {
    template <typename SrcImageView>
    using OutPixel = typename SrcImageView::pixel_type;

    template <typename SrcImageView, typename DstImageView, typename Sigma>
    inline void operator()(const SrcImageView& src, DstImageView& dst,
                           const Sigma& sigma, bool neumann = true) const
    {
      dst.copy(src);
      inplace_deriche_blur(dst, sigma, neumann);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
