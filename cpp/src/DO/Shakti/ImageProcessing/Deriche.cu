// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Shakti/MultiArray/MultiArrayView.hpp>

#include <array>
#include <cmath>


namespace DO { namespace Shakti {

  /*!
   *  @ingroup ImageProcessing
   *  @defgroup Deriche Deriche Filter
   *  Deriche Infinite Impulse Response (IIR) filters which approximates
   *  - Gaussian smoothing
   *  - Gaussian smoothed first-order derivative
   *  - Gaussian smoothed second-order derivative
   *  @{
   */

  // Reuse Nehab's implementation to have a really efficient implementation
  // available here: https://github.com/andmax/gpufilter

//  //! @brief Apply Deriche filter with specified order $o$ to dimension $d$.
//  __global__ template <typename T>
//  void inplace_deriche_2d(T* inout_signal, const Vector2i& sizes, T sigma,
//                          int derivative_order, int axis, bool neumann = true)
//  {
//    const auto p = coords<2>();
//
//    // Bound checking.
//    if (p.x() >= sizes[0] || p.y() >= sizes[1])
//      return;
//
//    // Increase if necessary.
//    __shared__ T y_causal[4096];
//    __shared__ T y_anticausal[4096];
//
//    // In 2D, we scan the beginning of each row/columns.
//    // We must be at the beginning of each row or column.
//    const auto i = p.dot(grid_strides<2>());
//    //
//    const auto ptr = inout_signal + i;
//
//    // Causal signal: i == 0.
//    T* forward_x[2] = {inout_signal + i, inout_signal + i + step};
//    y_causal[0] = sumg0 * *forward_x[0];
//
//    // Causal signal: i == 1.
//#pragma unroll
//    for (auto k = 0; k < 2; ++k)
//      forward_x[k] += step;
//    y_causal[1] = g0 * *forward_x[0] + sumg1 * *forward_x[1];
//
//    // Causal signal: i = 2 .. size-1
//#pragma unroll
//    for (auto i = 2; i < size; ++i)
//    {
//      for (auto k = 0; k < 2; ++k)
//        forward_x[k] += step;
//      y_causal[i] = a1 * *forward_x[0] + a2 * *forward_x[1] +
//                    b1 * y_causal[i - 1] + b2 * y_causal[i - 2];
//    }
//
//    // Anti-causal signal: i == size-1
//    T* backward_x[2] = {ptr + (size - 1) * step, ptr + size * step};
//    y_anticausal[size - 1] = parity * sumg1 * *backward_x[0];
//
//    // Anti-causal signal: i == size-2
//#pragma unroll
//    for (auto k = 0; k < 2; ++k)
//      forward_x[k] += step;
//    y_anticausal[size - 2] = y_anticausal[size - 1];
//
//    // Anti-causal signal: i == size-3 .. 0
//#pragma unroll
//    for (auto i = size - 3; i >= 0; --i)
//    {
//      for (auto k = 0; k < 2; ++k)
//        backward_x[k] -= step;
//      y_anticausal[i] = a3 * *backward_x[0] + a4 * *backward_x[1] +
//                        b1 * y_anticausal[i + 1] + b2 * y_anticausal[i + 2];
//    }
//
//    *ptr = y_causal[i] + y_anticausal[i];
//    // Store the sum of the two signals.
//#pragma unroll
//    for (auto i = 0; i < size; ++i)
//    {
//      *ptr = y_causal[i] + y_anticausal[i];
//      ptr += step;
//    }
//  }

  // void deriche()
  // {
  //   // Sanity check.
  //   if (sigma <= 0)
  //     throw std::runtime_error("sigma must be positive");
  //   if (derivative_order < 0 || derivative_order >= 3)
  //     throw std::runtime_error("derivative order must be between 0 and 2");
  //   if (axis < 0 || axis >= 2)
  //     throw std::runtime_error("axis of derivative must be between 0 and 1");

  //   // Compute the coefficients of the recursive filter.
  //   //
  //   // The constant 1.695 is mysterious... Also found in CImg library.
  //   // TODO: ask where this constant comes from.
  //   const auto alpha = static_cast<T>(1.695) / sigma;
  //   const auto ea = std::exp(alpha);
  //   const auto ema = std::exp(-alpha);
  //   const auto em2a = ema * ema;
  //   const auto b1 = 2 * ema;
  //   const auto b2 = -em2a;

  //   T ek, ekn;
  //   T parity;
  //   T a1, a2, a3, a4;
  //   T g0, sumg1, sumg0;

  //   switch (derivative_order)
  //   {
  //   // first-order derivative
  //   case 1:
  //     ek = -(1 - ema) * (1 - ema) * (1 - ema) / (2 * (ema + 1) * ema);
  //     a1 = a4 = 0;
  //     a2 = ek * ema;
  //     a3 = -ek * ema;
  //     parity = -1;
  //     if (neumann)
  //     {
  //       sumg1 = (ek * ea) / ((ea - 1) * (ea - 1));
  //       g0 = 0;
  //       sumg0 = g0 + sumg1;
  //     }
  //     else
  //       g0 = sumg0 = sumg1 = 0;
  //     break;

  //   // second-order derivative
  //   case 2:
  //     ekn = (-2 * (-1 + 3 * ea - 3 * ea * ea + ea * ea * ea) /
  //            (3 * ea + 1 + 3 * ea * ea + ea * ea * ea));
  //     ek = -(em2a - 1) / (2 * alpha * ema);
  //     a1 = ekn;
  //     a2 = -ekn * (1 + ek * alpha) * ema;
  //     a3 = ekn * (1 - ek * alpha) * ema;
  //     a4 = -ekn * em2a;
  //     parity = 1;
  //     if (neumann)
  //     {
  //       sumg1 = ekn / 2;
  //       g0 = ekn;
  //       sumg0 = g0 + sumg1;
  //     }
  //     else
  //       g0 = sumg0 = sumg1 = 0;
  //     break;

  //   // smoothing
  //   default:
  //     ek = (1 - ema) * (1 - ema) / (1 + 2 * alpha * ema - em2a);
  //     a1 = ek;
  //     a2 = ek * ema * (alpha - 1);
  //     a3 = ek * ema * (alpha + 1);
  //     a4 = -ek * em2a;
  //     parity = 1;
  //     if (neumann)
  //     {
  //       sumg1 = ek * (alpha * ea + ea - 1) / ((ea - 1) * (ea - 1));
  //       g0 = ek;
  //       sumg0 = g0 + sumg1;
  //     }
  //     else
  //       g0 = sumg0 = sumg1 = 0;
  //     break;
  //   }

  //   const auto start = Vector2i::Zero();
  //   auto end = sizes;
  //   end[axis] = 1;

  //   // Initialize two temporary arrays.
  //   const auto size = inout_signal.size(axis);
  //   const auto step = inout_signal.stride(axis);

  //   inplace_deriche<<<grid_sizes, block_sizes>>>(
  //       inout_signal.data(), inout_signal.sizes(), inout_signal.padded_width(),
  //       sigma, ek, a1, a2, a3, a4, b1, b2, g0, sumg0, sumg1);
  // }

  //! @}

}}  // namespace DO::Shakti
