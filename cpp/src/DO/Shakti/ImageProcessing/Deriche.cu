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


}}  // namespace DO::Shakti
