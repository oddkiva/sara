// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP
#define DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP


namespace DO { namespace Shakti {

  texture<float, 2> in_float_texture;
  texture<float2, 2> in_float2_texture;

  __constant__ float kernel[1024];
  __constant__ int kernel_size;

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP */