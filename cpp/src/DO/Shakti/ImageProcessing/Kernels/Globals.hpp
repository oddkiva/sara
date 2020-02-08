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

#pragma once


namespace DO { namespace Shakti {

  texture<float, 2> in_float_texture;
  texture<float2, 2> in_float2_texture;

  __constant__ float kernel[1024];
  __constant__ int kernel_size;
  __constant__ int2 image_sizes;

}}  // namespace DO::Shakti
