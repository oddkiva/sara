// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/ImageProcessing/Kernels/Globals.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>

//! TODO: write tests as this is untested.

namespace DO { namespace Shakti {

  __constant__ Vector2f scale;

  __global__ void scale(Vector2f* dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();
    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    // TODO: check that the default texture value is done by nearest
    // interpolation and not linear.
    dst[i] = tex2D(in_float_texture, p.x() * scale.x(), p.y() * scale.y())
  }

}}  // namespace DO::Shakti
