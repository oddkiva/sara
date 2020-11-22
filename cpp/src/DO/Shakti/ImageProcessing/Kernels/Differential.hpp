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

#include <DO/Shakti/ImageProcessing/Kernels/Globals.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>


namespace DO { namespace Shakti {

  __global__
  void apply_gradient_kernel(Vector2f *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    Vector2f nabla_f{tex2D(in_float_texture, p.x() + 1, p.y()) -
                         tex2D(in_float_texture, p.x() - 1, p.y()),
                     tex2D(in_float_texture, p.x(), p.y() + 1) -
                         tex2D(in_float_texture, p.x(), p.y() - 1)};
    nabla_f *= 0.5f;
    dst[i] = nabla_f;
  }

  __global__
  void apply_gradient_polar_coordinates_kernel(Vector2f *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    const auto f_x = tex2D(in_float_texture, p.x() + 1, p.y()) -
                     tex2D(in_float_texture, p.x() - 1, p.y());
    const auto f_y = tex2D(in_float_texture, p.x(), p.y() + 1) -
                     tex2D(in_float_texture, p.x(), p.y() - 1);

    dst[i] = {
      sqrt(f_x*f_x + f_y*f_y),
      atan2(f_y, f_x)
    };
  }

  __global__
  void apply_squared_norms_kernel(float *out, const Vector<float, 2> *in)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    const auto f_i = in[i];

    out[i] = f_i.squared_norm();
  }

  __global__
  void apply_gradient_squared_norms_kernel(float *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    auto u_x = 0.5f * (tex2D(in_float_texture, p.x() + 1, p.y()) -
                       tex2D(in_float_texture, p.x() - 1, p.y()));
    auto u_y = 0.5f * (tex2D(in_float_texture, p.x(), p.y() + 1) -
                       tex2D(in_float_texture, p.x(), p.y() - 1));
    dst[i] = u_x*u_x + u_y*u_y;
  }

  __global__
  void apply_laplacian_kernel(float *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes.x || p.y() >= image_sizes.y)
      return;

    const auto u_x = tex2D(in_float_texture, p.x(), p.y());
    const auto u_e = tex2D(in_float_texture, p.x() + 1, p.y());
    const auto u_w = tex2D(in_float_texture, p.x() - 1, p.y());
    const auto u_n = tex2D(in_float_texture, p.x(), p.y() - 1);
    const auto u_s = tex2D(in_float_texture, p.x(), p.y() + 1);

    dst[i] = u_e + u_w + u_n + u_s - 4 * u_x;
  }

} /* namespace Shakti */
} /* namespace DO */
