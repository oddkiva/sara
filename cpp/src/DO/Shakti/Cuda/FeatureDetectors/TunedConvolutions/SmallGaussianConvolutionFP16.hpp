// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

namespace DO::Shakti::Cuda::Gaussian {

  static constexpr auto kernel_max_radius = 20;
  static constexpr auto tile_size = 1024 / kernel_max_radius;
  static constexpr auto sdata_rows = tile_size + 2 * kernel_max_radius;

  static constexpr auto kernel_count_max = 16;
  static constexpr auto kernel_capacity = kernel_max_radius * kernel_count_max;

  __constant__ half kernels[kernel_capacity];
  __constant__ int kernel_radii[kernel_count_max];
  __constant__ int kernel_count;
  __constant__ int kernel_radius_max;

  template <typename T>
  __global__ auto rgba8u_to_gray16f(cudaSurfaceObject_t rgba8u,
                                    cudaSurfaceObject_t gray16f, int w, int h)
      -> void
  {

    // Calculate normalized texture coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w / 2 || y >= h / 2)
      return;

    // uchar4 rgba;
    // surf2Dread(                   //
    //     &rgba,                    //
    //     rgba8u,                   //
    //     x * int(sizeof(uchar4)),  //
    //     y,                        //
    //     cudaBoundaryModeClamp);

    // static constexpr half norminv = half(1. / 255);
    // const half r = half(rgba.x) * norminv;
    // const half g = half(rgba.y) * norminv;
    // const half b = half(rgba.z) * norminv;

    // static constexpr float one_third = half(1.f / 3);
    // half2 gray;
    // gray.x = (r + g + b) * one_third;
    // gray.y = (r + g + b) * one_third;
    // surf2Dwrite(                //
    //     gray,                   //
    //     gray16f,                //
    //     x * int(sizeof(half2)),  //
    //     y);
  }

}  // namespace DO::Shakti::Cuda::Gaussian

