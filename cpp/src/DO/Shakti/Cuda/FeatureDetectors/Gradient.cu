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

#include <DO/Shakti/Cuda/FeatureDetectors/DoG.hpp>


namespace DO::Shakti::Cuda {

  __global__ auto polar_gradient(cudaSurfaceObject_t gaussians,
                                 cudaSurfaceObject_t mag,  //
                                 cudaSurfaceObject_t ori,  //
                                 int w, int h, int d) -> void
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= w || y >= h || s >= d)
      return;

    float Ix1;
    surf2DLayeredread(&Ix1, gaussians, (x - 1) * sizeof(float), y, s,
                      cudaBoundaryModeClamp);
    float Ix2;
    surf2DLayeredread(&Ix2, gaussians, (x + 1) * sizeof(float), y, s,
                      cudaBoundaryModeClamp);

    float Iy1;
    surf2DLayeredread(&Iy1, gaussians, x * sizeof(float), y - 1, s,
                      cudaBoundaryModeClamp);
    float Iy2;
    surf2DLayeredread(&Iy2, gaussians, x * sizeof(float), y + 1, s,
                      cudaBoundaryModeClamp);

    const auto g = Vector2f{0.5f * (Ix2 - Ix1), 0.5f * (Iy2 - Iy1)};

    const auto mag_xys = sqrtf(g.x() * g.x() + g.y() * g.y());
    const auto ori_xys = atan2f(g.y(), g.x());

    surf2DLayeredwrite(mag_xys, mag, x * sizeof(float), y, s);
    surf2DLayeredwrite(ori_xys, ori, x * sizeof(float), y, s);
  }

  auto compute_polar_gradient_octave(const Octave<float>& gaussians,
                                     Octave<float>& grad_mag,
                                     Octave<float>& grad_ori) -> void
  {
    if (grad_mag.array().sizes() != gaussians.array().sizes())
      throw std::runtime_error{"Invalid gradient magnitude octave sizes!"};

    if (grad_ori.array().sizes() != gaussians.array().sizes())
      throw std::runtime_error{"Invalid gradient orientation octave sizes!"};

    if (!gaussians.surface_object().initialized())
      throw std::runtime_error{"Gaussian octave surface object must be initialized!"};

    if (!grad_mag.surface_object().initialized())
      grad_mag.init_surface();
    if (!grad_ori.surface_object().initialized())
      grad_ori.init_surface();

    static const auto block_sizes = dim3(32, 32, 1);
    static const auto grid_sizes =
        dim3((gaussians.width() + block_sizes.x - 1) / block_sizes.x,
             (gaussians.height() + block_sizes.y - 1) / block_sizes.y,
             (gaussians.scale_count() + block_sizes.z - 1) / block_sizes.z);

    polar_gradient<<<grid_sizes, block_sizes>>>(
        gaussians.surface_object(),  //
        grad_mag.surface_object(),   //
        grad_ori.surface_object(),   //
        gaussians.width(), gaussians.height(), gaussians.scale_count());
  }

}  // namespace DO::Shakti::Cuda
