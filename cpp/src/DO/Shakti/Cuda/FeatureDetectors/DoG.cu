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

  __global__ auto dog(cudaSurfaceObject_t gaussian_octave,
                      cudaSurfaceObject_t dog_octave,  //
                      int w, int h, int d) -> void
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= w || y >= h || z >= d)
      return;

    float val1;
    surf2DLayeredread(&val1, gaussian_octave, x * sizeof(float), y, z,
                      cudaBoundaryModeClamp);

    float val2;
    surf2DLayeredread(&val2, gaussian_octave, x * sizeof(float), y, z + 1,
                      cudaBoundaryModeClamp);

#ifdef USE_SHARED_MEMORY
    __shared__ float sdata[32][32][2];

    // No real benefit in using the shared memory with a 4K video:
    //
    // [DoG] Elapsed time = 1.44467 ms
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    sdata[ty][tx][0] = val1;
    sdata[ty][tx][1] = val2;
    __syncthreads();

    const float diff = sdata[ty][tx][1] - sdata[ty][tx][0];
#else
    // [DoG] Elapsed time = 1.42746 ms
    const float diff = val2 - val1;
#endif
    surf2DLayeredwrite(diff, dog_octave, x * sizeof(float), y, z);
  }

  auto compute_dog_octave(const Octave<float>& gaussians, Octave<float>& dogs)
      -> void
  {
    if (dogs.width() != gaussians.width() ||
        dogs.height() != gaussians.height() ||
        dogs.scale_count() != gaussians.scale_count() - 1)
      throw std::runtime_error{"Invalid octave sizes!"};

    if (!gaussians.surface_object().initialized())
      throw std::runtime_error{"Gaussian surface object is uninitialized!"};

    if (!dogs.surface_object().initialized())
      dogs.init_surface();

    static constexpr auto threadsperBlock = dim3(32, 32, 1);
    static const auto numBlocks =
        dim3((dogs.width() + threadsperBlock.x - 1) / threadsperBlock.x,
             (dogs.height() + threadsperBlock.y - 1) / threadsperBlock.y,
             (dogs.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
    dog<<<numBlocks, threadsperBlock>>>(gaussians.surface_object(),
                                        dogs.surface_object(), dogs.width(),
                                        dogs.height(), dogs.scale_count());
  }

}  // namespace DO::Shakti::Cuda
