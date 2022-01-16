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

    const float diff = val2 - val1;
    // printf("diff = %f\n", diff);
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
      throw std::runtime_error{"Gaussian surface object is unitialized!"};

    if (!dogs.surface_object().initialized())
      dogs.init_surface();

    const auto threadsperBlock = dim3(32, 16, 2);
    const auto numBlocks =
        dim3((dogs.width() + threadsperBlock.x - 1) / threadsperBlock.x,
             (dogs.height() + threadsperBlock.y - 1) / threadsperBlock.y,
             (dogs.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
    SARA_CHECK(threadsperBlock.x);
    SARA_CHECK(threadsperBlock.y);
    SARA_CHECK(threadsperBlock.z);
    SARA_CHECK(numBlocks.x);
    SARA_CHECK(numBlocks.y);
    SARA_CHECK(numBlocks.z);
    cudaSurfaceObject_t garr = gaussians.surface_object();
    cudaSurfaceObject_t darr = dogs.surface_object();
    SARA_CHECK(garr);
    SARA_CHECK(darr);
    dog<<<numBlocks, threadsperBlock>>>(gaussians.surface_object(),
                                        dogs.surface_object(), dogs.width(),
                                        dogs.height(), dogs.scale_count());
  }

}  // namespace DO::Shakti::Cuda
