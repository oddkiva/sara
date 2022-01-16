// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// Convolving in batch the input image does not seem very fast.
// Rather convolving sequentially seems much faster if we base ourselves from
// the computation time spent in the Halide implementation.


#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Octave"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;


__global__ void fill(cudaSurfaceObject_t output, int width, int height,
                     int scale_count)
{
  // Calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < scale_count)
  {
    const float out = z * width * height + y * width + x;
    surf2DLayeredwrite(out, output, x * sizeof(float), y, z);
  }
}

BOOST_AUTO_TEST_CASE(test_fill)
{
  static constexpr auto w = 3;
  static constexpr auto h = 5;
  static constexpr auto scale_count = 1;

  // Initialize the octave CUDA surface.
  auto octave = sc::make_gaussian_octave<float>(w, h, scale_count);
  octave.init_surface();

  // Initialize the octave in CUDA.
  const dim3 threadsperBlock(16, 16, 2);
  const dim3 numBlocks(
      (octave.width() + threadsperBlock.x - 1) / threadsperBlock.x,
      (octave.height() + threadsperBlock.y - 1) / threadsperBlock.y,
      (octave.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
  fill<<<numBlocks, threadsperBlock>>>(octave.surface_object().value(),  //
                                       octave.width(), octave.height(),
                                       octave.scale_count());

  // Check the result.
  auto result = sara::Image<float, 3>{w, h, octave.scale_count()};
  result.flat_array().fill(0);
  octave.array().copy_to(result);

  auto gt = sara::Image<float, 3>{w, h, octave.scale_count()};
  for (auto i = 0u; i < gt.size(); ++i)
    gt.data()[i] = i;

  // for (auto s = 0; s < result.depth(); ++s)
  //   SARA_DEBUG << s << "\n"
  //              << sara::tensor_view(result)[s].matrix() << std::endl;
  BOOST_CHECK(result == gt);
}
