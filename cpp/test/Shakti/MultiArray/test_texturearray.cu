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

#include <gtest/gtest.h>

#include <DO/Shakti/ImageProcessing/Kernels/Globals.hpp>

#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>


using namespace std;
using namespace DO::Shakti;


void check(const float *res, const Vector2i& sizes)
{
  for (int y = 0; y < sizes[1]; ++y)
  {
    for (int x = 0; x < sizes[0]; ++x)
      cout << res[sizes[0] * y + x] << " ";
    cout << endl;
  }
}

__global__
void copy_from_texture_to_global_memory(float *out)
{
  const auto i = offset<2>();
  const auto p = coords<2>();
  out[i] = tex2D(in_float_texture, p.x(), p.y());
}

TEST(TestCudaArray, test_constructor)
{
  const int w = 3;
  const int h = 4;
  float in[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
  };

  TextureArray<float> in_array{ in, { w, h } };

  float out[w * h];
  in_array.copy_to_host(out) ;
  EXPECT_TRUE(equal(in, in + w*h, out));
  //check(out, { w, h });
}

TEST(TestCudaArray, test_operation_from_cuda_array_to_multiarray)
{
  Vector2i sizes{ 3, 4 };
  float in[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
  };

  TextureArray<float> in_array{ in, { sizes[0], sizes[1] } };
  MultiArray<float, 2> out_array{ { sizes[0], sizes[1] } };
  {
    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in_array));
    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out_array);
    copy_from_texture_to_global_memory<<<grid_size, block_size>>>(out_array.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));
  }

  float out[3 * 4];
  out_array.copy_to_host(out);
  EXPECT_TRUE(equal(in, in + sizes[0] * sizes[1], out));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}