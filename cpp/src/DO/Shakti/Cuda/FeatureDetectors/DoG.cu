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

  // TODO: speed this up as it is very slow.
  static constexpr auto tile_x = 32;
  static constexpr auto tile_y = 16;
  static constexpr auto tile_z = 2;
  __global__ auto local_scale_space_extremum(cudaSurfaceObject_t dog_octave,
                                             std::int8_t* ext_map,  //
                                             int dog_w, int dog_h, int dog_d,
                                             int ext_pitch,
                                             float min_extremum_abs_value,
                                             float edge_ratio_thres) -> void
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dog_w || y >= dog_h || z >= dog_d)
      return;

    const auto gi = (z * dog_h + y) * ext_pitch + x;

    if (x == 0 || y == 0 || z == 0 ||  //
        x == dog_w - 1 || y == dog_h - 1 || z == dog_d - 1)
    {
      ext_map[gi] = 0;
      return;
    }

    // Avoid the local extremum loops.
    float val;
    surf2DLayeredread(&val, dog_octave, x * sizeof(float), y, z,
                      cudaBoundaryModeClamp);
    if (abs(val) < 0.8f * min_extremum_abs_value)  // 0.8f prefiltering ratio.
    {
      ext_map[gi] = 0;
      return;
    }

    // Use the shared memory to fully leverage the GPU speed.
    __shared__ float s_prev[tile_z][tile_y + 2][tile_x + 2];
    __shared__ float s_curr[tile_z][tile_y + 2][tile_x + 2];
    __shared__ float s_next[tile_z][tile_y + 2][tile_x + 2];

    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto& tz = threadIdx.z;

    // Populate the previous scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[tz][ty + 0][tx + 0] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[tz][ty + 0][tx + 2] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[tz][ty + 2][tx + 0] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[tz][ty + 2][tx + 2] = val;

    // Populate the current scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z,
                      cudaBoundaryModeClamp);
    s_curr[tz][ty + 0][tx + 0] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z,
                      cudaBoundaryModeClamp);
    s_curr[tz][ty + 0][tx + 2] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z,
                      cudaBoundaryModeClamp);
    s_curr[tz][ty + 2][tx + 0] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z,
                      cudaBoundaryModeClamp);
    s_curr[tz][ty + 2][tx + 2] = val;

    // Populate the next scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[tz][ty + 0][tx + 0] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[tz][ty + 0][tx + 2] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[tz][ty + 2][tx + 0] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[tz][ty + 2][tx + 2] = val;
    __syncthreads();

    // Make this check first.
    const auto on_edge =
        [&edge_ratio_thres](
            const volatile decltype(s_curr) s_data,  //
            auto tx, auto ty, auto tz) -> bool {
      //   const auto H = hessian(I, Point2i{x, y});
      //   return square(H.trace()) * edge_ratio >=
      //          square(edge_ratio + 1.f) * std::abs(H.determinant());
      return false;
    };

    if (on_edge(s_curr, tx, ty, tz))
    {
      ext_map[gi] = 0;
      return;
    }

    // Now the most expensive check.
    auto val_ext = val;
    if (val > 0)
    {
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = max(val_ext, s_prev[tz][ty + dy][tx + dx]);
        }
      }
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = max(val_ext, s_curr[tz][ty + dy][tx + dx]);
        }
      }
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = max(val_ext, s_next[tz][ty + dy][tx + dx]);
        }
      }
    }
    else
    {
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = min(val_ext, s_prev[tz][ty + dy][tx + dx]);
        }
      }
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = min(val_ext, s_curr[tz][ty + dy][tx + dx]);
        }
      }
#pragma unroll
      for (auto dy = 0; dy <= 2; ++dy)
      {
#pragma unroll
        for (auto dx = 0; dx <= 2; ++dx)
        {
          val_ext = min(val_ext, s_next[tz][ty + dy][tx + dx]);
        }
      }
    }

    auto extremum_type = std::int8_t{};
    if (val == val_ext && val > 0)
      extremum_type = 1;
    else if (val == val_ext && val < 0)
      extremum_type = -1;

    ext_map[gi] = extremum_type;
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

  auto compute_scale_space_extremum_map(
      const Octave<float>& dogs,
      MultiArrayView<std::int8_t, 3, RowMajorStrides>& extremum_map,
      float min_extremum_abs_value, float edge_ratio_thres) -> void
  {
    if (extremum_map.width() != dogs.width() ||
        extremum_map.height() != dogs.height() ||
        extremum_map.depth() != dogs.scale_count())
      throw std::runtime_error{"Invalid octave sizes!"};

    if (!dogs.surface_object().initialized())
      throw std::runtime_error{"DoG surface object is uninitialized!"};

    static constexpr auto threadsperBlock = dim3(tile_x, tile_y, tile_z);
    static const auto numBlocks =
        dim3((dogs.width() + threadsperBlock.x - 1) / threadsperBlock.x,
             (dogs.height() + threadsperBlock.y - 1) / threadsperBlock.y,
             (dogs.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
    local_scale_space_extremum<<<numBlocks, threadsperBlock>>>(
        dogs.surface_object(), extremum_map.data(),       //
        dogs.width(), dogs.height(), dogs.scale_count(),  //
        extremum_map.padded_width(), min_extremum_abs_value, edge_ratio_thres);
  }

}  // namespace DO::Shakti::Cuda
