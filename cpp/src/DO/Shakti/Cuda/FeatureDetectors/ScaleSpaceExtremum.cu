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
#include <DO/Shakti/Cuda/FeatureDetectors/ScaleSpaceExtremum.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iterator>


namespace DO::Shakti::Cuda {

  // TODO: speed this up as it is very slow.
  static constexpr auto tile_x = 32;
  static constexpr auto tile_y = 16;
  static constexpr auto tile_z = 2;

  __global__ auto local_scale_space_extremum(cudaSurfaceObject_t dog_octave,
                                             std::int8_t* ext_map,  //
                                             int dog_w, int dog_h, int dog_d,
                                             float min_extremum_abs_value,
                                             float edge_ratio_thres) -> void
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dog_w || y >= dog_h || z >= dog_d)
      return;

    const auto gi = (z * dog_h + y) * dog_w + x;

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
        [&edge_ratio_thres](const volatile decltype(s_curr) I,  //
                            auto x, auto y, auto z) -> bool {
      const auto h00 = I[z][y][x + 1] - 2 * I[z][y][x] + I[z][y][x - 1];
      const auto h11 = I[z][y + 1][x] - 2 * I[z][y][x] + I[z][y - 1][x];
      const auto h01 = (I[z][y + 1][x + 1] - I[z][y + 1][x - 1] -
                        I[z][y - 1][x + 1] + I[z][y - 1][x - 1]) *
                       0.25f;
      const auto& h10 = h01;

      const auto trace_H = h00 + h11;
      const auto det_H = h00 * h11 - h01 * h10;

      const auto trace_squared = trace_H * trace_H;
      const auto abs_det_H = abs(det_H);

      auto edge_ratio_1 = edge_ratio_thres + 1;
      edge_ratio_1 *= edge_ratio_1;

      const auto quantity =
          trace_squared * edge_ratio_thres - edge_ratio_1 * abs_det_H;

      return quantity >= 0;
    };

    if (on_edge(s_curr, tx + 1, ty + 1, tz))
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


  struct IsExtremum
  {
    __host__ __device__ inline auto operator()(std::int8_t val) -> bool
    {
      return val != 0;
    }
  };

  auto compute_scale_space_extremum_map(
      const Octave<float>& dogs,
      MultiArrayView<std::int8_t, 1, RowMajorStrides>& extremum_flat_map,
      float min_extremum_abs_value, float edge_ratio_thres) -> void
  {
    const auto& w = dogs.width();
    const auto& h = dogs.height();
    const auto& d = dogs.scale_count();
    if (extremum_flat_map.size() != w * h * d)
      throw std::runtime_error{"Invalid octave sizes!"};

    if (!dogs.surface_object().initialized())
      throw std::runtime_error{"DoG surface object is uninitialized!"};

    static constexpr auto threadsperBlock = dim3(tile_x, tile_y, tile_z);
    static const auto numBlocks =
        dim3((dogs.width() + threadsperBlock.x - 1) / threadsperBlock.x,
             (dogs.height() + threadsperBlock.y - 1) / threadsperBlock.y,
             (dogs.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
    local_scale_space_extremum<<<numBlocks, threadsperBlock>>>(
        dogs.surface_object(), extremum_flat_map.data(),  //
        dogs.width(), dogs.height(), dogs.scale_count(),  //
        min_extremum_abs_value, edge_ratio_thres);
  }

  auto count_extrema(
      const MultiArrayView<std::int8_t, 1, RowMajorStrides>& extremum_map)
      -> int
  {
    const auto dev_ptr = thrust::device_pointer_cast(extremum_map.data());
    const int num_extrema = thrust::count_if(
        thrust::device, dev_ptr, dev_ptr + extremum_map.size(), IsExtremum());
    return num_extrema;
  }

  auto
  compress_extremum_map(const MultiArrayView<std::int8_t, 1, RowMajorStrides>&
                            d_extremum_flat_map) -> QuantizedExtrema
  {
    // Count the number of extrema.
    const auto num_extrema = count_extrema(d_extremum_flat_map);

    const auto dev_ptr =
        thrust::device_pointer_cast(d_extremum_flat_map.data());

    // Recopy nonzero elements only.
    auto d_indices = thrust::device_vector<int>(num_extrema);
    thrust::copy_if(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(
                        static_cast<int>(d_extremum_flat_map.size())),
                    dev_ptr, d_indices.begin(), thrust::identity());

    // Recopy extremum types.
    auto d_extremum_sparse_map =
        thrust::device_vector<std::int8_t>(num_extrema);
    thrust::copy_if(dev_ptr, dev_ptr + d_extremum_flat_map.size(),
                    d_extremum_sparse_map.begin(), IsExtremum{});

    return {d_indices, d_extremum_sparse_map};
  }


  __global__ auto flat_index_to_3d_coords(const int* index, float* x, float* y,
                                          float* s, int n, int w, int h, int d)
      -> void
  {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
      return;

    const auto wh = w * h;
    const auto ii = index[i];
    const auto si = ii / wh;
    const auto yi = (ii - si * wh) / w;
    const auto xi = ii - si * wh - yi * w;

    x[i] = static_cast<float>(xi);
    y[i] = static_cast<float>(yi);
    s[i] = static_cast<float>(si);
  }

  auto initialize_oriented_extrema(QuantizedExtrema& qe, OrientedExtrema& oe,
                                   int w, int h, int d) -> void
  {
    oe.x.resize(qe.indices.size());
    oe.y.resize(qe.indices.size());
    oe.s.resize(qe.indices.size());

    const auto index_ptr = thrust::raw_pointer_cast(qe.indices.data());
    auto x_ptr = thrust::raw_pointer_cast(oe.x.data());
    auto y_ptr = thrust::raw_pointer_cast(oe.y.data());
    auto s_ptr = thrust::raw_pointer_cast(oe.s.data());

    static const auto block_size = dim3(32);
    static const auto grid_size = dim3(1024 / block_size.x);
    flat_index_to_3d_coords<<<grid_size, block_size>>>(
        index_ptr, x_ptr, y_ptr, s_ptr, qe.indices.size(), w, h, d);
  };

}  // namespace DO::Shakti::Cuda
