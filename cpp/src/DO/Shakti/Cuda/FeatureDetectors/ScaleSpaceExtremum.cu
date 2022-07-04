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
  static constexpr auto tile_y = 32;
  static constexpr auto tile_z = 1;

  template <typename Index>
  __device__ inline auto at(Index tx, Index ty, Index tz)
  {
    static constexpr auto offset = Index{1};
    static constexpr auto tile_height = tile_y + Index{2};
    static constexpr auto tile_width = tile_x + Index{2};
    return (tz * tile_height + ty + offset) * tile_width + tx + offset;
  };

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

    // Use the shared memory to fully leverage the GPU speed.
    __shared__ float s_prev[tile_z * (tile_y + 2) * (tile_x + 2)];
    __shared__ float s_curr[tile_z * (tile_y + 2) * (tile_x + 2)];
    __shared__ float s_next[tile_z * (tile_y + 2) * (tile_x + 2)];

    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto& tz = threadIdx.z;
    static_assert(std::is_same_v<decltype(tx), const unsigned int&>);

    float val;

    // Populate the previous scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[at(tx - 1, ty - 1, tz)] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[at(tx + 1, ty - 1, tz)] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[at(tx - 1, ty + 1, tz)] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z - 1,
                      cudaBoundaryModeClamp);
    s_prev[at(tx + 1, ty + 1, tz)] = val;

    // Populate the current scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z,
                      cudaBoundaryModeClamp);
    s_curr[at(tx - 1, ty - 1, tz)] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z,
                      cudaBoundaryModeClamp);
    s_curr[at(tx + 1, ty - 1, tz)] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z,
                      cudaBoundaryModeClamp);
    s_curr[at(tx - 1, ty + 1, tz)] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z,
                      cudaBoundaryModeClamp);
    s_curr[at(tx + 1, ty + 1, tz)] = val;

    // Populate the next scale.
    //
    // Top-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y - 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[at(tx - 1, ty - 1, tz)] = val;
    // Top-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y - 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[at(tx + 1, ty - 1, tz)] = val;
    // Bottom-left
    surf2DLayeredread(&val, dog_octave, (x - 1) * sizeof(float), y + 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[at(tx - 1, ty + 1, tz)] = val;
    // Bottom-right
    surf2DLayeredread(&val, dog_octave, (x + 1) * sizeof(float), y + 1, z + 1,
                      cudaBoundaryModeClamp);
    s_next[at(tx + 1, ty + 1, tz)] = val;

    // ========================================================================
    __syncthreads();
    // ========================================================================

    val = s_curr[at(tx, ty, tz)];

    const auto gi = (z * dog_h + y) * dog_w + x;

    if (x == 0 || y == 0 || z == 0 ||  //
        x == dog_w - 1 || y == dog_h - 1 || z == dog_d - 1)
    {
      ext_map[gi] = 0;
      return;
    }

    if (fabsf(val) < min_extremum_abs_value)
    {
      ext_map[gi] = 0;
      return;
    }


    // Make this check first.
    const auto on_edge = [&edge_ratio_thres](          //
                             const volatile float* I,  //
                             auto x, auto y, auto z) -> bool {
#ifdef DEBUG_EXTREMUM_MAP_LOCALIZATION
      printf("x=%d y=%d z=%d -> val=%f\n", x, y, z, I[at(x, y, z)]);
#endif

      const auto h00 =
          I[at(x + 1, y, z)] - 2 * I[at(x, y, z)] + I[at(x - 1, y, z)];
      const auto h11 =
          I[at(x, y + 1, z)] - 2 * I[at(x, y, z)] + I[at(x, y - 1, z)];
      const auto h01 = (I[at(x + 1, y + 1, z)] - I[at(x - 1, y + 1, z)] -
                        I[at(x + 1, y - 1, z)] + I[at(x - 1, y - 1, z)]) *
                       0.25f;
      const auto& h10 = h01;

      const auto trace_H = h00 + h11;
      const auto det_H = h00 * h11 - h01 * h10;

      const auto trace_squared = trace_H * trace_H;
      const auto abs_det_H = fabsf(det_H);

      auto edge_ratio_1 = edge_ratio_thres + 1;
      edge_ratio_1 *= edge_ratio_1;

      const auto quantity =
          trace_squared * edge_ratio_thres - edge_ratio_1 * abs_det_H;

      return quantity >= 0;
    };

    if (on_edge(s_curr, tx, ty, tz))
    {
      ext_map[gi] = 0;
      return;
    }


#ifdef DEBUG_EXTREMUM_MAP_LOCALIZATION
    printf("me = %f\n", val);
    printf("s_prev\n");
    for (auto dy = -1; dy <= 1; ++dy)
    {
      for (auto dx = -1; dx <= 1; ++dx)
        printf("%f ", s_prev[at(tx + dx, ty + dy, tz)]);
      printf("\n");
    }
    printf("s_curr\n");
    for (auto dy = -1; dy <= 1; ++dy)
    {
      for (auto dx = -1; dx <= 1; ++dx)
        printf("%f ", s_curr[at(tx + dx, ty + dy, tz)]);
      printf("\n");
    }
    printf("s_next\n");
    for (auto dy = -1; dy <= 1; ++dy)
    {
      for (auto dx = -1; dx <= 1; ++dx)
        printf("%f ", s_next[at(tx + dx, ty + dy, tz)]);
      printf("\n");
    }
#endif

    // Now the most expensive check.
    auto val_ext = s_prev[at(tx - 1, ty - 1, tz)];
    if (val > 0)
    {
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fmaxf(val_ext, s_prev[at(tx + dx, ty + dy, tz)]);
      }
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fmaxf(val_ext, s_curr[at(tx + dx, ty + dy, tz)]);
      }
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fmaxf(val_ext, s_next[at(tx + dx, ty + dy, tz)]);
      }
    }
    else
    {
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fminf(val_ext, s_prev[at(tx + dx, ty + dy, tz)]);
      }
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fminf(val_ext, s_curr[at(tx + dx, ty + dy, tz)]);
      }
#pragma unroll
      for (auto dy = -1; dy <= 1; ++dy)
      {
#pragma unroll
        for (auto dx = -1; dx <= 1; ++dx)
          val_ext = fminf(val_ext, s_next[at(tx + dx, ty + dy, tz)]);
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

    static const auto block_sizes = dim3(tile_x, tile_y, tile_z);
    static const auto grid_sizes =
        dim3((dogs.width() + block_sizes.x - 1) / block_sizes.x,
             (dogs.height() + block_sizes.y - 1) / block_sizes.y,
             (dogs.scale_count() + block_sizes.z - 1) / block_sizes.z);
    local_scale_space_extremum<<<grid_sizes, block_sizes>>>(
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
                            d_extremum_flat_map) -> DeviceExtrema
  {
    auto e = DeviceExtrema{};

    // Count the number of extrema.
    const auto num_extrema = count_extrema(d_extremum_flat_map);

    const auto dev_ptr =
        thrust::device_pointer_cast(d_extremum_flat_map.data());

    // Recopy nonzero elements only.
    auto& d_indices = e.indices;
    d_indices = thrust::device_vector<int>(num_extrema);
    thrust::copy_if(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(
                        static_cast<int>(d_extremum_flat_map.size())),
                    dev_ptr, d_indices.begin(), thrust::identity());

    // Recopy extremum types.
    auto& d_types = e.types;
    d_types = thrust::device_vector<std::int8_t>(num_extrema);
    thrust::copy_if(dev_ptr, dev_ptr + d_extremum_flat_map.size(),
                    d_types.begin(), IsExtremum{});

    return e;
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

  auto initialize_extrema(DeviceExtrema& e, int w, int h, int d) -> void
  {
    e.x.resize(e.indices.size());
    e.y.resize(e.indices.size());
    e.s.resize(e.indices.size());

    const auto index_ptr = thrust::raw_pointer_cast(e.indices.data());
    auto x_ptr = thrust::raw_pointer_cast(e.x.data());
    auto y_ptr = thrust::raw_pointer_cast(e.y.data());
    auto s_ptr = thrust::raw_pointer_cast(e.s.data());

    static const auto block_sizes = dim3(1024);
    static const auto grid_sizes =
        dim3((e.indices.size() + block_sizes.x - 1) / block_sizes.x);
    flat_index_to_3d_coords<<<grid_sizes, block_sizes>>>(
        index_ptr, x_ptr, y_ptr, s_ptr, e.indices.size(), w, h, d);
  };

  __device__ auto gradient_3d(cudaSurfaceObject_t surface,  //
                              int x,                        //
                              int y,                        //
                              int z) -> Vector<float, 3>
  {
    auto g = Vector<float, 3>{};
    auto f = [surface](int x, int y, int z) {
      float val;
      surf2DLayeredread(&val, surface, x * sizeof(float), y, z);
      return val;
    };

    g.x() = (f(x + 1, y, z) - f(x - 1, y, z)) * 0.5f;
    g.y() = (f(x, y + 1, z) - f(x, y - 1, z)) * 0.5f;
    g.z() = (f(x, y, z + 1) - f(x, y, z - 1)) * 0.5f;

    return g;
  }

  __device__ inline auto hessian_3d(cudaSurfaceObject_t surface,  //
                                    int x, int y, int z) -> Matrix<float, 3, 3>
  {
    auto f = [surface](int x, int y, int z) {
      float val;
      surf2DLayeredread(&val, surface, x * sizeof(float), y, z);
      return val;
    };

    const auto dxx = f(x + 1, y, z) - 2 * f(x, y, z) + f(x - 1, y, z);
    const auto dyy = f(x, y + 1, z) - 2 * f(x, y, z) + f(x, y - 1, z);
    const auto dss = f(x, y, z + 1) - 2 * f(x, y, z) + f(x, y, z - 1);

    const auto dxy = (f(x + 1, y + 1, z) - f(x - 1, y - 1, z) -  //
                      f(x + 1, y - 1, z) + f(x - 1, y - 1, z)) *
                     0.25f;

    const auto dxs = (f(x + 1, y, z + 1) - f(x - 1, y, z + 1) -  //
                      f(x + 1, y, z - 1) + f(x - 1, y, z - 1)) *
                     0.25f;

    const auto dys = (f(x, y + 1, z + 1) - f(x, y - 1, z + 1) -  //
                      f(x, y + 1, z - 1) + f(x, y - 1, z - 1)) *
                     0.25f;

    auto h = Matrix<float, 3, 3>{};
    // clang-format off
    h(0, 0) = dxx; h(0, 1) = dxy; h(0, 2) = dxs;
    h(1, 0) = dxy; h(1, 1) = dyy; h(1, 2) = dys;
    h(2, 0) = dxs; h(2, 1) = dys; h(2, 2) = dss;
    // clang-format on

    return h;
  }

  __global__ auto refine_extremum(cudaSurfaceObject_t surface,                //
                                  float* x, float* y, float* s,               //
                                  float* val,                                 //
                                  float scale_initial, float scale_exponent,  //
                                  int n) -> void
  {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("i=%d\n", i);
    if (i >= n)
      return;

    const auto xi = int(x[i]);
    const auto yi = int(y[i]);
    const auto si = int(s[i]);

    const auto gradient = gradient_3d(surface, xi, yi, si);
    static_assert(std::is_same_v<decltype(gradient), const Vector3f>);

    const auto hessian = hessian_3d(surface, xi, yi, si);
    static_assert(std::is_same_v<decltype(hessian), const Matrix3f>);

#ifdef DEBUG_REFINE_EXTREMUM
    printf("Coords   %3d %3d %3d\n", xi, yi, si);
    printf("Gradient %0.4f %0.4f %0.4f\n", gradient(0), gradient(1),
           gradient(2));

    printf("Hessian\n");
    for (auto k = 0; k < 3; ++k)
      printf("%0.4f %0.4f %0.4f\n",  //
             hessian(k, 0), hessian(k, 1), hessian(k, 2));
#endif

    const auto hessian_inv = inverse(hessian);
#ifdef DEBUG_REFINE_EXTREMUM
    printf("HessianInverse\n");
    for (auto k = 0; k < 3; ++k)
      printf("%0.4f %0.4f %0.4f\n",  //
             hessian_inv(k, 0), hessian_inv(k, 1), hessian_inv(k, 2));
#endif

    const auto residual = -(hessian_inv * gradient);
    static_assert(std::is_same_v<decltype(residual), const Vector3f>);
#ifdef DEBUG_REFINE_EXTREMUM
    printf("%0.4f %0.4f %0.4f\n", residual(0), residual(1), residual(2));
#endif

    float current_value;
    surf2DLayeredread(&current_value, surface, xi * sizeof(float), yi, si);

    const auto new_value = current_value + 0.5f * gradient.dot(residual);

    // Sometimes the residual explodes, probably because the matrix inversion
    // is not very stable.
    const auto refinement_successful =
        fabsf(residual.x()) < 1.5f &&             //
        fabsf(residual.y()) < 1.5f &&             //
        fabsf(residual.z()) < 1.5f &&             //
        fabsf(current_value) < fabsf(new_value);  //

    const auto log_scale = refinement_successful ? si + residual.z() : si;
    const auto scale = scale_initial * powf(scale_exponent, log_scale);

    if (refinement_successful)
    {
      x[i] += residual.x();
      y[i] += residual.y();
      val[i] = new_value;
    }
    else
    {
      val[i] = current_value;
    }
    s[i] = scale;
  }

  auto refine_extrema(const Octave<float>& dogs, DeviceExtrema& e,
                      float scale_initial, float scale_exponent) -> void
  {
    const auto& w = dogs.width();
    const auto& h = dogs.height();
    const auto& d = dogs.scale_count();
    if (!dogs.surface_object().initialized())
      throw std::runtime_error{"DoG surface object is uninitialized!"};

    static const auto block_sizes = dim3(1024);
    static const auto grid_sizes =
        dim3((e.indices.size() + block_sizes.x - 1) / block_sizes.x);

    e.values.resize(e.indices.size());

    auto x_ptr = thrust::raw_pointer_cast(e.x.data());
    auto y_ptr = thrust::raw_pointer_cast(e.y.data());
    auto s_ptr = thrust::raw_pointer_cast(e.s.data());
    auto val_ptr = thrust::raw_pointer_cast(e.values.data());

    refine_extremum<<<grid_sizes, block_sizes>>>(dogs.surface_object(),  //
                                                 x_ptr, y_ptr, s_ptr,    //
                                                 val_ptr,                //
                                                 scale_initial,          //
                                                 scale_exponent,         //
                                                 int(e.indices.size())   //
    );
  }

}  // namespace DO::Shakti::Cuda
