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

#pragma once

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


namespace DO::Shakti::Cuda {

  auto compute_scale_space_extremum_map(
      const Octave<float>& dogs,
      MultiArrayView<std::int8_t, 1, RowMajorStrides>& extremum_flat_map,
      float min_extremum_abs_value = 0.03f,  //
      float edge_ratio_thres = 10.f) -> void;

  struct HostExtrema
  {
    thrust::host_vector<int> indices;
    thrust::host_vector<float> x;
    thrust::host_vector<float> y;
    thrust::host_vector<float> s;
    thrust::host_vector<std::int8_t> log_s_quantized;
    thrust::host_vector<std::int8_t> types;
    thrust::host_vector<float> values;
  };

  struct DeviceExtrema
  {
    //! @brief Flattened and quantized 1D-indices.
    thrust::device_vector<int> indices;

    //! @brief 3D scale-space coordinates
    //! @{
    thrust::device_vector<float> x;
    thrust::device_vector<float> y;
    thrust::device_vector<float> s;
    thrust::host_vector<std::int8_t> log_s_quantized;
    //! @}

    //! @brief Extremum type.
    thrust::device_vector<std::int8_t> types;

    //! @brief Extremum response value.
    thrust::device_vector<float> values;

    inline auto copy_to_host() const -> HostExtrema
    {
      auto h_extrema = HostExtrema{};

      h_extrema.indices.resize(indices.size());
      h_extrema.x.resize(x.size());
      h_extrema.y.resize(y.size());
      h_extrema.s.resize(s.size());
      h_extrema.types.resize(types.size());
      h_extrema.values.resize(values.size());

      thrust::copy(indices.begin(), indices.end(), h_extrema.indices.begin());
      thrust::copy(x.begin(), x.end(), h_extrema.x.begin());
      thrust::copy(y.begin(), y.end(), h_extrema.y.begin());
      thrust::copy(s.begin(), s.end(), h_extrema.s.begin());
      thrust::copy(types.begin(), types.end(), h_extrema.types.begin());
      thrust::copy(values.begin(), values.end(), h_extrema.values.begin());

      return h_extrema;
    }
  };

  auto count_extrema(const MultiArrayView<std::int8_t, 1, RowMajorStrides>&)
      -> int;

  auto
  compress_extremum_map(const MultiArrayView<std::int8_t, 1, RowMajorStrides>&)
      -> DeviceExtrema;

  auto initialize_extrema(DeviceExtrema&, int w, int h, int d) -> void;

  auto refine_extrema(const Octave<float>& dogs, DeviceExtrema& e,
                      float scale_initial, float scale_exponent) -> void;

}  // namespace DO::Shakti::Cuda
