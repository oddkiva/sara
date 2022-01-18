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

  struct QuantizedExtrema
  {
    thrust::host_vector<int> indices;
    thrust::host_vector<std::int8_t> types;
  };

  struct DeviceQuantizedExtrema
  {
    thrust::device_vector<int> indices;
    thrust::device_vector<std::int8_t> types;
  };

  struct OrientedExtrema
  {
    thrust::device_vector<float> x;
    thrust::device_vector<float> y;
    thrust::device_vector<float> s;
    thrust::device_vector<float> o;
    thrust::device_vector<std::int8_t> types;
  };

  auto count_extrema(const MultiArrayView<std::int8_t, 1, RowMajorStrides>&)
      -> int;

  auto
  compress_extremum_map(const MultiArrayView<std::int8_t, 1, RowMajorStrides>&)
      -> QuantizedExtrema;

  auto initialize_oriented_extrema(QuantizedExtrema& qe, OrientedExtrema& oe,
                                   int w, int h, int d) -> void;

}  // namespace DO::Shakti::Cuda
