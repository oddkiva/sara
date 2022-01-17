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


namespace DO::Shakti::Cuda {

  auto compute_dog_octave(const Octave<float>& gaussians, Octave<float>& dogs)
      -> void;

  auto compute_scale_space_extremum_map(
      const Octave<float>& dogs,
      MultiArrayView<std::int8_t, 3, RowMajorStrides>& extremum_map,
      float min_extremum_abs_value = 0.03f,  //
      float edge_ratio_thres = 10.f) -> void;

}  // namespace DO::Shakti::Cuda
