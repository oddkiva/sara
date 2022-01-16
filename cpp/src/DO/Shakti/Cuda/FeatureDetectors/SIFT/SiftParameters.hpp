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

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/GaussianOctaveKernels.hpp>


namespace DO::Shakti::Cuda {

  struct SiftParameters
  {
    inline SiftParameters(int scale_count = 3, float scale_camera = 1.f,
                          float scale_initial = 1.6f,
                          float gaussian_truncation_factor = 4.f)
      : kernels{scale_count, scale_camera, scale_initial,
                gaussian_truncation_factor}
    {
    }

    // Gaussian kernel filters for a given octave.
    GaussianOctaveKernels<float> kernels;

    //! @brief Extremum detection thresholds.
    float edge_ratio = 10.0f;
    float extremum_thres = 0.01f;

    //! @brief Dominant gradient orientations.
    int num_orientation_bins = 36;
    float ori_gaussian_truncation_factor = 3.f;
    float scale_multiplying_factor = 1.5f;
    float peak_ratio_thres = 0.8f;
  };

}  // namespace DO::Shakti::Cuda
