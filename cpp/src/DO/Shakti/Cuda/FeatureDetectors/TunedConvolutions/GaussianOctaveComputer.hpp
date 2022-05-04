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

#include <DO/Shakti/Cuda/Utilities/Timer.hpp>

#include <DO/Shakti/Cuda/MultiArray/MultiArrayView.hpp>
#include <DO/Shakti/Cuda/MultiArray/PinnedMemoryAllocator.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/SmallGaussianConvolutionFP32.hpp>


namespace DO::Shakti::Cuda {

  struct GaussianOctaveComputer
  {
    GaussianOctaveComputer(int w, int h, int scale_count = 3);

    auto operator()(const MultiArrayView<float, 2, RowMajorStrides>& d_in,
                    Octave<float>& d_octave) -> void;

    // Gaussian kernels.
    GaussianOctaveKernels<float> host_kernels;
    Gaussian::DeviceGaussianFilterBank device_kernels;

    // Device work buffer for intermediate x-convolution results.
    MultiArray<float, 2> d_convx;

    // Profile.
    Timer d_timer;
  };

}  // namespace DO::Shakti::Cuda
