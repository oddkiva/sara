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

  auto compute_histogram_of_gradients(const Octave<float>& mag_fn,  //
                                      const Octave<float>& ori_fn,  //
                                      const float* x,               //
                                      const float* y,               //
                                      const float* s) -> void;

  auto compute_histogram_of_gradients(const Octave<float>& gaussians,  //
                                      const float* x,                  //
                                      const float* y,                  //
                                      const float* s) -> void;

}  // namespace DO::Shakti::Cuda
