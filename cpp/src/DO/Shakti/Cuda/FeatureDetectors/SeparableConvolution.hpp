// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <cuda_runtime.h>

#include <DO/Shakti/Cuda/MultiArray/TextureArray.hpp>


namespace DO::Shakti::Cuda {

  
  // TODO: apply convolution all at once and see vectorization using float4
  // makes thing faster.


}  // namespace DO::Shakti::Cuda
