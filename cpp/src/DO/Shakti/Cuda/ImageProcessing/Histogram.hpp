// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Defines.hpp>

#include <DO/Shakti/Cuda/MultiArray/MultiArray.hpp>
#include <DO/Shakti/Cuda/MultiArray/TextureArray.hpp>


namespace DO { namespace Shakti {

  DO_SHAKTI_EXPORT
  MultiArray<float, 3>
  compute_color_histogram(const MultiArray<Vector4f, 2>& in);

  DO_SHAKTI_EXPORT
  MultiArray<float, 3>
  normalize_color_histogram(const MultiArray<Vector4f, 2>& in);

}}  // namespace DO::Shakti


namespace DO { namespace Shakti {

  DO_SHAKTI_EXPORT
  void compute_color_distribution(float *out_histogram,
                                  const Vector4ub *in_image,
                                  const int *in_image_sizes,
                                  const int *quantization_steps);

  inline
  void compute_color_distribution(float *out_histogram,
                                  const Vector4ub *in_image,
                                  const int *in_image_sizes,
                                  int quantization_step = 8)
  {
    const int quantization_steps[] = {
      quantization_step,
      quantization_step,
      quantization_step
    };
    compute_color_distribution(out_histogram, in_image, in_image_sizes,
                               quantization_steps);
  }

}}  // namespace DO::Shakti
