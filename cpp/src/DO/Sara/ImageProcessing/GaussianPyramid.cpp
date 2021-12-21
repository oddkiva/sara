// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>

#ifdef DO_SARA_USE_HALIDE
#  include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#  include "shakti_subtract_32f_cpu.h"
#endif


namespace DO::Sara {

  auto difference_of_gaussians_pyramid(const ImagePyramid<float>& gaussians)
      -> ImagePyramid<float>
  {
    auto D = ImagePyramid<float>{};
    D.reset(gaussians.num_octaves(), gaussians.num_scales_per_octave() - 1,
            gaussians.scale_initial(), gaussians.scale_geometric_factor());

    for (auto o = 0; o < D.num_octaves(); ++o)
    {
      D.octave_scaling_factor(o) = gaussians.octave_scaling_factor(o);
      for (auto s = 0; s < D.num_scales_per_octave(); ++s)
      {
        D(s, o).resize(gaussians(s, o).sizes());

#ifdef DO_SARA_USE_HALIDE
        auto a = Shakti::Halide::as_runtime_buffer_4d(gaussians(s + 1, o));
        auto b = Shakti::Halide::as_runtime_buffer_4d(gaussians(s, o));
        auto out = Shakti::Halide::as_runtime_buffer_4d(D(s, o));

        shakti_subtract_32f_cpu(a, b, out);
#else
        tensor_view(D(s, o)).flat_array() =
            tensor_view(gaussians(s + 1, o)).flat_array() -
            tensor_view(gaussians(s, o)).flat_array();
#endif
      }
    }
    return D;
  }

}  // namespace DO::Sara
