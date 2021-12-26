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

//! @file

#include <DO/Sara/ImageProcessing/LocalExtremum.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_scale_space_dog_extremum_32f_cpu.h"


namespace DO::Sara {

  auto scale_space_dog_extremum_map(const ImageView<float>& a,
                                    const ImageView<float>& b,
                                    const ImageView<float>& c,
                                    float edge_ratio_thres,
                                    float extremum_thres,
                                    ImageView<std::int8_t>& out) -> void
  {
    auto a_ = Shakti::Halide::as_runtime_buffer_4d(a);
    auto b_ = Shakti::Halide::as_runtime_buffer_4d(b);
    auto c_ = Shakti::Halide::as_runtime_buffer_4d(c);
    auto out_ = Shakti::Halide::as_runtime_buffer_4d(out);
    shakti_scale_space_dog_extremum_32f_cpu(a_, b_, c_, edge_ratio_thres,
                                            extremum_thres, out_);
  }

}  // namespace DO::Sara
#endif
