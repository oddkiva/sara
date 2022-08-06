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

#include <DO/Sara/ImageProcessing/CartesianToPolarCoordinates.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_cartesian_to_polar_32f_cpu.h"
#endif


namespace DO::Sara {

  auto
  cartesian_to_polar_coordinates([[maybe_unused]] const ImageView<float>& fx,
                                 [[maybe_unused]] const ImageView<float>& fy,
                                 [[maybe_unused]] ImageView<float>& mag,
                                 [[maybe_unused]] ImageView<float>& ori) -> void
  {
#ifdef DO_SARA_USE_HALIDE
    if (fx.sizes() != fy.sizes() ||   //
        fx.sizes() != mag.sizes() ||  //
        fx.sizes() != ori.sizes())
      throw std::domain_error{
          "Cartesian to polar coordinates: image sizes are not equal!"};

    auto fx_buffer = Shakti::Halide::as_runtime_buffer_4d(fx);
    auto fy_buffer = Shakti::Halide::as_runtime_buffer_4d(fy);
    auto mag_buffer = Shakti::Halide::as_runtime_buffer_4d(mag);
    auto ori_buffer = Shakti::Halide::as_runtime_buffer_4d(ori);

    shakti_cartesian_to_polar_32f_cpu(fx_buffer, fy_buffer, mag_buffer,
                                      ori_buffer);
#else
    throw std::runtime_error{"Not Implemented!"};
#endif
  }

}  // namespace DO::Sara
