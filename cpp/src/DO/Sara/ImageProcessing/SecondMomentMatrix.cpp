// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image/Image.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_moment_matrix_32f_cpu.h"
#endif


namespace DO::Sara {

  auto second_moment_matrix(const ImageView<float>& fx,
                            const ImageView<float>& fy,  //
                            ImageView<float>& mxx,       //
                            ImageView<float>& myy,       //
                            ImageView<float>& mxy) -> void
  {
#ifdef DO_SARA_USE_HALIDE
    if (fx.sizes() != fy.sizes() ||   //
        fx.sizes() != mxx.sizes() ||  //
        fx.sizes() != myy.sizes() ||  //
        fx.sizes() != mxy.sizes())    //
      throw std::domain_error{"Second moment matrix: image sizes are not equal!"};

    auto fx_buffer = Shakti::Halide::as_runtime_buffer_4d(fx);
    auto fy_buffer = Shakti::Halide::as_runtime_buffer_4d(fy);
    auto mxx_buffer = Shakti::Halide::as_runtime_buffer_4d(mxx);
    auto myy_buffer = Shakti::Halide::as_runtime_buffer_4d(myy);
    auto mxy_buffer = Shakti::Halide::as_runtime_buffer_4d(mxy);

    shakti_moment_matrix_32f_cpu(fx_buffer, fy_buffer, mxx_buffer, myy_buffer,
                                 mxy_buffer);
#else
    throw std::runtime_error{"Not Implemented!"};
#endif
  }

}  // namespace DO::Sara
