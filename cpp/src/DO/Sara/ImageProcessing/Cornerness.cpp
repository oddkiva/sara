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

#include <DO/Sara/ImageProcessing/Cornerness.hpp>

#ifdef DO_SARA_USE_HALIDE
#  include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#  include "shakti_cornerness_32f_cpu.h"
#endif


namespace DO::Sara {

  auto compute_cornerness([[maybe_unused]] const ImageView<float>& mxx,  //
                          [[maybe_unused]] const ImageView<float>& myy,  //
                          [[maybe_unused]] const ImageView<float>& mxy,  //
                          [[maybe_unused]] const float kappa,            //
                          [[maybe_unused]] ImageView<float>& cornerness) -> void
  {
#ifdef DO_SARA_USE_HALIDE
    if (mxx.sizes() != myy.sizes() ||  //
        mxx.sizes() != mxy.sizes() ||  //
        mxx.sizes() != cornerness.sizes())
      throw std::domain_error{"Cornerness: image sizes are not equal!"};

    auto mxx_buffer = Shakti::Halide::as_runtime_buffer_4d(mxx);
    auto myy_buffer = Shakti::Halide::as_runtime_buffer_4d(myy);
    auto mxy_buffer = Shakti::Halide::as_runtime_buffer_4d(mxy);
    auto cornerness_buffer = Shakti::Halide::as_runtime_buffer_4d(cornerness);

    shakti_cornerness_32f_cpu(mxx_buffer, myy_buffer, mxy_buffer, kappa,
                              cornerness_buffer);
#else
    throw std::runtime_error{"Not Implemented!"};
#endif
  }

}  // namespace DO::Sara
