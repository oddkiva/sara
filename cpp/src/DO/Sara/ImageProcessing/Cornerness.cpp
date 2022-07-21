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

#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_cornerness_32f_cpu.h"


namespace DO::Sara {

  inline auto cornerness(const ImageView<float>& mxx,  //
                         const ImageView<float>& myy,  //
                         const ImageView<float>& mxy,  //
                         const float kappa,            //
                         ImageView<float>& cornerness) -> void;
  {
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
  }

}  // namespace DO::Sara
