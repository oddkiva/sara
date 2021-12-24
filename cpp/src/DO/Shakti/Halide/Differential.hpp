// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_polar_gradient_2d_32f_gpu_v2.h"


namespace DO::Shakti::Halide {

  //! @brief Calculate image gradients.
  inline auto polar_gradient_2d(Sara::ImageView<float>& in,   //
                                Sara::ImageView<float>& mag,  //
                                Sara::ImageView<float>& ori)  //
  {
    auto in_buffer = as_runtime_buffer_4d(in);
    auto mag_buffer = as_runtime_buffer_4d(mag);
    auto ori_buffer = as_runtime_buffer_4d(ori);

    in_buffer.set_host_dirty();
    mag_buffer.set_host_dirty();
    ori_buffer.set_host_dirty();

    shakti_polar_gradient_2d_32f_gpu_v2(in_buffer, mag_buffer, ori_buffer);

    mag_buffer.copy_to_host();
    ori_buffer.copy_to_host();
  }

}  // namespace DO::Shakti::Halide
