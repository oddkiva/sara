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

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_dominant_gradient_orientations_gpu_v2.h"


namespace DO::Shakti::Halide {

  auto dominant_gradient_orientations(
      Sara::ImageView<float>& gradient_magnitudes,    //
      Sara::ImageView<float>& gradient_orientations,  //
      std::vector<float>& x,                          //
      std::vector<float>& y,                          //
      std::vector<float>& scale,                      //
      float scale_upper_bound,                        //
      Sara::Tensor_<bool, 2>& peak_map,               //
      Sara::Tensor_<float, 2>& peak_residuals,        //
      int num_orientation_bins = 36,                  //
      float gaussian_truncation_factor = 3.f,         //
      float scale_multiplying_factor = 1.5f,          //
      float peak_ratio_thres = 0.8f)                  //
      -> void
  {
    // Input buffers.
    auto mag_buffer = as_runtime_buffer_4d(gradient_magnitudes);
    auto ori_buffer = as_runtime_buffer_4d(gradient_orientations);
    auto x_buffer = as_runtime_buffer(x);
    auto y_buffer = as_runtime_buffer(y);
    auto scale_buffer = as_runtime_buffer(scale);

    //  Output buffers.
    auto peak_map_buffer = as_runtime_buffer(peak_map);
    auto peak_residuals_buffer = as_runtime_buffer(peak_residuals);

    // Input buffer to GPU.
    mag_buffer.set_host_dirty();
    ori_buffer.set_host_dirty();
    x_buffer.set_host_dirty();
    y_buffer.set_host_dirty();
    scale_buffer.set_host_dirty();

    // Output buffer to GPU.
    peak_map_buffer.set_host_dirty();
    peak_residuals_buffer.set_host_dirty();

    // Run the algorithm.
    shakti_dominant_gradient_orientations_gpu_v2(mag_buffer, ori_buffer,      //
                                                 x_buffer,                    //
                                                 y_buffer,                    //
                                                 scale_buffer,                //
                                                 scale_upper_bound,           //
                                                 num_orientation_bins,        //
                                                 gaussian_truncation_factor,  //
                                                 scale_multiplying_factor,    //
                                                 peak_ratio_thres,            //
                                                 peak_map_buffer,             //
                                                 peak_residuals_buffer);

    // Copy back to GPU.
    peak_map_buffer.copy_to_host();
    peak_residuals_buffer.copy_to_host();
  }

}  // namespace DO::Shakti::HalideBackend
