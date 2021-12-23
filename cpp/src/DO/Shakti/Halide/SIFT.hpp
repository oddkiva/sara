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
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_sift_descriptor_gpu.h"
#include "shakti_sift_descriptor_gpu_v2.h"
#include "shakti_sift_descriptor_gpu_v3.h"
#include "shakti_sift_descriptor_gpu_v4.h"
#include "shakti_sift_descriptor_gpu_v5.h"


namespace DO::Shakti::HalideBackend {

  namespace v1 {

    auto compute_sift_descriptors(                      //
        Sara::ImageView<float>& gradient_magnitudes,    //
        Sara::ImageView<float>& gradient_orientations,  //
        std::vector<float>& x,                          //
        std::vector<float>& y,                          //
        std::vector<float>& scale,                      //
        std::vector<float>& orientation,                //
        float scale_upper_bound,                        //
        sara::Tensor_<float, 4>& kijo_tensor,           //
        float bin_length_in_scale_unit = 3.f,           //
        int N = 4,                                      //
        int O = 8)                                      //
        -> void
    {
      // Input buffers.
      auto mag_buffer = as_runtime_buffer(gradient_magnitudes);
      auto ori_buffer = as_runtime_buffer(gradient_orientations);
      auto x_buffer = as_runtime_buffer(x);
      auto y_buffer = as_runtime_buffer(y);
      auto scale_buffer = as_runtime_buffer(scale);
      auto orientation_buffer = as_runtime_buffer(orientation);

      //  Output buffers.
      auto kijo_tensor_buffer = as_runtime_buffer(kijo_tensor);

      // Input buffer to GPU.
      mag_buffer.set_host_dirty();
      ori_buffer.set_host_dirty();
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();
      orientation_buffer.set_host_dirty();

      // Output buffer to GPU.
      kijo_tensor_buffer.set_host_dirty();

      // Run the algorithm.
      shakti_sift_descriptor_gpu(mag_buffer, ori_buffer,    //
                                 x_buffer,                  //
                                 y_buffer,                  //
                                 scale_buffer,              //
                                 orientation_buffer,        //
                                 scale_upper_bound,         //
                                 bin_length_in_scale_unit,  //
                                 N, O,                      //
                                 kijo_tensor_buffer);

      // Copy back to GPU.
      kijo_tensor_buffer.copy_to_host();
    }

  }  // namespace v1

  namespace v2 {

    auto compute_sift_descriptors(                              //
        Sara::ImageView<float>& gradient_magnitudes,            //
        Sara::ImageView<float>& gradient_orientations,          //
        std::vector<float>& x,                                  //
        std::vector<float>& y,                                  //
        std::vector<float>& scale,                              //
        std::vector<float>& orientation,                        //
        float scale_upper_bound,                                //
        sara::Tensor_<float, 2>& sifts,                         //
        [[maybe_unused]] float bin_length_in_scale_unit = 3.f,  //
        int N = 4,                                              //
        int O = 8)                                              //
        -> void
    {
      // Input buffers.
      auto mag_buffer = as_runtime_buffer(gradient_magnitudes);
      auto ori_buffer = as_runtime_buffer(gradient_orientations);
      auto x_buffer = as_runtime_buffer(x);
      auto y_buffer = as_runtime_buffer(y);
      auto scale_buffer = as_runtime_buffer(scale);
      auto orientation_buffer = as_runtime_buffer(orientation);

      //  Output buffers.
      auto sift_tensor_buffer = as_runtime_buffer(sifts);

      // Input buffer to GPU.
      mag_buffer.set_host_dirty();
      ori_buffer.set_host_dirty();
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();
      orientation_buffer.set_host_dirty();

      // Output buffer to GPU.
      sift_tensor_buffer.set_host_dirty();

      // Run the algorithm.
      shakti_sift_descriptor_gpu_v2(mag_buffer, ori_buffer,  //
                                    x_buffer,                //
                                    y_buffer,                //
                                    scale_buffer,            //
                                    orientation_buffer,      //
                                    scale_upper_bound,       //
                                    N, O,                    //
                                    sift_tensor_buffer);

      // Copy back to GPU.
      sift_tensor_buffer.copy_to_host();
    }

  }  // namespace v2

  namespace v3 {

    auto compute_sift_descriptors(                              //
        Sara::ImageView<float>& gradient_magnitudes,            //
        Sara::ImageView<float>& gradient_orientations,          //
        std::vector<float>& x,                                  //
        std::vector<float>& y,                                  //
        std::vector<float>& scale,                              //
        std::vector<float>& orientation,                        //
        float scale_upper_bound,                                //
        sara::Tensor_<float, 3>& sifts,                         //
        [[maybe_unused]] float bin_length_in_scale_unit = 3.f,  //
        int N = 4,                                              //
        int O = 8)                                              //
        -> void
    {
      // Input buffers.
      auto mag_buffer = as_runtime_buffer(gradient_magnitudes);
      auto ori_buffer = as_runtime_buffer(gradient_orientations);
      auto x_buffer = as_runtime_buffer(x);
      auto y_buffer = as_runtime_buffer(y);
      auto scale_buffer = as_runtime_buffer(scale);
      auto orientation_buffer = as_runtime_buffer(orientation);

      //  Output buffers.
      auto sift_tensor_buffer = as_runtime_buffer(sifts);

      // Input buffer to GPU.
      mag_buffer.set_host_dirty();
      ori_buffer.set_host_dirty();
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();
      orientation_buffer.set_host_dirty();

      // Output buffer to GPU.
      sift_tensor_buffer.set_host_dirty();

      // Run the algorithm.
      shakti_sift_descriptor_gpu_v3(mag_buffer, ori_buffer,  //
                                    x_buffer,                //
                                    y_buffer,                //
                                    scale_buffer,            //
                                    orientation_buffer,      //
                                    scale_upper_bound,       //
                                    N, O,                    //
                                    sift_tensor_buffer);

      // Copy back to GPU.
      sift_tensor_buffer.copy_to_host();
    }

  }  // namespace v3

  namespace v4 {

    auto compute_sift_descriptors(                      //
        Sara::ImageView<float>& gradient_magnitudes,    //
        Sara::ImageView<float>& gradient_orientations,  //
        std::vector<float>& x,                          //
        std::vector<float>& y,                          //
        std::vector<float>& scale,                      //
        std::vector<float>& orientation,                //
        sara::Tensor_<float, 3>& sifts)                 //
        -> void
    {
      // Input buffers.
      auto mag_buffer = as_runtime_buffer(gradient_magnitudes);
      auto ori_buffer = as_runtime_buffer(gradient_orientations);
      auto x_buffer = as_runtime_buffer(x);
      auto y_buffer = as_runtime_buffer(y);
      auto scale_buffer = as_runtime_buffer(scale);
      auto orientation_buffer = as_runtime_buffer(orientation);

      //  Output buffers.
      auto sift_tensor_buffer = as_runtime_buffer(sifts);

      // Input buffer to GPU.
      mag_buffer.set_host_dirty();
      ori_buffer.set_host_dirty();
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();
      orientation_buffer.set_host_dirty();

      // Output buffer to GPU.
      sift_tensor_buffer.set_host_dirty();

      // Run the algorithm.
      shakti_sift_descriptor_gpu_v4(mag_buffer, ori_buffer,  //
                                    x_buffer,                //
                                    y_buffer,                //
                                    scale_buffer,            //
                                    orientation_buffer,      //
                                    sift_tensor_buffer);

      // Copy back to CPU.
      sift_tensor_buffer.copy_to_host();
    }

  }  // namespace v4

  namespace v5 {

    auto compute_sift_descriptors(                      //
        Sara::ImageView<float>& gradient_magnitudes,    //
        Sara::ImageView<float>& gradient_orientations,  //
        std::vector<float>& x,                          //
        std::vector<float>& y,                          //
        std::vector<float>& scale,                      //
        std::vector<float>& orientation,                //
        sara::Tensor_<float, 3>& sifts)                 //
        -> void
    {
      // Input buffers.
      auto mag_buffer = Shakti::Halide::as_runtime_buffer_4d(gradient_magnitudes);
      auto ori_buffer = Shakti::Halide::as_runtime_buffer_4d(gradient_orientations);
      auto x_buffer = as_runtime_buffer(x);
      auto y_buffer = as_runtime_buffer(y);
      auto scale_buffer = as_runtime_buffer(scale);
      auto orientation_buffer = as_runtime_buffer(orientation);

      //  Output buffers.
      auto sift_tensor_buffer = as_runtime_buffer(sifts);

      // Input buffer to GPU.
      mag_buffer.set_host_dirty();
      ori_buffer.set_host_dirty();
      x_buffer.set_host_dirty();
      y_buffer.set_host_dirty();
      scale_buffer.set_host_dirty();
      orientation_buffer.set_host_dirty();

      // Output buffer to GPU.
      sift_tensor_buffer.set_host_dirty();

      // Run the algorithm.
      shakti_sift_descriptor_gpu_v5(mag_buffer, ori_buffer,  //
                                    x_buffer,                //
                                    y_buffer,                //
                                    scale_buffer,            //
                                    orientation_buffer,      //
                                    sift_tensor_buffer);

      // Copy back to CPU.
      sift_tensor_buffer.copy_to_host();
    }

  }  // namespace v5

}  // namespace DO::Shakti::HalideBackend
