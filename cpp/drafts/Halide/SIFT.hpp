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

#include <drafts/Halide/ExtremaDataStructures.hpp>
#include <drafts/Halide/Utilities.hpp>

#include "shakti_sift_descriptor.h"


namespace DO { namespace Shakti { namespace HalideBackend {

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
    shakti_sift_descriptor(mag_buffer, ori_buffer,    //
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

  auto compute_sift_descriptors(                         //
      Sara::ImagePyramid<float>& gradient_magnitudes,    //
      Sara::ImagePyramid<float>& gradient_orientations,  //
      Pyramid<OrientedExtremumArray>& keypoints,         //
      float bin_length_in_scale_unit = 3.f,              //
      int N = 4,                                         //
      int O = 8)                                         //
  {
    auto descriptors = Pyramid<Sara::Tensor_<float, 4>>{};

    descriptors.scale_octave_pairs = keypoints.scale_octave_pairs;

    const auto& scale_factor = gradient_magnitudes.scale_geometric_factor();

    for (const auto& so : keypoints.scale_octave_pairs)
    {
      const auto& s = so.first.first;
      const auto& o = so.first.second;

      auto kit = keypoints.dict.find({s, o});
      if (kit == keypoints.dict.end())
        continue;

      auto& k = kit->second;

      auto& d = descriptors.dict[{s, o}];
      d.resize({k.size(), N, N, O});

      compute_sift_descriptors(gradient_magnitudes(s, o),         //
                               gradient_orientations(s, o),       //
                               k.x, k.y, k.s, k.orientations,     //
                               k.scale_quantized * scale_factor,  //
                               d,                                 //
                               bin_length_in_scale_unit,          //
                               N, O);                             //
    }

    return descriptors;
  }

}}}  // namespace DO::Shakti::HalideBackend
