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

#include <DO/Shakti/Halide/SIFT/V1/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_sift_descriptor_gpu.h"
#include "shakti_sift_descriptor_gpu_v2.h"
#include "shakti_sift_descriptor_gpu_v3.h"
#include "shakti_sift_descriptor_gpu_v4.h"


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

      for (const auto& so : keypoints.scale_octave_pairs)
      {
        const auto& s = so.first.first;
        const auto& o = so.first.second;

        auto kit = keypoints.dict.find({s, o});
        if (kit == keypoints.dict.end())
          continue;

        auto& k = kit->second;
        const auto& scale_max = *std::max_element(k.s.begin(), k.s.end());

        auto& descriptors_so = descriptors.dict[{s, o}];
        descriptors_so.resize({static_cast<int>(k.size()), N, N, O});

        compute_sift_descriptors(gradient_magnitudes(s, o),      //
                                 gradient_orientations(s, o),    //
                                 k.x, k.y, k.s, k.orientations,  //
                                 scale_max,                      //
                                 descriptors_so,                 //
                                 bin_length_in_scale_unit,       //
                                 N, O);                          //
      }

      return descriptors;
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

    auto compute_sift_descriptors(                                 //
        Sara::ImagePyramid<float>& gradient_magnitudes,            //
        Sara::ImagePyramid<float>& gradient_orientations,          //
        Pyramid<HalideBackend::OrientedExtremumArray>& keypoints,  //
        float bin_length_in_scale_unit = 3.f,                      //
        int N = 4,                                                 //
        int O = 8)                                                 //
    {
      auto descriptors = Pyramid<Sara::Tensor_<float, 2>>{};

      descriptors.scale_octave_pairs = keypoints.scale_octave_pairs;

      for (const auto& so : keypoints.scale_octave_pairs)
      {
        const auto& s = so.first.first;
        const auto& o = so.first.second;

        auto kit = keypoints.dict.find({s, o});
        if (kit == keypoints.dict.end())
          continue;

        auto& k = kit->second;
        const auto& scale_max = *std::max_element(k.s.begin(), k.s.end());

        auto& descriptors_so = descriptors.dict[{s, o}];
        descriptors_so.resize({static_cast<int>(k.size()), N * N * O});

        v2::compute_sift_descriptors(gradient_magnitudes(s, o),      //
                                     gradient_orientations(s, o),    //
                                     k.x, k.y, k.s, k.orientations,  //
                                     scale_max,                      //
                                     descriptors_so,                 //
                                     bin_length_in_scale_unit,       //
                                     N, O);                          //
      }

      return descriptors;
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

    auto compute_sift_descriptors(                         //
        Sara::ImagePyramid<float>& gradient_magnitudes,    //
        Sara::ImagePyramid<float>& gradient_orientations,  //
        Pyramid<OrientedExtremumArray>& keypoints,         //
        float bin_length_in_scale_unit = 3.f,              //
        int N = 4,                                         //
        int O = 8)                                         //
    {
      auto descriptors = Pyramid<Sara::Tensor_<float, 3>>{};

      descriptors.scale_octave_pairs = keypoints.scale_octave_pairs;

      for (const auto& so : keypoints.scale_octave_pairs)
      {
        const auto& s = so.first.first;
        const auto& o = so.first.second;

        auto kit = keypoints.dict.find({s, o});
        if (kit == keypoints.dict.end())
          continue;

        auto& k = kit->second;
        const auto& scale_min = *std::min_element(k.s.begin(), k.s.end());
        const auto& scale_max = *std::max_element(k.s.begin(), k.s.end());
        SARA_CHECK(s);
        SARA_CHECK(o);
        SARA_CHECK(so.second.first);
        SARA_CHECK(k.size());
        SARA_CHECK(k.x.size());
        SARA_CHECK(k.y.size());
        SARA_CHECK(k.s.size());
        SARA_CHECK(k.orientations.size());
        SARA_CHECK(scale_min);
        SARA_CHECK(scale_max);
        SARA_CHECK(scale_min * std::sqrt(2.) * 3 * (4 + 1) / 2.f);
        SARA_CHECK(scale_max * std::sqrt(2.) * 3 * (4 + 1) / 2.f);

        auto& descriptors_so = descriptors.dict[{s, o}];
        descriptors_so.resize({static_cast<int>(k.size()), N * N, O});

        auto timer = Sara::Timer{};
        timer.restart();
        v3::compute_sift_descriptors(gradient_magnitudes(s, o),      //
                                     gradient_orientations(s, o),    //
                                     k.x, k.y, k.s, k.orientations,  //
                                     scale_max,                      //
                                     descriptors_so,                 //
                                     bin_length_in_scale_unit,       //
                                     N, O);                          //
        auto elapsed_ms = timer.elapsed_ms();
        SARA_DEBUG << "SIFT v3 = " << elapsed_ms << "ms" << std::endl
                   << std::endl;
      }

      return descriptors;
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

    auto compute_sift_descriptors(                         //
        Sara::ImagePyramid<float>& gradient_magnitudes,    //
        Sara::ImagePyramid<float>& gradient_orientations,  //
        Pyramid<OrientedExtremumArray>& keypoints)         //
    {
#ifdef DEBUG
      auto timer = Sara::Timer{};
#endif

      constexpr auto N = 4;
      constexpr auto O = 8;

      auto descriptors = Pyramid<Sara::Tensor_<float, 3>>{};
      descriptors.scale_octave_pairs = keypoints.scale_octave_pairs;

      for (const auto& so : keypoints.scale_octave_pairs)
      {
        const auto& s = so.first.first;
        const auto& o = so.first.second;

        auto kit = keypoints.dict.find({s, o});
        if (kit == keypoints.dict.end())
          continue;
        auto& k = kit->second;

        auto& d = descriptors.dict[{s, o}];

        d.resize({static_cast<int>(k.size()), N * N, O});

#ifdef DEBUG
        SARA_CHECK(o);
        SARA_CHECK(s);
        SARA_CHECK(so.second.first);
        SARA_CHECK(d.sizes().transpose());

        timer.restart();
#endif

        v4::compute_sift_descriptors(gradient_magnitudes(s, o),      //
                                     gradient_orientations(s, o),    //
                                     k.x, k.y, k.s, k.orientations,  //
                                     d);                             //

#ifdef DEBUG
        auto elapsed_ms = timer.elapsed_ms();
        SARA_DEBUG << "SIFT v4 = " << elapsed_ms << "ms" << std::endl
                   << std::endl;
#endif
      }

      return descriptors;
    }

  }  // namespace v4

}  // namespace DO::Shakti::HalideBackend
