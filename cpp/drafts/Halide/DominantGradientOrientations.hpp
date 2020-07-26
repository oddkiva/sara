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

#include "shakti_dominant_gradient_orientations.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  auto dominant_gradient_orientations(
      Sara::ImageView<float>& gradient_magnitudes,    //
      Sara::ImageView<float>& gradient_orientations,  //
      std::vector<float>& x,                          //
      std::vector<float>& y,                          //
      std::vector<float>& scale,                      //
      float scale_upper_bound,                        //
      sara::Tensor_<bool, 2>& peak_map,               //
      sara::Tensor_<float, 2>& peak_residuals,        //
      int num_orientation_bins = 36,                  //
      float gaussian_truncation_factor = 3.f,         //
      float scale_multiplying_factor = 1.5f,          //
      float peak_ratio_thres = 0.8f)                  //
      -> void
  {
    // Input buffers.
    auto mag_buffer = as_runtime_buffer(gradient_magnitudes);
    auto ori_buffer = as_runtime_buffer(gradient_orientations);
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
    shakti_dominant_gradient_orientations(mag_buffer, ori_buffer,      //
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

  auto dominant_gradient_orientations(
      Sara::ImagePyramid<float>& gradient_mag_pyramid,                 //
      Sara::ImagePyramid<float>& gradient_ori_pyramid,                 //
      Pyramid<ExtremumArray>& dog_extrema,                              //
      Pyramid<DominantOrientationDenseMap>& dominant_orientations,  //
      int num_orientation_bins = 36,                                   //
      float gaussian_truncation_factor = 3.f,                          //
      float scale_multiplying_factor = 1.5f,                           //
      float peak_ratio_thres = 0.8f)                                   //
      -> void
  {
    for (auto o = 0; o < gradient_mag_pyramid.num_octaves(); ++o)
    {
      const auto oct_scale = gradient_mag_pyramid.octave_scaling_factor(o);

      for (auto s = 1; s < gradient_mag_pyramid.num_scales_per_octave() - 1; ++s)
      {
        auto& extrema = dog_extrema.dict[{s - 1, o}];
        if (extrema.size() == 0)
          continue;

        const auto& scale_max = *std::max_element(extrema.s.begin(), extrema.s.end());

        auto& dom_ori = dominant_orientations.dict[{s - 1, o}];
        dom_ori.resize(static_cast<std::int32_t>(extrema.size()),  //
                       num_orientation_bins);                      //

        dominant_gradient_orientations(       //
            gradient_mag_pyramid(s, o),       //
            gradient_ori_pyramid(s, o),       //
            extrema.x, extrema.y, extrema.s,  //
            scale_max,                //
            dom_ori.peak_map,                 //
            dom_ori.peak_residuals);
      }
    }
  }


  auto compress(const DominantOrientationDenseMap& dense_view)
  {
    auto sparse_view = std::multimap<int, float>{};

    const Eigen::VectorXi peak_count =
        dense_view.peak_map.matrix().rowwise().count().cast<int>();

    for (auto k = 0; k < dense_view.num_keypoints(); ++k)
    {
      if (peak_count(k) == 0)
      {
        sparse_view.insert({k, 0});
        continue;
      }

      const auto N = dense_view.num_orientation_bins();
      constexpr auto two_pi = 2 * static_cast<float>(M_PI);
      for (auto o = 0; o < dense_view.num_orientation_bins(); ++o)
      {
        if (!dense_view.peak_map(k, o))
          continue;

        auto ori = o + dense_view.peak_residuals(k, o);

        // Make sure that the angle is in the interval [0, N[.
        if (ori < 0)
          ori += N;
        else if (ori > N)
          ori -= N;
        // Convert to radians.
        ori = ori * two_pi /  N;

        sparse_view.insert({k, ori});
      }
    }

    return sparse_view;
  }

  auto compress(Pyramid<DominantOrientationDenseMap>& dense_views)
  {
    auto sparse_views = Pyramid<DominantOrientationMap>{};

    sparse_views.scale_octave_pairs = dense_views.scale_octave_pairs;

    std::transform(dense_views.dict.begin(), dense_views.dict.end(),
                   std::inserter(sparse_views.dict, sparse_views.dict.end()),
                   [](const auto& kv) {
                     return std::make_pair(                             //
                         kv.first,                                      //
                         DominantOrientationMap{compress(kv.second)});  //
                   });

    return sparse_views;
  }

}}}  // namespace DO::Shakti::HalideBackend
