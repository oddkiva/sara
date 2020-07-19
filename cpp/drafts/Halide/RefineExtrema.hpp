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

#include <drafts/Halide/ExtremaDataStructures.hpp>
#include <drafts/Halide/Utilities.hpp>

#include "shakti_refine_scale_space_extrema.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  inline auto populate_local_scale_space_extrema(
      Sara::ImagePyramid<std::int8_t>& extrema_map_pyramid)
  {
    sara::tic();
    const auto num_scales = extrema_map_pyramid.num_octaves() *
                            extrema_map_pyramid.num_scales_per_octave();

    auto extrema = Pyramid<QuantizedExtremaArray>{};

    for (auto o = 0; o < extrema_map_pyramid.num_octaves(); ++o)
    {
      for (auto s = 0; s < extrema_map_pyramid.num_scales_per_octave(); ++s)
      {
        const auto& dog_ext_map = extrema_map_pyramid(s, o);
        const auto num_extrema = std::count_if(      //
            dog_ext_map.begin(), dog_ext_map.end(),  //
            [](const auto& v) { return v != 0; }     //
        );

        if (num_extrema == 0)
          continue;

        // Map the index pair to the scale value, octave scaling factor.
        extrema.scale_octave_pairs[{s, o}] = {
            extrema_map_pyramid.scale_relative_to_octave(s),
            extrema_map_pyramid.octave_scaling_factor(o)};

        auto& extrema_so = extrema.dict[{s, o}];

        // Populate the list of extrema for the corresponding scale.
        extrema_so.resize(num_extrema);
        extrema_so.scale = extrema_map_pyramid.scale_relative_to_octave(s);
        extrema_so.scale_geometric_factor =
            extrema_map_pyramid.scale_geometric_factor();

        auto i = 0;
        for (auto y = 0; y < dog_ext_map.height(); ++y)
        {
          for (auto x = 0; x < dog_ext_map.width(); ++x)
          {
            if (dog_ext_map(x, y) == 0)
              continue;

            extrema_so.x[i] = x;
            extrema_so.y[i] = y;
            extrema_so.type[i] = dog_ext_map(x, y);
            ++i;
          }
        }
      }
    }
    sara::toc("Populating DoG extrema");

    return extrema;
  }


  inline auto
  refine_scale_space_extrema(Sara::ImageView<float>& a,               //
                             Sara::ImageView<float>& b,               //
                             Sara::ImageView<float>& c,               //
                             QuantizedExtremaArray& extrema_initial,  //
                             ExtremaArray& extrema_refined)           //
  {
    auto a_buffer = as_runtime_buffer(a);
    auto b_buffer = as_runtime_buffer(b);
    auto c_buffer = as_runtime_buffer(c);
    auto x_buffer = as_runtime_buffer(extrema_initial.x);
    auto y_buffer = as_runtime_buffer(extrema_initial.y);
    const auto w = a.width();
    const auto h = a.height();

    auto xf_buffer = as_runtime_buffer(extrema_refined.x);
    auto yf_buffer = as_runtime_buffer(extrema_refined.y);
    auto sf_buffer = as_runtime_buffer(extrema_refined.s);
    auto value_buffer = as_runtime_buffer(extrema_refined.value);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();
    x_buffer.set_host_dirty();
    y_buffer.set_host_dirty();

    shakti_refine_scale_space_extrema(
        a_buffer, b_buffer, c_buffer,            //
        x_buffer, y_buffer,                      //
        w, h,                                    //
        extrema_initial.scale,                   //
        extrema_initial.scale_geometric_factor,  //
        xf_buffer,                               //
        yf_buffer,                               //
        sf_buffer,                               //
        value_buffer);                           //

    xf_buffer.copy_to_host();
    yf_buffer.copy_to_host();
    sf_buffer.copy_to_host();
    value_buffer.copy_to_host();

    extrema_refined.type = extrema_initial.type;
  }

  //! @brief Extract local scale-space extrema.
  inline auto refine_scale_space_extrema(
      Sara::ImagePyramid<float>& dog,
      Pyramid<QuantizedExtremaArray>& extrema_initial)
  {
    sara::tic();
    auto extrema_refined = Pyramid<ExtremaArray>{};
    extrema_refined.scale_octave_pairs = extrema_initial.scale_octave_pairs;

    for (auto o = 0; o < dog.num_octaves(); ++o)
      for (auto s = 0; s < dog.num_scales_per_octave() -  2; ++s)
        extrema_refined.dict[{s, o}].resize(extrema_initial.dict[{s, o}].size());

    for (auto o = 0; o < dog.num_octaves(); ++o)
    {
      for (auto s = 0; s < dog.num_scales_per_octave() - 2; ++s)
      {
        if (extrema_initial.dict[{s, o}].x.empty())
          continue;

        refine_scale_space_extrema(dog(s, o), dog(s + 1, o), dog(s + 2, o),  //
                                   extrema_initial.dict[{s, o}],             //
                                   extrema_refined.dict[{s, o}]);            //
      }
    }
    sara::toc("Refining DoG extrema");

    return extrema_refined;
  }

}}}  // namespace DO::Shakti::HalideBackend
