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

#include <drafts/Halide/Utilities.hpp>

#include "shakti_refine_scale_space_extrema.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  struct DoGExtremaInitial
  {
    std::vector<std::int32_t> x;
    std::vector<std::int32_t> y;
    std::vector<std::int8_t> type;
    float scale;
    float scale_geometric_factor;

    auto resize(std::size_t size)
    {
      x.resize(size);
      y.resize(size);
      type.resize(size);
    }
  };

  struct DoGExtremaRefined
  {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> s;
    std::vector<float> value;
    std::vector<std::int8_t> type;

    auto resize(std::size_t size)
    {
      x.resize(size);
      y.resize(size);
      s.resize(size);
      value.resize(size);
      type.resize(size);
    }
  };


  inline auto populate_local_scale_space_extrema(
      Sara::ImagePyramid<std::int8_t>& dog_extrema_pyramid)
  {
    sara::tic();
    const auto num_scales = dog_extrema_pyramid.num_octaves() *
                            dog_extrema_pyramid.num_scales_per_octave();

    auto points = std::vector<DoGExtremaInitial>(num_scales);

    const auto at = [&](int s, int o) {
      return o * dog_extrema_pyramid.num_scales_per_octave() + s;
    };

    for (auto o = 0; o < dog_extrema_pyramid.num_octaves(); ++o)
    {
      for (auto s = 0; s < dog_extrema_pyramid.num_scales_per_octave(); ++s)
      {
        const auto& dog_ext_map = dog_extrema_pyramid(s, o);
        const auto num_extrema = std::count_if(      //
            dog_ext_map.begin(), dog_ext_map.end(),  //
            [](const auto& v) { return v != 0; }     //
        );

        if (num_extrema == 0)
          continue;

        auto& points_so = points[at(s, o)];
        points_so.resize(num_extrema);
        points_so.scale = dog_extrema_pyramid.scale_relative_to_octave(s);
        points_so.scale_geometric_factor =
            dog_extrema_pyramid.scale_geometric_factor();

        auto i = 0;
        for (auto y = 0; y < dog_ext_map.height(); ++y)
        {
          for (auto x = 0; x < dog_ext_map.width(); ++x)
          {
            if (dog_ext_map(x, y) == 0)
              continue;

            points_so.x[i] = x;
            points_so.y[i] = y;
            points_so.type[i] = dog_ext_map(x, y);
            ++i;
          }
        }
      }
    }
    sara::toc("Populating DoG extrema");

    return points;
  }


  inline auto refine_scale_space_extrema(Sara::ImageView<float>& a,  //
                                         Sara::ImageView<float>& b,  //
                                         Sara::ImageView<float>& c,  //
                                         DoGExtremaInitial& dogi,    //
                                         DoGExtremaRefined& dogf)    //
  {
    auto a_buffer = as_runtime_buffer(a);
    auto b_buffer = as_runtime_buffer(b);
    auto c_buffer = as_runtime_buffer(c);
    auto x_buffer = as_runtime_buffer(dogi.x);
    auto y_buffer = as_runtime_buffer(dogi.y);
    const auto w = a.width();
    const auto h = a.height();

    auto xf_buffer = as_runtime_buffer(dogf.x);
    auto yf_buffer = as_runtime_buffer(dogf.y);
    auto sf_buffer = as_runtime_buffer(dogf.s);
    auto value_buffer = as_runtime_buffer(dogf.value);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();
    x_buffer.set_host_dirty();
    y_buffer.set_host_dirty();

    shakti_refine_scale_space_extrema(a_buffer, b_buffer, c_buffer,  //
                                      x_buffer, y_buffer,            //
                                      w, h,                          //
                                      dogi.scale,                    //
                                      dogi.scale_geometric_factor,   //
                                      xf_buffer,                     //
                                      yf_buffer,                     //
                                      sf_buffer,                     //
                                      value_buffer);                 //

    xf_buffer.copy_to_host();
    yf_buffer.copy_to_host();
    sf_buffer.copy_to_host();
    value_buffer.copy_to_host();

    dogf.type = dogi.type;
  }

  //! @brief Extract local scale-space extrema.
  inline auto refine_scale_space_extrema(Sara::ImagePyramid<float>& dog,
                                         std::vector<DoGExtremaInitial>& dogi)
  {
    sara::tic();
    auto dogf = std::vector<DoGExtremaRefined>(dogi.size());

    const auto num_scales_per_octave = dog.num_scales_per_octave() - 2;
    const auto at = [&](int s, int o) {
      return o * num_scales_per_octave + s;
    };

    for (auto o = 0; o < dog.num_octaves(); ++o)
      for (auto s = 0; s < num_scales_per_octave; ++s)
        dogf[at(s, o)].resize(dogi[at(s, o)].x.size());

    for (auto o = 0; o < dog.num_octaves(); ++o)
    {
      for (auto s = 0; s < dog.num_scales_per_octave() - 2; ++s)
      {
        if (dogi[at(s, o)].x.empty())
          continue;

        refine_scale_space_extrema(dog(s, o), dog(s + 1, o), dog(s + 2, o),  //
                                   dogi[at(s, o)], dogf[at(s, o)]);          //
      }
    }
    sara::toc("Refining DoG extrema");

    return dogf;
  }

}}}  // namespace DO::Shakti::HalideBackend
