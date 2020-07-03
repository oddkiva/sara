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

#include "shakti_local_max_32f.h"
#include "shakti_local_scale_space_extremum_32f.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  inline auto local_max(Sara::ImageView<float>& a,  //
                        Sara::ImageView<float>& b,  //
                        Sara::ImageView<float>& c,  //
                        Sara::ImageView<std::uint8_t>& out)
  {
    auto a_tensor_view =
        tensor_view(a).reshape(Eigen::Vector3i{1, a.height(), a.width()});
    auto b_tensor_view =
        tensor_view(b).reshape(Eigen::Vector3i{1, b.height(), b.width()});
    auto c_tensor_view =
        tensor_view(c).reshape(Eigen::Vector3i{1, c.height(), c.width()});
    auto out_tensor_view =
        tensor_view(out).reshape(Eigen::Vector3i{1, out.height(), out.width()});

    auto a_buffer = as_runtime_buffer(a_tensor_view);
    auto b_buffer = as_runtime_buffer(b_tensor_view);
    auto c_buffer = as_runtime_buffer(c_tensor_view);
    auto out_buffer = as_runtime_buffer(out_tensor_view);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();
    shakti_local_max_32f(a_buffer, b_buffer, c_buffer, out_buffer);
    out_buffer.copy_to_host();
  }

  inline auto local_scale_space_extrema(Sara::ImageView<float>& a,
                                        Sara::ImageView<float>& b,
                                        Sara::ImageView<float>& c,
                                        Sara::ImageView<std::int8_t>& out,
                                        float edge_ratio = 10.0f,
                                        float extremum_thres = 0.03f)
  {
    auto a_buffer = as_runtime_buffer(a);
    auto b_buffer = as_runtime_buffer(b);
    auto c_buffer = as_runtime_buffer(c);
    auto out_buffer = as_runtime_buffer(out);

    a_buffer.set_host_dirty();
    b_buffer.set_host_dirty();
    c_buffer.set_host_dirty();

    shakti_local_scale_space_extremum_32f(
        a_buffer, b_buffer, c_buffer, edge_ratio, extremum_thres, out_buffer);

    out_buffer.copy_to_host();
  }

  //! @brief Extract local scale-space max.
  inline auto local_max(Sara::ImagePyramid<float>& in)
      -> Sara::ImagePyramid<std::uint8_t>
  {
    auto out = Sara::ImagePyramid<std::uint8_t>{};

    out.reset(in.num_octaves(),                                  //
              in.num_scales_per_octave() - 2,                    //
              in.scale_initial() * in.scale_geometric_factor(),  //
              in.scale_geometric_factor());                      //

    for (auto o = 0; o < in.num_octaves(); ++o)
    {
      out.octave_scaling_factor(o) = in.octave_scaling_factor(o);

      for (auto s = 0; s < in.num_scales_per_octave() - 2; ++s)
      {
        out(s, o).resize(in(s, o).sizes());
        // Sara::tic();
        local_max(in(s, o), in(s + 1, o), in(s + 2, o), out(s, o));
        // Sara::toc(Sara::format("Scale-space local max at (s=%d, o=%d)", s,
        // o));
      }
    }

    return out;
  }

  //! @brief Extract local scale-space extrema.
  inline auto local_scale_space_extrema(Sara::ImagePyramid<float>& in,
                                        float edge_ratio = 10.f,
                                        float extremum_thres = 0.03f)
      -> Sara::ImagePyramid<std::int8_t>
  {
    auto out = Sara::ImagePyramid<std::int8_t>{};

    out.reset(in.num_octaves(),                                  //
              in.num_scales_per_octave() - 2,                    //
              in.scale_initial() * in.scale_geometric_factor(),  //
              in.scale_geometric_factor());                      //

    for (auto o = 0; o < in.num_octaves(); ++o)
    {
      out.octave_scaling_factor(o) = in.octave_scaling_factor(o);

      for (auto s = 0; s < in.num_scales_per_octave() - 2; ++s)
      {
        out(s, o).resize(in(s, o).sizes());
        // Sara::tic();
        local_scale_space_extrema(in(s, o), in(s + 1, o), in(s + 2, o),
                                  out(s, o), edge_ratio, extremum_thres);
        // Sara::toc(Sara::format("Scale-space local extrema at (s=%d, o=%d)",
        // s, o));
      }
    }

    return out;
  }

}}}  // namespace DO::Shakti::HalideBackend
