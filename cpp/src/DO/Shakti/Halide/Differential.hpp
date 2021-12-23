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

#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_polar_gradient_2d_32f_gpu.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  //! @brief Calculate image gradients.
  //! @{
  inline auto polar_gradient_2d(Halide::Runtime::Buffer<float>& in,   //
                                Halide::Runtime::Buffer<float>& mag,  //
                                Halide::Runtime::Buffer<float>& ori)  //
  {
    shakti_polar_gradient_2d_32f_gpu(in, mag, ori);
  }

  inline auto polar_gradient_2d(Sara::ImageView<float>& in,   //
                                Sara::ImageView<float>& mag,  //
                                Sara::ImageView<float>& ori)  //
  {
    auto in_buffer = as_runtime_buffer(in);
    auto mag_buffer = as_runtime_buffer(mag);
    auto ori_buffer = as_runtime_buffer(ori);

    in_buffer.set_host_dirty();
    mag_buffer.set_host_dirty();
    ori_buffer.set_host_dirty();

    polar_gradient_2d(in_buffer, mag_buffer, ori_buffer);

    mag_buffer.copy_to_host();
    ori_buffer.copy_to_host();
  }

  inline auto polar_gradient_2d(Sara::ImagePyramid<float>& in)
    -> std::tuple<Sara::ImagePyramid<float>, Sara::ImagePyramid<float>>
  {
    auto mag = Sara::ImagePyramid<float>{};
    auto ori = Sara::ImagePyramid<float>{};

    mag.reset(in.octave_count(),              //
              in.scale_count_per_octave(),    //
              in.scale_initial(),            //
              in.scale_geometric_factor());  //

    ori.reset(in.octave_count(),              //
              in.scale_count_per_octave(),    //
              in.scale_initial(),            //
              in.scale_geometric_factor());  //

    for (auto o = 0; o < in.octave_count(); ++o)
    {
      mag.octave_scaling_factor(o) = in.octave_scaling_factor(o);
      ori.octave_scaling_factor(o) = in.octave_scaling_factor(o);

      for (auto s = 0; s < in.scale_count_per_octave(); ++s)
      {
        mag(s, o).resize(in(s, o).sizes());
        ori(s, o).resize(in(s, o).sizes());
        // Sara::tic();
        polar_gradient_2d(in(s, o), mag(s, o), ori(s, o));
        // Sara::toc(Sara::format("Scale-space local max at (s=%d, o=%d)", s,
        // o));
      }
    }

    return std::make_tuple(mag, ori);
  }
  //! @}.

}}}  // namespace DO::Shakti::HalideBackend
