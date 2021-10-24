// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Shakti::Halide::CPU {

  auto subtract(const Sara::ImageView<float>& a,  //
                const Sara::ImageView<float>& b,  //
                Sara::ImageView<float>& out) -> void;

  auto convolve(const Sara::TensorView_<float, 4>& src,
                const Sara::TensorView_<float, 4>& kernel,
                Sara::TensorView_<float, 4>& dst) -> void;

  auto gaussian_convolution(const Sara::ImageView<float>& src,
                            Sara::ImageView<float>& dst,  //
                            float sigma,                  //
                            int truncation_factor = 4) -> void;

}  // namespace DO::Shakti::Halide::CPU
