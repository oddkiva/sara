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


namespace DO::Shakti::Halide::cpu {

  auto scale(const Sara::ImageView<float>& src, Sara::ImageView<float>& dst) -> void;

  auto gaussian_convolution(const Sara::ImageView<float>& src,
                            Sara::ImageView<float>& dst, float sigma,
                            int truncation_factor = 4) -> void;

}  // namespace DO::Shakti::Halide::Cpu
