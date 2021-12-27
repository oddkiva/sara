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


namespace DO::Sara {

  auto scale_space_dog_extremum_map(const ImageView<float>& a,
                                    const ImageView<float>& b,
                                    const ImageView<float>& c,
                                    float edge_ratio_thres,
                                    float extremum_thres,
                                    ImageView<std::int8_t>& out) -> void;


}  // namespace DO::Sara
