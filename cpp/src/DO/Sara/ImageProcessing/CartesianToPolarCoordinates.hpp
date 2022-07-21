// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  inline auto cartesian_to_polar_coordinates(const ImageView<float>& fx,
                                             const ImageView<float>& fy,
                                             const ImageView<float>& f_mag,
                                             const ImageView<float>& f_ori)
      -> void;

}  // namespace DO::Sara
