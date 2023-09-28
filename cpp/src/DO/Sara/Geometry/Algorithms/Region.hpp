// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  //! @addtogroup GeometryAlgorithms
  //! @{

  DO_SARA_EXPORT
  std::vector<Point2i>
  compute_region_inner_boundary(const ImageView<int>& regions, int region_id);

  DO_SARA_EXPORT
  std::vector<std::vector<Point2i>>
  compute_region_inner_boundaries(const ImageView<int>& regions);

  //! @}

}  // namespace DO::Sara
