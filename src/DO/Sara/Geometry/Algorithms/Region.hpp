// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_ALGORITHMS_REGION_HPP
#define DO_SARA_GEOMETRY_ALGORITHMS_REGION_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  std::vector<Point2i>
  compute_region_inner_boundary(const Image<int>& regions, int region_id);

  DO_SARA_EXPORT
  std::vector<std::vector<Point2i>>
  compute_region_inner_boundaries(const Image<int>& regions);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_ALGORITHMS_REGION_HPP */
