// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "Geometry.hpp"
#include "Utilities.hpp"

#include <DO/Sara/Geometry.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;
namespace sara = DO::Sara;


auto compute_region_inner_boundaries(py::array_t<int> regions)
{
  auto im = to_image_view<int>(regions);
  return sara::compute_region_inner_boundaries(im);
}

void expose_geometry(py::module& m)
{
  m.def("compute_region_inner_boundaries", &compute_region_inner_boundaries);
  m.def("ramer_douglas_peucker", &sara::ramer_douglas_peucker);
}
