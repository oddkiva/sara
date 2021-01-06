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

#include "sfm.hpp"
#include "Utilities.hpp"

#include <DO/Sara/Core/MultiArray.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>


namespace py = pybind11;
namespace sara = DO::Sara;


auto compute_sift_keypoints(py::array_t<float> image)
{
  const auto imview = to_image_view<float>(image);
  return sara::compute_sift_keypoints(imview);
}


auto expose_sfm(pybind11::module& m) -> void
{
  m.doc() = "Sara Python API";  // optional module docstring

  // TODO: move this to somewhere else.
  py::class_<sara::Tensor_<float, 2>>(m, "Tensor2f")
      .def(py::init<>())
      .def("data",
           py::overload_cast<>(&sara::Tensor_<float, 2>::data, py::const_))
      .def("sizes",
           py::overload_cast<>(&sara::Tensor_<float, 2>::sizes, py::const_));

  py::class_<sara::OERegion>(m, "OERegion")
      .def(py::init<>())
      .def_readwrite("coords", &sara::OERegion::coords)
      .def_readwrite("shape_matrix", &sara::OERegion::shape_matrix)
      .def_readwrite("orientation", &sara::OERegion::orientation)
      .def_readwrite("type", &sara::OERegion::type)
      .def_readwrite("extremum_type", &sara::OERegion::extremum_type)
      .def(py::self == py::self);

  m.def("compute_sift_keypoints", &compute_sift_keypoints,
        "Compute SIFT keypoints for an input float image.");
}
