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

#include "FeatureDetectors.hpp"
#include "Utilities.hpp"

#include <DO/Sara/FeatureMatching.hpp>

#include <pybind11/stl.h>


namespace py = pybind11;
namespace sara = DO::Sara;


auto expose_feature_matching(pybind11::module& m) -> void
{
  py::class_<sara::Tensor_<float, 2>>(m, "Tensor2f", py::buffer_protocol())
      .def_buffer([](sara::Tensor_<float, 2>& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(float),                          /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style
                                                       format descriptor */
            2,                                      /* Number of dimensions */
            {m.rows(), m.cols()},                   /* Buffer dimensions */
            {sizeof(float) * m.cols(), /* Strides (in bytes) for each index */
             sizeof(float)});
      });

  py::class_<sara::Match>(m, "Match")
      .def(py::init<>())
      .def("x", py::overload_cast<>(&sara::Match::x_index, py::const_))
      .def("y", py::overload_cast<>(&sara::Match::y_index, py::const_));

  py::class_<sara::AnnMatcher>(m, "AnnMatcher")
      .def(py::init<const sara::KeypointList<sara::OERegion, float>&,
                    const sara::KeypointList<sara::OERegion, float>&,  //
                    float>())
      .def("compute_matches", &sara::AnnMatcher::compute_matches);
}
