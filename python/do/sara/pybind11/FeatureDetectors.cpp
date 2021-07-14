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

#include "Utilities.hpp"
#include "FeatureDetectors.hpp"

#include <DO/Sara/FeatureDetectors.hpp>


namespace py = pybind11;
namespace sara = DO::Sara;


auto expose_feature_detectors(pybind11::module& m) -> void
{
  py::class_<sara::EdgeDetector>(m, "EdgeDetector")
      .def(py::init<>())
      .def(py::init([](float high_threshold_ratio, float low_threshold_ratio,
                       float angular_threshold) {
        return sara::EdgeDetector{{
            high_threshold_ratio,  //
            low_threshold_ratio,   //
            angular_threshold      //
        }};
      }))
      // .def_readonly("edge_chains",
      // &sara::EdgeDetector::Pipeline::edges_as_list,
      //               "edge chains");
      ;
}
