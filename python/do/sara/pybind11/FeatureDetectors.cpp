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

#include <DO/Sara/FeatureDetectors.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>


namespace py = pybind11;
namespace sara = DO::Sara;

using namespace pybind11::literals;


auto expose_feature_detectors(pybind11::module& m) -> void
{
  py::class_<sara::KeypointList<sara::OERegion, float>>(m, "KeypointList")
      .def(py::init<>());

  py::class_<sara::OERegion>(m, "OERegion")
      .def(py::init<>())
      .def_readwrite("coords", &sara::OERegion::coords)
      .def_readwrite("shape_matrix", &sara::OERegion::shape_matrix)
      .def_readwrite("orientation", &sara::OERegion::orientation)
      .def_readwrite("type", &sara::OERegion::type)
      .def_readwrite("extremum_type", &sara::OERegion::extremum_type)
      .def(py::self == py::self)
      .def("radius", &sara::OERegion::radius, "radian"_a = 0.f);

  py::class_<sara::EdgeDetector::Pipeline>(m, "EdgeDetectorPipeline")
      .def(py::init<>())
      .def_readonly("edge_chains", &sara::EdgeDetector::Pipeline::edges_as_list,
                    "edge chains")
      .def_readonly("edge_polylines",
                    &sara::EdgeDetector::Pipeline::edges_simplified,
                    "edge polylines");

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
      .def("detect", &sara::EdgeDetector::operator(), "detect edges")
      .def_readonly("pipeline", &sara::EdgeDetector::pipeline, "pipeline data");
  ;

  m.def(
      "compute_sift_keypoints",
      [](py::array_t<float> image) {
        const auto imview = to_image_view<float>(image);
        return sara::compute_sift_keypoints(imview);
      },
      "Compute SIFT keypoints for an input float image.");
}
