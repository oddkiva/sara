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
  py::class_<sara::OERegion>(m, "OERegion")
      .def(py::init<>())
      .def_readwrite("coords", &sara::OERegion::coords)
      .def_readwrite("shape_matrix", &sara::OERegion::shape_matrix)
      .def_readwrite("orientation", &sara::OERegion::orientation)
      .def_readwrite("type", &sara::OERegion::type)
      .def_readwrite("extremum_type", &sara::OERegion::extremum_type)
      .def(py::self == py::self)
      .def("radius", &sara::OERegion::radius, "radian"_a = 0.f);

  py::class_<std::vector<sara::OERegion>>(m, "OERegionVector")
      .def(py::init<>())
      .def("clear", &std::vector<sara::OERegion>::clear)
      .def("pop_back", &std::vector<sara::OERegion>::pop_back)
      .def("__len__",
           [](const std::vector<sara::OERegion>& v) { return v.size(); })
      .def(
          "__iter__",
          [](std::vector<sara::OERegion>& v) {
            return py::make_iterator(v.begin(), v.end());
          },
          py::keep_alive<0, 1>());

  py::class_<sara::KeypointList<sara::OERegion, float>>(m, "KeypointList")
      .def(py::init<>());

  m.def(
      "features",
      [](const sara::KeypointList<sara::OERegion, float>& key) {
        return sara::features(key);
      },
      "Extract the geometric features for each keypoint");
  m.def(
      "descriptors",
      [](const sara::KeypointList<sara::OERegion, float>& key) {
        return sara::descriptors(key);
      },
      "Extract the descriptor vectors for each keypoint");

  py::class_<sara::ImagePyramidParams>(m, "ImagePyramidParams")
      .def(py::init<int, int, float, int, float, float>(),
           "first_octave_index"_a = 1, "scale_count_per_octave"_a = 3 + 3,
           "scale_geometric_factor"_a = std::pow(2.f, 1 / 3.f),
           "image_padding_size"_a = 1, "scale_camera"_a = 0.5f,
           "scale_initial"_a = 1.6f)
      .def_property_readonly("first_octave_index",
                             &sara::ImagePyramidParams::first_octave_index)
      .def_property_readonly("scale_count_per_octave",
                             &sara::ImagePyramidParams::scale_count_per_octave)
      .def_property_readonly("scale_camera",
                             &sara::ImagePyramidParams::scale_camera)
      .def_property_readonly("scale_initial",
                             &sara::ImagePyramidParams::scale_initial)
      .def_property_readonly("scale_geometric_factor",
                             &sara::ImagePyramidParams::scale_geometric_factor)
      .def_property_readonly("image_padding_size",
                             &sara::ImagePyramidParams::image_padding_size);

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

  m.def("compute_sift_keypoints", sara::compute_sift_keypoints,
        "pyramid_params"_a = sara::ImagePyramidParams(),
        "gauss_truncate"_a = 4.f,
        "extremum_thres"_a = 0.01f,
        "edge_ratio_thres"_a = 10.f,
        "extremum_refinement_iter"_a = 5,
        "parallel"_a = true,
        "Compute SIFT keypoints for an input float image.");
}
