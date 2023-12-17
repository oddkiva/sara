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

#include <DO/Sara/DisjointSets.hpp>

#include <pybind11/stl.h>

#include "DisjointSets.hpp"
#include "Utilities.hpp"


namespace py = pybind11;
namespace sara = DO::Sara;


auto compute_adjacency_list_2d(py::array_t<int> labels)
{
  using namespace sara;

  const auto im = to_image_view(labels);
  const auto adj_list = compute_adjacency_list_2d(im);

  return adj_list;
}

auto compute_connected_components(py::array_t<int> labels)
{
  using namespace sara;

  const auto im = to_image_view(labels);

  auto adj_list_data = compute_adjacency_list_2d(im);
  AdjacencyList adj_list{adj_list_data};

  auto disjoint_sets = DisjointSets{};
  disjoint_sets.compute_connected_components(adj_list);
  return disjoint_sets.get_connected_components();
}


auto expose_disjoint_sets(pybind11::module& m) -> void
{
  m.def("compute_adjacency_list_2d", &compute_adjacency_list_2d,
        "Compute the ajdacency list for the 2D image.");
  m.def("compute_connected_components", &compute_connected_components,
        "Compute the connected components of 2D image");
}
