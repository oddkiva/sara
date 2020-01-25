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

#include <boost/python.hpp>

#include <DO/Sara/DisjointSets.hpp>

#include "DisjointSets.hpp"
#include "Numpy.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;


bp::list compute_adjacency_list_2d(PyObject* labels)
{
  using namespace sara;

  const auto im = image_view_2d<int>(labels);
  auto adj_list = compute_adjacency_list_2d(im);

  auto adj_pylist = bp::list{};
  for (const auto& neighborhood : adj_list)
  {
    auto neighborhood_pylist = bp::list{};

    for (const auto& index : neighborhood)
      neighborhood_pylist.append(index);

    adj_pylist.append(neighborhood_pylist);
  }

  return adj_pylist;
}


bp::list compute_connected_components(PyObject* labels)
{
  using namespace sara;

  const auto im = image_view_2d<int>(labels);

  auto adj_list_data = compute_adjacency_list_2d(im);
  AdjacencyList adj_list{adj_list_data};

  auto disjoint_sets = DisjointSets{};
  disjoint_sets.compute_connected_components(adj_list);
  const auto components = disjoint_sets.get_connected_components();

  auto components_pylist = bp::list{};
  for (const auto& component : components)
  {
    auto component_pylist = bp::list{};

    for (const auto& vertex : component)
      component_pylist.append(vertex);

    components_pylist.append(component_pylist);
  }

  return components_pylist;
}


void expose_disjoint_sets()
{
#if BOOST_VERSION <= 106300
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
#else
  Py_Initialize();
  bp::numpy::initialize();
#endif

  // Import numpy array.
  import_numpy_array();

  bp::def("compute_adjacency_list_2d", &compute_adjacency_list_2d);
  bp::def("compute_connected_components", &compute_connected_components);
}
