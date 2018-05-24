#include <boost/python.hpp>

#include <DO/Sara/DisjointSets.hpp>

#include "DisjointSets.hpp"
#include "Numpy.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;


bp::list compute_adjacency_list_2d(PyObject* labels)
{
  using namespace sara;

  auto im = image_view_2d<int>(labels);
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

  auto disjoint_sets = DisjointSets{im.size(), adj_list};
  disjoint_sets.compute_connected_components();
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
