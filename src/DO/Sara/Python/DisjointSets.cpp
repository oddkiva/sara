#include <boost/python.hpp>

#include <DO/Sara/DisjointSets.hpp>

#include <DO/Sara/Python/DisjointSets.hpp>
#include <DO/Sara/Python/Numpy.hpp>


namespace bp = boost::python;
namespace sara = DO::Sara;


bp::list compute_adjacency_list_2d(PyObject *labels)
{
  using namespace sara;

  auto numpy_array = reinterpret_cast<PyArrayObject *>(labels);
  auto data = reinterpret_cast<int *>(PyArray_DATA(numpy_array));
  auto shape = PyArray_SHAPE(numpy_array);
  const auto& h = shape[0];
  const auto& w = shape[1];

  auto im = ImageView<int, 2>{ data, Vector2i{ w, h } };
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


void expose_disjoint_sets()
{
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_numpy_array();

  bp::def("compute_adjacency_list_2d", &compute_adjacency_list_2d);
}
