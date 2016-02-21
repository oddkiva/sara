#include <boost/python.hpp>

#include <DO/Sara/Geometry.hpp>

#include <DO/Sara/Python/Geometry.hpp>
#include <DO/Sara/Python/Numpy.hpp>


namespace bp = boost::python;
namespace sara = DO::Sara;


/**
 * Add view() and const_view() methods to wrap MultiArray in order to avoid
 * destroying the allocated memory.
 */
bp::list compute_region_inner_boundaries(PyObject *regions)
{
  return bp::list{};
}


void expose_geometry()
{
  bp::def("compute_region_inner_boundaries", &compute_region_inner_boundaries);
}
