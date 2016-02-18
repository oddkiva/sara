#include <boost/python.hpp>

#include <DO/Sara/Geometry.hpp>

#include "geometry.hpp"
#include "python.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;


/**
 * Add view() and const_view() methods to wrap MultiArray in order to avoid
 * destroying the allocated memory.
 */
bp::list compute_region_inner_boundaries(PyObject *regions)
{
  auto compute_inner_region
}


void expose_geometry()
{
  bp::def("compute_region_inner_boundaries", &compute_region_inner_boundaries);
}
