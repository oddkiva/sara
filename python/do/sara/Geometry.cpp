#include <boost/python.hpp>

#include <DO/Sara/Geometry.hpp>

#include <DO/Sara/Python/Geometry.hpp>
#include <DO/Sara/Python/Numpy.hpp>


namespace bp = boost::python;
namespace sara = DO::Sara;


bp::list compute_region_inner_boundaries(PyObject *regions)
{
  using namespace sara;

  auto im = image_view_2d<int>(regions);
  const auto region_polygons = compute_region_inner_boundaries(im);

  auto polys = bp::list{};
  for (const auto& region_poly : region_polygons)
  {
    auto poly = bp::list{};
    for (const auto& vertex : region_poly)
    {
      auto v_list = bp::list{};
      v_list.append(vertex.x());
      v_list.append(vertex.y());
      poly.append(v_list);
    }

    polys.append(poly);
  }

  return polys;
}


void expose_geometry()
{
  bp::def("compute_region_inner_boundaries", &compute_region_inner_boundaries);
}
