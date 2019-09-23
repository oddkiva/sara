#include <boost/python.hpp>
#include <boost/python/tuple.hpp>

#include <DO/Sara/Geometry.hpp>

#include "Geometry.hpp"
#include "Numpy.hpp"


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
    for (const auto& v : region_poly)
    {
      auto point = bp::make_tuple(v.x(), v.y());
      poly.append(point);
    }

    polys.append(poly);
  }

  return polys;
}

bp::list ramer_douglas_peucker(bp::list contours, double eps)
{
  using namespace std;
  using namespace sara;

  const auto sz = bp::len(contours);

  auto c = vector<Point2d>{};
  c.reserve(bp::len(contours));

  for (auto i = 0; i < sz; ++i)
  {
    bp::tuple p = bp::extract<bp::tuple>(contours[i]);
    double x = bp::extract<double>(p[0]);
    double y = bp::extract<double>(p[1]);
    c.push_back(Point2d(x, y));
  }

  c = sara::ramer_douglas_peucker(c, eps);

  auto c_pylist = bp::list{};
  for (const auto& v : c)
    c_pylist.append(bp::make_tuple(v.x(), v.y()));

  return c_pylist;
}

void expose_geometry()
{
  bp::def("compute_region_inner_boundaries", &compute_region_inner_boundaries);
  bp::def("ramer_douglas_peucker", &ramer_douglas_peucker);
}
