#include <boost/python.hpp>

#include <DO/Sara/Python/DisjointSets.hpp>
#include <DO/Sara/Python/Geometry.hpp>
#include <DO/Sara/Python/ImageIO.hpp>
#include <DO/Sara/Python/VideoIO.hpp>


BOOST_PYTHON_MODULE(pysara)
{
  using namespace std;

  expose_disjoint_sets();
  expose_geometry();
  expose_image_io();
  expose_video_io();
}
