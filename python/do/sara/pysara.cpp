#include <boost/python.hpp>

#include "DisjointSets.hpp"
#include "Geometry.hpp"
#include "ImageIO.hpp"
#include "VideoIO.hpp"


BOOST_PYTHON_MODULE(pysara)
{
  using namespace std;

  expose_disjoint_sets();
  expose_geometry();
  expose_image_io();
  expose_video_io();
}
