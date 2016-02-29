#include <boost/python.hpp>

#include "DisjointSets.hpp"
#include "Geometry.hpp"
#include "ImageIO.hpp"
#ifdef PYSARA_BUILD_VIDEOIO
# include "VideoIO.hpp"
#endif


BOOST_PYTHON_MODULE(pysara)
{
  using namespace std;

  expose_disjoint_sets();
  expose_geometry();
  expose_image_io();
#ifdef PYSARA_BUILD_VIDEOIO
  expose_video_io();
#endif
}
