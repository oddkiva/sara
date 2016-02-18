#include <boost/python.hpp>

#include "disjoint_sets.hpp"
#include "image_io.hpp"
#include "video_io.hpp"


BOOST_PYTHON_MODULE(sara)
{
  using namespace std;

  expose_disjoint_sets();
  expose_image_io();
  expose_video_io();
}
