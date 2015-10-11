#include <boost/python.hpp>

#include "imageio.hpp"
#include "videoio.hpp"


BOOST_PYTHON_MODULE(sara)
{
  using namespace std;

  expose_imageio();
  expose_videoio();
}
