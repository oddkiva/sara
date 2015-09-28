#include <boost/python.hpp>

#include <DO/Sara/VideoIO.hpp>


using namespace std;

namespace sara = DO::Sara;


BOOST_PYTHON_MODULE(sara)
{
  using namespace boost::python;

  // Create "sara.videoio" module name.
  string videoio_name{ extract<string>{
    scope().attr("__name__") + ".videoio"
  }};

  // Create "sara.videoio" module.
  object videoio_module{ handle<>{
    borrowed(PyImport_AddModule(videoio_name.c_str()))
  }};

  // Set the "sara.videoio" scope.
  scope().attr("videoio") = videoio_module;
  scope parent{ videoio_module };

  class_<sara::VideoStream, boost::noncopyable>("VideoStream")
    .def("open", &sara::VideoStream::open)
    .def("close", &sara::VideoStream::close)
    .def("seek", &sara::VideoStream::seek)
    .def("read", &sara::VideoStream::read)
    ;
}
