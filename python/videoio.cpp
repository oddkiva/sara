#include <boost/python.hpp>

#include <DO/Sara/VideoIO.hpp>

#include "python.hpp"
#include "videoio.hpp"


namespace sara = DO::Sara;

using namespace std;
using namespace boost::python;


class VideoStream : sara::VideoStream
{
public:
  VideoStream() = default;

  object read()
  {
    using namespace sara;

    auto video_frame = Image<Rgb8>{};
    if (!sara::VideoStream::read(video_frame))
      return object{};

    auto data = video_frame.data();
    auto ndims = 3;
    npy_intp sizes[] = { video_frame.height(), video_frame.width(), 3 };
    auto py_obj = PyArray_SimpleNewFromData(ndims, sizes, NPY_UINT8, data);

    boost::python::handle<> handle{ py_obj };
    boost::python::numeric::array arr{ handle };

    return arr.copy();
  }
};

void expose_videoio()
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  DO::Sara::python::import_numpy_array();

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

  class_<VideoStream, boost::noncopyable>("VideoStream")
    .def("open", &VideoStream::open)
    .def("close", &VideoStream::close)
    .def("seek", &VideoStream::seek)
    .def("read", &VideoStream::read)
    ;
}
