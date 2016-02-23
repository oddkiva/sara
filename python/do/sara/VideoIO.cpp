#include <boost/python.hpp>
#include <boost/python/tuple.hpp>

#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/Python/Numpy.hpp>
#include <DO/Sara/Python/VideoIO.hpp>


namespace bp = boost::python;
namespace sara = DO::Sara;


using namespace std;


class VideoStream : public sara::VideoStream
{
public:
  void read_rgb_frame(PyObject *inout)
  {
    using namespace sara;

    auto numpy_array = reinterpret_cast<PyArrayObject *>(inout);
    auto data = reinterpret_cast<Rgb8 *>(PyArray_DATA(numpy_array));

    Image<Rgb8> image{ data, sizes() };
    if (!read(image))
      throw std::runtime_error{ "Error: could not read image frame" };
  }

  bp::tuple sizes_tuple() const
  {
    return bp::make_tuple(height(), width(), 3);
  }
};


void expose_video_io()
{
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_numpy_array();

  bp::class_<VideoStream, boost::noncopyable>("VideoStream")
    .def("open", &VideoStream::open)
    .def("close", &VideoStream::close)
    .def("seek", &VideoStream::seek)
    .def("read", &VideoStream::read_rgb_frame)
    .def("width", &VideoStream::width)
    .def("height", &VideoStream::height)
    .def("sizes", &VideoStream::sizes_tuple)
    ;
}
