// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>

#include <DO/Sara/VideoIO.hpp>

#include "Numpy.hpp"
#include "VideoIO.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;


using namespace std;


class VideoStream : public sara::VideoStream
{
public:
  void read_rgb_frame(PyObject *inout)
  {
    using namespace sara;

    auto image = image_view_2d<Rgb8>(inout);
    if (!read(image))
      throw std::runtime_error{"Error: could not read image frame"};
  }

  bp::tuple sizes_tuple() const
  {
    return bp::make_tuple(height(), width(), 3);
  }
};


void expose_video_io()
{
#if BOOST_VERSION <= 106300
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
#else
  Py_Initialize();
  bp::numpy::initialize();
#endif

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
