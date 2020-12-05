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

#include <DO/Sara/VideoIO.hpp>

#include "VideoIO.hpp"
#include "utilities.hpp"


namespace py = pybind11;
namespace sara = DO::Sara;


using namespace std;


class VideoStream : public sara::VideoStream
{
public:
  bool read_rgb_frame(py::array_t<std::uint8_t> image)
  {
    using namespace sara;

    auto imview = to_interleaved_rgb_image_view(image);

    if (!read())
      return false;

    imview = this->frame();
    return true;
  }

  auto numpy_sizes() const
  {
    return Eigen::Vector3i{height(), width(), 3};
  }
};


auto expose_video_io(pybind11::module& m) -> void
{
  py::class_<VideoStream>(m, "VideoStream")
      .def(py::init<>())
      .def("open", &VideoStream::open)
      .def("close", &VideoStream::close)
      .def("seek", &VideoStream::seek)
      .def("read", &VideoStream::read_rgb_frame)
      .def("width", &VideoStream::width)
      .def("height", &VideoStream::height)
      .def("sizes", py::overload_cast<>(&VideoStream::numpy_sizes, py::const_));
}
