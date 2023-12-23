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

#include "Utilities.hpp"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>


namespace py = pybind11;
namespace sara = DO::Sara;


auto resize(py::array_t<float, py::array::c_style | py::array::forcecast> src,
            py::array_t<float, py::array::c_style | py::array::forcecast> dst)
    -> void
{
  const auto tensor_src = to_tensor_view_3d(src);
  auto tensor_dst = to_tensor_view_3d(dst);

  for (auto i = 0; i < src.shape(0); ++i)
  {
    const auto plane_src_i = sara::image_view(tensor_src[i]);
    auto plane_dst_i = sara::image_view(tensor_dst[i]);
    sara::resize_v2(plane_src_i, plane_dst_i);
  }
}

auto expose_image_processing(pybind11::module& m) -> void
{
  m.def("resize", &resize, "Resize image");
}
