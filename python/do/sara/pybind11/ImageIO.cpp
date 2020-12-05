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

#include <DO/Sara/ImageIO.hpp>

#include "ImageIO.hpp"
#include "Numpy.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;

using namespace std;


bp::object imread(const std::string& filepath)
{
  using namespace sara;

  auto image = imread<double>(filepath);
  auto data = image.data();

#if BOOST_VERSION <= 106300
  auto ndims = 2;
  npy_intp sizes[] = {image.height(), image.width()};
  auto py_obj = PyArray_SimpleNewFromData(ndims, sizes, NPY_DOUBLE, data);

  bp::handle<> handle{py_obj};
  bp::numeric::array arr{handle};
#else
  namespace np = bp::numpy;

  const auto shape = bp::make_tuple(image.height(), image.width());
  const auto dtype = np::dtype::get_builtin<double>();
  const auto stride = bp::make_tuple(sizeof(double));

  auto arr_owner = bp::object{};
  auto arr = np::from_data(data, dtype, shape, stride, arr_owner);
#endif

  return arr.copy();
}


void expose_image_io()
{
#if BOOST_VERSION <= 106300
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
#else
  Py_Initialize();
  bp::numpy::initialize();
#endif

  import_numpy_array();

  def("imread", &imread);
}
