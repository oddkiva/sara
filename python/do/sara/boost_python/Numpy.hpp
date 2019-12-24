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

#ifndef DO_SARA_PYTHON_NUMPY_HPP
#define DO_SARA_PYTHON_NUMPY_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <numpy/ndarrayobject.h>

#include <boost/version.hpp>

#if BOOST_VERSION <= 106300
# include <boost/python/numeric.hpp>
#else
# include <boost/python/numpy.hpp>
#endif

#include <DO/Sara/Core/Image.hpp>


inline
#if (PY_VERSION_HEX < 0x03000000)
void import_numpy_array()
#else
void * import_numpy_array()
#endif
{
  /* Initialise numpy API and use 2/3 compatible return */
  import_array();

#if (PY_VERSION_HEX >= 0x03000000)
  return 0;
#endif
}


namespace DO { namespace Sara {

  template <typename T>
  inline ImageView<T> image_view_2d(PyObject *object)
  {
    auto np_array = reinterpret_cast<PyArrayObject *>(object);
    auto data = reinterpret_cast<T *>(PyArray_DATA(np_array));
    auto shape = PyArray_SHAPE(np_array);

    const auto& h = shape[0];
    const auto& w = shape[1];

    return ImageView<T>{data, Vector2i{w, h}};
  }

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_PYTHON_NUMPY_HPP */
