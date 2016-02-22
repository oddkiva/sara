#ifndef DO_SARA_PYTHON_NUMPY_HPP
#define DO_SARA_PYTHON_NUMPY_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <numpy/ndarrayobject.h>

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

    return ImageView<T>{ data, Vector2i{ h, w } };
  }

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_PYTHON_NUMPY_HPP */
