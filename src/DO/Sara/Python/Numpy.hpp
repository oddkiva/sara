#ifndef DO_SARA_PYTHON_NUMPY_HPP
#define DO_SARA_PYTHON_NUMPY_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <numpy/ndarrayobject.h>


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


#endif /* DO_SARA_PYTHON_NUMPY_HPP */
