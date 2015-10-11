#ifndef DO_SARA_PYTHON_PYTHON_HPP
#define DO_SARA_PYTHON_PYTHON_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <numpy/ndarrayobject.h>


namespace DO { namespace Sara { namespace python {

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

} /* namespace python */
} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_PYTHON_PYTHON_HPP */
