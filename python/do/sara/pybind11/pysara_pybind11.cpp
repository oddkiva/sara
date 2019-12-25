#include <pybind11/pybind11.h>

#include "sfm.hpp"


PYBIND11_MODULE(pysara_pybind11, m)
{
  expose_sfm(m);
}
