#include <pybind11/pybind11.h>

#include "SfM.hpp"


PYBIND11_MODULE(pysara_pybind11, m)
{
  expose_sfm(m);
}
