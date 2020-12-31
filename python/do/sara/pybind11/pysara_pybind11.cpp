#include <pybind11/pybind11.h>

#include "DisjointSets.hpp"
#include "Geometry.hpp"
#include "VideoIO.hpp"
#include "sfm.hpp"


PYBIND11_MODULE(pysara_pybind11, m)
{
  expose_disjoint_sets(m);
  expose_geometry(m);
#ifdef PYSARA_BUILD_VIDEOIO
  expose_video_io(m);
#endif
  expose_sfm(m);
}
