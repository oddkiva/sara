#include <pybind11/pybind11.h>

#include "DisjointSets.hpp"
#include "FeatureDetectors.hpp"
#include "FeatureMatching.hpp"
#include "Geometry.hpp"
#include "VideoIO.hpp"


PYBIND11_MODULE(pysara_pybind11, m)
{
  m.doc() = "Sara Python API";  // optional module docstring

  expose_disjoint_sets(m);
  expose_geometry(m);
#ifdef PYSARA_BUILD_VIDEOIO
  expose_video_io(m);
#endif
  expose_feature_detectors(m);
  expose_feature_matching(m);
}
