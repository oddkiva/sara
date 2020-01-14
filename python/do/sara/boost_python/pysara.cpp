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

#include "DisjointSets.hpp"
#include "Geometry.hpp"
#include "ImageIO.hpp"
#ifdef PYSARA_BUILD_VIDEOIO
#  include "VideoIO.hpp"
#endif
#include "IPC.hpp"


BOOST_PYTHON_MODULE(pysara)
{
  using namespace std;

  expose_disjoint_sets();
  expose_geometry();
  expose_image_io();
#ifdef PYSARA_BUILD_VIDEOIO
  expose_video_io();
#endif
  expose_ipc();
}
