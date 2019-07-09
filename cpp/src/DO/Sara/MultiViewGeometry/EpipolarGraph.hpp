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

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>

#include <memory>


namespace DO::Sara {

struct PhotoAttributes
{
  std::vector<Image<Rgb8>> images;
  std::vector<KeypointList<OERegion, float>> keypoints;
  std::vector<PinholeCamera> cameras;
};

struct EpipolarEdge
{
  int i;  // left
  int j;  // right
  Eigen::Matrix3d m;
};

} /* namespace DO::Sara */
