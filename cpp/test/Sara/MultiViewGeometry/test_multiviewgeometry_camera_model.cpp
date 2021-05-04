// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Camera Model"

#include <DO/Sara/MultiViewGeometry/Camera/CameraModel.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/FisheyeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_camera_model_constructor)
{
  namespace sara = DO::Sara;
  auto cameras = std::vector<sara::CameraModel>{};

  cameras.emplace_back(sara::PinholeCamera<float>{});
  cameras.emplace_back(sara::FisheyeCamera<float>{});
  // cameras.emplace_back(sara::BrownConradyCamera<float>{});

  project(cameras.front(), Eigen::Vector3f{});
  project(cameras.back(), Eigen::Vector3f{});
}
