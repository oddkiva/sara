// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Camera Model"

#include <DO/Sara/MultiViewGeometry/Camera/CameraModel.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/FisheyeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_camera_model)
{
  namespace sara = DO::Sara;
  using CameraModelf = sara::CameraModel<float>;
  auto cameras = std::vector<CameraModelf>{};

  cameras.emplace_back(sara::PinholeCamera<float>{});
  cameras.emplace_back(sara::BrownConradyCamera32<float>{});
  cameras.emplace_back(sara::FisheyeCamera<float>{});
  cameras.emplace_back(sara::OmnidirectionalCamera<float>{});

  // clang-format off
  auto K = (Eigen::Matrix3f{} <<
    970,   0, 960,
      0, 970, 540,
      0,   0,   1
  ).finished();
  // clang-format on
  cameras.front().set_calibration_matrix(K);

  BOOST_CHECK(cameras.front().calibration_matrix() == K);
  BOOST_CHECK(cameras.front().inverse_calibration_matrix() == K.inverse());
  BOOST_CHECK(project(cameras.front(), Eigen::Vector3f{0, 0, 1}) ==
              Eigen::Vector2f(960, 540));
  BOOST_CHECK_SMALL((backproject(cameras.front(), Eigen::Vector2f{960, 540}) -
                     Eigen::Vector3f::UnitZ())
                        .norm(),
                    1e-6f);

  {
    const auto& K = cameras.front().calibration_matrix();
    const auto& Kinv = cameras.front().inverse_calibration_matrix();
    BOOST_CHECK_SMALL((K * Kinv - Eigen::Matrix3f::Identity()).norm(), 1e-6f);
  }
}
