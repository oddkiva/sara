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

#define BOOST_TEST_MODULE "MultiViewGeometry/Pinhole Camera Model"

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


auto make_pinhole_camera()
{
  auto camera_parameters = PinholeCamera<float>{};

  const auto w = 1920;
  const auto h = 1080;

  // Focal lengths in each dimension.
  const auto fx = 1063.30738864;
  const auto fy = 1064.20554291;
  // Shear component.
  const auto s = -1.00853432;
  // Principal point.
  const auto u0 = 969.55702157;
  const auto v0 = 541.26230733;

  camera_parameters.image_sizes << w, h;
  // clang-format off
  camera_parameters.K <<
      fx,  s, u0,
       0, fy, v0,
       0,  0,  1;
  // clang-format on

  camera_parameters.calculate_inverse_calibration_matrix();

  return camera_parameters;
}


BOOST_AUTO_TEST_CASE(test_pinhole_camera_model)
{
  const auto camera = make_pinhole_camera();
  const auto& K = camera.K;
  const auto& K_inv = camera.K_inverse;
  BOOST_CHECK_LE((K * K_inv - Eigen::Matrix3f::Identity()).norm(), 1e-4f);
}
