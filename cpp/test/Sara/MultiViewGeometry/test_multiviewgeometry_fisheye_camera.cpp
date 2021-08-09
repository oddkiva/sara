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

#define BOOST_TEST_MODULE "MultiViewGeometry/Fisheye Camera Model"

#include <DO/Sara/MultiViewGeometry/Camera/FisheyeCamera.hpp>

#include <boost/test/unit_test.hpp>

#include <array>


using namespace DO::Sara;


auto make_fisheye_camera()
{
  auto camera_parameters = FisheyeCamera<float>{};

  const auto w = 1920;
  const auto h = 1080;

  const auto f = 677.3246133600308f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;

  camera_parameters.image_sizes << w, h;
  // clang-format off
  camera_parameters.K <<
      f, 0, u0,
      0, f, v0,
      0, 0,  1;
  // clang-format on
  camera_parameters.k <<      //
      -0.20f,                 //
      0.1321295087447987f,    //
      -0.06844064024539671f,  //
      0.01237548905484928f;

  camera_parameters.calculate_inverse_calibration_matrix();

  return camera_parameters;
}


BOOST_AUTO_TEST_CASE(test_fisheye_camera_model)
{
  const auto camera = make_fisheye_camera();
  const auto& w = camera.image_sizes.x();
  const auto& h = camera.image_sizes.y();

  // Check the projection and backprojection.
  {
    const auto center = Eigen::Vector2f(960, 540);

    // The backprojected ray must have positive depth, i.e., z > 0?
    //
    // The light ray that hits the center of the film plane must be in front
    // of the camera.
    const auto ray = camera.backproject(center);
    BOOST_CHECK(ray.z() > 0);

    // The reprojected ray must hit the center of the image.
    const auto projected_ray = camera.project(ray);
    BOOST_CHECK_LE((projected_ray - center).norm(), 1e-3f);

    // Another property is that the center should be not too distorted.
    //
    // This is not very rigorous but this test is there to serve as a geometric
    // insight.
    const auto center_distorted = camera.distort(center);
    BOOST_CHECK_LE((center_distorted - center).norm(), 10.f);
    const auto center_undistorted = camera.undistort(center);
    BOOST_CHECK_LE((center_undistorted - center).norm(), 15.f);
  }

  // Some chosen points.
  const auto points = std::array<Eigen::Vector2f, 4>{
      Eigen::Vector2f{w * 0.25f, h * 0.25f},
      Eigen::Vector2f{w * 0.75f, h * 0.25f},
      Eigen::Vector2f{w * 0.75f, h * 0.75f},
      Eigen::Vector2f{w * 0.25f, h * 0.75f},
  };
  for (const auto& pd : points)
  {
    // Check that the corners are behind the cameras
    const auto ray = camera.backproject(pd);
    // This is a non-negotiable check.
    BOOST_CHECK(ray.z() > 0);

    // The reprojected ray must hit the center of the image.
    const auto projected_ray = camera.project(ray);
    // We allow ourselves to set generous thresholds for the reprojection of
    // rays because the corners are extreme cases.
    BOOST_CHECK_LE((projected_ray - pd).norm(), 1e-3f);

    const auto pu = camera.undistort(pd);
    const auto pd2 = camera.distort(pu);

    BOOST_CHECK_LE((pd2 - pd).norm(), 1e-3f);
  }
}
