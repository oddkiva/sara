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

#define BOOST_TEST_MODULE "MultiViewGeometry/Omnidirectional Camera Model"

#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


auto make_omnidirectional_camera()
{
  auto camera_parameters = OmnidirectionalCamera<float>{};

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
  camera_parameters.radial_distortion_coefficients <<
      0.50776095,
      -0.16478652;
  camera_parameters.tangential_distortion_coefficients <<
      0.00023093,
      0.00078712;
  // clang-format on
  camera_parameters.xi = 1.50651524;

  camera_parameters.calculate_inverse_calibration_matrix();

  return camera_parameters;
}


BOOST_AUTO_TEST_CASE(test_omnidirectional_camera_model)
{
  const auto camera = make_omnidirectional_camera();
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
    std::cout << (projected_ray - center).norm() << std::endl;
    BOOST_CHECK_LE((projected_ray - center).norm(), 1e-3f);

    // Another property is that the center should be not too distorted.
    //
    // This is not very rigorous but this test is there to serve as a geometric
    // insight.
    const auto center_distorted= camera.distort(center);
    BOOST_CHECK_LE((center_distorted - center).norm(), 10.f);
    const auto center_undistorted= camera.undistort(center);
    BOOST_CHECK_LE((center_undistorted - center).norm(), 15.f);

    std::cout << center_distorted.transpose() << std::endl;
    std::cout << center_undistorted.transpose() << std::endl;
  }

  // Check the corners.
  const auto corners = std::array{
    Eigen::Vector2f{0, 0},
    Eigen::Vector2f{w, 0},
    Eigen::Vector2f{w, h},
    Eigen::Vector2f{0, h},
  };
  for (const auto& c: corners)
  {
    // Check that the corners are behind the cameras
    const auto ray = camera.backproject(c);
    // This is a non-negotiable check.
    BOOST_CHECK(ray.z() < 0);

    // The reprojected ray must hit the center of the image.
    const auto projected_ray = camera.project(ray);
    // We allow ourselves to set generous thresholds for the reprojection of
    // rays because the corners are extreme cases.
    BOOST_CHECK_LE((projected_ray - c).norm(), 6.f);
  }
}
