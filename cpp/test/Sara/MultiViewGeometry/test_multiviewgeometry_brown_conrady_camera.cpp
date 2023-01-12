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

#define BOOST_TEST_MODULE                                                      \
  "MultiViewGeometry/Brown-Conrady Camera Distortion Model"

#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyDistortionModel.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>

#include <boost/test/unit_test.hpp>

#include <array>


using namespace DO::Sara;


auto make_gopro4_camera()
{
  auto camera = BrownConradyCamera32<float>{};

  const auto w = 1920;
  const auto h = 1080;

  camera.image_sizes << w, h;

  // clang-format off
  // Calibration matrix.
  camera.set_calibration_matrix((Eigen::Matrix3f{} <<
    8.7217820124018249e+02f,                        0, 960,
                          0., 8.7432544275392024e+02f, 540,
                          0,                        0,   1).finished());

  // Distortion model.
  camera.distortion_model.k <<
    -2.9131825247866094e-01f,
    1.1617227061207483e-01f,
    1.3047182418863507e-02f;
  camera.distortion_model.p <<
    4.7283170922189409e-03f,
    2.0814815668955206e-03f;
  // clang-format on

  return camera;
}


BOOST_AUTO_TEST_CASE(test_brown_conrady_camera_model)
{
  const auto camera = make_gopro4_camera();
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

  // Because the distortion model is not a very accurate mathematical model, we
  // cannot use the the image corners. So we use the points closer to the image
  // center.
  //
  // So camera correction model should be used instead.
  const auto points =
      std::array<Eigen::Vector2f, 4>{Eigen::Vector2f{w * 0.25f, h * 0.25f},
                                     Eigen::Vector2f{w * 0.75f, h * 0.25f},
                                     Eigen::Vector2f{w * 0.75f, h * 0.75f},
                                     Eigen::Vector2f{w * 0.25f, h * 0.75f}};

  for (const auto& pd : points)
  {
    // Check that the corners are behind the cameras
    const auto ray = camera.backproject(pd);
    // This is a non-negotiable check.
    BOOST_CHECK(ray.z() > 0);

    // The reprojected ray must be imaged in the distorted point.
    const auto imaged_pixel_coords = camera.project(ray);
    // We allow ourselves to set generous thresholds for the reprojection of
    // rays because the corners are extreme cases.
    BOOST_CHECK_LE((imaged_pixel_coords - pd).norm(), 1e-3f);
    const auto pu = camera.undistort(pd);
    const auto pd2 = camera.distort(pu);

    BOOST_CHECK_LE((pd2 - pd).norm(), 1e-3f);

#ifdef DEBUG_ME
    SARA_CHECK(ray.transpose());
    SARA_CHECK(imaged_pixel_coords.transpose());
    SARA_CHECK(pd.transpose());
    SARA_CHECK(pu.transpose());
    SARA_CHECK(pd2.transpose());
    SARA_CHECK("");
#endif
  }
}

BOOST_AUTO_TEST_CASE(test_brown_conrady_camera_model_v2)
{
  auto camera = v2::BrownConradyDistortionModel<float>{};

  // Calibration matrix.
  camera.fx() = 8.7217820124018249e+02f;
  camera.fy() = 8.7432544275392024e+02f;
  camera.shear() = 0.f;
  camera.u0() = 960.f;
  camera.v0() = 540.f;

  // Distortion model.
  // clang-format off
  camera.k() <<
    -2.9131825247866094e-01f,
    1.1617227061207483e-01f,
    1.3047182418863507e-02f;
  camera.p() <<
    4.7283170922189409e-03f,
    2.0814815668955206e-03f;
  // clang-format on

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

}
