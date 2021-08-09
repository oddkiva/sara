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

#define BOOST_TEST_MODULE "Geometry/Rotation 3D"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/Core/PhysicalQuantities.hpp>

#include <boost/test/unit_test.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_yaw_rotation)
{
  const double psi = 5._deg;
  const auto r = yaw(psi);
  BOOST_CHECK_LE((r.col(0) - Eigen::Vector3d(cos(psi), sin(psi), 0)).norm(),
                 1e-6f);
  BOOST_CHECK_LE((r.col(1) - Eigen::Vector3d(-sin(psi), cos(psi), 0)).norm(),
                 1e-6f);
  BOOST_CHECK_LE((r.col(2) - Eigen::Vector3d::UnitZ()).norm(), 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_pitch_rotation)
{
  const double theta = 5._deg;
  const auto r = pitch(theta);
  BOOST_CHECK_LE(
      (r.col(0) - Eigen::Vector3d(cos(theta), 0, -sin(theta))).norm(), 1e-6);
  BOOST_CHECK_LE((r.col(1) - Eigen::Vector3d::UnitY()).norm(), 1e-6);
  BOOST_CHECK_LE((r.col(2) - Eigen::Vector3d(sin(theta), 0, cos(theta))).norm(),
                 1e-6);
}

BOOST_AUTO_TEST_CASE(test_roll_rotation)
{
  const double phi = 5._deg;
  const auto r = roll(phi);
  BOOST_CHECK_LE((r.col(0) - Eigen::Vector3d::UnitX()).norm(), 1e-6);
  BOOST_CHECK_LE((r.col(1) - Eigen::Vector3d(0, cos(phi), sin(phi))).norm(),
                 1e-6);
  BOOST_CHECK_LE((r.col(2) - Eigen::Vector3d(0, -sin(phi), cos(phi))).norm(),
                 1e-6);
}

BOOST_AUTO_TEST_CASE(test_yaw_then_pitch_then_roll_rotation)
{
  const double psi = 5._deg;
  const double theta = 5._deg;
  const double phi = 5._deg;
  const auto r = rotation(psi, theta, phi);
  for (auto i = 0; i < 3; ++i)
    BOOST_CHECK_GE(r(i, i), 0.9);
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 3; ++j)
      if (i != j)
        BOOST_CHECK_LE(std::abs(r(i, j)), 0.1);
}
