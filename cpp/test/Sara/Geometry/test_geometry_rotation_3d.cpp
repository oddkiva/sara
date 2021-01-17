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
#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>

#include <boost/test/unit_test.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_yaw_rotation)
{
  const float psi = 5._deg;
  const auto r = yaw(psi);
  BOOST_CHECK_LE(
      (r.col(0) - Eigen::Vector3f(cos(psi), sin(psi), 0)).norm(), 1e-6f);
  BOOST_CHECK_LE(
      (r.col(1) - Eigen::Vector3f(-sin(psi), cos(psi), 0)).norm(), 1e-6f);
  BOOST_CHECK_LE((r.col(2) - Eigen::Vector3f::UnitZ()).norm(), 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_pitch_rotation)
{
  const float theta = 5._deg;
  const auto r = pitch(theta);
  BOOST_CHECK_LE(
      (r.col(0) - Eigen::Vector3f(cos(theta), 0, -sin(theta))).norm(), 1e-6f);
  BOOST_CHECK_LE((r.col(1) - Eigen::Vector3f::UnitY()).norm(), 1e-6f);
  BOOST_CHECK_LE(
      (r.col(2) - Eigen::Vector3f(sin(theta), 0, cos(theta))).norm(), 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_roll_rotation)
{
  const float phi = 5._deg;
  const auto r = roll(phi);
  BOOST_CHECK_LE((r.col(0) - Eigen::Vector3f::UnitX()).norm(), 1e-6f);
  BOOST_CHECK_LE(
      (r.col(1) - Eigen::Vector3f(0, cos(phi), sin(phi))).norm(), 1e-6f);
  BOOST_CHECK_LE(
      (r.col(2) - Eigen::Vector3f(0, -sin(phi), cos(phi))).norm(), 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_yaw_then_pitch_then_roll_rotation)
{
  const float psi = 5._deg;
  const float theta = 5._deg;
  const float phi = 5._deg;
  const auto r = rotation(psi, theta, phi);
  for (auto i = 0; i < 3; ++i)
    BOOST_CHECK_GE(r(i, i), 0.9f);
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 3; ++j)
      if (i != j)
        BOOST_CHECK_LE(std::abs(r(i, j)), 0.1f);
}
