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

#include <boost/test/unit_test.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_pitch_rotation)
{
  const float theta = 5._deg;
  auto r = Eigen::AngleAxisf{theta, Eigen::Vector3f::UnitY()};
  std::cout << r.matrix() << std::endl;
}
