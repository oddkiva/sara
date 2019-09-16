// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "FeatureDescriptors/SIFT Descriptor"

#include <DO/Sara/FeatureDescriptors/SIFT.hpp>

#include <boost/test/unit_test.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestSIFTDescriptors)

BOOST_AUTO_TEST_CASE(test_computation)
{
  constexpr int N{5};

  auto grad_polar_coords = Image<Vector2f>{N, N};
  const Point2f c = grad_polar_coords.sizes().cast<float>() / 2.f;

  const auto theta = atan2(0 - c.y(), 0 - c.x());

  // Set all gradients to zero except at coords (gx, gy).
  for (int y = 0; y < grad_polar_coords.height(); ++y)
    for (int x = 0; x < grad_polar_coords.width(); ++x)
      grad_polar_coords(x, y) = Vector2f::Zero();
  grad_polar_coords(0, 0) = Vector2f{1.f, theta};

  auto feature = OERegion{c, 1.f};
  auto sift = ComputeSIFTDescriptor<>{}(feature, grad_polar_coords);

  BOOST_CHECK(sift.matrix() != decltype(sift)::Zero());
}

BOOST_AUTO_TEST_SUITE_END()
