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

#define BOOST_TEST_MODULE "ImageProcessing/Level Sets/Narrow Band"

#include <DO/Sara/ImageProcessing/LevelSets/NarrowBand.hpp>

#include "../AssertHelpers.hpp"

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


auto radial_distance(sara::Image<float>& phi,        //
                     const Eigen::Vector2f& center,  //
                     float radius)
{
  for (auto y = 0; y < phi.height(); ++y)
    for (auto x = 0; x < phi.width(); ++x)
      phi(x, y) = (Eigen::Vector2f(x, y) - center).norm() - radius;
}


BOOST_AUTO_TEST_CASE(test_narrow_band)
{
  // Load level set function
  const auto w = 128;
  const auto h = 128;
  auto phi = sara::Image<float, 2>(w, h);
  radial_distance(phi, Eigen::Vector2f(w, h) / 2, w / 5.);

  // Can we instantiate it?
  auto narrow_band = sara::NarrowBand<float, 2>{phi};
}
