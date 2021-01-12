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

#include <DO/Sara/ImageProcessing/LevelSets/FiniteDifferences.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/Flux.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/NarrowBand.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/TimeIntegrators.hpp>

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

  // Initialize the time integrator with the initial level set function.
  auto euler = sara::EulerIntegrator<float, 2>{phi};

  const auto thickness1 = 6.f;
  const auto thickness2 = 3.f;
  narrow_band.init<sara::Weno3>(thickness1, euler);

  for (auto t = 0; t < 100; ++t)
  {
    if (narrow_band.reinit_needed(thickness2))
      narrow_band.reinit<sara::Weno3>(thickness1, euler);

    for (auto p = narrow_band._band_map.begin_array(); !p.end(); ++p)
    {
      if (!*p)
        continue;
      euler._df(*p) = sara::mean_curvature_motion(phi, p.position());
      // euler._df(*p) = sara::normal_motion<sara::Weno3>(phi, p.position(), -0.1f);
      // euler._df(*p) = sara::normal_motion<sara::Weno3>(phi, p.position(), 0.1f);
      // euler._df(*p) = sara::advection<sara::Weno3>(phi, p.position(), {0.1f,  0.0f});
    }
  }
}
