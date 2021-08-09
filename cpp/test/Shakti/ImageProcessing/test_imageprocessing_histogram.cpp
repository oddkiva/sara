// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Histogram"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>

#include <DO/Shakti/Cuda/ImageProcessing/Histogram.hpp>

#include "../AssertHelpers.hpp"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace sara;


BOOST_AUTO_TEST_CASE(test_histogram)
{
  using sara::Rgba8;

  auto image = sara::Image<Rgba8>{ 5, 5 };
  image.flat_array().fill(Rgba8::Zero());

  image(0, 0) = Rgba8(1, 1, 1, 0);
  image(1, 0) = Rgba8(1, 1, 1, 0);
  image(2, 0) = Rgba8(16, 0, 0, 0);
  image(4, 4) = Rgba8(17, 0, 0, 0);

  auto quantization_step = 8;
  auto histogram_sizes = Vector3i::Ones() * 256 / quantization_step;
  auto histogram = sara::Image<float, 3>{ histogram_sizes };

  shakti::compute_color_distribution(
    histogram.data(), reinterpret_cast<shakti::Vector4ub *>(image.data()),
    image.sizes().data(), quantization_step);

  for (int z = 0; z < histogram.depth(); ++z)
  {
    for (int y = 0; y < histogram.height(); ++y)
    {
      for (int x = 0; x < histogram.width(); ++x)
      {
        const auto coord = Vector3i{ x, y, z };
        if (coord == Vector3i::Zero())
          BOOST_REQUIRE_CLOSE(23.f/(image.size()), histogram(x,y,z), 1e-6f);
        else if (coord == Vector3i{ 2, 0, 0 })
          BOOST_REQUIRE_CLOSE(2.f/(image.size()), histogram(x,y,z), 1e-6f);
        else
          BOOST_REQUIRE_CLOSE(0.f, histogram(coord), 1e-6f);
      }
    }
  }
}
