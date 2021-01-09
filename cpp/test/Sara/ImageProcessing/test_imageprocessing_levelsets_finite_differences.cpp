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

#define BOOST_TEST_MODULE "ImageProcessing/Level Sets/Finite Differences"

#include <DO/Sara/Core/MultiArray/Padding.hpp>
#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/FiniteDifferences.hpp>

#include <boost/test/unit_test.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;

class TestFixtureForDifferentialOperators
{
protected:
  Image<float> _src_image;
  vector<float> _kernel;

public:
  TestFixtureForDifferentialOperators()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() <<
        1, 2, 3,  //
        1, 2, 3,  //
        1, 2, 3;  //

    _kernel.resize(3);
    _kernel[0] = -1. / 2;
    _kernel[1] = 0;
    _kernel[2] = 1. / 2;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestDifferentialOperator,
                         TestFixtureForDifferentialOperators)

BOOST_AUTO_TEST_CASE(test_centered_finite_differences)
{
  auto& f = _src_image;
  Vector2i x{1, 1};

  Vector2f gradf_x{
    CentralDifference::at(f, x, 0),
    CentralDifference::at(f, x, 1)
  };
  BOOST_CHECK_CLOSE_L2_DISTANCE(Vector2f(1, 0), gradf_x, 1e-5f);

  auto gradf = gradient(f);
  auto gradf_2 = f.compute<Gradient>();
  for (int y = 0; y < gradf.height(); ++y)
  {
    for (int x = 0; x < gradf.width(); ++x)
    {
      auto true_gradf = Vector2f{};
      true_gradf[0] = x == 1 ? 1 : 0.5f;
      true_gradf[1] = 0;

      const auto& gradf_xy = gradf(x, y);
      const auto& gradf_xy_2 = gradf_2(x, y);

      BOOST_CHECK_CLOSE_L2_DISTANCE(true_gradf, gradf_xy, 1e-5f);
      BOOST_CHECK_CLOSE_L2_DISTANCE(true_gradf, gradf_xy_2, 1e-5f);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
