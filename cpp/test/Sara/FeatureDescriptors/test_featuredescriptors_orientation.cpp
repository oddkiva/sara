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

#define BOOST_TEST_MODULE "FeatureDescriptors/Orientation"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/FeatureDescriptors/Orientation.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestComputeDominantOrientations)

BOOST_AUTO_TEST_CASE(test_lowe_smooth_histogram)
{
  constexpr auto O = 36;
  auto h = Array<float, O, 1>{};
  h.fill(0);
  h(0, 0) = 1;
  h(14, 0) = 1;

  SARA_DEBUG << "Before box blur" << std::endl;
  SARA_DEBUG << h.transpose() << std::endl;

  const auto num_iters = 1;
  lowe_smooth_histogram(h, num_iters);

  SARA_DEBUG << "After box blur" << std::endl;
  SARA_DEBUG << h.transpose() << std::endl;

  BOOST_CHECK_CLOSE(h(35, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h(0, 0),  1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h(1, 0),  1. / 3., 1e-3f);

  BOOST_CHECK_CLOSE(h(13, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h(14, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h(15, 0), 1. / 3., 1e-3f);
}

BOOST_AUTO_TEST_CASE(test_orientation_histogram)
{
  const int N{5};
  const int M{N * N - 1};

  auto grad_polar_coords = Image<Vector2f>{N, N};
  const Point2f c{grad_polar_coords.sizes().cast<float>() / 2.f};

  for (int gy = 0; gy < grad_polar_coords.height(); ++gy)
  {
    for (int gx = 0; gx < grad_polar_coords.width(); ++gx)
    {
      const auto theta = [&]() {
        auto t = atan2(gy - c.y(), gx - c.x());
        if (t < 0)
          t += 2 * float(M_PI);
        return t;
      }();
      const auto theta_bin = int(floor(theta / float(2 * M_PI) * M)) % M;

      // Set all gradients to zero.
      for (int y = 0; y < grad_polar_coords.height(); ++y)
        for (int x = 0; x < grad_polar_coords.width(); ++x)
          grad_polar_coords(x, y) = Vector2f::Zero();

      // Except at coords (gx, gy).
      grad_polar_coords(gx, gy) = Vector2f{1.f, theta};

      Array<float, M, 1> histogram;
      compute_orientation_histogram(histogram, grad_polar_coords, c.x(), c.y(),
                                    1.f);
      histogram /= histogram.sum();

      Matrix<float, M, 1> expected_histogram;
      expected_histogram.setZero();
      expected_histogram[theta_bin] = 1.f;

      const auto &expe= expected_histogram;
      const auto &calc= histogram;
      SARA_CHECK(expe.transpose());
      SARA_CHECK(calc.transpose());
      std::cout << std::endl;

      BOOST_REQUIRE_SMALL((expected_histogram - histogram.matrix()).norm(),
                          1e-6f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_detect_single_peak)
{
  const int N{5};

  auto grad_polar_coords = Image<Vector2f>{N, N};
  const Point2f c{grad_polar_coords.sizes().cast<float>() / 2.f};

  const Vector2f g{Vector2f::Zero()};
  const auto theta = atan2(0 - c.y(), 0 - c.x());

  // Set all gradients to zero except at coords (gx, gy).
  for (int y = 0; y < grad_polar_coords.height(); ++y)
    for (int x = 0; x < grad_polar_coords.width(); ++x)
      grad_polar_coords(x, y) = Vector2f::Zero();
  grad_polar_coords(0, 0) = Vector2f{1.f, theta};

  auto dominant_orientations =
      ComputeDominantOrientations{}(grad_polar_coords, c.x(), c.y(), 1.f);

  BOOST_CHECK_EQUAL(dominant_orientations.size(), 1u);

  auto& ori = dominant_orientations.front();
  BOOST_CHECK_SMALL(theta - ori, 1e-6f);
}

BOOST_AUTO_TEST_SUITE_END()
