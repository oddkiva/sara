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

#define BOOST_TEST_MODULE "FeatureDetectors/Difference of Gaussians Detector"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/FeatureDetectors/DoG.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestExtremumRefinement)

BOOST_AUTO_TEST_CASE(test_on_edge)
{
  // TODO.
}

BOOST_AUTO_TEST_CASE(test_refine_extremum)
{
  // TODO.
}

BOOST_AUTO_TEST_CASE(test_local_scale_space_extrema)
{
  // TODO.
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestDoG)

BOOST_AUTO_TEST_CASE(test_compute_dog_extrema)
{
  // Create a centered gaussian.
  constexpr auto N = 2 * 5 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);

  const auto xc = N / 2;
  const auto yc = N / 2;
  const auto r = 2;
  for (int y = yc - r; y <= yc + r; ++y)
    for (int x = xc - r; x <= xc + r; ++x)
      I(x, y) = 1;

  using namespace std;

  // Create the detector of DoG extrema.
  static constexpr auto scale_count = 3;
  const auto pyramid_params = ImagePyramidParams{
    0,
    scale_count + 3,
    std::pow(2.f, 1.f / scale_count),
    1, // image border size
    1.f, // camera scale
    1.6f}; // initial scale of the pyramid
  auto compute_DoGs = ComputeDoGExtrema{pyramid_params, 1e-6f, 1e-6f};

  auto scale_octave_pairs = vector<Point2i>{};
  auto features = compute_DoGs(I, &scale_octave_pairs);
  const auto& o_index = scale_octave_pairs[0](1);

  BOOST_REQUIRE(!features.empty());

  // There should be only one extrema at only one scale.
  SARA_CHECK(features.size());
  for (const auto& f: features)
    SARA_DEBUG << f << std::endl;

  // N.B.: the other are detected at the corners if we use Halide
  // implementation, these are artefacts because of the boundary checks... It
  // should not matter too much anyways...

  // The first is the one we want anyways.
  const auto& f = features.front();
  const auto& D = compute_DoGs.diff_of_gaussians();
  const auto z = static_cast<float>(D.octave_scaling_factor(o_index));

  BOOST_CHECK_SMALL(f.x() * z - xc, 1e-2f);
  BOOST_CHECK_SMALL(f.y() * z - yc, 1e-2f);

//  SARA_CHECK(f.extremum_value);
//  SARA_CHECK(f.scale() * z * M_SQRT2);  // Estimate of the blob radius.
//  SARA_CHECK((f.center() * z).transpose());
}

BOOST_AUTO_TEST_SUITE_END()
