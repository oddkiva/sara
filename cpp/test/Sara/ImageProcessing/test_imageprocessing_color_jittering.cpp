// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Color Jittering"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/ColorJitter.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestNormalDistribution)

BOOST_AUTO_TEST_CASE(test_random_seed)
{
  auto dist = NormalDistribution{false};
  auto dist2 = NormalDistribution{true};

  auto samples1 = vector<float>{};
  auto samples2 = vector<float>{};
  auto sample1_i = back_inserter(samples1);
  auto sample2_i = back_inserter(samples2);

  for (int i = 0; i < 10; ++i)
  {
    *sample1_i++ = dist();
    *sample2_i++ = dist2();
  }

  BOOST_CHECK_EQUAL(samples1.size(), 10u);
  BOOST_CHECK_EQUAL(samples2.size(), 10u);
  BOOST_CHECK(samples1 != samples2);
}

BOOST_AUTO_TEST_CASE(test_sampling)
{
  auto dist = NormalDistribution{false};

  auto samples = vector<float>{};
  auto sample_i = back_inserter(samples);
  for (int i = 0; i < 10; ++i)
    *sample_i++ = dist();

  BOOST_CHECK_EQUAL(samples.size(), 10u);
  for (const auto& s : samples)
    BOOST_CHECK_NE(s, 0.f);
}

BOOST_AUTO_TEST_CASE(test_randn_vector)
{
  const auto randn = NormalDistribution{false};
  auto v = Vector3f{};
  randn(v);
  BOOST_CHECK(v != Vector3f::Zero());

  auto v2 = MatrixXf{3, 1};
  randn(v2);
  BOOST_CHECK(v2 != Vector3f::Zero());
}

BOOST_AUTO_TEST_CASE(test_randn_image)
{
  const auto randn = NormalDistribution{false};
  auto image = Image<Rgb32f>{5, 5};
  image.flat_array().fill(Rgb32f::Zero());

  randn(image);
  for (const auto& pixel : image)
    BOOST_CHECK(pixel != Vector3f::Zero());
}

BOOST_AUTO_TEST_CASE(test_add_randn_noise)
{
  auto image = Image<Rgb32f>{5, 5};
  image.flat_array().fill(Rgb32f::Zero());

  const auto randn = NormalDistribution{false};
  add_randn_noise(image, 0.5, randn);
  for (const auto& pixel : image)
    BOOST_CHECK(pixel != Vector3f::Zero());
}

BOOST_AUTO_TEST_SUITE_END()
