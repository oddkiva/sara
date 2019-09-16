// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Local Extrema"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Extrema.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestLocalExtremum)

BOOST_AUTO_TEST_CASE(test_local_extremum)
{
  // Simple test case.
  Image<float> I(10,10);
  I.matrix().fill(1.f);

  // Local maximality and minimality
  CompareWithNeighborhood3<greater_equal, float> greater_equal33;
  CompareWithNeighborhood3<less_equal, float> less_equal33;
  // Strict local maximality and minimality
  CompareWithNeighborhood3<greater, float> greater33;
  CompareWithNeighborhood3<less, float> less33;

  // Check local maximality
  BOOST_CHECK(!greater33(I(1,1), 1, 1, I, true));
  BOOST_CHECK(!greater33(I(1,1), 1, 1, I, false));
  BOOST_CHECK(greater_equal33(I(1,1),1,1,I,true));
  BOOST_CHECK(greater_equal33(I(1,1),1,1,I,false));

  // Check local minimality
  BOOST_CHECK(!less33(I(1,1), 1, 1, I, true));
  BOOST_CHECK(!less33(I(1,1), 1, 1, I, false));
  BOOST_CHECK(less_equal33(I(1,1),1,1,I,true));
  BOOST_CHECK(less_equal33(I(1,1),1,1,I,false));

  // Check that aliases are correctly defined.
  BOOST_CHECK(!StrictLocalMax<float>()(1, 1, I));
  BOOST_CHECK(!StrictLocalMin<float>()(1, 1, I));
  vector<Point2i> maxima;
  vector<Point2i> minima;

  maxima = strict_local_maxima(I);
  BOOST_CHECK(maxima.empty());
  maxima = local_maxima(I);
  BOOST_CHECK(maxima.size() == 8*8);

  minima = strict_local_minima(I);
  BOOST_CHECK(minima.empty());
  minima = local_minima(I);
  BOOST_CHECK(minima.size() == 8*8);

  I(1,1) = 10.f;
  I(7,7) = 10.f;
  BOOST_CHECK(greater33(I(1,1), 1, 1, I, false));
  BOOST_CHECK(!greater33(I(1,1), 1, 1, I, true));
  BOOST_CHECK(LocalMax<float>()(1, 1, I));
  BOOST_CHECK(StrictLocalMax<float>()(1, 1, I));
  BOOST_CHECK(!LocalMin<float>()(1, 1, I));
  BOOST_CHECK(!StrictLocalMin<float>()(1, 1, I));

  maxima = strict_local_maxima(I);
  BOOST_CHECK_EQUAL(maxima.size(), 2u);
  minima = strict_local_minima(I);
  BOOST_CHECK(minima.empty());

  I.matrix() *= -1;
  BOOST_CHECK(less33(I(1,1), 1, 1, I, false));
  BOOST_CHECK(!less33(I(1,1), 1, 1, I, true));
  BOOST_CHECK(StrictLocalMin<float>()(1, 1, I));
  BOOST_CHECK(LocalMin<float>()(1, 1, I));
  maxima = strict_local_maxima(I);
  minima = strict_local_minima(I);

  BOOST_CHECK(maxima.empty());
  BOOST_CHECK_EQUAL(minima.size(), 2u);
}

BOOST_AUTO_TEST_CASE(test_local_scale_space_extremum)
{
  ImagePyramid<double> I;
  I.reset(1,3,1.6f,pow(2., 1./3.));
  for (int i = 0; i < 3; ++i)
  {
    I(i,0).resize(10,10);
    I(i,0).matrix().fill(1);
  }
  BOOST_CHECK(!StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  BOOST_CHECK(!StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));

  // Local scale-space extrema test 1
  I(1,1,1,0) = 10.f;
  I(7,7,1,0) = 10.f;
  BOOST_CHECK(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  BOOST_CHECK(!StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));

  vector<Point2i> maxima, minima;
  maxima = strict_local_scale_space_maxima(I,1,0);
  minima = strict_local_scale_space_minima(I,1,0);
  BOOST_CHECK_EQUAL(maxima.size(), 2u);
  BOOST_CHECK(minima.empty());

  // Local scale-space extrema test 2
  I(1,1,1,0) *= -1.f;
  I(7,7,1,0) *= -1.f;
  maxima = strict_local_scale_space_maxima(I,1,0);
  minima = strict_local_scale_space_minima(I,1,0);
  BOOST_CHECK(!LocalScaleSpaceMax<double>()(1,1,1,0,I));
  BOOST_CHECK(!StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  BOOST_CHECK(LocalScaleSpaceMin<double>()(1,1,1,0,I));
  BOOST_CHECK(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));
  BOOST_CHECK(maxima.empty());
  BOOST_CHECK_EQUAL(minima.size(), 2u);
}

BOOST_AUTO_TEST_SUITE_END()
