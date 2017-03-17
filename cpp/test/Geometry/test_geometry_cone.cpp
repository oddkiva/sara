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

#define BOOST_TEST_MODULE "Geometry/Objects/Affine Cone"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects/Cone.hpp>

#include "../AssertHelpers.hpp"

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForAffineCone : public TestFixtureForPolygon
{
protected:
  // Generators of the 2D cone
  Vector2d _alpha;
  Vector2d _beta;

  TestFixtureForAffineCone() : TestFixtureForPolygon()
  {
    _alpha << 1, 0;
    _beta << 1, 1;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestAffineCone, TestFixtureForAffineCone)

BOOST_AUTO_TEST_CASE(test_convex_affine_cone)
{
  const auto K = AffineCone2{_alpha, _beta, _center, AffineCone2::Convex};

  auto convex_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto convex_ground_truth = [&](const Point2d& p) {
    return
      p.x() > _width/2. &&
      p.y() > _height/2. &&
      p.x() > p.y();
  };

  sweep_check(convex_predicate, convex_ground_truth);
}


BOOST_AUTO_TEST_CASE(test_blunt_affine_cone)
{
  const auto K = AffineCone2{_alpha, _beta, _center, AffineCone2::Blunt};

  auto blunt_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto blunt_ground_truth = [&](const Point2d& p) {
    return
      p.x() >= _width/2. &&
      p.y() >= _height/2. &&
      p.x() >= p.y();
  };

  sweep_check(blunt_predicate, blunt_ground_truth);
}


BOOST_AUTO_TEST_CASE(test_convex_pointed_affine_cone)
{
  AffineCone2 K(_alpha, _alpha, _center, AffineCone2::Convex);

  auto convex_pointed_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto convex_pointed_ground_truth = [&](const Point2d& p) {
    (void) p;
    return false;
  };

  sweep_check(convex_pointed_predicate, convex_pointed_ground_truth);

  AffineCone2 K2 { affine_cone2(0, 0, _center)};
  BOOST_CHECK_CLOSE_L2_DISTANCE(K.basis(), K2.basis(), 1e-6);
  BOOST_CHECK_EQUAL(K.vertex(), K2.vertex());
}


BOOST_AUTO_TEST_CASE(test_blunt_pointed_cone)
{
  const auto K = AffineCone2{_alpha, _alpha, _center, AffineCone2::Blunt};

  auto blunt_pointed_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto blunt_pointed_ground_truth = [&](const Point2d& p) {
    return
      p.x() >= _width/2. &&
      p.y() == _height/2.;
  };

  sweep_check(blunt_pointed_predicate, blunt_pointed_ground_truth);
}


// Degenerate case where the affine cone is actually empty.
BOOST_AUTO_TEST_CASE(test_degenerate_convex_affine_cone)
{
  const auto K = AffineCone2{_alpha, -_alpha, _center, AffineCone2::Convex};

  auto convex_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto convex_ground_truth = [&](const Point2d& p) {
    (void) p;
    return false;
  };

  sweep_check(convex_predicate, convex_ground_truth);
}

// Degenerate case where the affine cone is actually a half-space.
BOOST_AUTO_TEST_CASE(test_degenerate_blunt_pointed_affine_cone)
{
  const auto K = AffineCone2{_alpha, -_alpha, _center, AffineCone2::Blunt};

  auto blunt_pointed_predicate = [&](const Point2d& p) {
    return K.contains(p);
  };

  auto blunt_pointed_ground_truth = [&](const Point2d& p) {
    return p.y() == _height/2.;
  };

  sweep_check(blunt_pointed_predicate, blunt_pointed_ground_truth);
}

BOOST_AUTO_TEST_SUITE_END()
