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

#define BOOST_TEST_MODULE "Geometry/Objects/BBox"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Objects.hpp>

#include "TestPolygon.hpp"

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForBBox : TestFixtureForPolygon
{
};

BOOST_FIXTURE_TEST_SUITE(TestBBox, TestFixtureForPolygon)

BOOST_AUTO_TEST_CASE(test_constructor_and_accessors)
{
  BBox bbox(_p1, _p2);
  BOOST_CHECK_EQUAL(bbox.top_left(), Point2d(_a, _a));
  BOOST_CHECK_EQUAL(bbox.top_right(), Point2d(_b, _a));
  BOOST_CHECK_EQUAL(bbox.bottom_right(), Point2d(_b, _b));
  BOOST_CHECK_EQUAL(bbox.bottom_left(), Point2d(_a, _b));

  auto predicate = [&](const Point2d& p) { return inside(p, bbox); };
  auto groundTruth = [&](const Point2d& p) {
    return p.cwiseMin(_p1) == _p1 && p.cwiseMax(_p2) == _p2;
  };
  sweep_check(predicate, groundTruth);
}

BOOST_AUTO_TEST_CASE(test_constructor_from_point_set)
{
  Point2d points[] = {Point2d::Zero(), Point2d(_a, _a), _center};

  BBox bbox(points, points + 3);
  BOOST_CHECK_EQUAL(bbox.top_left(), points[0]);
  BOOST_CHECK_EQUAL(bbox.bottom_right(), points[2]);

  vector<Point2d> points_vector(points, points + 3);
  BBox bbox2(points_vector);
  BOOST_CHECK_EQUAL(bbox.top_left(), bbox2.top_left());
  BOOST_CHECK_EQUAL(bbox.bottom_right(), bbox2.bottom_right());
}

BOOST_AUTO_TEST_CASE(test_point_inside_bbox)
{
  BBox bbox(_p1, _p2);

  Point2d points[] = {Point2d::Zero(), Point2d(_a, _a), _center};
  BOOST_CHECK(!inside(points[0], bbox));
  BOOST_CHECK(inside(points[1], bbox));
  BOOST_CHECK(inside(points[2], bbox));
}

BOOST_AUTO_TEST_CASE(test_bbox_ostream)
{
  const auto bbox = BBox{};

  stringstream buffer;
  CoutRedirect cout_redirect{buffer.rdbuf()};
  cout << bbox << endl;

  auto text = buffer.str();

  BOOST_CHECK(text.find("top-left: [") != string::npos);
  BOOST_CHECK(text.find("bottom-right: [") != string::npos);
}

BOOST_AUTO_TEST_CASE(test_bbox_degenerate)
{
  const auto bbox = BBox::zero();
  BOOST_CHECK(degenerate(bbox));
}

BOOST_AUTO_TEST_CASE(test_intersection)
{
  const auto b1 = BBox{Point2d{0, 0}, Point2d{1, 1}};
  const auto b2 = BBox{Point2d{0.5, 0.5}, Point2d{1.5, 1.5}};

  const auto inter = BBox{Point2d{0.5, 0.5}, Point2d{1, 1}};
  BOOST_CHECK(intersect(b1, b2));
  BOOST_CHECK_EQUAL(inter, intersection(b1, b2));

  auto expected_jaccard_distance =
      1 - area(inter) / (area(b1) + area(b2) - area(inter));
  BOOST_CHECK_EQUAL(expected_jaccard_distance, jaccard_distance(b1, b2));
}

BOOST_AUTO_TEST_SUITE_END()
