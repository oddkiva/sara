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

#define BOOST_TEST_MODULE "Geometry/Objects/Ellipse"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Geometry/Objects/CSG.hpp>
#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/Triangle.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Geometry/Algorithms/EllipseIntersection.hpp>

#include "../AssertHelpers.hpp"

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;
namespace sara = DO::Sara;


class TestFixtureForEllipse : public TestFixtureForPolygon
{
public:
  TestFixtureForEllipse()
    : TestFixtureForPolygon()
  {
    _width = 300;
    _height = 300;
  }

protected:
  double _a{250.};
  double _b{150.};
  double _theta_degree{30.};
};


BOOST_FIXTURE_TEST_SUITE(TestEllipse, TestFixtureForEllipse)

BOOST_AUTO_TEST_CASE(test_rho)
{
  const auto E = sara::Ellipse{_a, _b, to_radian(_theta_degree), _center};

  // Change of variable in polar coordinates.
  // x = rho * cos(theta)
  // y = rho * sin(theta)
  auto expected_rho = [&](double theta) {
    const auto c = cos(theta);
    const auto s = sin(theta);
    const auto r = (_a * _b) / sqrt(_b * _b * c * c + _a * _a * s * s);
    return Vector2d{r * unit_vector2(theta)};
  };

  constexpr auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = 2 * M_PI * i / N;
    BOOST_REQUIRE_SMALL_L2_DISTANCE(E.rho(theta), expected_rho(theta), 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_overloaded_operator)
{
  const auto E = sara::Ellipse{_a, _b, to_radian(_theta_degree), _center};

  // Change of variable in polar coordinates.
  // x = rho * cos(theta)
  // y = rho * sin(theta)
  auto expected_rho = [&](double theta) {
    const auto c = cos(theta);
    const auto s = sin(theta);
    const auto rho = (_a * _b) / sqrt(_b * _b * c * c + _a * _a * s * s);
    const Matrix2d R{rotation2(to_radian(_theta_degree))};
    return Vector2d{_center + R * rho * unit_vector2(theta)};
  };

  constexpr auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = 2 * M_PI * i / N;
    BOOST_REQUIRE_SMALL_L2_DISTANCE(E(theta), expected_rho(theta), 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_orientation)
{
  const auto E = sara::Ellipse{_a, _b, to_radian(_theta_degree), _center};

  const auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = -M_PI + 2 * M_PI * i / N;
    const Point2d p{_center + unit_vector2(theta + to_radian(_theta_degree))};
    BOOST_CHECK_SMALL(orientation(p, E) - theta, 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_ellipse_ostream)
{
  const auto E = sara::Ellipse{250, 150, to_radian(_theta_degree), _center};

  stringstream buffer;
  CoutRedirect cout_redirect{buffer.rdbuf()};
  cout << E << endl;

  auto text = buffer.str();

  BOOST_CHECK(text.find("a = ") != string::npos);
  BOOST_CHECK(text.find("b = ") != string::npos);
  BOOST_CHECK(text.find("o = ") != string::npos);
  BOOST_CHECK(text.find("c = ") != string::npos);
}

 BOOST_AUTO_TEST_CASE(test_sector_area)
{
  const auto E = sara::Ellipse{_a, _b, to_radian(_theta_degree), _center};

  const auto ell = CSG::Singleton<sara::Ellipse>{E};

  try
  {
    const auto steps = 18;
    const auto i0 = 0;
    const auto theta_0 = 2 * M_PI * i0 / steps;
    const auto num_tests = 2;

    for (int i1 = i0 + 1; i1 < i0 + num_tests; ++i1)
    {
      auto theta_1 = 2 * M_PI * i1 / steps;

      // Compute the sector area in a closed form.
      double analytic_sector_area = sector_area(E, theta_0, theta_1);

      // Build constructive solid geometry.
      CSG::Singleton<AffineCone2> cone{affine_cone2(
          theta_0 + E.orientation(), theta_1 + E.orientation(), E.center())};
      auto E_and_Cone = ell * cone;
      auto E_minus_Cone = ell - E_and_Cone;

      // Use the CSGs to estimate the sector area by pixel counting.
      auto inside_E_and_Cone = [&](const Point2d& p) {
        return E_and_Cone.contains(p);
      };

      auto inside_E_minus_Cone = [&](const Point2d& p) {
        return E_minus_Cone.contains(p);
      };

      // Use the lambda functors to estimate the elliptic sector area.
      auto estimated_sector_area = double{};
      if (i1 - i0 < steps / 2)
        estimated_sector_area = sweep_count_pixels(inside_E_and_Cone);
      else if (abs(i1 - i0) == steps / 2)
        estimated_sector_area = area(E) / 2;
      else
        estimated_sector_area = sweep_count_pixels(inside_E_minus_Cone);

      // Absolute error and relative error.
      const auto thres = 1e-1;
      BOOST_CHECK_CLOSE(estimated_sector_area, analytic_sector_area, thres);
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
}

BOOST_AUTO_TEST_CASE(test_segment_area)
{
  const auto E = sara::Ellipse{_a, _b, to_radian(_theta_degree), _center};

  try
  {
    const auto steps = 18;
    const auto i0 = 0;
    const auto theta_0 = 2 * M_PI * i0 / steps;
    const auto num_tests = 2;

    for (int i1 = i0 + 1; i1 < i0 + num_tests; ++i1)
    {
      const auto theta_1 = 2 * M_PI * i1 / steps;

      const Point2d a{E(theta_0)};
      const Point2d b{E(theta_1)};

      auto inside_segment = [&](const Point2d& p) {
        return (ccw(a, b, p) == -1) && E.contains(p);
      };

      const auto seg_area1 = sweep_count_pixels(inside_segment);
      const auto seg_area2 = segment_area(E, theta_0, theta_1);

      const auto thres = 0.5;
      BOOST_CHECK_CLOSE(seg_area1, seg_area2, thres);
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
}

BOOST_AUTO_TEST_CASE(test_construct_from_shape_matrix)
{
  const auto E =
      construct_from_shape_matrix(Matrix2d::Identity(), Vector2d::Zero());
  BOOST_CHECK_CLOSE_L2_DISTANCE(shape_matrix(E), Matrix2d::Identity(), 1e-8);
  BOOST_CHECK_CLOSE(E.radius1(), 1., 1e-6);
  BOOST_CHECK_CLOSE(E.radius2(), 1., 1e-6);
  BOOST_CHECK_SMALL_L2_DISTANCE(E.center(), Point2d::Zero(), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_oriented_bbox)
{
  const auto E = sara::Ellipse{_a, _b, 0, Point2d::Zero()};
  const auto actual_bbox = oriented_bbox(E);
  const auto expected_bbox = Quad{BBox{-Point2d{_a, _b}, Point2d{_a, _b}}};
  for (int i = 0; i < 4; ++i)
    BOOST_CHECK_SMALL_L2_DISTANCE(expected_bbox[i], actual_bbox[i], 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestEllipseIntersection)

BOOST_AUTO_TEST_CASE(test_first_ellipse_is_inside_the_second_ellipse)
{
  const auto e1 = sara::Ellipse{100, 100, 0, Point2d::Zero()};
  const auto e2 = sara::Ellipse{200, 200, 0, Point2d::Zero()};

  const auto intersection_points = sara::compute_intersection_points(e1, e2);
  BOOST_CHECK_EQUAL(intersection_points.size(), 0u);

  const auto intersection_area = sara::analytic_intersection_area(e1, e2);
  const auto e1_area = area(e1);
  BOOST_CHECK_CLOSE(intersection_area, e1_area, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_two_intersection_points)
{
  const auto e1 = sara::Ellipse{10, 10, 0, Point2d::Zero()};
  const auto e2 = sara::Ellipse{10, 20, 0, 8 * Point2d::Ones()};

  const auto intersection_points = sara::compute_intersection_points(e1, e2);
  BOOST_CHECK_EQUAL(intersection_points.size(), 2u);

  const auto m1 = shape_matrix(e1);
  const auto m2 = shape_matrix(e2);
  for (auto i = 0u; i < intersection_points.size(); ++i)
  {
    const auto& x = intersection_points[i];

    const auto diff1 =
        std::abs((x - e1.center()).transpose() * m1 * (x - e1.center()) - 1);
    const auto diff2 =
        std::abs((x - e2.center()).transpose() * m2 * (x - e2.center()) - 1);

    BOOST_CHECK_SMALL(diff1, 1e-6);
    BOOST_CHECK_SMALL(diff2, 1e-6);
  }

  const auto intersection_area = sara::analytic_intersection_area(e1, e2);
  const auto intersection_area_approx = sara::area(sara::approximate_intersection(e1, e2, 720));

  BOOST_CHECK_CLOSE(intersection_area, intersection_area_approx, 1e-2);
}

BOOST_AUTO_TEST_CASE(test_four_intersection_points)
{
  const auto e1 = sara::Ellipse{10, 10, 0, Point2d::Zero()};
  const auto e2 = sara::Ellipse{5, 30, 0, Point2d::Zero()};

  const auto intersection_points = sara::compute_intersection_points(e1, e2);
  BOOST_CHECK_EQUAL(intersection_points.size(), 4u);

  const auto intersection_area = sara::analytic_intersection_area(e1, e2);
  const auto intersection_area_approx =
      sara::area(sara::approximate_intersection(e1, e2, 720));

  BOOST_CHECK_CLOSE(intersection_area, intersection_area_approx, 1e-2);
}

BOOST_AUTO_TEST_SUITE_END()