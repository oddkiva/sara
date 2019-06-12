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

#define BOOST_TEST_MODULE "Geometry/Tools/Metric"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Geometry/Tools/Metric.hpp>

#include "../AssertHelpers.hpp"


using namespace DO::Sara;

using DistanceTypes =
    boost::mpl::list<SquaredRefDistance<float, 2>, SquaredDistance<float, 2>>;

BOOST_AUTO_TEST_SUITE(TestSquaredDistanceAndOpenBall)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_computations, Distance, DistanceTypes)
{
  static_assert(Distance::Dim == 2, "Wrong dimension");

  const Matrix2f A = Matrix2f::Identity();

  const Vector2f a = Vector2f::Zero();
  const Vector2f b = Vector2f{1.f, 0.f};
  const Vector2f c = Vector2f{0.f, 1.f};

  const auto d = Distance{A};
  BOOST_CHECK_EQUAL(d.covariance_matrix(), A);
  BOOST_CHECK(is_quasi_isotropic(d));
  BOOST_CHECK_CLOSE(d(b, a), A(0, 0), 1e-6f);
  BOOST_CHECK_CLOSE(d(c, a), A(1, 1), 1e-6f);

  const auto ball = OpenBall<Distance>{Point2f::Zero(), 1.1f, d};
  BOOST_CHECK_EQUAL(ball.center(), Point2f::Zero());
  BOOST_CHECK_EQUAL(ball.radius(), 1.1f);
  BOOST_CHECK_EQUAL(ball.squared_distance().covariance_matrix(), A);
  BOOST_CHECK(ball.contains(a));
  BOOST_CHECK(ball.contains(b));
  BOOST_CHECK(ball.contains(c));
}

BOOST_AUTO_TEST_SUITE_END()
