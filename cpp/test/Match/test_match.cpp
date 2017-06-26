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

#define BOOST_TEST_MODULE "Match/Data Structures"

#include <limits>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Match.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestMatch)

BOOST_AUTO_TEST_CASE(test_default_constructor)
{
  auto m = Match{};
  BOOST_CHECK(m.x_pointer() == nullptr);
  BOOST_CHECK(m.y_pointer() == nullptr);
  BOOST_CHECK_THROW(m.x(), runtime_error);
  BOOST_CHECK_THROW(m.y(), runtime_error);
  BOOST_CHECK_EQUAL(m.x_index(), -1);
  BOOST_CHECK_EQUAL(m.y_index(), -1);
  BOOST_CHECK_EQUAL(m.score(), numeric_limits<float>::max());
  BOOST_CHECK_EQUAL(m.rank(), -1);
  BOOST_CHECK(m.matching_direction() == Match::Direction::SourceToTarget);

  const auto& m2 = m;
  BOOST_CHECK(m2.x_pointer() == nullptr);
  BOOST_CHECK(m2.y_pointer() == nullptr);
  BOOST_CHECK_THROW(m2.x(), runtime_error);
  BOOST_CHECK_THROW(m2.y(), runtime_error);
  BOOST_CHECK_EQUAL(m2.x_index(), -1);
  BOOST_CHECK_EQUAL(m2.y_index(), -1);
  BOOST_CHECK_EQUAL(m2.score(), numeric_limits<float>::max());
  BOOST_CHECK_EQUAL(m2.rank(), -1);
  BOOST_CHECK(m2.matching_direction() == Match::Direction::SourceToTarget);
}

BOOST_AUTO_TEST_CASE(test_custom_constructor)
{
  auto f_x = OERegion{ Point2f::Zero(), 1.f };
  f_x.orientation() = 0.f;
  f_x.type() = OERegion::Type::DoG;
  auto f_y = OERegion{ Point2f::Ones(), 1.f };
  f_y.orientation() = 0.f;
  f_y.type() = OERegion::Type::DoG;

  auto m = Match{ &f_x, &f_y, 0.5f };
  const auto const_m = Match{ &f_x, &f_y, 0.5f };
  BOOST_CHECK_EQUAL(m.x(), f_x);
  BOOST_CHECK_EQUAL(m.y(), f_y);
  BOOST_CHECK_EQUAL(const_m.x(), f_x);
  BOOST_CHECK_EQUAL(const_m.y(), f_y);

  auto m2 = Match{ &f_x, &f_y, 0.5f };
  BOOST_CHECK_EQUAL(m, m2);
}

BOOST_AUTO_TEST_CASE(test_make_index_match)
{
  auto m = make_index_match(0, 1000);
  BOOST_CHECK_EQUAL(m.x_index(), 0);
  BOOST_CHECK_EQUAL(m.y_index(), 1000);
}

BOOST_AUTO_TEST_CASE(test_read_write)
{
  auto X = vector<OERegion>{ 10 };
  auto Y = vector<OERegion>{ 10 };

  // Write dummy matches.
  auto matches = vector<Match>(10);
  for (size_t i = 0; i < matches.size(); ++i)
  {
    matches[i] = Match{
      &X[i], &Y[i], float(i), Match::Direction::SourceToTarget,
      int(i), int(i)
    };
  }
  BOOST_CHECK(write_matches(matches, "match.txt"));

  // Read the saved matches.
  auto saved_matches = vector<Match>{};
  BOOST_CHECK(read_matches(saved_matches, "match.txt"));

  // Compare the saved matches.
  BOOST_CHECK_EQUAL(matches.size(), saved_matches.size());
  for (size_t i = 0; i < saved_matches.size(); ++i)
  {
    BOOST_REQUIRE(saved_matches[i].x_pointer() == nullptr);
    BOOST_REQUIRE(saved_matches[i].y_pointer() == nullptr);
    BOOST_REQUIRE_EQUAL(matches[i].x_index(), saved_matches[i].x_index());
    BOOST_REQUIRE_EQUAL(matches[i].y_index(), saved_matches[i].y_index());
    BOOST_REQUIRE_EQUAL(matches[i].score(), saved_matches[i].score());
  }

  // Read the saved matches.
  auto saved_matches_2 = vector<Match>{};
  BOOST_CHECK(read_matches(saved_matches_2, X, Y, "match.txt"));
  BOOST_CHECK_EQUAL(matches.size(), saved_matches_2.size());
  for (size_t i = 0; i < saved_matches_2.size(); ++i)
  {
    BOOST_REQUIRE_EQUAL(&X[i], saved_matches_2[i].x_pointer());
    BOOST_REQUIRE_EQUAL(&Y[i], saved_matches_2[i].y_pointer());
    BOOST_REQUIRE_EQUAL(matches[i].x_index(), saved_matches_2[i].x_index());
    BOOST_REQUIRE_EQUAL(matches[i].y_index(), saved_matches_2[i].y_index());
    BOOST_REQUIRE_EQUAL(matches[i].score(), saved_matches_2[i].score());
  }

}

BOOST_AUTO_TEST_CASE(test_ostream_operator)
{
  auto x = OERegion{ Point2f::Zero(), 1.f };
  auto y = OERegion{ Point2f::Ones(), 1.f };

  auto m = Match(&x, &y, 0.6f);

  ostringstream os;
  os << m;
  auto str = os.str();
  BOOST_CHECK(str.find("source=") != string::npos);
  BOOST_CHECK(str.find("target=") != string::npos);
  BOOST_CHECK(str.find("score=") != string::npos);
}

BOOST_AUTO_TEST_SUITE_END()