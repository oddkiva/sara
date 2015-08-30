// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <limits>

#include <gtest/gtest.h>

#include <DO/Sara/Match.hpp>


using namespace std;
using namespace DO::Sara;

TEST(TestMatch, test_default_constructor)
{
  auto m = Match{};
  EXPECT_TRUE(m.x_pointer() == nullptr);
  EXPECT_TRUE(m.y_pointer() == nullptr);
  EXPECT_THROW(m.x(), runtime_error);
  EXPECT_THROW(m.y(), runtime_error);
  EXPECT_EQ(m.x_index(), -1);
  EXPECT_EQ(m.y_index(), -1);
  EXPECT_EQ(m.score(), numeric_limits<float>::max());
  EXPECT_EQ(m.rank(), -1);
  EXPECT_EQ(m.matching_direction(), Match::SourceToTarget);

  const auto& m2 = m;
  EXPECT_TRUE(m2.x_pointer() == nullptr);
  EXPECT_TRUE(m2.y_pointer() == nullptr);
  EXPECT_THROW(m2.x(), runtime_error);
  EXPECT_THROW(m2.y(), runtime_error);
  EXPECT_EQ(m2.x_index(), -1);
  EXPECT_EQ(m2.y_index(), -1);
  EXPECT_EQ(m2.score(), numeric_limits<float>::max());
  EXPECT_EQ(m2.rank(), -1);
  EXPECT_EQ(m2.matching_direction(), Match::SourceToTarget);
}

TEST(TestMatch, test_custom_constructor)
{
  auto f_x = OERegion{};
  auto f_y = OERegion{};
  auto m = Match{ &f_x, &f_y, 0.5f };
  const auto const_m = Match{ &f_x, &f_y, 0.5f };
  EXPECT_EQ(m.x(), f_x);
  EXPECT_EQ(m.y(), f_y);
  EXPECT_EQ(const_m.x(), f_x);
  EXPECT_EQ(const_m.y(), f_y);

  auto m2 = Match{ &f_x, &f_y, 0.5f };
  EXPECT_EQ(m, m2);
}

TEST(TestMatch, test_make_index_match)
{
  auto m = make_index_match(0, 1000);
  EXPECT_EQ(m.x_index(), 0);
  EXPECT_EQ(m.y_index(), 1000);
}

TEST(TestMatch, test_read_write)
{
  auto X = vector<OERegion>{ 10 };
  auto Y = vector<OERegion>{ 10 };

  // Write dummy matches.
  auto matches = vector<Match>(10);
  for (size_t i = 0; i < matches.size(); ++i)
  {
    matches[i] = Match{
      &X[i], &Y[i], float(i), Match::SourceToTarget, int(i), int(i)
    };
  }
  EXPECT_TRUE(write_matches(matches, "match.txt"));

  // Read the saved matches.
  auto saved_matches = vector<Match>{};
  EXPECT_TRUE(read_matches(saved_matches, "match.txt"));

  // Compare the saved matches.
  EXPECT_EQ(matches.size(), saved_matches.size());
  for (size_t i = 0; i < saved_matches.size(); ++i)
  {
    ASSERT_EQ(saved_matches[i].x_pointer(), nullptr);
    ASSERT_EQ(saved_matches[i].y_pointer(), nullptr);
    ASSERT_EQ(matches[i].x_index(), saved_matches[i].x_index());
    ASSERT_EQ(matches[i].y_index(), saved_matches[i].y_index());
    ASSERT_EQ(matches[i].score(), saved_matches[i].score());
  }

  // Read the saved matches.
  auto saved_matches_2 = vector<Match>{};
  EXPECT_TRUE(read_matches(saved_matches_2, X, Y, "match.txt"));
  EXPECT_EQ(matches.size(), saved_matches_2.size());
  for (size_t i = 0; i < saved_matches_2.size(); ++i)
  {
    ASSERT_EQ(&X[i], saved_matches_2[i].x_pointer());
    ASSERT_EQ(&Y[i], saved_matches_2[i].y_pointer());
    ASSERT_EQ(matches[i].x_index(), saved_matches_2[i].x_index());
    ASSERT_EQ(matches[i].y_index(), saved_matches_2[i].y_index());
    ASSERT_EQ(matches[i].score(), saved_matches_2[i].score());
  }

}

TEST(TestMatch, test_ostream_operator)
{
  auto x = OERegion{ Point2f::Zero(), 1.f };
  auto y = OERegion{ Point2f::Ones(), 1.f };

  auto m = Match(&x, &y, 0.6f);

  ostringstream os;
  os << m;
  auto str = os.str();
  EXPECT_TRUE(str.find("source=") != string::npos);
  EXPECT_TRUE(str.find("target=") != string::npos);
  EXPECT_TRUE(str.find("score=") != string::npos);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
