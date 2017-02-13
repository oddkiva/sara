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

#include <gtest/gtest.h>

#include <DO/Sara/Features.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestInterestPoint, test_methods)
{
  auto f = InterestPoint{Point2f::Ones()};

  f.type() = InterestPoint::Type::Harris;
  f.extremum_type() = InterestPoint::ExtremumType::Saddle;
  f.extremum_value() = 0.f;
  EXPECT_EQ(f.coords(), Point2f::Ones());
  EXPECT_EQ(f.center(), Point2f::Ones());
  EXPECT_EQ(f.x(), 1.f);
  EXPECT_EQ(f.y(), 1.f);
  EXPECT_EQ(f.extremum_type(), InterestPoint::ExtremumType::Saddle);
  EXPECT_EQ(f.extremum_value(), 0.f);
  EXPECT_EQ(f.type(), InterestPoint::Type::Harris);

  // Check output stream operator.
  pair<InterestPoint::Type, string> types[] = {
      make_pair(InterestPoint::Type::DoG, "DoG"),
      make_pair(InterestPoint::Type::HarAff, "Harris-Affine"),
      make_pair(InterestPoint::Type::HesAff, "Hessian-Affine"),
      make_pair(InterestPoint::Type::MSER, "MSER"),
      make_pair(InterestPoint::Type::SUSAN, "")
  };

  for (auto i = 0; i < 5; ++i)
  {
    f.type() = types[i].first;
    ostringstream oss;
    oss << f;
    auto str = oss.str();
    EXPECT_TRUE(str.find(types[i].second) != string::npos);
  }
}

TEST(TestOERegion, test_methods)
{
  auto f = OERegion{Point2f::Zero(), 1.f};
  f.orientation() = 0;
  EXPECT_EQ(f.shape_matrix(), Matrix2f::Identity());
  EXPECT_MATRIX_NEAR(f.affinity(), Matrix3f::Identity(), 1e-3);
  EXPECT_EQ(f.radius(), 1.f);
  EXPECT_EQ(f.scale(), 1.f);

  // Check output stream operator.
  ostringstream oss;
  oss << f;
  auto str = oss.str();
  EXPECT_TRUE(str.find("shape matrix") != string::npos);
  EXPECT_TRUE(str.find("orientation:\t0 degrees") != string::npos);
}

TEST(TestIO, test_read_write)
{
  const auto num_features = size_t{10};

  // Test construction.
  auto features = vector<OERegion>{num_features};
  auto descriptors = DescriptorMatrix<float>{num_features, 3};
  for (size_t i = 0; i < num_features; ++i)
  {
    descriptors[i] = (Vector3f::Ones() * float(i)).eval();
    OERegion& f = features[i];
    f.type() = OERegion::Type::DoG;
    f.coords() = Point2f::Ones() * float(i);
    f.shape_matrix() = Matrix2f::Identity();
    f.orientation() = float(i);
    f.extremum_type() = OERegion::ExtremumType::Max;
    f.extremum_value() = 0.f;
  }

  // Test write function.
  write_keypoints(features, descriptors, "keypoints.txt");

  // Test read function.
  auto features2 = vector<OERegion>{};
  auto descriptors2 = DescriptorMatrix<float>{};
  read_keypoints(features2, descriptors2, "keypoints.txt");

  ASSERT_EQ(features.size(), features2.size());
  ASSERT_EQ(descriptors.size(), descriptors2.size());

  for (size_t i = 0; i < num_features; ++i)
  {
    ASSERT_EQ(features[i], features2[i]);
    ASSERT_EQ(descriptors[i], descriptors2[i]);
  }
}

TEST(TestSet, test_methods)
{
  // Test constructor.
  auto set = Set<OERegion, RealDescriptor>{};
  EXPECT_EQ(set.size(), 0);

  // Test resize function.
  set.features.resize(10);
  EXPECT_THROW(set.size(), std::runtime_error);

  set.resize(10, 2);
  EXPECT_EQ(set.size(), 10);
  EXPECT_EQ(set.features.size(), 10);
  EXPECT_EQ(set.descriptors.size(), 10);  // Test swap.

  auto set2 = Set<OERegion, RealDescriptor>{};
  set2.resize(20, 2);

  set.swap(set2);
  EXPECT_EQ(set.size(), 20);
  EXPECT_EQ(set2.size(), 10);

  // Test append.
  set.append(set2);
  EXPECT_EQ(set.size(), 30);

  // Test accessors.
  const auto& const_set = set;

  set.f(0).coords() = Point2f::Ones();
  EXPECT_EQ(set.f(0).coords(), Point2f::Ones());
  EXPECT_EQ(const_set.f(0).coords(), Point2f::Ones());

  set.v(0) = Point2f::Ones();
  EXPECT_EQ(set.v(0), Point2f::Ones());
  EXPECT_EQ(const_set.v(0), Point2f::Ones());
}

TEST(TestSet, test_remove_redundant_features)
{
  auto set = Set<OERegion, RealDescriptor>{};

  // Check corner case.
  set.features.resize(10);
  EXPECT_THROW(remove_redundant_features(set), runtime_error);

  // Check normal case.
  set.resize(7, 2);
  set.descriptors[0] = Vector2f::Zero();
  set.descriptors[1] = Vector2f::Zero();
  set.descriptors[2] = Vector2f::Ones();
  set.descriptors[3] = Vector2f::Ones();
  set.descriptors[4] = Vector2f::Ones();
  set.descriptors[5] = Vector2f::Zero();
  set.descriptors[6] = Vector2f::Zero();

  remove_redundant_features(set);
  Matrix2f expected_descriptor_matrix;
  expected_descriptor_matrix.col(0) = Vector2f::Zero();
  expected_descriptor_matrix.col(1) = Vector2f::Ones();

  EXPECT_MATRIX_EQ(expected_descriptor_matrix, set.descriptors.matrix());
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
