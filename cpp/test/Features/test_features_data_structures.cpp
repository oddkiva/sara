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

#define BOOST_TEST_MODULE "Features/Data Structures"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Features.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestOERegion)

BOOST_AUTO_TEST_CASE(test_methods)
{
  auto f = OERegion{Point2f::Ones()};

  f.type = OERegion::Type::Harris;
  f.extremum_type = OERegion::ExtremumType::Saddle;
  f.extremum_value = 0.f;
  BOOST_CHECK_EQUAL(f.coords, Point2f::Ones());
  BOOST_CHECK_EQUAL(f.center(), Point2f::Ones());
  BOOST_CHECK_EQUAL(f.x(), 1.f);
  BOOST_CHECK_EQUAL(f.y(), 1.f);
  BOOST_CHECK(f.extremum_type == OERegion::ExtremumType::Saddle);
  BOOST_CHECK_EQUAL(f.extremum_value, 0.f);
  BOOST_CHECK(f.type == OERegion::Type::Harris);

  // Check output stream operator.
  pair<OERegion::Type, string> types[] = {
    make_pair(OERegion::Type::DoG, "DoG"),
    make_pair(OERegion::Type::HarAff, "Harris-Affine"),
    make_pair(OERegion::Type::HesAff, "Hessian-Affine"),
    make_pair(OERegion::Type::MSER, "MSER"),
    make_pair(OERegion::Type::SUSAN, "")
  };

  for (int i = 0; i < 5; ++i)
  {
    f.type = types[i].first;
    ostringstream oss;
    oss << f;
    auto str = oss.str();
    BOOST_CHECK(str.find(types[i].second) != string::npos);
  }
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestOrientedEllipticRegion)

BOOST_AUTO_TEST_CASE(test_methods)
{
  OERegion f{Point2f::Zero(), 1.f};
  f.orientation = 0;
  BOOST_CHECK_EQUAL(f.shape_matrix, Matrix2f::Identity());
  BOOST_CHECK_SMALL((f.affinity() - Matrix3f::Identity()).norm(), 1e-3f);
  BOOST_CHECK_EQUAL(f.radius(), 1.f);
  BOOST_CHECK_EQUAL(f.scale(), 1.f);

  // Check output stream operator.
  ostringstream oss;
  oss << f;
  const auto str = oss.str();
  BOOST_CHECK(str.find("shape matrix") != string::npos);
  BOOST_CHECK(str.find("orientation:\t0 degrees") != string::npos);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestIO)

BOOST_AUTO_TEST_CASE(test_read_write)
{
  const auto num_features = size_t{ 10 };

  // Test construction.
  auto features = vector<OERegion>{ num_features };
  auto descriptors = DescriptorMatrix<float>{ num_features, 3 };
  for (size_t i = 0; i < num_features; ++i)
  {
    descriptors[i] = (Vector3f::Ones() * float(i)).eval();
    OERegion& f = features[i];
    f.type = OERegion::Type::DoG;
    f.coords = Point2f::Ones() * float(i);
    f.shape_matrix = Matrix2f::Identity();
    f.orientation = float(i);
    f.extremum_type = OERegion::ExtremumType::Max;
    f.extremum_value = 0.f;
  }

  // Test write function.
  write_keypoints(features, descriptors, "keypoints.txt");

  // Test read function.
  auto features2 = vector<OERegion>{};
  auto descriptors2 = DescriptorMatrix<float>{};
  read_keypoints(features2, descriptors2, "keypoints.txt");

  BOOST_REQUIRE_EQUAL(features.size(), features2.size());
  BOOST_REQUIRE_EQUAL(descriptors.size(), descriptors2.size());

  for (size_t i = 0; i < num_features; ++i)
  {
    BOOST_REQUIRE_EQUAL(features[i], features2[i]);
    BOOST_REQUIRE_EQUAL(descriptors[i], descriptors2[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestSet)

BOOST_AUTO_TEST_CASE(test_methods)
{
  // Test constructor.
  Set<OERegion, RealDescriptor> set;
  BOOST_CHECK_EQUAL(set.size(), 0u);

  // Test resize function.
  set.features.resize(10);
  BOOST_CHECK_THROW(set.size(), std::runtime_error);

  set.resize(10, 2);
  BOOST_CHECK_EQUAL(set.size(), 10u);
  BOOST_CHECK_EQUAL(set.features.size(), 10u);
  BOOST_CHECK_EQUAL(set.descriptors.size(), 10u); // Test swap.

  Set<OERegion, RealDescriptor> set2;
  set2.resize(20, 2);

  set.swap(set2);
  BOOST_CHECK_EQUAL(set.size(), 20u);
  BOOST_CHECK_EQUAL(set2.size(), 10u);

  // Test append.
  set.append(set2);
  BOOST_CHECK_EQUAL(set.size(), 30u);

  // Test accessors.
  const auto& const_set = set;

  set.f(0).coords = Point2f::Ones();
  BOOST_CHECK_EQUAL(set.f(0).coords, Point2f::Ones());
  BOOST_CHECK_EQUAL(const_set.f(0).coords, Point2f::Ones());

  set.v(0) = Point2f::Ones();
  BOOST_CHECK_EQUAL(set.v(0), Point2f::Ones());
  BOOST_CHECK_EQUAL(const_set.v(0), Point2f::Ones());
}

BOOST_AUTO_TEST_CASE(test_remove_redundant_features)
{
  Set<OERegion, RealDescriptor> set;

  // Check corner case.
  set.features.resize(10);
  BOOST_CHECK_THROW(remove_redundant_features(set), runtime_error);

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

  BOOST_CHECK_EQUAL(expected_descriptor_matrix, set.descriptors.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
