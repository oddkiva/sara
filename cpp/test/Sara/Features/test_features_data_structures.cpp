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
  const auto num_features = 10;

  // Test construction.
  auto features = vector<OERegion>(num_features);
  auto descriptors = Tensor_<float, 2>{num_features, 3};
  auto dmat = descriptors.matrix();
  for (size_t i = 0; i < num_features; ++i)
  {
    dmat.row(i) = (RowVector3f::Ones() * float(i)).eval();
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
  auto descriptors2 = Tensor_<float, 2>{};
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
  KeypointList<OERegion, float> set;
  BOOST_CHECK_EQUAL(size(set), 0);

  // Test resize function.
  features(set).resize(10);
  BOOST_CHECK(!size_consistency_predicate(set));

  resize(set, 10, 2);
  BOOST_CHECK(size_consistency_predicate(set));
  BOOST_CHECK_EQUAL(size(set), 10);
  BOOST_CHECK_EQUAL(features(set).size(), 10u);
  BOOST_CHECK_EQUAL(descriptors(set).rows(), 10); // Test swap.

  KeypointList<OERegion, float> set2;
  resize(set2, 20, 2);

  std::swap(set, set2);
  BOOST_CHECK_EQUAL(size(set), 20);
  BOOST_CHECK_EQUAL(size(set2), 10);

  // Test append.
  set = DO::Sara::stack(set, set2);
  BOOST_CHECK_EQUAL(size(set), 30);

  // Test accessors.
  const auto& const_set = set;

  auto& [f, d] = set;
  const auto& [fc, dc] = const_set;

  auto dmat = d.matrix();
  auto dmatc = dc.matrix();

  f[0].coords = Point2f::Ones();
  BOOST_CHECK_EQUAL(f[0].coords, Point2f::Ones());
  BOOST_CHECK_EQUAL(fc[0].coords, Point2f::Ones());

  dmat.row(0) = RowVector2f::Ones();
  BOOST_CHECK_EQUAL(dmat.row(0), RowVector2f::Ones());
  BOOST_CHECK_EQUAL(dmatc.row(0), RowVector2f::Ones());
}

BOOST_AUTO_TEST_CASE(test_remove_redundant_features)
{
  auto features = std::vector<OERegion>{};
  auto descriptors = Tensor_<float, 2>{};

  // Check corner case.
  features.resize(10);
  BOOST_CHECK_THROW(remove_redundant_features(features, descriptors), runtime_error);

  // Check normal case.
  features.resize(7);
  descriptors.resize(7, 2);
  descriptors.matrix().row(0) = RowVector2f::Zero();
  descriptors.matrix().row(1) = RowVector2f::Zero();
  descriptors.matrix().row(2) = RowVector2f::Ones();
  descriptors.matrix().row(3) = RowVector2f::Ones();
  descriptors.matrix().row(4) = RowVector2f::Ones();
  descriptors.matrix().row(5) = RowVector2f::Zero();
  descriptors.matrix().row(6) = RowVector2f::Zero();

  remove_redundant_features(features, descriptors);
  Matrix2f expected_descriptor_matrix;
  expected_descriptor_matrix.row(0) = RowVector2f::Zero();
  expected_descriptor_matrix.row(1) = RowVector2f::Ones();

  BOOST_CHECK_EQUAL(expected_descriptor_matrix, descriptors.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
