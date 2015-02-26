// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Features/Feature.hpp>


using namespace DO;
using namespace std;


TEST(TestFeatures, test_interest_point)
{
  InterestPoint f(Point2f::Ones());
  EXPECT_EQ(f.coords(), Point2f::Ones());
  EXPECT_EQ(f.center(), Point2f::Ones());
  EXPECT_EQ(f.x(), 1.f);
  EXPECT_EQ(f.y(), 1.f);
}

TEST(TestFeatures, test_oe_region_shape)
{
  OERegion f(Point2f::Zero(), 1.f);
  EXPECT_EQ(f.shape_matrix(), Matrix2f::Identity());
  EXPECT_EQ(f.affinity(), Matrix3f::Identity());
  EXPECT_EQ(f.radius(), 1.f);
}


TEST(TestFeatures, test_io)
{
  const size_t num_features = 10;

  // Test construction.
  vector<OERegion> features(num_features);
  DescriptorMatrix<float> descriptors(num_features, 3);
  for (size_t i = 0; i < num_features; ++i)
  {
    descriptors[i] = (Vector3f::Ones() * float(i)).eval();
    OERegion& f = features[i];
    f.type() = OERegion::DoG;
    f.coords() = Point2f::Ones() * float(i);
    f.shape_matrix() = Matrix2f::Identity();
    f.orientation() = float(i);
    f.extremum_type() = OERegion::Max;
    f.extremum_value() = 0.f;
    cout << f << endl;
  }

  // Test write function.
  write_keypoints(features, descriptors, "keypoints.txt");

  // Test read function.
  vector<OERegion> features2;
  DescriptorMatrix<float> descriptors2;
  read_keypoints(features2, descriptors2, "keypoints.txt");

  ASSERT_EQ(features.size(), features2.size());
  ASSERT_EQ(descriptors.size(), descriptors2.size());

  for (size_t i = 0; i < num_features; ++i)
  {
    ASSERT_EQ(features[i], features2[i]);
    ASSERT_EQ(descriptors[i], descriptors2[i]);
  }
}


int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}

#undef main
int main(int argc, char **argv)
{
  GraphicsApplication gui_app_(argc, argv);
  gui_app_.register_user_main(worker_thread);
  int return_code = gui_app_.exec();
  return return_code;
}
