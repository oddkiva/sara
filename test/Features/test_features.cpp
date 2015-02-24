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

#include <DO/Features.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>


using namespace DO;
using namespace std;


const bool draw_feature_center_only = false;
const Rgb8& c = Cyan8;


void check_affine_adaptation(const OERegion& f)
{
  // Define the window width and height.
  const int w = 500;
  const int h = 500;
  Image<float> image(w, h);
  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      float T = 10.f*2*static_cast<float>(M_PI);
      float u = (x - w/2.f) / T;
      float v = (y - h/2.f) / T;
      image(x, y) = (sin(u) + 1) * (cos(v) + 1);
    }
  }

  if (!active_window())
    create_window(w, h);
  display(image);
  f.draw(Red8);
  get_key();

  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);

  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shape_matrix() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3d A(f.affinity().cast<double>());
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = 2*float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = 2*float(x-r)/r;
      Point3d pp(u, v, 1.);
      pp = A*pp;

      Point2d p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = interpolate(image, p);
    }
  }

  Window w1 = active_window();
  Window w2 = create_window(patchSz, patchSz);
  set_active_window(w2);
  set_antialiasing();
  display(patch);
  rg.draw(Blue8);
  millisleep(40);
  close_window(w2);

  millisleep(40);
  set_active_window(w1);
  get_key();
}

TEST(TestFeatures, test_misc)
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


int worker_thread_task(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}

#undef main
int main(int argc, char **argv)
{
  GraphicsApplication gui_app_(argc, argv);
  gui_app_.register_user_main(worker_thread_task);
  int return_code = gui_app_.exec();
  return return_code;
}