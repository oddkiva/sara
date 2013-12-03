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

const bool drawFeatureCenterOnly = false;
const Rgb8& c = Cyan8;

void checkAffineAdaptation(const Image<unsigned char>& image,
                           const OERegion& f)
{
  int w = image.width();
  int h = image.height();
  display(image);
  f.draw(Blue8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);

  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shapeMat() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3f A(f.affinity());
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = 2*float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = 2*float(x-r)/r;
      Point3f pp(u, v, 1.);
      pp = A*pp;

      Point2f p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = interpolate(image, p);
    }
  }

  Window w1 = activeWindow();
  Window w2 = openWindow(patchSz, patchSz);
  setActiveWindow(w2);
  setAntialiasing();
  display(patch);
  rg.draw(Blue8);
  milliSleep(1000);
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void readFeatures(const Image<unsigned char>& image,
                  const string& filepath)
{
  cout << "Reading DoG features... " << endl;
  vector<OERegion> features;
  DescriptorMatrix<float> descriptors;

  cout << "Reading keypoints..." << endl;
  readKeypoints(features, descriptors, filepath);

  for (int i = 0; i < 10; ++i)
    checkAffineAdaptation(image, features[i]);

  string ext = filepath.substr(filepath.find_last_of("."), filepath.size());
  string name = filepath.substr(0, filepath.find_last_of("."));
  string copy_filepath = name + "_copy" + ext;
  writeKeypoints(features, descriptors, name + "_copy" + ext);

  vector<OERegion> features2;
  DescriptorMatrix<float> descriptors2;
  cout << "Checking written file..." << endl;
  readKeypoints(features2, descriptors2, copy_filepath);

  ASSERT_EQ(features.size(), features2.size());
  ASSERT_EQ(descriptors.size(), descriptors2.size());

  for(int i = 0; i < 10; ++i)
  {
    ASSERT_EQ(features[i], features2[i]);
    ASSERT_EQ(descriptors[i], descriptors2[i]);
  }


  cout << "Printing the 10 first keypoints..." << endl;
  for(size_t i = 0; i < 10; ++i)
    cout << features[i] << endl;

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawOERegions(features, Red8);
  cout << "done!" << endl;
  milliSleep(1000);
}

TEST(DO_Features_Test, testFeaturesIO)
{
  Image<unsigned char> I;
  load(I, srcPath("../../datasets/obama_2.jpg"));

  setActiveWindow(openWindow(I.width(), I.height()));
  setAntialiasing(activeWindow());
  readFeatures(I, srcPath("../../datasets/test.dogkey"));
  readFeatures(I, srcPath("../../datasets/test.haraffkey"));
  readFeatures(I, srcPath("../../datasets/test.mserkey"));
}

int main()
{
  testing::InitGoogleTest(&guiApp()->argc, guiApp()->argv);
  return RUN_ALL_TESTS();
}
