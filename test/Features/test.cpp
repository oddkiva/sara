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

#include <DO/Features.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace DO;
using namespace std;

const bool drawFeatureCenterOnly = false;
const Rgb8& c = Cyan8;

void checkAffineAdaptation(const Image<unsigned char>& image,
                           const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Blue8);
    

  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shapeMat() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3f A(k.feat().affinity());
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
  rg.drawOnScreen(Blue8);
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void testDoGSift(const Image<unsigned char>& image, bool drawFeatureCenterOnly = false)
{
  // Run DoG Detector
  cout << "Detecting DoG features... " << endl;
  vector<Keypoint> keys;

  cout << "Reading keypoints..." << endl;
  readKeypoints(keys, srcPath("test.dogkey"));

  cout << "Printing the 10 first keypoints..." << endl;
  for(size_t i = 0; i < 10; ++i)
    cout << keys[i] << endl;

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  cout << "done!" << endl;
  getKey();

  for (int i = 0; i < 10; ++i)
    checkAffineAdaptation(image, keys[i]);
}

void testHarAffSift(const Image<unsigned char>& image,
                    bool drawFeatureCenterOnly = false)
{
  // Run Harris Affine Detector
  cout << "Detecting Harris-Affine features... " << endl;
  vector<Keypoint> keys;
  cout << "Reading keypoints..." << endl;
  readKeypoints(keys, srcPath("test.haraffkey"));

  cout << "Printing the 10 first keypoints..." << endl;
  for(size_t i = 0; i < 10; ++i)
    cout << keys[i] << endl;

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  cout << "done!" << endl;
  click();

  for (int i = 0; i < 10; ++i)
    checkAffineAdaptation(image, keys[i]);
}

void testMserSift(const Image<unsigned char>& image,
                  bool drawFeatureCenterOnly = false)
{
  // Run MSER Detector
  cout << "Detecting MSER features... " << endl;
  vector<Keypoint> keys;
  cout << "Reading keypoints..." << endl;
  readKeypoints(keys, srcPath("test.mserkey"));

  cout << "Printing the 10 first keypoints..." << endl;
  for(size_t i = 0; i < 10; ++i)
    cout << keys[i] << endl;

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  
  cout << "done!" << endl;
  click();
  for (int i = 0; i < 10; ++i)
    checkAffineAdaptation(image, keys[i]);
}

int main()
{
  Image<unsigned char> I;
  load(I, srcPath("obama_2.jpg"));

  setActiveWindow(openWindow(I.width(), I.height()));
  setAntialiasing(activeWindow());
  testDoGSift(I);
  testHarAffSift(I);
  testMserSift(I);
  getKey();

  return 0;
}