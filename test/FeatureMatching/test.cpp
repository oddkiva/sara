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

#include <DO/FeatureDetectors.hpp>
#include <DO/FeatureMatching.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace std;
using namespace DO;

const bool drawFeatureCenterOnly = false;
const Rgb8 c = Cyan8;

//#define MSER_KEYS
#define HARRIS_KEYS

string file1 = srcPath("All.tif");
string file2 = srcPath("GuardOnBlonde.tif");

void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          vector<Keypoint>& keys1, vector<Keypoint>& keys2,
          vector<Match>& matches)
{
  cout << "Loading images" << endl;
  load(image1, file1);
  load(image2, file2);

  cout << "Computing/Reading keypoints" << endl;
#ifdef HARRIS_KEYS
  vector<Keypoint> har1 = HarAffSiftDetector().run(image1.convert<unsigned char>());
  vector<Keypoint> har2 = HarAffSiftDetector().run(image2.convert<unsigned char>());
  keys1.insert(keys1.end(), har1.begin(), har1.end());
  keys2.insert(keys2.end(), har2.begin(), har2.end());
#endif // HARRIS_KEYS
#ifdef MSER_KEYS
  vector<Keypoint> mser1 = MserSiftDetector().run(image1.convert<unsigned char>());
  vector<Keypoint> mser2 = MserSiftDetector().run(image2.convert<unsigned char>());
  keys1.insert(keys1.end(), mser1.begin(), mser1.end());
  keys2.insert(keys2.end(), mser2.begin(), mser2.end());
#endif // MSER_KEYS
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  // Compute/read matches
  cout << "Computing Matches" << endl;
  AnnMatcher matcher(keys1, keys2, 1.0f);
  matches = matcher.computeMatches();
  cout << matches.size() << " matches" << endl;
}

DO::Ellipse ellFromFeature(const OERegion& f)
{ return fromShapeMat(f.shapeMat().cast<double>(), f.center().cast<double>()); }

void checkPatch(const Image<unsigned char>& image,
                const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("whole_picture.png"));


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);

  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.shapeMat() /= 4.;

  Matrix3f A;
  A.fill(0.f);
  A(0,0) = A(1,1) = r/2.;
  A(0,2) = k.feat().x();
  A(1,2) = k.feat().y();
  A(2,2) = 1.f;
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = float(x-r)/r;
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
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation(const Image<unsigned char>& image,
                           const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Yellow8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  Matrix2f Q = Rotation2D<float>(k.feat().orientation()).matrix();
  rg.shapeMat() = Q.transpose()*k.feat().shapeMat()*Q/4.;


  Matrix3f A(k.feat().affinity());  
  A.fill(0.f);
  A.block(0,0,2,2) = Q*r/2.;
  A(0,2) = k.feat().x();
  A(1,2) = k.feat().y();
  A(2,2) = 1.f;
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = float(x-r)/r;
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
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("rotated_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation2(const Image<unsigned char>& image,
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
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("normalized_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

int main()
{
	// Load images.
	Image<Rgb8> image1, image2;
	vector<Keypoint> keys1, keys2;
	vector<Match> matches;
	load(image1, image2, keys1, keys2, matches);

	float scale = 1.0f;
  int w = int((image1.width()+image2.width())*scale);
  int h = max(image1.height(), image2.height());
  Point2f off(float(image1.width()), 0.f);

  openWindow(w, h);
  setAntialiasing();
  //checkMatches(image1, image2, matches, true, scale);

  for (int i = 0; i < matches.size(); ++i)
  {
    drawImPair(image1, image2, off, scale);
    drawMatch(matches[i], Red8, off, scale);
    cout << matches[i] << endl;
    getKey();

    //checkPatch(image1.convert<unsigned char>(), matches[i].x());
    //checkAffineAdaptation(image1.convert<unsigned char>(), matches[i].x());
    //checkAffineAdaptation2(image1.convert<unsigned char>(), matches[i].x());
    //checkPatch(image2.convert<unsigned char>(), matches[i].y());
    //checkAffineAdaptation(image2.convert<unsigned char>(), matches[i].y());
    //checkAffineAdaptation2(image2.convert<unsigned char>(), matches[i].y());
  }

  return 0;
}