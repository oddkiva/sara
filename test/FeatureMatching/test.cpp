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

#include <DO/FeatureDetectorWrappers.hpp>
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
          Set<OERegion, RealDescriptor>& keys1,
          Set<OERegion, RealDescriptor>& keys2,
          vector<Match>& matches)
{
  cout << "Loading images" << endl;
  load(image1, file1);
  load(image2, file2);

  cout << "Computing/Reading keypoints" << endl;
#ifdef HARRIS_KEYS
  Set<OERegion, RealDescriptor> har1 = HarAffSiftDetector().run(image1.convert<unsigned char>());
  Set<OERegion, RealDescriptor> har2 = HarAffSiftDetector().run(image2.convert<unsigned char>());
  keys1.append(har1);
  keys2.append(har2);
#endif // HARRIS_KEYS
#ifdef MSER_KEYS
  vector<Keypoint> mser1 = MserSiftDetector().run(image1.convert<unsigned char>());
  vector<Keypoint> mser2 = MserSiftDetector().run(image2.convert<unsigned char>());
  keys1.append(mser1);
  keys2.append(mser2);
#endif // MSER_KEYS
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  // Compute/read matches
  cout << "Computing Matches" << endl;
  AnnMatcher matcher(keys1, keys2, 1.0f);
  matches = matcher.computeMatches();
  cout << matches.size() << " matches" << endl;
}

int main()
{
	// Load images.
	Image<Rgb8> image1, image2;
	Set<OERegion, RealDescriptor> keys1, keys2;
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
  }

  return 0;
}