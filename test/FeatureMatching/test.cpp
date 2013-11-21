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
#include <DO/FeatureDescriptors.hpp>
#include <DO/FeatureMatching.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace std;
using namespace DO;

//#define MSER_KEYS
//#define HARRIS_KEYS

string file1 = srcPath("../../datasets/All.tif");
string file2 = srcPath("../../datasets/GuardOnBlonde.tif");


Set<OERegion, RealDescriptor> computeSIFT(const Image<float>& image)
{
  // Time everything.
  HighResTimer timer;
  double elapsed = 0.;
  double DoGDetTime, oriAssignTime, siftCompTime, gradGaussianTime;

  // We describe the work flow of the feature detection and description.
  Set<OERegion, RealDescriptor> keys;
  vector<OERegion>& DoGs = keys.features;
  DescriptorMatrix<float>& SIFTs = keys.descriptors;

  // 1. Feature extraction.
  printStage("Computing DoG extrema");
  timer.restart();
  ImagePyramidParams pyrParams(0);
  ComputeDoGExtrema computeDoGs(pyrParams);
  vector<Point2i> scaleOctPairs;
  DoGs = computeDoGs(image, &scaleOctPairs);
  DoGDetTime = timer.elapsedMs();
  elapsed += DoGDetTime;
  cout << "DoG detection time = " << DoGDetTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  printStage("Computing gradients of Gaussians");
  timer.restart();
  ImagePyramid<Vector2f> gradG;
  const ImagePyramid<float>& gaussPyr = computeDoGs.gaussians();
  gradG = gradPolar(gaussPyr);
  gradGaussianTime = timer.elapsedMs();
  elapsed += gradGaussianTime;
  cout << "gradient of Gaussian computation time = " << gradGaussianTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // Find dominant gradient orientations.
  printStage("Assigning (possibly multiple) dominant orientations to DoG extrema");
  timer.restart();
  ComputeDominantOrientations assignOrientations;
  assignOrientations(gradG, DoGs, scaleOctPairs);
  oriAssignTime = timer.elapsedMs();
  elapsed += oriAssignTime;
  cout << "orientation assignment time = " << oriAssignTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // 3. Feature description.
  printStage("Describe DoG extrema with SIFT descriptors");
  timer.restart();
  ComputeSIFTDescriptor<> computeSIFT;
  SIFTs = computeSIFT(DoGs, scaleOctPairs, gradG);
  siftCompTime = timer.elapsedMs();
  elapsed += siftCompTime;
  cout << "description time = " << siftCompTime << " ms" << endl;
  cout << "sifts.size() = " << SIFTs.size() << endl;


  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    float octScaleFact = gradG.octaveScalingFactor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shapeMat() /= pow(octScaleFact, 2);
  }

  removeRedundancies(keys.features, keys.descriptors);

  return keys;
}


void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          Set<OERegion, RealDescriptor>& keys1,
          Set<OERegion, RealDescriptor>& keys2,
          vector<Match>& matches)
{
  cout << "Loading images" << endl;
  if (!load(image1, file1))
  {
    cerr << "Error: cannot load image file 1: " << file1 << endl;
    return;
  }
  if (!load(image2, file2))
  {
    cerr << "Error: cannot load image file 2: " << file2 << endl;
    return;
  }

  cout << "Computing/Reading keypoints" << endl;
  Set<OERegion, RealDescriptor> SIFTs1 = computeSIFT(image1.convert<float>());
  Set<OERegion, RealDescriptor> SIFTs2 = computeSIFT(image2.convert<float>());
  keys1.append(SIFTs1);
  keys2.append(SIFTs2);
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  CHECK(keys1.features.size());
  CHECK(keys1.descriptors.size());
  CHECK(keys2.features.size());
  CHECK(keys2.descriptors.size());

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