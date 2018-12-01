// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include "sift.hpp"

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>


using namespace std;
using namespace DO::Sara;


void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          Set<OERegion, RealDescriptor>& keys1,
          Set<OERegion, RealDescriptor>& keys2, vector<Match>& matches)
{
  cout << "Loading images" << endl;
  imread(image1, file1);
  imread(image2, file2);

  cout << "Computing/Reading keypoints" << endl;
  auto SIFTs1 = compute_sift_keypoints(image1.convert<float>());
  auto SIFTs2 = compute_sift_keypoints(image2.convert<float>());
  keys1.append(SIFTs1);
  keys2.append(SIFTs2);
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  SARA_CHECK(keys1.features.size());
  SARA_CHECK(keys1.descriptors.size());
  SARA_CHECK(keys2.features.size());
  SARA_CHECK(keys2.descriptors.size());

  // Compute/read matches
  cout << "Computing Matches" << endl;
  AnnMatcher matcher{keys1, keys2, 1.0f};
  matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;
}

GRAPHICS_MAIN()
{
  // Load images.
  auto image1 = Image<Rgb8>{};
  auto image2 = Image<Rgb8>{};
  auto keys1 = Set<OERegion, RealDescriptor>{};
  auto keys2 = Set<OERegion, RealDescriptor>{};
  auto matches = vector<Match>{};
  load(image1, image2, keys1, keys2, matches);

  auto scale = 1.0f;
  auto w = int((image1.width() + image2.width()) * scale);
  auto h = max(image1.height(), image2.height());
  auto off = Point2f{float(image1.width()), 0.f};

  create_window(w, h);
  set_antialiasing();
  // checkMatches(image1, image2, matches, true, scale);

  for (size_t i = 0; i < matches.size(); ++i)
  {
    draw_image_pair(image1, image2, off, scale);
    draw_match(matches[i], Red8, off, scale);
    cout << matches[i] << endl;
    get_key();
  }

  return 0;
}
