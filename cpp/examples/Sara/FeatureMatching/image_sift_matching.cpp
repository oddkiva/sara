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

//! @example

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/Visualization.hpp>


using namespace std;
using namespace DO::Sara;


const auto file1 = src_path("../../../../data/All.tif");
const auto file2 = src_path("../../../../data/GuardOnBlonde.tif");


void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          KeypointList<OERegion, float>& keys1,
          KeypointList<OERegion, float>& keys2,
          vector<Match>& matches)
{
  cout << "Loading images" << endl;
  image1 = imread<Rgb8>(file1);
  image2 = imread<Rgb8>(file2);

  cout << "Computing/Reading keypoints" << endl;
  keys1 = compute_sift_keypoints(image1.convert<float>());
  keys2 = compute_sift_keypoints(image2.convert<float>());

  auto& [features1, dmat1] = keys1;
  auto& [features2, dmat2] = keys2;

  cout << "Image 1: " << features1.size() << " keypoints" << endl;
  cout << "Image 2: " << features2.size() << " keypoints" << endl;

  SARA_CHECK(features1.size());
  SARA_CHECK(dmat1.size());

  SARA_CHECK(features2.size());
  SARA_CHECK(dmat2.size());

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
  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};
  auto matches = vector<Match>{};
  load(image1, image2, keys1, keys2, matches);

  auto scale = 1.0f;
  auto w = int((image1.width() + image2.width()) * scale);
  auto h = max(image1.height(), image2.height());
  auto off = Point2f{float(image1.width()), 0.f};

  create_window(w, h);
  set_antialiasing();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    draw_image_pair(image1, image2, off, scale);
    draw_match(matches[i], Red8, off, scale);
    cout << matches[i] << endl;
    get_key();
  }

  return 0;
}
