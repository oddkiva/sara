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
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>


using namespace std;
using namespace DO::Sara;


void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          Set<OERegion, RealDescriptor>& keys1,
          Set<OERegion, RealDescriptor>& keys2,  //
          vector<Match>& matches)
{
  cout << "Loading images" << endl;

  auto data_dir = std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto file1 = "0000.png";
  auto file2 = "0001.png";

  image1 = imread<Rgb8>(data_dir + "/" + file1);
  image2 = imread<Rgb8>(data_dir + "/" + file2);

#ifdef COMPUTE_KEYPOINTS
  cout << "Computing/Reading keypoints" << endl;
  auto sifts1 = compute_sift_keypoints(image1.convert<float>());
  auto sifts2 = compute_sift_keypoints(image2.convert<float>());
  keys1.append(sifts1);
  keys2.append(sifts2);
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  write_keypoints(sifts1.features, sifts1.descriptors,
                  data_dir + "/" + "0000.key");
  write_keypoints(sifts2.features, sifts2.descriptors,
                  data_dir + "/" + "0001.key");

#else
  read_keypoints(keys1.features, keys1.descriptors,
                 data_dir + "/" + "0000.key");
  read_keypoints(keys2.features, keys2.descriptors,
                 data_dir + "/" + "0001.key");
#endif

  // Compute/read matches
  cout << "Computing Matches" << endl;
  AnnMatcher matcher{keys1, keys2, 1.0f};
  matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  // Debug this.
  //write_matches(matches, data_dir + "/" + "0000_0001.match");
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

  auto scale = 0.25f;
  auto w = int((image1.width() + image2.width()) * scale);
  auto h = max(image1.height(), image2.height()) * scale;
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
