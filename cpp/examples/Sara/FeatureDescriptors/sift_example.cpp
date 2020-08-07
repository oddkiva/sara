// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace DO::Sara;
using namespace std;


GRAPHICS_MAIN()
{
  const auto image_path = src_path("../../../../data/sunflowerField.jpg");
  const auto image = imread<float>(image_path);

  print_stage("Detecting SIFT features");
  const auto pyramid_params = ImagePyramidParams(-1);
  auto keypoints = compute_sift_keypoints(image, pyramid_params, true);
  const auto& features = std::get<0>(keypoints);

#ifdef REMOVE_REDUNDANCIES
  print_stage("Removing existing redundancies");
  remove_redundant_features(features, descriptors);
  SARA_CHECK(features.size());
  SARA_CHECK(descriptors.sizes().transpose());
#endif

  // Check the features visually.
  print_stage("Draw features");
  create_window(image.width(), image.height());
  set_antialiasing();
  display(image);
  for (const auto& f: features)
  {
    const auto& color =
        f.extremum_type == OERegion::ExtremumType::Max ? Red8 : Blue8;
    f.draw(color);
  }
  get_key();

  return 0;
}
