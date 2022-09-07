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

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Visualization.hpp>


using namespace DO::Sara;
using namespace std;


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  const auto image_path = argc < 2  //
                              ? src_path("../../../../data/sunflowerField.jpg")
                              : argv[1];
  const auto image = imread<Rgb8>(image_path);

  const auto first_octave = argc < 3 ? 0 : std::stoi(argv[2]);
  const auto octave_max =
      argc < 4 ? std::numeric_limits<int>::max() : std::stoi(argv[3]);

  print_stage("Detecting SIFT features");
  const auto pyramid_params = ImagePyramidParams(  //
      first_octave,                                //
      3 + 3,                                       //
      std::pow(2.f, 1.f / 3.f),                    //
      1,                                           //
      std::pow(2.f, first_octave),                 //
      1.6f,                                        //
      octave_max);
  auto keypoints =
      compute_sift_keypoints(image.convert<float>(), pyramid_params, true);
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
  for (const auto& f : features)
  {
    const auto& color =
        f.extremum_type == OERegion::ExtremumType::Max ? Red8 : Blue8;
    draw(f, color);
  }
  get_key();

  return 0;
}
