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

#include <filesystem>


using namespace DO::Sara;
using namespace std;

static const auto RedBerry8 = Rgb8{0xe3, 0x57, 0x60};
static const auto BlueBerry8 = Rgb8{0x6f, 0x84, 0x9c};
// static const auto GreenLeaf8 = Rgb8{0x80, 0x9a, 0x41};


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  namespace fs = std::filesystem;
  const auto image_path = fs::path{
    argc < 2  //
      ? src_path("../../../../data/sunflowerField.jpg")
      : argv[1]
  };
  const auto image = imread<Rgb8>(image_path.string());

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

  auto image_annotated = image;
  for (const auto& f : features)
  {
    const auto& color = f.extremum_type == OERegion::ExtremumType::Max
      ? RedBerry8
      : BlueBerry8;
    draw(image_annotated, f, color);
  }
  display(image_annotated);
  get_key();

  const auto dir_path = image_path.parent_path();
  const auto filename = fs::path{image_path}.filename().string();
  static constexpr auto quality = 95;
  imwrite(image_annotated, (dir_path / (filename + "-annotated.jpg")).string(),
          quality);

  return 0;
}
