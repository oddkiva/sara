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

#include <algorithm>
#include <cmath>

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Visualization.hpp>


using namespace DO::Sara;
using namespace std;


// A helper function
// Be aware that detection parameters are those set by default, e.g.,
// - thresholds like on extremum responses,
// - number of iterations in the keypoint localization,...
// Keypoints are described with the SIFT descriptor.
vector<OERegion> compute_harris_laplace_affine_corners(const Image<float>& I,
                                                       bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
    tic();
  auto compute_corners = ComputeHarrisLaplaceCorners{};
  auto scale_octave_pairs = vector<Point2i>{};
  auto corners = compute_corners(I, &scale_octave_pairs);
  if (verbose)
    toc("Harris-Laplace Extrema");

  const auto& G = compute_corners.gaussians();
  const auto& H = compute_corners.harris();

  // 2. Affine shape adaptation
  if (verbose)
    tic();
  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(corners.size(), 0);
  for (size_t i = 0; i != corners.size(); ++i)
  {
    const int s = scale_octave_pairs[i](0);
    const int o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adapt_shape(affine_adapt_transform, G(s,o), corners[i]))
    {
      corners[i].shape_matrix =
          affine_adapt_transform * corners[i].shape_matrix;
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc("Affine Shape Adaptation");

  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features = std::accumulate(
    keep_features.begin(), keep_features.end(), 0);

  auto kept_corners = vector<OERegion>{};
  kept_corners.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_corners.push_back(corners[i]);
      const auto fact = H.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_corners.back().shape_matrix /= square(fact);
      kept_corners.back().coords *= fact;
    }
  }

  SARA_CHECK(kept_corners.size());
  return kept_corners;
}

void check_keys(const Image<float>& image, const vector<OERegion>& features)
{
  display(image);
  set_antialiasing();
  draw_oe_regions(features, Red8);
  get_key();
}

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  const auto image_filepath =
      argc < 2 ? src_path("../../../../data/sunflowerField.jpg") : argv[1];
  auto image = Image<float>{};
  if (!load(image, image_filepath))
  {
    cerr << "Could not open file " << image_filepath << endl;
    return -1;
  }

  auto features = compute_harris_laplace_affine_corners(image);

  create_window(image.width(), image.height());
  check_keys(image, features);

  return 0;
}
