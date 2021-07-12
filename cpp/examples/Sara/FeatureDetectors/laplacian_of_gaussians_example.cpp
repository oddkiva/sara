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
#include <DO/Sara/Graphics.hpp>


using namespace DO::Sara;
using namespace std;


vector<OERegion> compute_LoG_extrema(const Image<float>& image,
                                     bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
    tic();

  auto pyramid_params = ImagePyramidParams{0, 3 + 2};
  ComputeLoGExtrema computeLoGs{pyramid_params};
  auto scale_octave_pairs = vector<Point2i>{};
  auto LoGs = computeLoGs(image, &scale_octave_pairs);
  if (verbose)
    toc("LoG Extrema");
  SARA_CHECK(LoGs.size());

  // 2. Rescale detected features to original image dimension.
  const auto& L = computeLoGs.laplacians_of_gaussians();
  for (size_t i = 0; i < LoGs.size(); ++i)
  {
    float octave_scale_factor =
        L.octave_scaling_factor(scale_octave_pairs[i](1));
    LoGs[i].center() *= octave_scale_factor;
    LoGs[i].shape_matrix /= pow(octave_scale_factor, 2);
  }

  return LoGs;
}

vector<OERegion> compute_LoG_affine_extrema(const Image<float>& image,
                                            bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
    tic();

  auto pyramid_params = ImagePyramidParams{0};
  auto compute_LoGs = ComputeLoGExtrema{pyramid_params};
  auto scale_octave_pairs = vector<Point2i>{};
  auto LoGs = compute_LoGs(image, &scale_octave_pairs);
  if (verbose)
    toc("LoG Extrema");
  SARA_CHECK(LoGs.size());

  const auto& G = compute_LoGs.gaussians();
  const auto& L = compute_LoGs.laplacians_of_gaussians();

  // 2. Affine shape adaptation
  if (verbose)
    tic();

  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(LoGs.size(), 0);
  for (size_t i = 0; i != LoGs.size(); ++i)
  {
    const auto& s = scale_octave_pairs[i](0);
    const auto& o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adapt_shape(affine_adapt_transform, G(s, o), LoGs[i]))
    {
      LoGs[i].shape_matrix = affine_adapt_transform * LoGs[i].shape_matrix;
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc("Affine Shape Adaptation");

  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features =
      std::accumulate(keep_features.begin(), keep_features.end(), 0);

  auto kept_DoGs = vector<OERegion>{};
  kept_DoGs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoGs.push_back(LoGs[i]);
      const float fact = L.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_DoGs.back().shape_matrix *= pow(fact, -2);
      kept_DoGs.back().coords *= fact;
    }
  }

  SARA_CHECK(kept_DoGs.size());

  return kept_DoGs;
}

void check_keys(const Image<float>& image, const vector<OERegion>& features)
{
  display(image);
  set_antialiasing();
  for (size_t i = 0; i != features.size(); ++i)
    draw(features[i], features[i].extremum_type == OERegion::ExtremumType::Max
                          ? Red8
                          : Blue8);
  get_key();
}

GRAPHICS_MAIN()
{
  const auto image_filepath = src_path("../../../../data/sunflowerField.jpg");
  auto image = Image<float>{};
  if (!load(image, image_filepath))
    return -1;

  auto features = compute_LoG_extrema(image);
  create_window(image.width(), image.height());
  check_keys(image, features);

  features = compute_LoG_affine_extrema(image);
  check_keys(image, features);

  return 0;
}
