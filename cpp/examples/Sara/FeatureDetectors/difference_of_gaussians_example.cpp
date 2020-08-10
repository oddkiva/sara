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

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>


using namespace DO::Sara;
using namespace std;


static Timer timer;

void tic()
{
  timer.restart();
}

void toc()
{
  auto elapsed = timer.elapsed_ms();
  cout << "Elapsed time = " << elapsed << " ms" << endl << endl;
}

vector<OERegion> compute_dog_extrema(const Image<float>& I, bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoG extrema");
    tic();
  }
  auto pyramid_params = ImagePyramidParams{-1};
  auto compute_DoGs = ComputeDoGExtrema{pyramid_params};
  auto scale_octave_pairs = vector<Point2i>{};
  auto DoGs = compute_DoGs(I, &scale_octave_pairs);
  if (verbose)
    toc();
  SARA_CHECK(DoGs.size());

  // 2. Rescale detected features to original image dimension.
  const auto& DoG = compute_DoGs.diff_of_gaussians();
  for (size_t i = 0; i < DoGs.size(); ++i)
  {
    auto octave_scale_factor =
        DoG.octave_scaling_factor(scale_octave_pairs[i](1));
    DoGs[i].center() *= octave_scale_factor;
    DoGs[i].shape_matrix /= pow(octave_scale_factor, 2);
  }

  return DoGs;
}

vector<OERegion> compute_dog_affine_extrema(const Image<float>& I,
                                            bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoG affine-adapted extrema");
    tic();
  }

  auto pyramid_params = ImagePyramidParams{0};
  auto compute_DoGs = ComputeDoGExtrema{pyramid_params};
  auto scale_octave_pairs = vector<Point2i>{};
  auto DoGs = compute_DoGs(I, &scale_octave_pairs);
  if (verbose)
    toc();
  SARA_CHECK(DoGs.size());

  const auto& G = compute_DoGs.gaussians();
  const auto& D = compute_DoGs.diff_of_gaussians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    print_stage("Affine shape adaptation");
    tic();
  }
  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(DoGs.size(), 0);
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    const auto& s = scale_octave_pairs[i](0);
    const auto& o = scale_octave_pairs[i](1);

    Matrix2f affine_adaptation_transform;
    if (adapt_shape(affine_adaptation_transform, G(s, o), DoGs[i]))
    {
      DoGs[i].shape_matrix = affine_adaptation_transform * DoGs[i].shape_matrix;
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features =
      std::accumulate(keep_features.begin(), keep_features.end(), 0);

  auto kept_DoGs = vector<OERegion>{};
  kept_DoGs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoGs.push_back(DoGs[i]);
      const auto fact = D.octave_scaling_factor(scale_octave_pairs[i](1));
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
    features[i].draw(features[i].extremum_type == OERegion::ExtremumType::Max
                         ? Red8
                         : Blue8);
  get_key();
}

GRAPHICS_MAIN()
{
  try
  {
    auto image = Image<float>{};
    auto image_filepath = src_path("../../../../data/sunflowerField.jpg");
    if (!load(image, image_filepath))
      return EXIT_FAILURE;

    cout << "Loaded image successfully" << endl;
    create_window(image.width(), image.height());

    auto features = compute_dog_extrema(image);
    check_keys(image, features);

    features = compute_dog_affine_extrema(image);
    check_keys(image, features);
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }

  return EXIT_SUCCESS;
}
