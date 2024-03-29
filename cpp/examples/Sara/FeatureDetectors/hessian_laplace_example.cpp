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


vector<OERegion> compute_hessian_laplace_affine_maxima(const Image<float>& I)
{
  // 1. Feature extraction.
  tic();
  auto compute_hessian_laplace = ComputeHessianLaplaceMaxima{};
  auto scale_octave_pairs = vector<Point2i>{};
  auto hessian_laplace_maxima = compute_hessian_laplace(I, &scale_octave_pairs);
  toc("Hessian-Laplace");
  SARA_CHECK(hessian_laplace_maxima.size());

  const auto& G = compute_hessian_laplace.gaussians();
  const auto& DoHs = compute_hessian_laplace.det_of_hessians();

  // 2. Affine shape adaptation
  tic();
  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(hessian_laplace_maxima.size(), 0);
  for (size_t i = 0; i != hessian_laplace_maxima.size(); ++i)
  {
    const auto& s = scale_octave_pairs[i](0);
    const auto& o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adapt_shape(affine_adapt_transform, G(s, o), hessian_laplace_maxima[i]))
    {
      hessian_laplace_maxima[i].shape_matrix =
          affine_adapt_transform * hessian_laplace_maxima[i].shape_matrix;
      keep_features[i] = 1;
    }
  }
  toc("Affine-Shape Adaptation");

  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features =
      std::accumulate(keep_features.begin(), keep_features.end(), 0);

  auto kept_DoHs = vector<OERegion>{};
  kept_DoHs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoHs.push_back(hessian_laplace_maxima[i]);
      const auto fact = DoHs.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_DoHs.back().shape_matrix /= square(fact);
      kept_DoHs.back().coords *= fact;
    }
  }
  SARA_CHECK(kept_DoHs.size());

  return kept_DoHs;
}

vector<OERegion> compute_DoH_extrema(const Image<float>& image)
{
  // 1. Feature extraction.
  tic();

  constexpr auto first_octave = 0;
  constexpr auto num_scales = 3;
  const auto image_pyramid_params = ImagePyramidParams(
      first_octave, num_scales + 2, pow(2.f, 1.f / num_scales), 2);
  const auto extremum_thres = 1e-5f;
  const auto edge_ratio_thres = 10.f;
  auto compute_DoHs =
      ComputeDoHExtrema{image_pyramid_params, extremum_thres, edge_ratio_thres};
  auto scale_octave_pairs = vector<Point2i>{};
  auto DoHs = compute_DoHs(image, &scale_octave_pairs);
  toc("Determinant of Hessians");
  SARA_CHECK(DoHs.size());

  const auto& DoH = compute_DoHs.det_of_hessians();

  // 2. Rescale feature points to original image dimensions.
  for (size_t i = 0; i != DoHs.size(); ++i)
  {
    const auto fact = DoH.octave_scaling_factor(scale_octave_pairs[i](1));
    DoHs[i].shape_matrix /= square(fact);
    DoHs[i].coords *= fact;
  }

  return DoHs;
}

vector<OERegion> compute_DoH_affine_extrema(const Image<float>& image)
{
  // 1. Feature extraction.
  tic();

  constexpr auto first_octave = 0;
  constexpr auto num_scales = 3;
  const auto image_pyramid_params = ImagePyramidParams(
      first_octave, num_scales + 2, pow(2.f, 1.f / num_scales), 2);
  const auto extremum_thres = 1e-5f;
  const auto edge_ratio_thres = 10.f;
  auto compute_DoHs =
      ComputeDoHExtrema{image_pyramid_params, extremum_thres, edge_ratio_thres};
  auto scale_octave_pairs = vector<Point2i>{};
  auto DoHs = compute_DoHs(image, &scale_octave_pairs);
  toc("Determinants of Hessian");
  SARA_CHECK(DoHs.size());

  const auto& G = compute_DoHs.gaussians();
  const auto& DoH = compute_DoHs.det_of_hessians();

  // 2. Affine shape adaptation
  tic();
  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(DoHs.size(), 0);
  for (size_t i = 0; i != DoHs.size(); ++i)
  {
    const auto& s = scale_octave_pairs[i](0);
    const auto& o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adapt_shape(affine_adapt_transform, G(s, o), DoHs[i]))
    {
      DoHs[i].shape_matrix = affine_adapt_transform * DoHs[i].shape_matrix;
      keep_features[i] = 1;
    }
  }
  toc("Affine Shape Adaptation");


  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features =
      std::accumulate(keep_features.begin(), keep_features.end(), 0);

  auto kept_DoHs = vector<OERegion>{};
  kept_DoHs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoHs.push_back(DoHs[i]);
      const auto fact = DoH.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_DoHs.back().shape_matrix /= square(fact);
      kept_DoHs.back().coords *= fact;
    }
  }

  return kept_DoHs;
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
  // Input.
  const auto image_filepath = src_path("../../../../data/sunflowerField.jpg");
  auto image = Image<float>{};
  if (!load(image, image_filepath))
    return -1;

  // Output.
  auto features = vector<OERegion>{};

  // Visualize.
  create_window(image.width(), image.height());

  features = compute_hessian_laplace_affine_maxima(image);
  check_keys(image, features);

  features = compute_DoH_extrema(image);
  check_keys(image, features);

  features = compute_DoH_affine_extrema(image);
  check_keys(image, features);

  return 0;
}
