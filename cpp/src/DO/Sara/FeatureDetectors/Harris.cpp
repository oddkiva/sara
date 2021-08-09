// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Core/StdVectorHelpers.hpp>


using namespace std;


namespace DO { namespace Sara {

  Image<float> scale_adapted_harris_cornerness(const ImageView<float>& I,
                                               float sigma_I, float sigma_D,
                                               float kappa)
  {
    // Derive the smoothed function $g_{\sigma_I} * I$
    const auto M = I.compute<Gaussian>(sigma_D)
                       .compute<Gradient>()
                       .compute<SecondMomentMatrix>()
                       .compute<Gaussian>(sigma_I);

    auto cornerness = Image<float>{I.sizes()};
    std::transform(M.begin(), M.end(), cornerness.begin(),
                   [kappa](const auto& m) {
                     return m.determinant() - kappa * pow(m.trace(), 2);
                   });

    // Rescale the cornerness function.
    cornerness.flat_array() *= sigma_D * sigma_D;

    return cornerness;
  }

  ImagePyramid<float> harris_cornerness_pyramid(const ImageView<float>& image,
                                                float kappa,
                                                const ImagePyramidParams& params)
  {
    // Resize the image with the appropriate factor.
    const auto resize_factor = std::pow(2.f, -params.first_octave_index());
    auto I = enlarge(image, resize_factor);

    // Deduce the new camera sigma with respect to the dilated image.
    const auto camera_sigma =
        static_cast<float>(params.scale_camera()) * resize_factor;

    // Blur the image so that its new sigma is equal to the initial sigma.
    const auto scale_initial = static_cast<float>(params.scale_initial());
    if (camera_sigma < scale_initial)
    {
      const auto sigma =
          sqrt(scale_initial * scale_initial - camera_sigma * camera_sigma);
      I = deriche_blur(I, sigma);
    }

    // Deduce the maximum number of octaves.
    const auto l = std::min(image.width(), image.height());
    const auto b = params.image_padding_size();
    // l/2^k > 2b
    // 2^k < l/(2b)
    // k < log(l/(2b))/log(2)
    const auto num_octaves = static_cast<int>(log(l/(2.f*b))/log(2.f));

    // Shorten names.
    const auto num_scales = params.num_scales_per_octave();
    const auto k = float(params.scale_geometric_factor());

    // Create the image pyramid
    auto cornerness = ImagePyramid<float>{};
    cornerness.reset(num_octaves, num_scales, scale_initial, k);
    for (auto o = 0; o < num_octaves; ++o)
    {
      // Compute the octave scaling factor
      cornerness.octave_scaling_factor(o) =
          (o == 0) ? 1.f / resize_factor
                   : cornerness.octave_scaling_factor(o - 1) * 2;

      // Compute the gaussians in octave $o$
      if (o != 0)
        I = downscale(I, 2);

      for (auto s = 0; s < num_scales; ++s)
      {
        const auto sigma_I =
            static_cast<float>(cornerness.scale_relative_to_octave(s));
        const auto sigma_D = sigma_I / sqrt(2.f);
        cornerness(s, o) =
            scale_adapted_harris_cornerness(I, sigma_I, sigma_D, kappa);
      }
    }

    return cornerness;
  }

  bool local_min_x(int x, int y, ImageView<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x, y) > I(x + u, y))
        return false;
    return true;
  }

  bool local_min_y(int x, int y, ImageView<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x, y) < I(x + u, y))
        return false;
    return true;
  }

  vector<OERegion>
  ComputeHarrisLaplaceCorners::operator()(const ImageView<float>& I,
                                          vector<Point2i> *scale_octave_pairs)
  {
    auto& G = _gaussians;
    auto& cornerness = _harris;

    G = Sara::gaussian_pyramid(I, _pyr_params);
    cornerness = harris_cornerness_pyramid(I, _kappa, _pyr_params);

    auto corners = vector<OERegion>{};
    corners.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(int(1e4));
    }

    for (auto o = 0; o < cornerness.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (auto s = 1; s < cornerness.num_scales_per_octave(); ++s)
      {
        auto new_corners = laplace_maxima(cornerness, G, s, o, _extremum_thres,
                                          _img_padding_sz, _num_scales,
                                          _extremum_refinement_iter);

        append(corners, new_corners);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_corners.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s, o));
        }
      }
    }
    shrink_to_fit(corners);
    return corners;
  }

} /* namespace Sara */
} /* namespace DO */
