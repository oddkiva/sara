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

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Core/StdVectorHelpers.hpp>


using namespace std;


namespace DO { namespace Sara {

  Image<float> scale_adapted_harris_cornerness(const Image<float>& I,
                                               float sigma_I, float sigma_D,
                                               float kappa)
  {
    Image<Matrix2f> M;

    // Derive the smoothed function $g_{\sigma_I} * I$
    M = I.
      compute<Gaussian>(sigma_D).
      compute<Gradient>().
      compute<SecondMomentMatrix>().
      compute<Gaussian>(sigma_I);

    // Compute the cornerness function.
    Image<float> cornerness(I.sizes());
    Image<Matrix2f>::const_iterator M_it = M.begin();
    Image<float>::iterator c_it= cornerness.begin();
    for ( ; c_it != cornerness.end(); ++c_it, ++M_it)
      *c_it = M_it->determinant() - kappa*pow(M_it->trace(), 2);

    // Normalize the cornerness function.
    cornerness.array() *= pow(sigma_D, 2);
    return cornerness;
  }

  ImagePyramid<float> harris_cornerness_pyramid(const Image<float>& image,
                                              float kappa,
                                              const ImagePyramidParams& params)
  {
    // Resize the image with the appropriate factor.
    float resize_factor = pow(2.f, -params.first_octave_index());
    Image<float> I(enlarge(image, resize_factor) );

    // Deduce the new camera sigma with respect to the dilated image.
    float cameraSigma = float(params.scale_camera())*resize_factor;

    // Blur the image so that its new sigma is equal to the initial sigma.
    float scale_initial = float(params.scale_initial());
    if (cameraSigma < scale_initial)
    {
      float sigma = sqrt(scale_initial*scale_initial - cameraSigma*cameraSigma);
      I = deriche_blur(I, sigma);
    }

    // Deduce the maximum number of octaves.
    int l = std::min(image.width(), image.height());
    int b = params.image_padding_size();
    // l/2^k > 2b
    // 2^k < l/(2b)
    // k < log(l/(2b))/log(2)
    int num_octaves = static_cast<int>(log(l/(2.f*b))/log(2.f));

    // Shorten names.
    int num_scales = params.num_scales_per_octave();
    float k = float(params.scale_geometric_factor());

    // Create the image pyramid
    ImagePyramid<float> cornerness;
    cornerness.reset(num_octaves, num_scales, scale_initial, k);
    for (int o = 0; o < num_octaves; ++o)
    {
      // Compute the octave scaling factor
      cornerness.octave_scaling_factor(o) =
        (o == 0) ? 1.f/resize_factor : cornerness.octave_scaling_factor(o-1)*2;

      // Compute the gaussians in octave $o$
      if (o != 0)
        I = downscale(I, 2);
      for (int s = 0; s < num_scales; ++s)
      {
        float sigma_I = cornerness.scale_relative_to_octave(s);
        float sigma_D = sigma_I/sqrt(2.f);
        cornerness(s,o) = scale_adapted_harris_cornerness(I, sigma_I, sigma_D, kappa);
      }
    }

    return cornerness;
  }

  bool local_min_x(int x, int y, Image<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x,y) > I(x+u,y))
        return false;
    return true;
  }

  bool local_min_y(int x, int y, Image<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x,y) < I(x+u,y))
        return false;
    return true;
  }

  vector<OERegion>
  ComputeHarrisLaplaceCorners::operator()(const Image<float>& I,
                                          vector<Point2i> *scale_octave_pairs)
  {
    ImagePyramid<float>& G = _gaussians;
    ImagePyramid<float>& cornerness = _harris;

    G = Sara::gaussian_pyramid(I, _pyr_params);
    cornerness = harris_cornerness_pyramid(I, _kappa, _pyr_params);

    vector<OERegion> corners;
    corners.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(int(1e4));
    }

    for (int o = 0; o < cornerness.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < cornerness.num_scales_per_octave(); ++s)
      {
        vector<OERegion> new_corners(laplace_maxima(
          cornerness, G, s, o, _extremum_thres, _img_padding_sz,
          _num_scales, _extremum_refinement_iter) );

        append(corners, new_corners);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_corners.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(corners);
    return corners;
  }

} /* namespace Sara */
} /* namespace DO */