// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/FeatureDetectors.hpp>
#include <DO/Core/StdVectorHelpers.hpp>

using namespace std;

namespace DO {

  Image<float> scaleAdaptedHarrisCornerness(const Image<float>& I,
                                            float sigma_I, float sigma_D, 
                                            float kappa)
  {
    Image<Matrix2f> M;
    // Derive the smoothed function $g_{\sigma_I} * I$
    M = I.compute<Gaussian>(sigma_D).
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

  ImagePyramid<float> harrisCornernessPyramid(const Image<float>& image, 
                                              float kappa,
                                              const ImagePyramidParams& params)
  {
    // Resize the image with the appropriate factor.
    float resizeFactor = pow(2.f, -params.initOctaveIndex());
    Image<float> I(enlarge(image, resizeFactor) );
    // Deduce the new camera sigma with respect to the dilated image.
    float cameraSigma = float(params.cameraSigma())*resizeFactor;
    // Blur the image so that its new sigma is equal to the initial sigma.
    float initSigma = float(params.initSigma());
    if (cameraSigma < initSigma)
    {
      float sigma = sqrt(initSigma*initSigma - cameraSigma*cameraSigma);
      I = dericheBlur(I, sigma);
    }

    // Deduce the maximum number of octaves.
    int l = std::min(image.width(), image.height());
    int b = params.imagePaddingSize();
    // l/2^k > 2b
    // 2^k < l/(2b)
    // k < log(l/(2b))/log(2)
    int numOctaves = static_cast<int>(log(l/(2.f*b))/log(2.f));

    // Shorten names.
    int numScales = params.numScalesPerOctave();
    float k = float(params.scaleGeomFactor());

    // Create the image pyramid
    ImagePyramid<float> cornerness;
    cornerness.reset(numOctaves, numScales, initSigma, k);
    for (int o = 0; o < numOctaves; ++o)
    {
      // Compute the octave scaling factor
      cornerness.octaveScalingFactor(o) = 
        (o == 0) ? 1.f/resizeFactor : cornerness.octaveScalingFactor(o-1)*2;

      // Compute the gaussians in octave $o$
      if (o != 0)
        I = downscale(I, 2);
      for (int s = 0; s < numScales; ++s)
      {
        float sigma_I = cornerness.octRelScale(s);
        float sigma_D = sigma_I/sqrt(2.f);
        cornerness(s,o) = scaleAdaptedHarrisCornerness(I, sigma_I, sigma_D, kappa);
      }
    }

    return cornerness;
  }

  bool localMinX(int x, int y, Image<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x,y) > I(x+u,y))
        return false;
    return true;
  }

  bool localMinY(int x, int y, Image<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x,y) < I(x+u,y))
        return false;
    return true;
  }

  vector<OERegion>
  ComputeHarrisLaplaceCorners::operator()(const Image<float>& I,
                                          vector<Point2i> *scaleOctavePairs)
  {
    ImagePyramid<float>& G = gaussians_;
    ImagePyramid<float>& cornerness = harris_;

    G = DO::gaussianPyramid(I, pyr_params_);
    cornerness = harrisCornernessPyramid(I, kappa_, pyr_params_);

    vector<OERegion> corners;
    corners.reserve(int(1e4));
    if (scaleOctavePairs)
    {
      scaleOctavePairs->clear();
      scaleOctavePairs->reserve(1e4);
    }

    for (int o = 0; o < cornerness.numOctaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < cornerness.numScalesPerOctave(); ++s)
      {
        vector<OERegion> newCorners(laplaceMaxima(
          cornerness, G, s, o, extremum_thres_, img_padding_sz_,
          num_scales_, extremum_refinement_iter_) );

        append(corners, newCorners);

        if (scaleOctavePairs)
        {
          for (size_t i = 0; i != newCorners.size(); ++i)
            scaleOctavePairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(corners);
    return corners;
  }

} /* namespace DO */
