// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>

using namespace std;

namespace DO {

  vector<OERegion>
  ComputeHessianLaplaceMaxima::
  operator()(const Image<float>& I, vector<Point2i> *scaleOctavePairs)
  {
    ImagePyramid<float>& gaussPyr = gaussians_;
    ImagePyramid<float>& detHessPyr = det_hessians_;

    gaussPyr = DO::gaussianPyramid(I, pyr_params_);
    detHessPyr = DoHPyramid(gaussPyr);

    vector<OERegion> detHessMaxima;
    detHessMaxima.reserve(int(1e4));
    if (scaleOctavePairs)
    {
      scaleOctavePairs->clear();
      scaleOctavePairs->reserve(1e4);
    }

    for (int o = 0; o < detHessPyr.numOctaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < detHessPyr.numScalesPerOctave(); ++s)
      {
        vector<OERegion> newDetHessMaxima(laplaceMaxima(
          detHessPyr, gaussPyr, s, o, extremum_thres_, img_padding_sz_,
          num_scales_, extremum_refinement_iter_) );

        append(detHessMaxima, newDetHessMaxima);

        if (scaleOctavePairs)
        {
          for (size_t i = 0; i != newDetHessMaxima.size(); ++i)
            scaleOctavePairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(detHessMaxima);
    return detHessMaxima;
  }

  vector<OERegion>
  ComputeDoHExtrema::
  operator()(const Image<float>& I, vector<Point2i> *scaleOctavePairs)
  {
    ImagePyramid<float>& gaussPyr = gaussians_;
    ImagePyramid<float>& detHessPyr = det_hessians_;

    gaussPyr = DO::gaussianPyramid(I, pyr_params_);
    detHessPyr = DoHPyramid(gaussPyr);

    vector<OERegion> detHessExtrema;
    detHessExtrema.reserve(int(1e4));
    if (scaleOctavePairs)
    {
      scaleOctavePairs->clear();
      scaleOctavePairs->reserve(1e4);
    }

    for (int o = 0; o < detHessPyr.numOctaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < detHessPyr.numScalesPerOctave()-1; ++s)
      {
        vector<OERegion> newDetHessExtrema(localScaleSpaceExtrema(
          detHessPyr, s, o, extremum_thres_, edge_ratio_thres_,
          img_padding_sz_, extremum_refinement_iter_) );

        append(detHessExtrema, newDetHessExtrema);

        if (scaleOctavePairs)
        {
          for (size_t i = 0; i != newDetHessExtrema.size(); ++i)
            scaleOctavePairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(detHessExtrema);
    return detHessExtrema;
  }


} /* namespace DO */