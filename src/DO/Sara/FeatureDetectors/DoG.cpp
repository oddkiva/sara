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

namespace DO { namespace Sara {

  vector<OERegion>
  ComputeDoGExtrema::operator()(const Image<float>& I,
                                vector<Point2i> *scaleOctavePairs)
  {
    ImagePyramid<float>& G = gaussians_;
    ImagePyramid<float>& D = diff_of_gaussians_;
    G = DO::gaussianPyramid(I, params_);
    D = DO::DoGPyramid(G);

    vector<OERegion> extrema;
    extrema.reserve(int(1e4));
    if (scaleOctavePairs)
    {
      scaleOctavePairs->clear();
      scaleOctavePairs->reserve(10000);
    }

    for (int o = 0; o < D.numOctaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < D.numScalesPerOctave()-1; ++s)
      {
        vector<OERegion> newExtrema(localScaleSpaceExtrema(
          D, s, o, extremum_thres_, edge_ratio_thres_,
          img_padding_sz_, extremum_refinement_iter_) );
        append(extrema, newExtrema);

        if (scaleOctavePairs)
        {
          for (size_t i = 0; i != newExtrema.size(); ++i)
            scaleOctavePairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(extrema);
    return extrema;
  }

} /* namespace Sara */
} /* namespace DO */
