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
  ComputeLoGExtrema::operator()(const Image<float>& I,
                                vector<Point2i> *scale_octave_pairs)
  {
    ImagePyramid<float>& G = gaussians_;
    ImagePyramid<float>& L = laplacians_of_gaussians_;
    G = gaussian_pyramid(I, params_);
    L = laplacian_pyramid(G);

    vector<OERegion> extrema;
    extrema.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(1e4);
    }

    for (int o = 0; o < L.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < L.num_scales_per_octave()-1; ++s)
      {
        vector<OERegion> newExtrema(local_scale_space_extrema(
          L, s, o, extremum_thres_, edge_ratio_thres_,
          img_padding_sz_, extremum_refinement_iter_) );
        append(extrema, newExtrema);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != newExtrema.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(extrema);
    return extrema;
  }

} /* namespace Sara */
} /* namespace DO */
