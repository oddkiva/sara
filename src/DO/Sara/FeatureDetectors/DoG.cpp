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


using namespace std;


namespace DO { namespace Sara {

  vector<OERegion>
  ComputeDoGExtrema::operator()(const Image<float>& image,
                                vector<Point2i> *scale_octave_pairs)
  {
    auto& G = _gaussians;
    auto& D = _diff_of_gaussians;
    G = gaussian_pyramid(image, _pyramid_params);
    D = difference_of_gaussians_pyramid(G);

    auto extrema = vector<OERegion>{};
    extrema.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(10000);
    }

    for (int o = 0; o < D.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < D.num_scales_per_octave()-1; ++s)
      {
        auto new_extrema = local_scale_space_extrema(
          D, s, o, _extremum_thres, _edge_ratio_thres,
          _img_padding_sz, _extremum_refinement_iter);
        append(extrema, new_extrema);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_extrema.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(extrema);
    return extrema;
  }

} /* namespace Sara */
} /* namespace DO */