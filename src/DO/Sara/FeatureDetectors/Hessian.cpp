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


using namespace std;


namespace DO { namespace Sara {

  vector<OERegion>
  ComputeHessianLaplaceMaxima::
  operator()(const ImageView<float>& I, vector<Point2i> *scale_octave_pairs)
  {
    auto& gauss_pyr = _gaussians;
    auto& det_hess_pyr = _det_hessians;

    gauss_pyr = gaussian_pyramid(I, _pyr_params);
    det_hess_pyr = det_of_hessian_pyramid(gauss_pyr);

    auto det_hess_maxima = vector<OERegion>{};
    det_hess_maxima.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(int(1e4));
    }

    for (auto o = 0; o < det_hess_pyr.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < det_hess_pyr.num_scales_per_octave(); ++s)
      {
        vector<OERegion> new_det_hess_maxima(laplace_maxima(
          det_hess_pyr, gauss_pyr, s, o, _extremum_thres, _img_padding_sz,
          _num_scales, _extremum_refinement_iter) );

        append(det_hess_maxima, new_det_hess_maxima);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_det_hess_maxima.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(det_hess_maxima);
    return det_hess_maxima;
  }

  vector<OERegion>
  ComputeDoHExtrema::
  operator()(const ImageView<float>& I, vector<Point2i> *scale_octave_pairs)
  {
    auto& gauss_pyr = _gaussians;
    auto& det_hess_pyr = _det_hessians;

    gauss_pyr = gaussian_pyramid(I, pyr_params_);
    det_hess_pyr = det_of_hessian_pyramid(gauss_pyr);

    vector<OERegion> det_hess_extrema;
    det_hess_extrema.reserve(int(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(int(1e4));
    }

    for (auto o = 0; o < det_hess_pyr.num_octaves(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (auto s = 1; s < det_hess_pyr.num_scales_per_octave()-1; ++s)
      {
        auto new_det_hess_extrema = local_scale_space_extrema(
            det_hess_pyr, s, o, _extremum_thres, _edge_ratio_thres,
            _img_padding_sz, _extremum_refinement_iter);

        append(det_hess_extrema, new_det_hess_extrema);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_det_hess_extrema.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s,o));
        }
      }
    }
    shrink_to_fit(det_hess_extrema);
    return det_hess_extrema;
  }


} /* namespace Sara */
} /* namespace DO */
