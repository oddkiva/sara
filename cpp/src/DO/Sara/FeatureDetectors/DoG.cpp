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


namespace DO::Sara {

  auto ComputeDoGExtrema::operator()(const ImageView<float>& image,
                                     vector<Point2i>* scale_octave_pairs)
      -> vector<OERegion>
  {
    auto& G = _gaussians;
    auto& D = _diff_of_gaussians;

    auto timer = Timer{};
    auto tic_ = [&timer]() { timer.restart(); };
    auto toc_ = [&timer](const std::string& what) {
      const auto elapsed = timer.elapsed_ms();
      SARA_DEBUG << "[" << what << "] " << elapsed << "ms\n";
    };

    tic_();
    G = gaussian_pyramid(image, _pyramid_params, _gauss_truncate);
    toc_("Gaussian Pyramid");

    tic_();
    D = difference_of_gaussians_pyramid(G);
    toc_("DoG Pyramid");

    tic_();
    _extrema.resize(D.scale_count());
    for (int o = 0; o < D.octave_count(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (int s = 1; s < D.scale_count_per_octave() - 1; ++s)
      {
        // N.B.: the parallelization happens here so there is no point
        // parallelizing the outer loop.
        const auto so = o * D.scale_count_per_octave() + s;
        _extrema[so] = local_scale_space_extrema(
            D, s, o, _extremum_thres, _edge_ratio_thres, _img_padding_sz,
            _extremum_refinement_iter);

      }
    }

    // FIXME: remove this legacy code where we flatten the list of extrema. This
    // makes it hard to optimize the calculation of SIFT descriptors.
    auto extrema = vector<OERegion>{};
    extrema.reserve(static_cast<int>(1e4));
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(10000);
    }

    for (int o = 0; o < D.octave_count(); ++o)
    {
      for (int s = 0; s < D.scale_count_per_octave(); ++s)
      {
        const auto so = o * D.scale_count_per_octave() + s;
        append(extrema, _extrema[so]);
        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != _extrema[so].size(); ++i)
            scale_octave_pairs->push_back(Point2i(s, o));
        }
      }
    }

    toc_("DoG Extrema");
    shrink_to_fit(extrema);
    return extrema;
  }

}  // namespace DO::Sara
