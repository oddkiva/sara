// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "Corner.hpp"


namespace DO::Sara {

  auto select(const DO::Sara::ImageView<float>& cornerness,
              const float image_scale, const float sigma_I,
              const float cornerness_adaptive_thres, const int border)
      -> std::vector<Corner<int>>
  {
    namespace sara = DO::Sara;

    const auto extrema = sara::local_maxima(cornerness);

    const auto cornerness_max = cornerness.flat_array().maxCoeff();
    const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

    auto extrema_filtered = std::vector<Corner<int>>{};
    extrema_filtered.reserve(extrema.size());
    for (const auto& p : extrema)
    {
      const auto in_image_domain =
          border <= p.x() && p.x() < cornerness.width() - border &&  //
          border <= p.y() && p.y() < cornerness.height() - border;
      if (in_image_domain && cornerness(p) > cornerness_thres)
        extrema_filtered.push_back({p, cornerness(p), image_scale * sigma_I});
    }

    return extrema_filtered;
  };

}  // namespace DO::Sara
