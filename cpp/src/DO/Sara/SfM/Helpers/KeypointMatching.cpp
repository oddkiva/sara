// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>

#include <DO/Sara/FeatureMatching.hpp>


namespace DO::Sara {

  auto match(const KeypointList<OERegion, float>& keys1,
             const KeypointList<OERegion, float>& keys2, float lowe_ratio)
      -> std::vector<Match>
  {
    AnnMatcher matcher{keys1, keys2, lowe_ratio};
    return matcher.compute_matches();
  }

} /* namespace DO::Sara */
