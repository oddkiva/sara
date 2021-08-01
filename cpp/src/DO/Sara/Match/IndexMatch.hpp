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

#pragma once

#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/Match/Match.hpp>


namespace DO::Sara {

  //! @addtogroup Match
  //! @{

  struct IndexMatch
  {
    int i;
    int j;
    float score;
  };

  inline auto to_match(const IndexMatch& m,
                       const KeypointList<OERegion, float>& k1,
                       const KeypointList<OERegion, float>& k2)
  {

    const auto& f1 = std::get<0>(k1);
    const auto& f2 = std::get<0>(k2);
    return Match{&f1[m.i], &f2[m.j],  //
                 m.score,  Match::Direction::SourceToTarget,
                 m.i,      m.j};
  };

  inline auto to_match(const std::vector<IndexMatch>& im,
                       const KeypointList<OERegion, float>& k1,
                       const KeypointList<OERegion, float>& k2)
  {
    auto m = std::vector<Match>{};
    m.reserve(im.size());
    std::transform(std::begin(im), std::end(im), std::back_inserter(m),
                   [&](const auto& im) { return to_match(im, k1, k2); });
    return m;
  };

  //! @}

} /* namespace DO::Sara */
