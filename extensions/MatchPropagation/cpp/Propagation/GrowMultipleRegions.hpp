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

// ========================================================================== //
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#pragma once

#include "GrowRegion.hpp"

namespace DO { namespace Sara { namespace Extensions {

  class GrowMultipleRegions
  {
  public: /* interface. */
    GrowMultipleRegions(const std::vector<Match>& matches,
                        const GrowthParams& params,
                        int verbose = 0);

    std::vector<Region> operator()(size_t N,
                                   RegionGrowingAnalyzer *pAnalyzer = 0,
                                   const PairWiseDrawer *pDrawer = 0);

    void build_hat_N_Ks()
    {
      G_.buildHatN_Ks();
    }

  private:
    void mark_reliable_matches(Region& allR, const Region& R) const;

    void
    merge_regions(std::vector<Region>& RR,
                  const std::pair<Region, std::vector<size_t>>& result) const;

    void check_regions(const std::vector<Region>& RR,
                       const PairWiseDrawer* pDrawer) const;

  private: /* data members. */
    //! Dynamic graph of matches containing the set of initial matches
    //! $\mathcal{M}$.
    DynamicMatchGraph G_;

    //! Growth parameters.
    GrowthParams params_;

    //! Verbose flag for debugging.
    int verbose_;
  };

} /* namespace Extensions */
} /* namespace Sara */
} /* namespace DO */
