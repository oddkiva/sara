// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#ifndef DO_GROWREGION_GROWMULTIPLEREGIONS_HPP
#define DO_GROWREGION_GROWMULTIPLEREGIONS_HPP

#include "GrowRegion.hpp"

namespace DO {

  class GrowMultipleRegions
  {
  public: /* interface. */
    GrowMultipleRegions(const std::vector<Match>& matches,
                        const GrowthParams& params,
                        int verbose = 0);

    std::vector<Region> operator()(size_t N,
                                   RegionGrowingAnalyzer *pAnalyzer = 0,
                                   const PairWiseDrawer *pDrawer = 0);

    void buildHatN_Ks() { G_.buildHatN_Ks(); }

  private:
    void markReliableMatches(Region& allR, const Region& R) const;
    void mergeRegions(std::vector<Region>& RR,
                      const std::pair<Region, std::vector<size_t> >& result) const;
    void checkRegions(const std::vector<Region>& RR,
                      const PairWiseDrawer *pDrawer) const;

  private: /* data members. */
    //! Dynamic graph of matches containing the set of initial matches
    //! $\mathcal{M}$.
    DynamicMatchGraph G_;
    //! Growth parameters.
    GrowthParams params_;
    //! Verbose flag for debugging.
    int verbose_;
  };

} /* namespace DO */

#endif /* DO_GROWREGION_GROWMULTIPLEREGIONS_HPP */