// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_MATCHREDUNDANCY_HPP
#define DO_MATCHREDUNDANCY_HPP

#include <DO/Core.hpp>
#include <DO/FeatureMatching.hpp>

namespace DO {

  void getRedundancyComponentsAndRepresenters(
    std::vector<std::vector<int> >& redundancies,
    std::vector<int>& representers,
    const std::vector<Match>& initialMatches,
    double thres = 1.5);
}

#endif /* DO_MATCHREDUNDANCY_HPP */