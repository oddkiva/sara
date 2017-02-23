// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_STUDY_N_K_M_HPP
#define DO_STUDY_N_K_M_HPP

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class Study_N_K_m : public StudyOnMikolajczykDataset
  {
  public:
    Study_N_K_m(const std::string& absParentFolderPath,
                const std::string& name,
                const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
    {}
    bool operator()(float inlierThres, float squaredEll,
                    size_t K, double squaredRhoMin);

  private:
    void getStat(Stat& stat_N_K, Stat& stat_hatN_K, Stat& stat_diff,
                 const std::vector<std::vector<size_t> >& N_K,
                 const std::vector<std::vector<size_t> >& hatN_K);

    void getStat(Stat& stat_N_K, Stat& stat_hatN_K, Stat& stat_diff,
                 const std::vector<size_t>& indices,
                 const std::vector<std::vector<size_t> >& N_K,
                 const std::vector<std::vector<size_t> >& hatN_K);

    
    void checkNeighborhood(const std::vector<std::vector<size_t> >& N_K,
                           const std::vector<Match>& M,
                           const PairWiseDrawer& drawer);

    bool saveStats(const std::string& name,
                   const std::vector<Stat>& stat_N_Ks,
                   const std::vector<Stat>& stat_hatN_Ks,
                   const std::vector<Stat>& stat_diffs);
  };

} /* namespace DO */

#endif /* DO_STUDY_N_K_M_HPP */