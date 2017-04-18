// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_LEARN_P_F_HPP
#define DO_LEARN_P_F_HPP

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class LearnPf : public StudyOnMikolajczykDataset
  {
  public:
    LearnPf(const std::string& absParentFolderPath,
            const std::string& name,
            const std::string& featType,
            bool approxEllInterArea = false)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(false)
      , display_(false)
      , approx_ell_inter_area_(approxEllInterArea)
    {}
    bool operator()(float squaredEll) const;

  private:
    bool saveStats(const std::string& name,
                   const std::vector<Stat>& stat_overlaps,
                   const std::vector<Stat>& stat_angles) const;

    void doTheJob(Stat& stat_overlap, Stat& stat_angle,
                  const std::vector<Match>& M,
                  const std::vector<IndexDist>& sortedM,
                  const Matrix3f& H,
                  float lb, float ub,
                  const PairWiseDrawer *pDrawer = 0) const;

  private:
    bool debug_;
    bool display_;
    //bool find_dilation_;
    bool approx_ell_inter_area_;
  };

} /* namespace DO */

#endif /* DO_LEARN_P_F_HPP */