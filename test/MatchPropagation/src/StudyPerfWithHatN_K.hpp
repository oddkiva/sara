// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_STUDY_PERF_WITH_HAT_N_K_HPP
#define DO_STUDY_PERF_WITH_HAT_N_K_HPP

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class StudyPerfWithHat_N_K : public StudyOnMikolajczykDataset
  {
  public:
    StudyPerfWithHat_N_K(const std::string& absParentFolderPath,
                         const std::string& name,
                         const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(false)
      , display_(false)
    {}
    bool operator()(float squaredEll, size_t numRegionGrowths,
                    size_t K, double rho_min);

  private:
    bool doTheJob(const std::vector<Match>& M,
                  const Matrix3f& H, size_t imgIndex,
                  float squaredEll, float inlierThres,
                  size_t numRegionGrowths,
                  size_t K, double rho_min,
                  bool useHatN_K,
                  const PairWiseDrawer *pDrawer = 0) const;
  private:
    bool debug_;
    bool display_;
  };

} /* namespace DO */

#endif /* DO_STUDY_PERF_WITH_HAT_N_K_HPP */