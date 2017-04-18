// ========================================================================== //
// This file is part of Sara.
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#pragma once

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class EvalQualityOfLocalAffApprox : public StudyOnMikolajczykDataset
  {
  public:
    EvalQualityOfLocalAffApprox(const std::string& absParentFolderPath,
                                const std::string& name,
                                const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , _debug(false)
      , _display(false)
    {}
    bool operator()(float squaredEll, size_t numRegionGrowths,
                    size_t K, double rho_min) const;

  private:
    bool run(const std::vector<Match>& M, const std::vector<IndexDist>& sortedM,
             const Matrix3f& H, size_t imgIndex, float squaredEll, float lb,
             float ub, size_t numRegionGrowths, size_t K, double rho_min,
             const PairWiseDrawer* pDrawer = 0) const;

  private:
    bool _debug;
    bool _display;
  };

} /* namespace DO */
