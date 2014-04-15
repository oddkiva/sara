// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_STUDY_AFFINE_METRIC_HPP
#define DO_STUDY_AFFINE_METRIC_HPP

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class StudySeedTriple : public StudyOnMikolajczykDataset
  {
  public:
    StudySeedTriple(const std::string& absParentFolderPath,
                    const std::string& name,
                    const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
    {}
    bool operator()(float inlierThres, float squaredEll,
                    size_t K, size_t k, double squaredRhoMin);
  };

} /* namespace DO */

#endif /* DO_STUDY_N_K_M_HPP */