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
            const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
    {}
    bool operator()(float inlierThres, float squaredEll);

  };

} /* namespace DO */

#endif /* DO_LEARN_P_F_HPP */