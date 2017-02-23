// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_DEBUGELLIPSEINTERAREA_HPP
#define DO_DEBUGELLIPSEINTERAREA_HPP

#include "StudyOnMikolajczykDataset.hpp"

namespace DO {

  class DebugEllipseInterArea : public StudyOnMikolajczykDataset
  {
  public:
    DebugEllipseInterArea(const std::string& absParentFolderPath,
            const std::string& name,
            const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(true)
    {}
    bool operator()(float inlierThres, float squaredEll);

  private:
    void checkReprojectedEllipse(const Match& m, const PairWiseDrawer& drawer,
                                 Ellipse& y,
                                 Ellipse& H_Sx,
                                 double polyApproxOverlap,
                                 double analyticalOverlap,
                                 double angle_phi_ox,
                                 double angle_y,
                                 double error) const;

    bool saveStats(const std::string& name,
                    const std::vector<Stat>& stat_overlaps,
                    const std::vector<Stat>& stat_angles) const;

  private:
    bool debug_;
  };

} /* namespace DO */

#endif /* DO_DEBUGELLIPSEINTERAREA_HPP */