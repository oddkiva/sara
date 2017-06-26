// ========================================================================== //
// This file is part of Sara which was presented in:
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

  class EvaluateOutlierResistance : public StudyOnMikolajczykDataset
  {
  public:
    EvaluateOutlierResistance(const std::string& abs_parent_folder_path,
                              const std::string& name,
                              const std::string& feature_type)
      : StudyOnMikolajczykDataset{abs_parent_folder_path, name, feature_type}
      , _debug{false}
      , _display{false}
    {
    }

    bool operator()(float squared_ell, size_t num_region_growths, size_t K,
                    size_t k, double rho_min) const;

  private:
    bool run(const std::vector<Match>& M, const Matrix3f& H,
             size_t img_index, float squared_ell, float inlier_thres,
             size_t num_region_growths, size_t K, size_t k, double rho_min,
             const PairWiseDrawer *drawer = 0) const;

  private:
    bool _debug;
    bool _display;
  };

} /* namespace DO */
