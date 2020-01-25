// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include "../HyperParameterLearning/Stat.hpp"

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Match.hpp>

#include <set>

namespace DO::Sara {

  //! In any case we need to decide good thresholds value,
  //! For that, one option is to think an extremely permissive thresholds.

  //! In any case, it is unavoidable that outliers will be added anyway !!!!
  //! The issue is to get the best parameters to optimize both accuracy and
  //! recall rate.
  class RegionGrowingAnalyzer
  {
  public:
    RegionGrowingAnalyzer(const std::vector<Match>& M, const Matrix3f& H,
                          bool verbose = false)
      : M_(M)
      , H_(H.cast<double>())
      , verbose_(verbose)
      , num_fusions_(0)
      , num_regions_(0)
    {
    }

    //! Set data of interest.
    void setInliers(const std::vector<size_t>& inliers);
    void setSubsetOfInterest(const std::vector<size_t>& subsetOfInterest);
    void analyzeQuad(const size_t t[3], size_t m);

    //! Easy accessors.
    const Match& M(size_t i) const
    {
      return M_[i];
    }
    bool isInlier(size_t i) const
    {
      return inliers_.find(i) != inliers_.end();
    }
    bool isOfInterest(size_t i) const
    {
      return subset_of_interest_.find(i) != subset_of_interest_.end();
    }

    //! Quantify the quality of affine estimation.
    void computeLocalAffineConsistencyStats();
    bool saveLocalAffineConsistencyStats(const std::string& name) const;

    //! Evaluate the performance of the region growing.
    void computePosAndNeg(const std::vector<size_t>& foundMatches);
    bool savePrecRecallEtc(const std::string& name) const;

    //! Monitoring the number of regions and the number of fusions.
    void incrementNumFusions()
    {
      ++num_fusions_;
    }
    void setNumRegions(int numRegions)
    {
      num_regions_ = numRegions;
    }
    void setNumAttemptedGrowths(int numGrowths)
    {
      num_attempted_growths_ = numGrowths;
    }
    int numFusions() const
    {
      return num_fusions_;
    }
    int numRegions() const
    {
      return num_regions_;
    }
    int numGrowths() const
    {
      return num_attempted_growths_;
    }
    bool saveNumRegionStats(const std::string& name) const;

    //! Monitor the number of $\partial R$.
    void resetEvolDR()
    {
      size_dR_.clear();
      good_.clear();
      time_.clear();
    }
    void push(size_t size_dR, size_t good, double time)
    {
      size_dR_.push_back(size_dR);
      good_.push_back(good);
      time_.push_back(time);
    }
    bool saveEvolDR(const std::string& name) const;

  private:
    // Input data.
    const std::vector<Match>& M_;
    Matrix3d H_;
    std::set<size_t> inliers_;
    std::set<size_t> outliers_;
    std::set<size_t> subset_of_interest_;

    //! Debugging flag.
    bool verbose_;

    //! For speed measurements.
    Timer timer_;

    // ====================================================================== //
    /*! Whenever an affine-consistent quadruple is found, check the quality
     *  of the affine estimation.
     *
     *  One thing to do first:
     *  If a match $m$ is an inlier, we analyze and store the needed values.
     *
     *  Then when $m=(x,y)$ is such that
     *  $\| \mathbf{H} \mathbf{x} - \mathbf{y} \|_2 \in [\delta_i,
     * \delta_{i+1}[$. with
     *  $\delta_i \in \{0, 1.5, 5, 10, 20, 30, 40, 50, 100, 200 \}$.
     *
     *  Therefore we need to store the following data:
     *
     */
    //! To quantify the level of triple degeneracy.
    std::vector<double> triple_angles_[3];
    //! To quantify the quality of affine estimation of (t,m);
    std::vector<double> pix_dist_error_, ell_overlap_error_,
        angle_est_error_degree_;
    //! Quality of local affine approximation.
    std::vector<Matrix3d> diff_approx_;
    std::vector<double> abs_diff_approx_;
    std::vector<double> rel_diff_approx_;
    //! Statistics on the quality of affine approximation.
    Stat pix_dist_error_stat_, ell_overlap_error_stat_, angle_est_error_stat_;
    /*Stat diff_approx_stat_, abs_diff_approx_stat_, rel_diff_approx_stat_;*/


    // ====================================================================== //
    //! Below are data dedicated for the analysis of the
    //! class 'GrowMultipleRegions'.
    int num_fusions_;
    int num_regions_;
    int num_attempted_growths_;
    int tp, fp, tn, fn;


    // ====================================================================== //
    //! Monitor the number of $\partial R$.
    std::vector<size_t> size_dR_;
    std::vector<size_t> good_;
    std::vector<double> time_;
    std::vector<double> spurious_pix_dist_error_;
  };

}  // namespace DO::Sara
