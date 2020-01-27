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

#include "Statistics.hpp"

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Match.hpp>

#include <set>


namespace DO::Sara {

  //! In any case we need to decide good thresholds value,
  //! For that, one option is to think an extremely permissive thresholds.

  //! In any case, it is unavoidable that outliers will be added anyway !!!!
  //! The issue is to get the best parameters to optimize both accuracy and
  //! recall rate.
  class DO_SARA_EXPORT RegionGrowingAnalyzer
  {
  public:
    RegionGrowingAnalyzer(const std::vector<Match>& M, const Matrix3f& H,
                          bool verbose = false)
      : _M(M)
      , _H(H.cast<double>())
      , _verbose(verbose)
    {
    }

    //! Set data of interest.
    void set_inliers(const std::vector<size_t>& inliers);
    void set_subset_of_interest(const std::vector<size_t>& subsetOfInterest);
    void analyze_quad(const size_t t[3], size_t m);

    //! Easy accessors.
    const Match& M(size_t i) const
    {
      return _M[i];
    }

    bool is_inlier(size_t i) const
    {
      return _inliers.find(i) != _inliers.end();
    }

    bool is_of_interest(size_t i) const
    {
      return _subset_of_interest.find(i) != _subset_of_interest.end();
    }

    //! Quantify the quality of affine estimation.
    void compute_local_affine_consistency_statistics();
    bool
    save_local_affine_consistency_statistics(const std::string& name) const;

    //! Evaluate the performance of the region growing.
    void
    compute_positives_and_negatives(const std::vector<size_t>& foundMatches);
    bool save_precision_recall_etc(const std::string& name) const;

    //! Monitoring the number of regions and the number of fusions.
    void increment_num_fusions()
    {
      ++_num_fusions;
    }

    void set_num_regions(int num_regions)
    {
      _num_regions = num_regions;
    }

    void set_num_attempted_growths(int num_growths)
    {
      _num_attempted_growths = num_growths;
    }

    int num_fusions() const
    {
      return _num_fusions;
    }

    int num_regions() const
    {
      return _num_regions;
    }

    int num_growths() const
    {
      return _num_attempted_growths;
    }

    bool save_region_number_statistics(const std::string& name) const;

    //! Monitor the number of $\partial R$.
    void reset_region_boundary_evolution()
    {
      _size_dR.clear();
      _good.clear();
      _time.clear();
    }
    void push(size_t size_dR, size_t good, double time)
    {
      _size_dR.push_back(size_dR);
      _good.push_back(good);
      _time.push_back(time);
    }
    bool save_boundary_region_evolution(const std::string& name) const;

  private:
    // Input data.
    const std::vector<Match>& _M;
    Matrix3d _H;
    std::set<size_t> _inliers;
    std::set<size_t> _outliers;
    std::set<size_t> _subset_of_interest;

    //! Debugging flag.
    bool _verbose;

    //! For speed measurements.
    Timer _timer;

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
    std::vector<double> _triple_angles[3];
    //! To quantify the quality of affine estimation of (t,m);
    std::vector<double> _pix_dist_error, _ell_overlap_error,
        _angle_est_error_degree;
    //! Quality of local affine approximation.
    std::vector<Matrix3d> _diff_approx;
    std::vector<double> _abs_diff_approx;
    std::vector<double> _rel_diff_approx;
    //! Statistics on the quality of affine approximation.
    Statistics _pix_dist_error_stat, _ell_overlap_error_stat, _angle_est_error_stat;
    /*Stat diff_approx_stat_, abs_diff_approx_stat_, rel_diff_approx_stat_;*/


    // ====================================================================== //
    //! Below are data dedicated for the analysis of the
    //! class 'GrowMultipleRegions'.
    int _num_fusions{};
    int _num_regions{};
    int _num_attempted_growths;
    int tp, fp, tn, fn;


    // ====================================================================== //
    //! Monitor the number of $\partial R$.
    std::vector<size_t> _size_dR;
    std::vector<size_t> _good;
    std::vector<double> _time;
    std::vector<double> _spurious_pix_dist_error;
  };

}  // namespace DO::Sara
