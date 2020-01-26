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

#include "RegionGrowingAnalyzer.hpp"

#include "LocalAffineConsistency.hpp"

#include <DO/Sara/Graphics.hpp>


using namespace std;


namespace DO::Sara {

  void RegionGrowingAnalyzer::set_inliers(const vector<size_t>& inliers)
  {
    // Fill set of inliers.
    for (size_t i = 0; i != inliers.size(); ++i)
      _inliers.insert(inliers[i]);
    // Fill set of outliers.
    for (size_t i = 0; i != _M.size(); ++i)
      if (!is_inlier(i))
        _outliers.insert(i);
  }

  void RegionGrowingAnalyzer::set_subset_of_interest(
      const vector<size_t>& subsetOfInterest)
  {
    for (size_t i = 0; i != subsetOfInterest.size(); ++i)
      _subset_of_interest.insert(subsetOfInterest[i]);
  }

  void RegionGrowingAnalyzer::analyze_quad(const size_t t[3], size_t m)
  {
    if (_verbose)
    {
      cout << "With inlier M[" << m << "]:\n" << M(m) << endl;
      cout << "Treating the quality of the triple: " << endl;
      cout << "t = { ";
      for (int i = 0; i < 3; ++i)
      {
        cout << t[i];
        if (i != 2)
          cout << ", ";
      }
      cout << " }" << endl;
      get_key();
    }

    // Shortcut.
    const OERegion& x = M(m).x();
    const OERegion& y = M(m).y();

    // Compute the local affinity.
    Matrix3f phi, inv_phi;
    phi = affinity_from_x_to_y(M(t[0]), M(t[1]), M(t[2]));
    inv_phi = phi.inverse();

    // Otherwise compute intersection area of intersecting ellipses.
    OERegion phi_x(transform_oeregion(x, phi));
    OERegion inv_phi_y(transform_oeregion(y, inv_phi));


    // Compute quality of approximations.
    Matrix3d true_phi(affinity(_H, x.center().cast<double>()));
    Matrix3d phid(phi.cast<double>());
    Matrix3d diff_approx(true_phi - phid);
    _diff_approx.push_back(diff_approx);
    _abs_diff_approx.push_back(diff_approx.norm());
    _rel_diff_approx.push_back(diff_approx.norm() / true_phi.norm());

    // Compute quality of triangles.
    double angleX[3], angleY[3];
    Point2d px[3], py[3];
    for (int i = 0; i < 3; ++i)
    {
      px[i] = M(t[i]).x_pos().cast<double>();
      py[i] = M(t[i]).y_pos().cast<double>();
    }
    get_triangle_angles_in_degree(angleX, px);
    get_triangle_angles_in_degree(angleY, py);
    for (int i = 0; i < 3; ++i)
    {
      _triple_angles[i].push_back(angleX[i]);
      _triple_angles[i].push_back(angleY[i]);
    }
    if (_verbose)
    {
      for (int i = 0; i < 3; ++i)
      {
        cout << "angleX[" << i << "] = " << angleX[i] << " ";
        cout << "angleY[" << i << "] = " << angleY[i] << endl;
      }
      get_key();
    }


    // Compute comparisons.
    float pix_dist_error[2];
    double angle_est_error_radian[2];
    double overlapRatio[2];
    compare_oeregions(pix_dist_error[0], angle_est_error_radian[0],
                      overlapRatio[0], phi_x, y);
    compare_oeregions(pix_dist_error[1], angle_est_error_radian[1],
                      overlapRatio[1], inv_phi_y, x);
    for (int i = 0; i < 2; ++i)
    {
      _pix_dist_error.push_back(pix_dist_error[i]);
      _ell_overlap_error.push_back(1 - overlapRatio[i]);
      _angle_est_error_degree.push_back(to_degree(angle_est_error_radian[i]));
    }
    if (_verbose)
    {
      for (int i = 0; i < 2; ++i)
      {
        cout << "pix_dist_error[" << i << "] = " << pix_dist_error[i] << endl;
        cout << "ell_overlap_error[" << i << "] = " << 1. - overlapRatio[i]
             << endl;
        cout << "_angle_est_error_degree[" << i
             << "] = " << to_degree(angle_est_error_radian[i]) << endl;
      }
      get_key();
    }
  }

  void RegionGrowingAnalyzer::compute_local_affine_consistency_statistics()
  {
    if (_subset_of_interest.empty())
    {
      const char* msg = "FATAL ERROR: cannot perform analysis because the "
                        "subset of matches of interest is empty!";
      throw std::runtime_error(msg);
    }

    _pix_dist_error_stat.compute_statistics(_pix_dist_error);
    _ell_overlap_error_stat.compute_statistics(_ell_overlap_error);
    _angle_est_error_stat.compute_statistics(_angle_est_error_degree);
    // diff_approx_stat_.computeStats(_pix_dist_error);
    // cout << "diff_approx_stat_OK" << endl;
    // abs_diff_approx_stat_.computeStats(abs_diff_approx_);
    // cout << "abs_diff_approx_stat_OK" << endl;
    // rel_diff_approx_stat_.computeStats(rel_diff_approx_);
    // cout << "rel_diff_approx_stat_OK" << endl;
  }

  bool RegionGrowingAnalyzer::save_local_affine_consistency_statistics(
      const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: diff dist center" << endl;
    out << _pix_dist_error_stat;
    out << "Statistics: jaccard" << endl;
    out << _ell_overlap_error_stat;
    out << "Statistics: diff angle error" << endl;
    out << _angle_est_error_stat;
    // out << "Statistics: diff_approx_stat_" << endl;
    // out << diff_approx_stat_;
    // out << "Statistics: abs_diff_approx_stat_" << endl;
    // out << abs_diff_approx_stat_;
    // out << "Statistics: rel_diff_approx_stat_" << endl;
    // out << rel_diff_approx_stat_;
    out.close();

    return true;
  }

  void
  RegionGrowingAnalyzer::compute_positives_and_negatives(const vector<size_t>& foundMatches)
  {
    if (_inliers.empty())
    {
      const char* msg = "FATAL ERROR: set of inliers is empty!";
      throw std::runtime_error(msg);
    }

    tp = fp = tn = fn = 0;
    for (size_t i = 0; i != foundMatches.size(); ++i)
    {
      if (is_inlier(foundMatches[i]))
        ++tp;
      else
        ++fp;
    }

    vector<int> myNeg(_M.size(), 1);
    for (size_t i = 0; i != foundMatches.size(); ++i)
      myNeg[foundMatches[i]] = 0;
    for (size_t i = 0; i != myNeg.size(); ++i)
    {
      if (myNeg[i] == 1)
      {
        bool isOutlier = !is_inlier(i);
        if (isOutlier)
          ++tn;
        else
          ++fn;
      }
    }

    if (fn + tn + tp + fp != _M.size())
    {
      const char* msg = "FATAL ERROR: fn+tn+tp+fp != _M.size()";
      throw std::runtime_error(msg);
    }
  }

  bool RegionGrowingAnalyzer::save_precision_recall_etc(const string& name) const
  {
    double prec = double(tp) / double(tp + fp);
    double recall = double(tp) / double(tp + fn);
    double tnr = double(tn) / double(tn + fp);
    double acc = double(tp + tn) / double(tp + tn + fp + fn);
    double fmeas = 2 * (prec * recall) / (prec + recall);

    ofstream out(name.c_str());
    if (!out.is_open())
    {
      cerr << "Cannot write file:\n" << name << endl;
      return false;
    }
    out << "tp = " << tp << endl;
    out << "fp = " << fp << endl;
    out << "tn = " << tn << endl;
    out << "fn = " << fn << endl;
    out << "prec = " << prec << endl;
    out << "recall = " << recall << endl;
    out << "tnr = " << tnr << endl;
    out << "acc = " << acc << endl;
    out << "fmeas = " << fmeas << endl;
    out.close();
    return true;
  }

  bool RegionGrowingAnalyzer::save_region_number_statistics(const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
    {
      cerr << "Cannot write file:\n" << name << endl;
      return false;
    }

    out << "num_fusion = " << _num_fusions << endl;
    out << "num_regions = " << _num_regions << endl;
    out << "num_attempted_growths = " << _num_attempted_growths << endl;
    out.close();

    return true;
  }

  bool RegionGrowingAnalyzer::save_boundary_region_evolution(
      const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
    {
      cerr << "Cannot write file:\n" << name << endl;
      return false;
    }
    out << "iter  size_dR  good  time_sec" << endl;
    for (size_t i = 0; i != _size_dR.size(); ++i)
      out << i << " " << _size_dR[i] << " " << _good[i] << " " << _time[i]
          << endl;
    out.close();
    return true;
  }

}  // namespace DO::Sara
