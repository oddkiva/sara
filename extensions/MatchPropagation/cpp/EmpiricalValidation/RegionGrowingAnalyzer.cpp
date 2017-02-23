// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#include "RegionGrowingAnalyzer.hpp"
#include "LocalAffineConsistency.hpp"
#include <DO/Graphics.hpp>

using namespace std;

namespace DO {

  void RegionGrowingAnalyzer::setInliers(const vector<size_t>& inliers)
  {
    // Fill set of inliers.
    for (size_t i = 0; i != inliers.size(); ++i)
      inliers_.insert(inliers[i]);
    // Fill set of outliers.
    for (size_t i = 0; i != M_.size(); ++i)
      if (!isInlier(i))
        outliers_.insert(i);
  }

  void RegionGrowingAnalyzer::setSubsetOfInterest(const vector<size_t>& subsetOfInterest)
  {
    for (size_t i = 0; i != subsetOfInterest.size(); ++i)
      subset_of_interest_.insert(subsetOfInterest[i]);
  }

  void RegionGrowingAnalyzer::analyzeQuad(const size_t t[3], size_t m)
  {
    if (verbose_)
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
      getKey();
    }

    // Shortcut.
    const OERegion& x = M(m).x();
    const OERegion& y = M(m).y();

    // Compute the local affinity.
    Matrix3f phi, inv_phi;
    phi = affinityFromXToY(M(t[0]), M(t[1]), M(t[2]));
    inv_phi = phi.inverse();

    // Otherwise compute intersection area of intersecting ellipses.
    OERegion phi_x(transformOERegion(x,phi));
    OERegion inv_phi_y(transformOERegion(y,inv_phi));


    // Compute quality of approximations.
    Matrix3d true_phi(affinity(H_, x.center().cast<double>()));
    Matrix3d phid(phi.cast<double>());
    Matrix3d diff_approx(true_phi - phid);
    diff_approx_.push_back(diff_approx);
    abs_diff_approx_.push_back(diff_approx.norm());
    rel_diff_approx_.push_back(diff_approx.norm()/true_phi.norm());

    // Compute quality of triangles.
    double angleX[3], angleY[3];
    Point2d px[3], py[3];
    for (int i = 0; i < 3; ++i)
    {
      px[i] = M(t[i]).posX().cast<double>();
      py[i] = M(t[i]).posY().cast<double>();
    }
    getTriangleAnglesDegree(angleX, px);
    getTriangleAnglesDegree(angleY, py);
    for (int i = 0; i < 3; ++i)
    {
      triple_angles_[i].push_back(angleX[i]);
      triple_angles_[i].push_back(angleY[i]);
    }
    if (verbose_)
    {
      for (int i = 0; i < 3; ++i)
      {
        cout << "angleX[" << i << "] = " << angleX[i] << " ";
        cout << "angleY[" << i << "] = " << angleY[i] << endl;
      }
      getKey();
    }


    // Compute comparisons.
    float pix_dist_error[2];
    double angle_est_error_radian[2];
    double overlapRatio[2];
    compareOERegion(pix_dist_error[0],
                    angle_est_error_radian[0],
                    overlapRatio[0],
                    phi_x, y);
    compareOERegion(pix_dist_error[1],
                    angle_est_error_radian[1],
                    overlapRatio[1],
                    inv_phi_y, x);
    for (int i = 0; i < 2; ++i)
    {
      pix_dist_error_.push_back(pix_dist_error[i]);
      ell_overlap_error_.push_back(1-overlapRatio[i]);
      angle_est_error_degree_.push_back(toDegree(angle_est_error_radian[i]));
    }
    if (verbose_)
    {
      for (int i = 0; i < 2; ++i)
      {
        cout << "pix_dist_error["<<i<<"] = "
             << pix_dist_error[i] << endl;
        cout << "ell_overlap_error["<<i<<"] = "
             << 1. - overlapRatio[i] << endl;
        cout << "angle_est_error_degree_["<<i<<"] = "
             << toDegree(angle_est_error_radian[i]) << endl;
      }
      getKey();
    }
  }

  void
  RegionGrowingAnalyzer::computeLocalAffineConsistencyStats()
  {
    if (subset_of_interest_.empty())
    {
      const char *msg = "FATAL ERROR: cannot perform analysis because the subset of matches of interest is empty!";
      throw std::runtime_error(msg);
    }

    pix_dist_error_stat_.computeStats(pix_dist_error_);
    ell_overlap_error_stat_.computeStats(ell_overlap_error_);
    angle_est_error_stat_.computeStats(angle_est_error_degree_);
    //diff_approx_stat_.computeStats(pix_dist_error_);
    //cout << "diff_approx_stat_OK" << endl;
    //abs_diff_approx_stat_.computeStats(abs_diff_approx_);
    //cout << "abs_diff_approx_stat_OK" << endl;
    //rel_diff_approx_stat_.computeStats(rel_diff_approx_);
    //cout << "rel_diff_approx_stat_OK" << endl;
  }

  bool
  RegionGrowingAnalyzer::
  saveLocalAffineConsistencyStats(const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: diff dist center" << endl;
    out << pix_dist_error_stat_;
    out << "Statistics: jaccard" << endl;
    out << ell_overlap_error_stat_;
    out << "Statistics: diff angle error" << endl;
    out << angle_est_error_stat_;
    //out << "Statistics: diff_approx_stat_" << endl;
    //out << diff_approx_stat_;
    //out << "Statistics: abs_diff_approx_stat_" << endl;
    //out << abs_diff_approx_stat_;
    //out << "Statistics: rel_diff_approx_stat_" << endl;
    //out << rel_diff_approx_stat_;
    out.close();

    return true;
  }

  void
  RegionGrowingAnalyzer::
  computePosAndNeg(const vector<size_t>& foundMatches)
  {
    if (inliers_.empty())
    {
      const char *msg = "FATAL ERROR: set of inliers is empty!";
      throw std::runtime_error(msg);
    }

    tp = fp = tn = fn = 0;
    for (size_t i = 0; i != foundMatches.size(); ++i)
    {
      if (isInlier(foundMatches[i]))
        ++tp;
      else
        ++fp;
    }
    
    vector<int> myNeg(M_.size(), 1);
    for (size_t i = 0; i != foundMatches.size(); ++i)
      myNeg[foundMatches[i]] = 0;
    for (size_t i = 0; i != myNeg.size(); ++i)
    {
      if (myNeg[i] == 1)
      {
        bool isOutlier = !isInlier(i);
        if (isOutlier)
          ++tn;
        else
          ++fn;
      }
    }

    if (fn+tn+tp+fp != M_.size())
    {
      const char *msg = "FATAL ERROR: fn+tn+tp+fp != M_.size()";
      throw std::runtime_error(msg);
    }
  }

  bool RegionGrowingAnalyzer::savePrecRecallEtc(const string& name) const
  {
    double prec = double(tp) / double(tp+fp);
    double recall = double(tp) / double(tp+fn);
    double tnr = double(tn) / double(tn+fp);
    double acc = double(tp+tn) / double(tp+tn+fp+fn);
    double fmeas = 2 * (prec*recall) /(prec+recall);

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

  bool RegionGrowingAnalyzer::saveNumRegionStats(const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
    {
      cerr << "Cannot write file:\n" << name << endl;
      return false;
    }
    
    out << "num_fusion = " << num_fusions_ << endl;
    out << "num_regions = " << num_regions_ << endl;
    out << "num_attempted_growths = " << num_attempted_growths_ << endl;
    out.close();

    return true;
  }

  bool RegionGrowingAnalyzer::saveEvolDR(const string& name) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
    {
      cerr << "Cannot write file:\n" << name << endl;
      return false;
    }
    out << "iter  size_dR  good  time_sec" << endl;
    for (size_t i = 0; i != size_dR_.size(); ++i)
      out << i << " " << size_dR_[i] << " " << good_[i] << " " << time_[i] << endl;
    out.close();
    return true;
  }

} /* namespace DO */