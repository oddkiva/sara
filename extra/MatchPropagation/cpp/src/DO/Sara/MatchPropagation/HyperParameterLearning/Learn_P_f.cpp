// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "Learn_P_f.hpp"
#include "MatchNeighborhood.hpp"
#include "LocalAffineConsistency.hpp"
#ifdef _OPENMP
# include <omp.h>
#endif

using namespace std;

namespace DO {

  bool LearnPf::operator()(float squaredEll) const
  {
    // ====================================================================== //
    /* Below: Mikolajczyk et al.'s parameter in their IJCV 2005 paper. 
     *
     * Let (x,y) be a match. It is an inlier if it satisfies:
     * $$\| \mathbf{H} \mathbf{x} - \mathbf{y} \|_2 < 1.5 \ \textrm{pixels}$$
     *
     * where $\mathbf{H}$ is the ground truth homography.
     * 1.5 pixels is used in the above-mentioned paper.
     */
    //float mikolajczykInlierThres = 1.5f;
    // Set of thresholds.
    vector<float> thres;
    thres.push_back(0.f);
    thres.push_back(1.5f);
    thres.push_back(5.f);
    thres.push_back(10.f);
    thres.push_back(20.f);
    thres.push_back(30.f);
    thres.push_back(40.f);
    thres.push_back(50.f);
    thres.push_back(100.f);
    thres.push_back(200.f);

    // ====================================================================== //
    // Array of stats.
    vector<vector<Stat> >
      stat_overlaps(thres.size()-1),
      stat_angles(thres.size()-1);


    // ====================================================================== //
    // Let's go.
    for (int j = 1; j < 6; ++j)
    {
      PairWiseDrawer *pDrawer;
      if (display_)
      {
        // View the image pair.
        pDrawer = new PairWiseDrawer(dataset().image(0), dataset().image(j));
        openWindowForImagePair(0, j);
        pDrawer->setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
        pDrawer->displayImages();
      }

      // The job is here.
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const Set<OERegion, RealDescriptor>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const Set<OERegion, RealDescriptor>& Y = dataset().keys(j);
        // Compute initial matches $\mathcal{M}$.
        vector<Match> M(computeMatches(X, Y, squaredEll));
        // Get ground truth homography
        const Matrix3f& H = dataset().H(j);
        // Extract the subset of matches of interest.
        vector<IndexDist> sortedM(sortMatchesByReprojError(M, H));

        for (size_t lb = 0; lb != thres.size()-1; ++lb)
        {
          int ub = lb+1;
          Stat stat_overlap, stat_angle;
          doTheJob(stat_overlap, stat_angle, M, sortedM, H,
                   thres[lb], thres[ub]/*, &drawer*/);
          stat_overlaps[lb].push_back(stat_overlap);
          stat_angles[lb].push_back(stat_angle);
        }
      }

      if (display_)
      {
        closeWindowForImagePair();
        if (pDrawer)
          delete pDrawer;
      }
    }

    // ====================================================================== //
    // Save stats.
    string folder; 
    folder = approx_ell_inter_area_ ? dataset().name()+"/P_f_approx" :
                                      dataset().name()+"/P_f";
    folder = stringSrcPath(folder);
#pragma omp critical
    {
      createDirectory(folder);
    }

    for (size_t lb = 0; lb != thres.size()-1; ++lb)
    {
      size_t ub = lb+1;
      const string name( dataset().name() 
                      + "_lb_" + toString(thres[lb]) 
                      + "_ub_" + toString(thres[ub])
                      + "_squaredEll_" + toString(squaredEll)
                      + dataset().featType()
                      + ".txt");

      bool success;
#pragma omp critical 
      {
        success = saveStats(folder+"/"+name, stat_overlaps[lb], stat_angles[lb]);
      }

      if (!success)
      {
        cerr << "Could not save stats:\n" << string(folder+"/"+name) << endl;
        return false;
      }
    }
    return true; 
  }

  bool LearnPf::saveStats(const string& name,
                           const vector<Stat>& stat_overlaps,
                           const vector<Stat>& stat_angles) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: overlaps" << endl;
    writeStats(out, stat_overlaps);
    out << "Statistics: angles" << endl;
    writeStats(out, stat_angles);
    out.close();

    return true;
  }

  void LearnPf::doTheJob(Stat& stat_overlap, Stat& stat_angle,
                         const vector<Match>& M,
                         const vector<IndexDist>& sortedM,
                         const Matrix3f& H,
                         float lb, float ub,
                         const PairWiseDrawer *pDrawer) const
  {
    // Get subset of matches.
    vector<size_t> I(getMatches(sortedM, lb, ub));
    // Store overlaps
    vector<double> overlaps(I.size()), angles(I.size());
    // Compute stuff for statistics.
#pragma omp parallel for
    for (int i = 0; i < I.size(); ++i)
    {
      const Match& m = M[I[i]];
      const OERegion& x = m.x();
      const OERegion& y = m.y();
      OERegion H_x = transformOERegion(x, H);

      float dist;
      double angle_diff_radian, overlapRatio;
      compareOERegion(dist, angle_diff_radian, overlapRatio, H_x, y,
                      approx_ell_inter_area_);

      if (debug_)
      {
        cout << "dist = " << dist << endl;
        cout << "(Analytical Comp. ) Overlap ratio = " << overlapRatio << endl;
        cout << "angle_H_ox = " << toDegree(H_x.orientation()) << " deg" << endl;
        cout << "angle_y     = " << toDegree(y.orientation()) << " deg" << endl;
        cout << "|angle_H_ox - angle_y| = " << angle_diff_radian << " deg" << endl << endl;
      }

      if (pDrawer)
      {
        pDrawer->displayImages();
        pDrawer->drawFeature(1, H_x, Blue8);
        pDrawer->drawFeature(1, y, Red8);
        getKey();
      }

      overlaps[i] = 1 - overlapRatio;
      angles[i] = toDegree(angle_diff_radian);
    }
    // Compute the stats.
    stat_overlap.computeStats(overlaps);
    stat_angle.computeStats(angles);
  }

} /* namespace DO */