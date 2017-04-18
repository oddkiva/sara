// ========================================================================== //
// This file is part of Sara which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifdef _OPENMP
#include <omp.h>
#endif

#include "EvaluateOutlierResistance.hpp"
#include "MatchNeighborhood.hpp"
#include "GrowMultipleRegions.hpp"


using namespace std;


namespace DO {

  bool EvalOutlierResistance::operator()(float squaredEll,
                                         size_t numRegionGrowths, size_t K,
                                         size_t k, double rho_min) const
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
    const auto mikolajczyk_inlier_thres = 1.5f;
    // Set of thresholds.
    auto thres = vector<float>{};
    thres.push_back(mikolajczyk_inlier_thres);
    thres.push_back(5.f);

    // ====================================================================== //
    // Let's go.
    for (int j = 1; j < 6; ++j)
    {
      auto pDrawer = unique_ptr<PairWiseDrawer>{};
      if (_display)
      {
        // View the image pair.
        pDrawer.reset(new PairWiseDrawer(dataset().image(0), dataset().image(j)));
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

        for (size_t t = 0; t != thres.size(); ++t)
        {
          bool success;
          success = doTheJob(M, H, j, squaredEll, thres[t], numRegionGrowths, K,
                             k, rho_min, pDrawer);
          if (!success)
          {
            if (_display)
              closeWindowForImagePair();

            return false;
          }
        }
      }

      if (_display)
        closeWindowForImagePair();
    }

    return true;
  }

  bool EvalOutlierResistance::doTheJob(const vector<Match>& M,
                                       const Matrix3f& H, size_t img_index,
                                       float squared_ell, float inlier_thres,
                                       size_t num_growths, size_t K, size_t k,
                                       double rho_min,
                                       const PairWiseDrawer *drawer) const
  {
    auto comment = std::string{};
    comment = "Evaluating outlier resistance on dataset '";
    comment += dataset().name() + "' :\n\tpair 1-" + to_string(img_index + 1);
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + to_string(squaredEll);
    comment += "\n\tK = " + to_string(K);
    comment += "\n\trho_min = " + to_string(rho_min);
    comment + "_inlierThres_" + to_string(inlierThres);
    print_stage(comment);

    // Get subset of matches.
    vector<size_t> inliers, outliers;
    getInliersAndOutliers(inliers, outliers, M, H, inlierThres);

    // We want to perform our analysis on this particular subset of matches of
    // interest.
    bool verbose = debug_ && pDrawer;
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.setInliers(inliers);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params, debug_ ? 1 : 0);
    vector<Region> RR(growMultipleRegions(numGrowths, &analyzer, pDrawer));
    cout << "Done!" << endl;

    // Compute the statistics.
    cout << "Computing stats... ";
    // Get found matches in a proper container.
    vector<size_t> allMatches;
    {
      Region allR;
      for (size_t i = 0; i != RR.size(); ++i)
        for (Region::iterator j = RR[i].begin(); j != RR[i].end(); ++j)
          allR.insert(*j);
      allMatches.reserve(allR.size());
      for (Region::iterator i = allR.begin(); i != allR.end(); ++i)
        allMatches.push_back(*i);
    }
    analyzer.computePosAndNeg(allMatches);

    // Save stats.
    cout << "Saving stats... ";
    string folder;
    folder = dataset().name() + "/outlier_resistance";
    folder = stringSrcPath(folder);
#pragma omp critical
    {
      createDirectory(folder);
    }

    const string name(
        dataset().name() + "_" + toString(1) + "_" + toString(imgIndex + 1) +
        "_sqEll_" + toString(squaredEll) + "_nReg_ " + toString(numGrowths) +
        "_K_" + toString(K) + "_rhoMin_" + toString(rho_min) + "_inlierThres_" +
        toString(inlierThres) + dataset().featType() + ".txt");

    bool success;
#pragma omp critical
    {
      success = analyzer.savePrecRecallEtc(string(folder + "/" + name));
    }
    if (!success)
    {
      cerr << "Could not save stats:\n" << string(folder + "/" + name) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true;
  }

  } /* namespace DO */
