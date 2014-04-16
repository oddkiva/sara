// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "StudyPerfWithHatN_K.hpp"
#include "GrowMultipleRegions.hpp"

using namespace std;

namespace DO {

  bool
  StudyPerfWithHat_N_K::
  operator()(float squaredEll, size_t numRegionGrowths,
             size_t K, double rho_min)
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
    float mikolajczykInlierThres = 1.5f;    
    // Set of thresholds.
    vector<float> thres;
    thres.push_back(mikolajczykInlierThres);
    thres.push_back(5.f);

    for (int j = 1; j < 6; ++j)
    {
      PairWiseDrawer *pDrawer = 0;
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

        for (size_t t = 0; t != thres.size(); ++t)
        {
          bool success;
          success = doTheJob(M, H, j, squaredEll,
                             thres[t],
                             numRegionGrowths, K, rho_min, false, pDrawer);
          success = doTheJob(M, H, j, squaredEll,
                             thres[t],
                             numRegionGrowths, K, rho_min, true, pDrawer);
          if (!success)
          {
            if (display_)
            {
              closeWindowForImagePair();
              if (pDrawer)
                delete pDrawer;
            }
            return false;
          }
        }
      }
      if (display_)
      {
        closeWindowForImagePair();
        if (pDrawer)
          delete pDrawer;
      }
    }

    
    return true;
  }

  bool
  StudyPerfWithHat_N_K::
  doTheJob(const vector<Match>& M,
           const Matrix3f& H, size_t imgIndex,
           float squaredEll, float inlierThres,
           size_t numRegionGrowths,
           size_t K, double rho_min,
           bool useHatN_K,
           const PairWiseDrawer *pDrawer) const
  {
    string comment;
    comment  = "Evaluating outlier resistance on dataset '";
    comment += dataset().name() + ":\n\tpair 1-"+toString(imgIndex+1);
    comment += (useHatN_K ? "\n\thatN_K" : "\n\tN_K");
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + toString(squaredEll);
    comment += "\n\tK = " + toString(K);
    comment += "\n\trho_min = " + toString(rho_min);
    comment += "\n\tinlierThres = " + toString(inlierThres);
    printStage(comment);

    // Get subset of matches.
    vector<size_t> inliers, outliers;
    getInliersAndOutliers(inliers, outliers, M, H, inlierThres);
    // We want to perform our analysis on this particular subset of matches of interest.
    bool verbose = debug_ && pDrawer;
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.setInliers(inliers);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params, debug_ ? 1 : 0);
    if (useHatN_K)
      growMultipleRegions.buildHatN_Ks();
    vector<Region> RR(growMultipleRegions(numRegionGrowths, &analyzer, pDrawer));
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
    folder = dataset().name()+"/performance_hat_N_K";
    folder = stringSrcPath(folder);
#pragma omp critical
    {
      createDirectory(folder);
    }

    string neighborhoodName = useHatN_K ? "_HatN_K_" : "_N_K_";

    const string namePrecRecall("prec_recall_"
                              + dataset().name() 
                              + "_" + toString(1) + "_" + toString(imgIndex+1)
                              + neighborhoodName
                              + "_sqEll_" + toString(squaredEll)
                              + "_nReg_ " + toString(numRegionGrowths)
                              + "_K_"  + toString(K)
                              + "_rhoMin_" + toString(rho_min)
                              + "_inlierThres_" + toString(inlierThres)
                              + dataset().featType()
                              + ".txt");

    const string nameStatRegions("stat_regions_"
                               + dataset().name() 
                               + "_" + toString(1) + "_" + toString(imgIndex+1)
                               + neighborhoodName
                               + "_sqEll_" + toString(squaredEll)
                               + "_nReg_ " + toString(numRegionGrowths)
                               + "_K_"  + toString(K)
                               + "_rhoMin_" + toString(rho_min)
                               + "_inlierThres_" + toString(inlierThres)
                               + dataset().featType()
                               + ".txt");

    bool success;
#pragma omp critical 
    {
      success = analyzer.savePrecRecallEtc(string(folder+"/"+namePrecRecall)) &&
                analyzer.saveNumRegionStats(string(folder+"/"+nameStatRegions));
    }
    if (!success)
    {
      cerr << "Could not save stats:" << endl
           << string(folder+"/"+namePrecRecall)  << endl
           << string(folder+"/"+nameStatRegions) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true; 
  }

} /* namespace DO */