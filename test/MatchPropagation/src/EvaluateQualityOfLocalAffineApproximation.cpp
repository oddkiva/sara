// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "EvaluateQualityOfLocalAffineApproximation.hpp"
#include "MatchNeighborhood.hpp"
#include "GrowMultipleRegions.hpp"
#ifdef _OPENMP
# include <omp.h>
#endif

using namespace std;

namespace DO {

  bool
  EvalQualityOfLocalAffApprox::
  operator()(float squaredEll,size_t numRegionGrowths,
             size_t K, double rho_min) const
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

    // ====================================================================== //
    // Let's go.
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
        // Extract the subset of matches of interest.
        vector<IndexDist> sortedM(sortMatchesByReprojError(M, H));

        for (size_t lb = 0; lb != thres.size()-1; ++lb)
        {
          int ub = lb+1;
          bool success;
          success = doTheJob(M, sortedM, H, j, squaredEll,
                             thres[lb], thres[ub],
                             numRegionGrowths, K, rho_min, pDrawer);
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
  EvalQualityOfLocalAffApprox::
  doTheJob(const vector<Match>& M, const vector<IndexDist>& sortedM,
           const Matrix3f& H, size_t imgIndex, 
           float squaredEll, float lb, float ub,
           size_t numGrowths, size_t K, double rho_min,
           const PairWiseDrawer *pDrawer) const
  {
    string comment;
    comment  = dataset().name() + ":\n\tpair 1-"+toString(imgIndex+1);
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + toString(squaredEll);
    comment += "\n\tK = " + toString(K);
    comment += "\n\trho_min = " + toString(rho_min);
    comment += "\n\tlb = " + toString(lb);
    comment += "\n\tub = " + toString(ub);
    printStage(comment);

    // Get subset of matches.
    vector<size_t> I(getMatches(sortedM, lb, ub));
    // We want to perform our analysis on this particular subset of matches of interest.
    bool verbose = debug_ && pDrawer;
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.setSubsetOfInterest(I);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params,  debug_ ? 1 : 0);
    vector<Region> RR(growMultipleRegions(numGrowths, &analyzer, pDrawer));
    cout << "Done!" << endl;

    // Compute the statistics.
    cout << "Computing stats... ";
    analyzer.computeLocalAffineConsistencyStats();
    cout << "Done!" << endl;

    // Save stats.
    cout << "Saving stats... ";
    string folder; 
    folder = dataset().name()+"/Quality_Local_Aff";
    folder = stringSrcPath(folder);
#pragma omp critical
    {
      createDirectory(folder);
    }

    const string name( dataset().name() 
                    + "_" + toString(1) + "_" + toString(imgIndex+1)
                    + "_sqEll_" + toString(squaredEll)
                    + "_nReg_ " + toString(numGrowths)
                    + "_K_"  + toString(K)
                    + "_rhoMin_" + toString(rho_min)
                    + "_lb_" + toString(lb) 
                    + "_ub_" + toString(ub)
                    + dataset().featType()
                    + ".txt");
    
    bool success;
#pragma omp critical 
    {
      success = analyzer.saveLocalAffineConsistencyStats(string(folder+"/"+name));
    }
    if (!success)
    {
      cerr << "Could not save stats:\n" << string(folder+"/"+name) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true; 
  }

} /* namespace DO */