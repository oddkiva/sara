// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "StudySeedTriple.hpp"

using namespace std;

namespace DO {

  bool StudySeedTriple::operator()(float inlierThres, float squaredEll, 
                                   size_t K, double squaredRhoMin)
  {
    vector<Stat> statSeedTriple;
    vector<double> successRates;
    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const vector<Keypoint>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const vector<Keypoint>& Y = dataset().keys(j);
        // Compute initial matches.
        vector<Match> M(computeMatches(X, Y, squaredEll));
        // Find inliers and outliers using ground truth homography.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, dataset().H(j), inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        for (size_t i = 0; i != inliers.size(); ++i)
          drawer.drawMatch(M[inliers[i]], Cyan8);
        getKey();

        GrowthParams growthParams;
        size_t maxRegionSize = 50;
        size_t criticalSize = 7;
        DynamicMatchGraph G(M, growthParams.K(), growthParams.rhoMin());
        int numSuccess = 0;
        for (size_t i = 0; i != inliers.size(); ++i)
        {
          // Grow region from inliers seeds.
          size_t seed = inliers[i];
          GrowRegion growRegion(seed, G, growthParams);
          Region R(growRegion(0, &drawer, maxRegionSize));
          if (R.size() > criticalSize)
            ++numSuccess;
        }
        double successRate = double(numSuccess)/inliers.size();
        successRates.push_back(successRate);
      }
      closeWindowForImagePair();
    }

    string folder(dataset().name()); folder = stringSrcPath(folder);
    createDirectory(folder);

    return true;
  }

} /* namespace DO */