// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#include <DO/LoweSIFTWrapper.hpp>
#include "StudyOnMikolajczykDataset.hpp"
#include "DO/MatchPropagation.hpp"

using namespace std;
using namespace DO;

class TestGrowRegion : public StudyOnMikolajczykDataset
{
public:
  TestGrowRegion(const string& absParentFolderPath,
                 const string& name,
                 const string& featType)
    : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
  {}

  void operator()()
  {
    float ell = 1.0f;
    float inlierThres = 5.f;
    size_t K = 200;
    double rho_min = 0.5;
    //
    double angleDeg1 = 15;
    double angleDeg2 = 25;
    //
    bool displayInliers = false;

    for (int j = 5; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Set of keypoints $\mathcal{X}$ in image 1.
        const Set<OERegion, RealDescriptor>& X = dataset().keys(0);
        // Set of keypoints $\mathcal{Y}$ in image 2.
        const Set<OERegion, RealDescriptor>& Y = dataset().keys(j);
        // Ground truth homography from image 1 to image 2.
        const Matrix3f& H = dataset().H(j);
        // Compute initial matches.
        vector<Match> M(computeMatches(X, Y, ell*ell));
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, H, inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        if (displayInliers)
        {
          for (size_t i = 0; i != inliers.size(); ++i)
            drawer.drawMatch(M[inliers[i]], Cyan8);
          getKey();
          drawer.displayImages();
        }

        RegionGrowingAnalyzer analyzer(M, H);
        analyzer.setSubsetOfInterest(inliers);

        // Grow region from the first seed
        size_t seed = inliers[0];
        GrowthParams growthParams(K, rho_min, angleDeg1, angleDeg2);
        DynamicMatchGraph G(M, growthParams.K(), growthParams.rhoMin());
        GrowRegion growRegion(seed, G, growthParams);
        Region R(growRegion(numeric_limits<size_t>::max(), &drawer, &analyzer));

        analyzer.computeLocalAffineConsistencyStats();
        /*string aff_stats_name = "local_aff_stat_" + toString(1) + "_" + toString(j+1)
                              + dataset().featType()
                              + ".txt";
        aff_stats_name = stringSrcPath(aff_stats_name);
        analyzer.saveLocalAffineConsistencyStats(aff_stats_name);*/

        string dR_stat_name = "evol_dR_size_"
                            + toString(1) + "_" + toString(j+1)
                            + "_ell_" + toString(ell)
                            + dataset().featType()
                            + ".txt";
        dR_stat_name = stringSrcPath(dR_stat_name);
        analyzer.saveEvolDR(dR_stat_name);
      }
      closeWindowForImagePair();
    }
  }
};

class TestGrowMultipleRegions : public StudyOnMikolajczykDataset
{
public:
  TestGrowMultipleRegions(const string& absParentFolderPath,
                          const string& name,
                          const string& featType)
    : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
  {}

  void operator()()
  {
    float ell = 1.0f;
    float inlierThres = 5.f;
    size_t K = 200;
    double rho_min = 0.3;
    //
    double angleDeg1 = 15;
    double angleDeg2 = 25;
    //
    bool displayInliers = false;

    for (int j = 4; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const Set<OERegion, RealDescriptor>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const Set<OERegion, RealDescriptor>& Y = dataset().keys(j);
        // Compute initial matches.
        vector<Match> M(computeMatches(X, Y, ell*ell));
//#define REDUNDANCY
#ifdef REDUNDANCY
        printStage("Removing redundant matches");
        // Get the redundancy components.
        vector<vector<size_t> > components;
        vector<size_t> representers;
        double thres = 3.0;
        ComputeN_K eliminateRedundancies(M, 1e3);
        eliminateRedundancies(components, representers, M, thres);
        // Only keep the best representers.
        vector<Match> filteredM(representers.size());
        for (size_t i = 0; i != filteredM.size(); ++i)
          filteredM[i] = M[representers[i]];
#else
        vector<Match>& filteredM = M;
#endif
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, filteredM, dataset().H(j), inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        if (displayInliers)
        {
          for (size_t i = 0; i != inliers.size(); ++i)
            drawer.drawMatch(M[inliers[i]], Cyan8);
          getKey();
          drawer.displayImages();
        }

        // Grow multiple regions.
        size_t N = 5000;
        GrowthParams params(K, rho_min, angleDeg1, angleDeg2);
        int verbose = 2;
        GrowMultipleRegions growMultipleRegions(M, params, verbose);
        //growMultipleRegions.buildHatN_Ks();
        vector<Region> RR(growMultipleRegions(N, 0, &drawer));
      }
      closeWindowForImagePair();
    }
  }
};

void testOnImage(const string& file1, const string& file2)
{
  Image<Rgb8> image1, image2;
  load(image1, file1);
  load(image2, file2);

  // View the image pair.
  printStage("Display image pair and the features");
  float scale = 1.f;
  int w = int((image1.width()+image2.width())*scale);
  int h = int(max(image1.height(), image2.height())*scale);
  openWindow(w, h);
  setAntialiasing(activeWindow());

  // Setup viewing.
  PairWiseDrawer drawer(image1, image2);
  drawer.setVizParams(scale, scale, PairWiseDrawer::CatH);
  drawer.displayImages();
  getKey();

  // Compute keypoints.
  Set<OERegion, RealDescriptor> keys1 = DoGSiftDetector().run(image1.convert<unsigned char>());
  Set<OERegion, RealDescriptor> keys2 = DoGSiftDetector().run(image2.convert<unsigned char>());
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  // Compute initial matches
  float ell = 1.0f;
  AnnMatcher matcher(keys1, keys2, ell);
  vector<Match> M = matcher.computeMatches();
  cout << M.size() << " matches" << endl;

  // Growing parameters.
  size_t K = 80;
  double rho_min = 0.3;
  //
  double angleDeg1 = 15;
  double angleDeg2 = 25;
  //
  size_t N = 1000;
  GrowthParams params(K, rho_min, angleDeg1, angleDeg2);
  // Grow multiple regions.
  int verbose = 2;
  GrowMultipleRegions growMultipleRegions(M, params, verbose);
  vector<Region> RR(growMultipleRegions(N, 0, &drawer));
  saveScreen(activeWindow(), srcPath("result.png"));
}

int main()
{
#ifdef VM_DATA_DIR
# define VM_STRINGIFY(s)  #s
# define VM_EVAL(s) VM_STRINGIFY(s)"/"
#endif

  // Dataset paths.
  string mikolajczyk_dataset_folder = string(VM_EVAL(VM_DATA_DIR)) + "Mikolajczyk/";
  cout << mikolajczyk_dataset_folder << endl;
  const string folders[8] = { 
    "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall" 
  };
  const string ext[4] = { ".dog", ".haraff", ".hesaff", ".mser" };

  // Select the test module you want to run.
  bool test_growRegionFromBestSeed = false;
  bool test_growMultipleRegions = false;
  size_t dataset = 0;
  size_t ext_index = 0;

  // Call the desired modules.
  if (test_growRegionFromBestSeed)
  {
    TestGrowRegion testGrowRegion(mikolajczyk_dataset_folder,
                                  folders[dataset], ext[ext_index]);
    testGrowRegion();
  }
  if (test_growMultipleRegions)
  {
    TestGrowMultipleRegions 
      testGrowMultipleRegions(mikolajczyk_dataset_folder,
                              folders[dataset], ext[ext_index]);
    testGrowMultipleRegions();
  }

  testOnImage(mikolajczyk_dataset_folder + "bark/img1.ppm",
              mikolajczyk_dataset_folder + "bark/img4.ppm");
  
  return 0;
}