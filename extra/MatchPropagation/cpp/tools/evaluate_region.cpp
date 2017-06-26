// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "StudyOnMikolajczykDataset.hpp"
#include "Region.hpp"
#include "RegionBoundary.hpp"

using namespace std;
using namespace DO;

class TestRegion : public StudyOnMikolajczykDataset
{
public:
  TestRegion(const string& absParentFolderPath,
             const string& name,
             const string& featType)
    : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
  {}

  void operator()()
  {
    for (int j = 1; j < 6; ++j)
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
        vector<Match> M(computeMatches(X, Y, 1.2f*1.2f));
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, dataset().H(j), 1.5f);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        for (size_t i = 0; i != inliers.size(); ++i)
          drawer.drawMatch(M[inliers[i]]);

        // Region
        Region R;
        for (size_t i = 0; i != inliers.size(); ++i)
        {
          //cout << "Inserting M["<<inliers[i]<<"]" << endl;
          R.insert(M[inliers[i]], M);
          //cout << "Is M["<<inliers[i]<<"] correctly inserted?" << endl;
          if (!R.find(M[inliers[i]], M))
          {
            cerr << "Cannot find match:\n" << M[inliers[i]] << endl;
            cerr << "Index is = " << inliers[i] << endl;
            break;
          }
          /*if (R.find(M[inliers[i]], M))
            cout << "M["<<inliers[i]<<"] correctly inserted" << endl;*/
        }
        cout << "R.size() = " << R.size() << endl;
        getKey();

      }
      closeWindowForImagePair();
    }
  }
};

class TestRegionBoundary : public StudyOnMikolajczykDataset
{
public:
  TestRegionBoundary(const string& absParentFolderPath,
                     const string& name,
                     const string& featType)
    : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
  {}

  void operator()()
  {
    for (int j = 1; j < 6; ++j)
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
        vector<Match> M(computeMatches(X, Y, 1.2f*1.2f));
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, dataset().H(j), 1.5f);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        for (size_t i = 0; i != inliers.size(); ++i)
          drawer.drawMatch(M[inliers[i]]);
        
        // ================================================================== //
        // Region boundary.
        // Testing insertion and query.
        RegionBoundary dR(M);
        for (size_t i = 0; i != inliers.size(); ++i)
        {
          //cout << "Inserting M["<<inliers[i]<<"]" << endl;
          dR.insert(M[inliers[i]]);
          //cout << "Is M["<<inliers[i]<<"] correctly inserted?" << endl;
          if (!dR.find(M[inliers[i]]))
          {
            cerr << "Cannot find match:\n" << M[inliers[i]] << endl;
            cerr << "Index is = " << inliers[i] << endl;
            break;
          }
          if (dR.find(M[inliers[i]]))
            cout << "M["<<inliers[i]<<"] correctly inserted" << endl;
          cout << "dR.size() = " << dR.size() << endl;
        }
        cout << "dR.size() = " << dR.size() << endl;
        getKey();
        // Testing iterators.
        for (RegionBoundary::const_iterator m = dR.begin(); m != dR.end(); ++m)
        {
          cout << "M[" << m.index() << "] = \n" << *m << endl;
          drawer.drawMatch(*m, Red8);
        }
        getKey();
        // Testing erasing.
        RegionBoundary dR2(M);
        dR2.insert(inliers[0]);
        // Testing iterators.
        for (RegionBoundary::const_iterator m = dR2.begin(); m != dR2.end(); ++m)
        {
          cout << "M[" << m.index() << "] = \n" << *m << endl;
          drawer.drawMatch(*m, Red8);
        }
        getKey();
        dR2.erase(inliers[0]);
        cout << "dR2.size() = " << dR2.size() << endl;
        getKey();
      }
      closeWindowForImagePair();
    }
  }
};

int main()
{
#ifdef VM_DATA_DIR
# define VM_STRINGIFY(s)  #s
# define VM_EVAL(s) VM_STRINGIFY(s)"/"
#endif

  // Dataset paths.
  const string mikolajczyk_dataset_folder = VM_EVAL(VM_DATA_DIR);
  const string folders[8] = { 
    "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall" 
  };
  const string ext[4] = { ".dog", ".haraff", ".hesaff", ".mser" };
  TestRegion testRegion(mikolajczyk_dataset_folder, folders[0], ext[0]);
  testRegion();

  TestRegionBoundary testRegionBoundary(mikolajczyk_dataset_folder, folders[0], ext[0]);
  testRegionBoundary();
  return 0;
}