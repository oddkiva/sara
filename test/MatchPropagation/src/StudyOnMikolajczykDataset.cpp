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

using namespace std;

namespace DO {

  StudyOnMikolajczykDataset::
  StudyOnMikolajczykDataset(const std::string& absParentFolderPath,
                            const std::string& name,
                            const std::string& featType)
    : dataset_(absParentFolderPath, name)
  {
    dataset_.loadKeys(featType);
    //dataset_.check();
  }
  
  void StudyOnMikolajczykDataset::openWindowForImagePair(size_t i, size_t j) const
  {
    int w = int((dataset().image(0).width()+dataset().image(1).width()));
    int h = max(dataset().image(0).height(), dataset().image(1).height());
    openWindow(w, h);
    setAntialiasing();
  }

  void StudyOnMikolajczykDataset::closeWindowForImagePair() const
  {
    closeWindow();
  }

  vector<Match>
  StudyOnMikolajczykDataset::
  computeMatches(const Set<OERegion, RealDescriptor>& X,
                 const Set<OERegion, RealDescriptor>& Y,
                 float squaredEll) const
  {
    printStage("Computing initial matches $\\mathcal{M}$ with $\\ell = "
              + toString(sqrt(squaredEll)) + "$");
    vector<Match> M;
    AnnMatcher matcher(X, Y, squaredEll);
    M = matcher.computeMatches();
    return M;
  }

  void 
  StudyOnMikolajczykDataset::getInliersAndOutliers(vector<size_t>& inliers,
                                                   vector<size_t>& outliers,
                                                   const vector<Match>& matches,
                                                   const Matrix3f& H,
                                                   float thres) const
  {
    inliers.reserve(matches.size());
    for (size_t i = 0; i != matches.size(); ++i)
    {
      Vector3f x;
      x << matches[i].posX(), 1.f;
      const Vector2f& y = matches[i].posY();
      
      Vector3f Hx_ = H*x; Hx_ /= Hx_(2);
      Vector2f Hx(Hx_(0), Hx_(1));

      if ( (Hx-y).squaredNorm() < thres*thres )
        inliers.push_back(i);
      else
        outliers.push_back(i);
    }
  }

  vector<StudyOnMikolajczykDataset::IndexDist>
  StudyOnMikolajczykDataset::sortMatchesByReprojError(const vector<Match>& M,
                                                      const Matrix3f& H) const
  {
    CompareIndexDist cmp;
    vector<IndexDist> indexDists(M.size());
    for (size_t i = 0; i != M.size(); ++i)
    {
      Vector3f xh; xh << M[i].posX(), 1.f;
      Vector3f H_xh = H*xh; H_xh/= H_xh(2);
      Vector2f H_x(H_xh.block(0,0,2,1));
      const Vector2f& y = M[i].posY();
      indexDists[i] = make_pair(i, (H_x - y).norm());
    }
    sort(indexDists.begin(), indexDists.end(), cmp);

    return indexDists;
  }

  vector<size_t>
  StudyOnMikolajczykDataset::getMatches(const vector<IndexDist>& sortedMatches,
                                        float reprojLowerBound,
                                        float reprojUpperBound) const
  {
    vector<size_t> indices;
    indices.reserve(sortedMatches.size());
    for (int i = 0; i != sortedMatches.size(); ++i)
    {
      if (sortedMatches[i].second < reprojLowerBound)
        continue;
      if (sortedMatches[i].second >= reprojUpperBound)
        break;
      indices.push_back(sortedMatches[i].first);
    }
    return indices;
  }


} /* namespace DO */