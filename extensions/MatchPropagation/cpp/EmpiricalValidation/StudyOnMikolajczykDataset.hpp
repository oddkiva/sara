// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_STUDY_ON_MIKOLAJCZYK_DATASET_HPP
#define DO_STUDY_ON_MIKOLAJCZYK_DATASET_HPP

#include <DO/FileSystem.hpp>
#include "MikolajczykDataset.hpp"
#include "Stat.hpp"

namespace DO {

  class StudyOnMikolajczykDataset
  {
  public:
    // Index Distance Pair
    typedef std::pair<size_t, float> IndexDist;
    struct CompareIndexDist
    {
      bool operator()(const IndexDist& p1, const IndexDist& p2) const
      { return p1.second < p2.second; }
    };

    // Constructor
    StudyOnMikolajczykDataset(const std::string& absParentFolderPath,
                              const std::string& name,
                              const std::string& featType);

    // Viewing, convenience functions...
    const MikolajczykDataset& dataset() const { return dataset_; }
    void openWindowForImagePair(size_t i, size_t j) const;
    void closeWindowForImagePair() const;
    // Match related functions.
    std::vector<Match> computeMatches(const Set<OERegion, RealDescriptor>& X,
                                      const Set<OERegion, RealDescriptor>& Y,
                                      float squaredEll) const;
    void getInliersAndOutliers(std::vector<size_t>& inliers,
                               std::vector<size_t>& outliers,
                               const std::vector<Match>& matches,
                               const Matrix3f& H,
                               float thres) const;
    std::vector<IndexDist> sortMatchesByReprojError(const std::vector<Match>& M,
                                                    const Matrix3f& H) const;
    std::vector<size_t> getMatches(const std::vector<IndexDist>& sortedMatches, 
                                   float reprojLowerBound,
                                   float reprojUpperBound) const;
    std::vector<size_t> getMatches(const std::vector<Match>& M,
                                   const Matrix3f& H,
                                   float reprojLowerBound,
                                   float reprojUpperBound) const
    {
      return getMatches(sortMatchesByReprojError(M, H),
                        reprojLowerBound, reprojUpperBound);
    }

  private:
    MikolajczykDataset dataset_;
  };

} /* namespace DO */

#endif /* DO_STUDY_ON_MIKOLAJCZYK_DATASET_HPP */