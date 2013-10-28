// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// Disable FLANN warnings
#ifdef _MSC_VER 
# pragma warning ( disable : 4244 4267 4800 4305 4291 4996)
#endif

#include <DO/FeatureMatching.hpp>
#include <flann/flann.hpp>

//#define DEBUG_FLANN
using namespace std;

namespace DO {

  //! Create FLANN matrix
  flann::Matrix<float> createFlannMatrix(const DescriptorMatrix<float>& descs)
  {
    if (descs.size() == 0)
    {
      std::cerr << "Error: list of key-points is empty!" << endl;
      throw 0;
    }

    const int sz = descs.size();
    const int dim = descs.dimension();
    // Print summary.
    cout << "Gentle Warning: make sure every key has distinct descriptors..." << endl;
    cout << "Number of descriptors = " << sz << endl;
    // Create a matrix that will contain a set of descriptors.
    flann::Matrix<float> matrix(
      const_cast<float *>(descs.matrix().data()), sz, dim );
    return matrix;
  }

  //! Find the nearest neighbors in the descriptor space using FLANN.
  void
  appendNearestNeighbors(size_t i1,
                         const Set<OERegion, RealDescriptor>& keys1,
                         const Set<OERegion, RealDescriptor>& keys2,
                         vector<Match>& matches,
                         const flann::Matrix<float>& data2,
                         flann::Index<flann::L2<float> >& tree2,
                         float sqRatioT, Match::MatchingDirection dir,
                         bool selfMatching,  const KeyProximity& isRedundant, // self-matching
                         vector<int>& vecIndices, vector<float>& vecDists, size_t maxNeighbors) // internal storage parameters
  {
    // Prepare the query matrix
    flann::Matrix<float> query(
      const_cast<float *>(&(keys1.descriptors.matrix()(0,i1))),
      1, keys1.descriptors.dimension() );

    // Prepare the indices and distances.
    flann::Matrix<int> indices(&vecIndices[0], 1, maxNeighbors);
    flann::Matrix<float> dists(&vecDists[0], 1, maxNeighbors);

    // Create search parameters.
    flann::SearchParams searchParams;

    // Search the nearest neighbor.
    tree2.knnSearch(query, indices, dists, 3, searchParams);

    // This is to avoid the source key matches with himself in case of intra image matching.
    const int startIndex = i1 == indices[0][1] ? 1 : 0;
    const float bestScore = dists[0][startIndex]/dists[0][startIndex+1];
    int K = 1;

    // Sanity check.
    if (dists[0][2] < std::numeric_limits<float>::epsilon())
    {
      std::cerr << "AnnMatcher: All distances are 0!" << endl;
      std::cerr << "AnnMatcher: Stopping the program voluntarily!" << endl;
      throw 0;
    }

    // Determine the number of nearest neighbors.
    if (sqRatioT > 1.f)
    {
      // Performs an adaptive radius search.
      const float radius = (dists[0][startIndex])*sqRatioT;
      K = tree2.radiusSearch(query, indices, dists, radius, searchParams);
    }

    // Retrieve the right key points.
    for (int rank = 0; rank < K; ++rank)
    {
      float score = 0.f;
      if (rank == startIndex)
        score = bestScore;
      else if (rank > startIndex)
        score = dists[0][rank] / dists[0][startIndex];

      int i2 = indices[0][rank];

      // Ignore the match if keys1 == keys2.
      if (selfMatching && isRedundant(keys1.features[i1], keys2.features[i2]))
        continue;

      Match m(&keys1.features[i1], &keys2.features[i2], score, dir, i1, i2);
      m.rank() = (startIndex == 0) ? rank+1 : rank;
      if(dir == Match::TargetToSource)
      {
        swap(m.ptrX(), m.ptrY());
        swap(m.indX(), m.indY());
      }
      
      matches.push_back(m);
    }
  }

  AnnMatcher::AnnMatcher(const Set<OERegion, RealDescriptor>& keys1,
                         const Set<OERegion, RealDescriptor>& keys2,
                         float siftRatioT)
    : keys1_(keys1)
    , keys2_(keys2)
    , sqRatioT(siftRatioT*siftRatioT)
    , max_neighbors_(std::max(keys1.size(), keys2.size()))
    , self_matching_(false)
  {
    vec_indices_.resize(max_neighbors_);
    vec_dists_.resize(max_neighbors_);
  }

  AnnMatcher::AnnMatcher(const Set<OERegion, RealDescriptor>& keys,
                         float siftRatioT,
                         float minMaxMetricDistT,
                         float pixelDistT)
    : keys1_(keys)
    , keys2_(keys)
    , sqRatioT(siftRatioT*siftRatioT)
    , is_too_close_(minMaxMetricDistT, pixelDistT)
    , max_neighbors_(keys.size())
    , self_matching_(true)
  {
    vec_indices_.resize(max_neighbors_);
    vec_dists_.resize(max_neighbors_);
  }

  //! Compute candidate matches using the Euclidean distance.
  vector<Match> AnnMatcher::computeMatches()
  {
    Timer t;

    flann::KDTreeIndexParams params(8);
    flann::Matrix<float> data1, data2;
    data1 = createFlannMatrix(keys1_.descriptors);
    data2 = createFlannMatrix(keys2_.descriptors);

    flann::Index<flann::L2<float> > tree1(data1, params);
    flann::Index<flann::L2<float> > tree2(data2, params);
    tree1.buildIndex();
    tree2.buildIndex();
    cout << "Built trees in " << t.elapsed() << " seconds." << endl;

    vector<Match> matches;
    matches.reserve(1e5);

    t.restart();
    for (int i1 = 0; i1 < keys1_.size(); ++i1)
    {
      appendNearestNeighbors(
        i1, keys1_, keys2_, matches, data2, tree2,
        sqRatioT, Match::SourceToTarget,
        self_matching_, is_too_close_, vec_indices_, vec_dists_, max_neighbors_);
    }
    for (int i2 = 0; i2 < keys2_.size(); ++i2)
    {
      appendNearestNeighbors(
        i2, keys2_, keys1_, matches, data1, tree1,
        sqRatioT, Match::TargetToSource,
        self_matching_, is_too_close_, vec_indices_, vec_dists_, max_neighbors_);
    }

    // Sort by indices and by score.
    struct CompareMatch {
      bool operator()(const Match& m1, const Match& m2) const {
        if (m1.indX() < m2.indX())
          return true;
        if (m1.indX() == m2.indX() && m1.indY() < m2.indY())
          return true;
        if (m1.indX() == m2.indX() && m1.indY() == m2.indY() && m1.score() < m2.score())
          return true;
        return false;
      }
    };
    sort(matches.begin(), matches.end(), CompareMatch());
    // Remove redundant matches in each consecutive group of identical matches.
    // We keep the one with the best score, which is the first one according to 'CompareMatch'.
    matches.resize(unique(matches.begin(), matches.end()) - matches.begin());

    struct CompareByScore {
      bool operator()(const Match& m1, const Match& m2) const {
        return m1.score() < m2.score();
      }
    };
    sort(matches.begin(), matches.end(), CompareByScore());

    cout << "Computed " << matches.size() << " matches in " << t.elapsed() << " seconds." << endl;

    return matches;
  }

}