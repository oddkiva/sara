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

#include <DO/FeatureMatching.hpp>
#include <flann/flann.hpp>

//#define DEBUG_FLANN
using namespace std;

namespace DO {

  //! Create FLANN matrix
  flann::Matrix<float> createFlannMatrix(const vector<Keypoint>& keys)
  {
    if (keys.empty())
    {
      std::cerr << "Error: list of key-points is empty!" << endl;
      throw 0;
    }

    const int sz = keys.size();
    // Print summary.
    cout << "Gentle Warning: make sure every key has distinct descriptors..." << endl;
    cout << "Number of descriptors = " << sz << endl;
    // Create a matrix that will contain a set of descriptors.
    flann::Matrix<float> matrix(new float[sz*128], sz, 128);    
    for (size_t i = 0; i != keys.size(); ++i)
      copy(keys[i].desc().data(), keys[i].desc().data()+128, matrix[i]);
    return matrix;
  }

  //! Find the nearest neighbors in the descriptor space using FLANN.
  void
  appendNearestNeighbors(size_t i1,
                         const vector<Keypoint>& keys1,
                         const vector<Keypoint>& keys2,
                         vector<Match>& matches,
                         const flann::Matrix<float>& data2,
                         flann::Index<flann::L2<float> >& tree2,
                         float sqRatioT, Match::MatchingDirection dir,
                         bool selfMatching,  const KeyProximity& isRedundant, // self-matching
                         vector<int>& vecIndices, vector<float>& vecDists, size_t maxNeighbors) // internal storage parameters
  {
    // Prepare the query matrix
    Desc128f desc1(keys1[i1].desc());
    flann::Matrix<float> query(desc1.data(), 1, 128);

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
      if (selfMatching && isRedundant(keys1[i1], keys2[i2]))
        continue;

      Match m(&keys1[i1], &keys2[i2], score, dir, i1, i2);
      m.rank() = (startIndex == 0) ? rank+1 : rank;
      if(dir == Match::TargetToSource)
      {
        swap(m.ptrX(), m.ptrY());
        swap(m.indX(), m.indY());
      }
      
      matches.push_back(m);
    }
  }

  AnnMatcher::AnnMatcher(const vector<Keypoint>& keys1,
                         const vector<Keypoint>& keys2,
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

  AnnMatcher::AnnMatcher(const std::vector<Keypoint>& keys,
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
    data1 = createFlannMatrix(keys1_);
    data2 = createFlannMatrix(keys2_);

    flann::Index<flann::L2<float> > tree1(data1, params);
    flann::Index<flann::L2<float> > tree2(data2, params);
    tree1.buildIndex();
    tree2.buildIndex();
    cout << "Built trees in " << t.elapsed() << " seconds." << endl;

    vector<Match> matches;
    matches.reserve(1e5);

    t.restart();
    for (size_t i1 = 0; i1 != keys1_.size(); ++i1)
    {
      appendNearestNeighbors(
        i1, keys1_, keys2_, matches, data2, tree2,
        sqRatioT, Match::SourceToTarget,
        self_matching_, is_too_close_, vec_indices_, vec_dists_, max_neighbors_);
    }
    for (size_t i2 = 0; i2 != keys2_.size(); ++i2)
    {
      appendNearestNeighbors(
        i2, keys2_, keys1_, matches, data1, tree1,
        sqRatioT, Match::TargetToSource,
        self_matching_, is_too_close_, vec_indices_, vec_dists_, max_neighbors_);
    }

    delete[] data1.ptr();
    delete[] data2.ptr();

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