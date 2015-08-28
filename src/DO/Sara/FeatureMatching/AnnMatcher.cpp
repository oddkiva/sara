// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
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

#include <flann/flann.hpp>

#include <DO/Sara/Core/Timer.hpp>

#include <DO/Sara/FeatureMatching.hpp>


using namespace std;


namespace DO { namespace Sara {

  //! Create FLANN matrix
  flann::Matrix<float>
  create_flann_matrix(const DescriptorMatrix<float>& descs)
  {
    if (descs.size() == 0)
      throw runtime_error{ "Error: list of key-points is empty!"};

    const int sz = descs.size();
    const int dim = descs.dimension();
    // Print summary.
    cout << "Gentle Warning: make sure every key has distinct descriptors..." << endl;
    cout << "Number of descriptors = " << sz << endl;
    // Create a matrix that will contain a set of descriptors.
    flann::Matrix<float> matrix(
      const_cast<float *>(descs.matrix().data()), sz, dim);
    return matrix;
  }

  //! Find the nearest neighbors in the descriptor space using FLANN.
  void append_nearest_neighbors(
    size_t i1,
    const Set<OERegion, RealDescriptor>& keys1,
    const Set<OERegion, RealDescriptor>& keys2,
    vector<Match>& matches,
    const flann::Matrix<float>& data2,
    flann::Index<flann::L2<float>>& tree2,
    float squared_ratio_thres,
    Match::MatchingDirection dir,
    bool self_matching, const KeyProximity& is_redundant, // self-matching
    vector<int>& vec_indices,
    vector<float>& vec_dists,
    size_t num_max_neighbors) // internal storage parameters
  {
    // Prepare the query matrix
    flann::Matrix<float> query(
      const_cast<float *>(&(keys1.descriptors.matrix()(0,i1))),
      1, keys1.descriptors.dimension() );

    // Prepare the indices and distances.
    flann::Matrix<int> indices(&vec_indices[0], 1, num_max_neighbors);
    flann::Matrix<float> dists(&vec_dists[0], 1, num_max_neighbors);

    // Create search parameters.
    flann::SearchParams search_params;

    // Search the nearest neighbor.
    tree2.knnSearch(query, indices, dists, 3, search_params);

    // This is to avoid the source key matches with himself in case of intra
    // image matching.
    const int startIndex = i1 == indices[0][1] ? 1 : 0;
    const float bestScore = dists[0][startIndex]/dists[0][startIndex+1];
    int K = 1;

    // Sanity check.
    if (dists[0][2] < std::numeric_limits<float>::epsilon())
      throw runtime_error{ "AnnMatcher: All distances are 0!" };

    // Determine the number of nearest neighbors.
    if (squared_ratio_thres > 1.f)
    {
      // Performs an adaptive radius search.
      const float radius = (dists[0][startIndex])*squared_ratio_thres;
      K = tree2.radiusSearch(query, indices, dists, radius, search_params);
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
      if (self_matching && is_redundant(keys1.features[i1], keys2.features[i2]))
        continue;

      Match m(&keys1.features[i1], &keys2.features[i2], score, dir, i1, i2);
      m.rank() = (startIndex == 0) ? rank+1 : rank;
      if(dir == Match::TargetToSource)
      {
        swap(m.x_pointer(), m.y_pointer());
        swap(m.x_index(), m.y_index());
      }

      matches.push_back(m);
    }
  }

  AnnMatcher::AnnMatcher(const Set<OERegion, RealDescriptor>& keys1,
                         const Set<OERegion, RealDescriptor>& keys2,
                         float sift_ratio_thres)
    : _keys1(keys1)
    , _keys2(keys2)
    , _squared_ratio_thres(sift_ratio_thres*sift_ratio_thres)
    , _max_neighbors(std::max(keys1.size(), keys2.size()))
    , _self_matching(false)
  {
    _vec_indices.resize(_max_neighbors);
    _vec_dists.resize(_max_neighbors);
  }

  AnnMatcher::AnnMatcher(const Set<OERegion, RealDescriptor>& keys,
                         float sift_ratio_thres,
                         float min_max_metric_dist_thres,
                         float pixel_dist_thres)
    : _keys1(keys)
    , _keys2(keys)
    , _squared_ratio_thres(sift_ratio_thres*sift_ratio_thres)
    , _is_too_close(min_max_metric_dist_thres, pixel_dist_thres)
    , _max_neighbors(keys.size())
    , _self_matching(true)
  {
    _vec_indices.resize(_max_neighbors);
    _vec_dists.resize(_max_neighbors);
  }

  //! Compute candidate matches using the Euclidean distance.
  vector<Match> AnnMatcher::compute_matches()
  {
    Timer t;

    flann::KDTreeIndexParams params(8);
    flann::Matrix<float> data1, data2;
    data1 = create_flann_matrix(_keys1.descriptors);
    data2 = create_flann_matrix(_keys2.descriptors);

    flann::Index<flann::L2<float> > tree1(data1, params);
    flann::Index<flann::L2<float> > tree2(data2, params);
    tree1.buildIndex();
    tree2.buildIndex();
    cout << "Built trees in " << t.elapsed() << " seconds." << endl;

    vector<Match> matches;
    matches.reserve(1e5);

    t.restart();
    for (int i1 = 0; i1 < _keys1.size(); ++i1)
    {
      append_nearest_neighbors(
        i1, _keys1, _keys2, matches, data2, tree2,
        _squared_ratio_thres, Match::SourceToTarget,
        _self_matching, _is_too_close, _vec_indices,
        _vec_dists, _max_neighbors);
    }

    for (int i2 = 0; i2 < _keys2.size(); ++i2)
    {
      append_nearest_neighbors(
        i2, _keys2, _keys1, matches, data1, tree1,
        _squared_ratio_thres, Match::TargetToSource,
        _self_matching, _is_too_close, _vec_indices, _vec_dists, _max_neighbors);
    }

    // Lexicographical comparison between matches.
    auto compare_match = [](const Match& m1, const Match& m2)
    {
      if (m1.x_index() < m2.x_index())
        return true;
      if (m1.x_index() == m2.x_index() && m1.y_index() < m2.y_index())
        return true;
      if (m1.x_index() == m2.x_index() && m1.y_index() == m2.y_index() &&
          m1.score() < m2.score())
        return true;
      return false;
    };
    sort(matches.begin(), matches.end(), compare_match);

    // Remove redundant matches in each consecutive group of identical matches.
    // We keep the one with the best Lowe score.
    matches.resize(unique(matches.begin(), matches.end()) - matches.begin());

    sort(matches.begin(), matches.end(),
      [&](const Match& m1, const Match& m2) {
        return m1.score() < m2.score();
      }
    );

    cout << "Computed " << matches.size() << " matches in " << t.elapsed() << " seconds." << endl;

    return matches;
  }

} /* namespace Sara */
} /* namespace DO */
