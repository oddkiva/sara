// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// Disable FLANN warnings
#ifdef _MSC_VER
#pragma warning(disable : 4244 4267 4800 4305 4291 4996)
#endif

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Timer.hpp>

#include <DO/Sara/FeatureMatching.hpp>

#include <flann/flann.hpp>



using namespace std;


namespace DO { namespace Sara {

  //! Create FLANN matrix
  auto create_flann_matrix(const Tensor_<float, 2>& descriptors)
  {
    if (descriptors.size() == 0)
      throw runtime_error{ "Error: the list of key-points is empty!"};

    SARA_DEBUG
        << "Gentle Warning: make sure every key has distinct descriptors..."
        << endl;

    SARA_DEBUG << "Number of descriptors = " << descriptors.rows() << endl;
    SARA_DEBUG << "Descriptor dimension = " << descriptors.cols() << endl;

    auto matrix = flann::Matrix<float>{const_cast<float*>(descriptors.data()),
                                       size_t(descriptors.rows()),
                                       size_t(descriptors.cols())};
    return matrix;
  }

  //! Find the nearest neighbors in the descriptor space using FLANN.
  void append_nearest_neighbors(
      int i1, const KeypointList<OERegion, float>& keys1,
      const KeypointList<OERegion, float>& keys2, vector<Match>& matches,
      const flann::Matrix<float>& /*data2*/,
      flann::Index<flann::L2<float>>& tree2, float squared_ratio_thres,
      Match::Direction dir, bool self_matching,
      const KeyProximity& is_redundant,  // self-matching
      vector<int>& vec_indices, vector<float>& vec_dists,
      size_t num_max_neighbors)  // internal storage parameters
  {
    const auto& [features1, dmat1] = keys1;
    const auto& features2 = features(keys2);

    // Prepare the query matrix
    auto query = flann::Matrix<float>{const_cast<float*>(dmat1[i1].data()), 1u,
                                      size_t(dmat1.cols())};

    // Prepare the indices and distances.
    auto indices = flann::Matrix<int>{vec_indices.data(), 1, num_max_neighbors};
    auto dists = flann::Matrix<float>{vec_dists.data(), 1, num_max_neighbors};

    // Create search parameters.
    flann::SearchParams search_params;

    // N.B.: We should not be in the boundary case in practice, in which case the
    // ambiguity score does not really make sense.
    //
    // Boundary case 1.
    if (features2.size() == 0)
      return;

    // Boundary case 2.
    if (features2.size() == 1 && !self_matching)
    {
      auto m = Match{&features1[i1], &features2[0], 1.f, dir, i1, 0};
      m.rank() = 1;

      if (dir == Match::Direction::TargetToSource)
      {
        swap(m.x_pointer(), m.y_pointer());
        swap(m.x_index(), m.y_index());
      }

      if (m.score() < squared_ratio_thres)
        matches.push_back(m);
      return;
    }

    // Boundary case 3.
    if (features2.size() == 2 && self_matching)
    {
      tree2.knnSearch(query, indices, dists, 2, search_params);

      const auto i2 = indices[0][1]; // The first index can't be indices[0][0], which is i1.
      auto m = Match{&features1[i1], &features2[i2], 1.f, dir, i1, i2};
      m.rank() = 1;

      if(dir == Match::Direction::TargetToSource)
      {
        swap(m.x_pointer(), m.y_pointer());
        swap(m.x_index(), m.y_index());
      }

      if (m.score() < squared_ratio_thres)
        matches.push_back(m);

      return;
    }

    // Now treat the generic case.
    //
    // Search the nearest neighbor.
    tree2.knnSearch(query, indices, dists, 3, search_params);

    // This is to avoid the source key matches with himself in case of intra
    // image matching.
    const auto top1_index = self_matching ? 1 : 0;
    auto top1_score = dists[0][top1_index + 1] > 0.f ?
      dists[0][top1_index] / dists[0][top1_index + 1] : 0.f;
    auto K = 1;

    // Determine the number of nearest neighbors.
    if (squared_ratio_thres > 1.f)
    {
      // Performs an adaptive radius search.
      const auto radius = dists[0][top1_index] * squared_ratio_thres;
      K = tree2.radiusSearch(query, indices, dists, radius, search_params);
    }

    // Retrieve the right key points.
    for (int rank = top1_index; rank < K; ++rank)
    {
      auto score = 0.f;
      if (rank == top1_index)
        score = top1_score;
      else if (dists[0][top1_index])
        score = dists[0][rank] / dists[0][top1_index];

      // We still need this check as FLANN can still return wrong neighbors.
      if (score > squared_ratio_thres)
        break;

      auto i2 = indices[0][rank];

      // Ignore the match if keys1 == keys2.
      if (self_matching && is_redundant(features1[i1], features2[i2]))
        continue;

      Match m(&features1[i1], &features2[i2], score, dir, i1, i2);
      m.rank() = top1_index == 0 ? rank + 1 : rank;
      if (dir == Match::Direction::TargetToSource)
      {
        swap(m.x_pointer(), m.y_pointer());
        swap(m.x_index(), m.y_index());
      }

      matches.push_back(m);
    }
  }

  AnnMatcher::AnnMatcher(const KeypointList<OERegion, float>& keys1,
                         const KeypointList<OERegion, float>& keys2,
                         float sift_ratio_thres)
    : _keys1(keys1)
    , _keys2(keys2)
    , _squared_ratio_thres(sift_ratio_thres * sift_ratio_thres)
    , _max_neighbors(std::max(size(keys1), size(keys2)))
    , _self_matching(false)
  {
    if (!size_consistency_predicate(_keys1) ||
        !size_consistency_predicate(_keys2))
      throw std::runtime_error{
          "The list of keypoints are inconsistent in size!"};

    _vec_indices.resize(_max_neighbors);
    _vec_dists.resize(_max_neighbors);
  }

  AnnMatcher::AnnMatcher(const KeypointList<OERegion, float>& keys,
                         float sift_ratio_thres,
                         float min_max_metric_dist_thres,
                         float pixel_dist_thres)
    : _keys1(keys)
    , _keys2(keys)
    , _squared_ratio_thres(sift_ratio_thres*sift_ratio_thres)
    , _is_too_close(min_max_metric_dist_thres, pixel_dist_thres)
    , _max_neighbors(size(keys))
    , _self_matching(true)
  {
    if (!size_consistency_predicate(_keys1))
      throw std::runtime_error{
          "The list of keypoints are inconsistent in size!"};

    _vec_indices.resize(_max_neighbors);
    _vec_dists.resize(_max_neighbors);
  }

  //! Compute candidate matches using the Euclidean distance.
  auto AnnMatcher::compute_matches() -> vector<Match>
  {
    auto t = Timer{};

    const auto& dmat1 = descriptors(_keys1);
    const auto& dmat2 = descriptors(_keys2);

    flann::KDTreeIndexParams params{8};
    auto data1 = create_flann_matrix(dmat1);
    auto data2 = create_flann_matrix(dmat2);

    flann::Index<flann::L2<float>> tree1(data1, params);
    flann::Index<flann::L2<float>> tree2(data2, params);
    tree1.buildIndex();
    tree2.buildIndex();
    SARA_DEBUG << "Built trees in " << t.elapsed() << " seconds." << endl;

    auto matches = vector<Match>{};
    matches.reserve(1e5);

    t.restart();
    for (auto i1 = 0; i1 < dmat1.rows(); ++i1)
      append_nearest_neighbors(
          i1, _keys1, _keys2, matches, data2, tree2, _squared_ratio_thres,
          Match::Direction::SourceToTarget, _self_matching, _is_too_close,
          _vec_indices, _vec_dists, _max_neighbors);

    for (auto i2 = 0; i2 < dmat2.rows(); ++i2)
      append_nearest_neighbors(
          i2, _keys2, _keys1, matches, data1, tree1, _squared_ratio_thres,
          Match::Direction::TargetToSource, _self_matching, _is_too_close,
          _vec_indices, _vec_dists, _max_neighbors);

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

    // Reorder the matches again.
    sort(matches.begin(), matches.end(), [&](const Match& m1, const Match& m2) {
      return m1.score() < m2.score();
    });

    SARA_DEBUG << "Computed " << matches.size() << " matches in " << t.elapsed()
               << " seconds." << endl;

    return matches;
  }

} /* namespace Sara */
} /* namespace DO */
