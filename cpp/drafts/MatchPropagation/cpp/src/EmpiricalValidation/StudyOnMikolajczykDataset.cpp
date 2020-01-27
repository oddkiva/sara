// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#include "StudyOnMikolajczykDataset.hpp"

using namespace std;

namespace DO::Sara {

  StudyOnMikolajczykDataset::StudyOnMikolajczykDataset(
      const std::string& abs_parent_folder_path, const std::string& name,
      const std::string& feature_type)
    : _dataset(abs_parent_folder_path, name)
  {
    _dataset.load_keys(feature_type);
    // dataset_.check();
  }

  void StudyOnMikolajczykDataset::open_window_for_image_pair(size_t i,
                                                             size_t j) const
  {
    const auto w = dataset().image(0).width() + dataset().image(1).width();
    const auto h = max(dataset().image(0).height(),  //
                       dataset().image(1).height());
    create_window(w, h);
    set_antialiasing();
  }

  void StudyOnMikolajczykDataset::close_window_for_image_pair() const
  {
    close_window();
  }

  vector<Match> StudyOnMikolajczykDataset::compute_matches(
      const KeypointList<OERegion, float>& X,
      const KeypointList<OERegion, float>& Y, float squared_ell) const
  {
    print_stage("Computing initial matches $\\mathcal{M}$ with $\\ell = " +
                to_string(sqrt(squared_ell)) + "$");
    AnnMatcher matcher(X, Y, squared_ell);
    const auto M = matcher.compute_matches();
    return M;
  }

  void StudyOnMikolajczykDataset::get_inliers_and_outliers(
      vector<size_t>& inliers, vector<size_t>& outliers,
      const vector<Match>& matches, const Matrix3f& H, float thres) const
  {
    inliers.reserve(matches.size());
    for (size_t i = 0; i != matches.size(); ++i)
    {
      const auto& x = matches[i].x_pos();
      const auto& y = matches[i].y_pos();

      const Vector2f Hx = (H * x.homogeneous()).hnormalized();

      if ((Hx - y).squaredNorm() < thres * thres)
        inliers.push_back(i);
      else
        outliers.push_back(i);
    }
  }

  vector<StudyOnMikolajczykDataset::IndexDist>
  StudyOnMikolajczykDataset::sort_matches_by_reprojection_error(
      const vector<Match>& M, const Matrix3f& H) const
  {
    auto index_dists = vector<IndexDist>(M.size());

    for (size_t i = 0; i != M.size(); ++i)
    {
      const Vector2f H_x = (H * M[i].x_pos().homogeneous()).hnormalized();
      const Vector2f& y = M[i].y_pos();
      index_dists[i] = make_pair(i, (H_x - y).norm());
    }

    auto cmp = CompareIndexDist{};
    sort(index_dists.begin(), index_dists.end(), cmp);

    return index_dists;
  }

  auto StudyOnMikolajczykDataset::get_matches(
      const vector<IndexDist>& sorted_matches,  //
      float reprojection_lower_bound,           //
      float reprojection_upper_bound) const     //
      -> vector<size_t>
  {
    auto indices = vector<size_t>{};
    indices.reserve(sorted_matches.size());

    for (int i = 0; i != sorted_matches.size(); ++i)
    {
      if (sorted_matches[i].second < reprojection_lower_bound)
        continue;

      if (sorted_matches[i].second >= reprojection_upper_bound)
        break;

      indices.push_back(sorted_matches[i].first);
    }
    return indices;
  }

}  // namespace DO::Sara
