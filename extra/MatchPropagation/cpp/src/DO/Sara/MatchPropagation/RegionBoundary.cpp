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

// ========================================================================== //
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#include "RegionBoundary.hpp"

using namespace std;


namespace DO { namespace Sara { namespace MatchPropagation {

  auto RegionBoundary::empty() const -> bool
  {
    if (indices_.size() != lowe_score_index_pairs_.size())
    {
      std::ostringstream oss;
      oss << "FATAL ERROR: indices_.size() != lowe_score_index_pairs_.size()" << endl;
      oss << "indices_.size() == " << indices_.size() << endl;
      oss << "lowe_score_index_pairs_.size() == " << lowe_score_index_pairs_.size() << endl;
      throw std::runtime_error(oss.str());
    }

    return indices_.empty();
  }

  auto RegionBoundary::size() const -> size_t
  {
    if (indices_.size() != lowe_score_index_pairs_.size())
    {
      std::ostringstream oss;
      oss << "FATAL ERROR: indices_.size() != lowe_score_index_pairs_.size()" << endl;
      oss << "indices_.size() == " << indices_.size() << endl;
      oss << "lowe_score_index_pairs_.size() == " << lowe_score_index_pairs_.size() << endl;
      throw std::runtime_error(oss.str());
    }

    return indices_.size();
  }

  auto RegionBoundary::top() const -> const Match&
  {
    if (empty())
    {
      const char *msg = "FATAL ERROR: Region Boundary is empty!";
      throw std::runtime_error(msg);
    }

    return M_[lowe_score_index_pairs_.begin()->second];
  }

  auto RegionBoundary::find(size_t i) const -> bool
  {
    return indices_.find(i) != indices_.end();
  }

  auto RegionBoundary::insert(size_t i) -> bool
  {
    if (find(i))
      return false;

    indices_.insert(i);
    lowe_score_index_pairs_.insert(make_pair(M_[i].score(), i));

    return true;
  }

  auto RegionBoundary::erase(size_t i) -> void
  {
    if (!find(i))
      return;

    indices_.erase(i);
    auto ret = lowe_score_index_pairs_.equal_range(M_[i].score());
    for (auto it = ret.first; it != ret.second; ++it)
    {
      if (it->second == i)
      {
        lowe_score_index_pairs_.erase(it);
        return;
      }
    }
  }

  auto RegionBoundary::erase(RegionBoundary::iterator m) -> void
  {
    indices_.erase(m.index());
    lowe_score_index_pairs_.erase(m());
  };

  auto RegionBoundary::matches() const -> vector<Match>
  {
    auto R = vector<Match>{};

    R.reserve(indices_.size());
    for (auto i = indices_.begin(); i != indices_.end(); ++i)
      R.push_back(M_[*i]);

    return R;
  }

  auto RegionBoundary::view(const PairWiseDrawer& drawer) const -> void
  {
    for (auto i = indices_.begin(); i != indices_.end(); ++i)
      drawer.draw_match(M_[*i]);
  }

} /* namespace MatchPropagation */
} /* namespace Sara */
} /* namespace DO */
