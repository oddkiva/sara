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

namespace DO {

  bool RegionBoundary::empty() const
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

  size_t RegionBoundary::size() const
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

  const Match& RegionBoundary::top() const
  {
    if (empty())
    {
      const char *msg = "FATAL ERROR: Region Boundary is empty!";
      throw std::runtime_error(msg);
    }
    return M_[lowe_score_index_pairs_.begin()->second];
  }

  bool RegionBoundary::find(size_t i) const
  {
    return indices_.find(i) != indices_.end();
  }

  bool RegionBoundary::insert(size_t i)
  {
    if (find(i))
      return false;
    indices_.insert(i);
    lowe_score_index_pairs_.insert(make_pair(M_[i].score(), i));
    return true;
  }

  void RegionBoundary::erase(size_t i)
  {
    if (!find(i))
      return;
    indices_.erase(i);
    pair<multimap_type::iterator, multimap_type::iterator> ret;
    ret = lowe_score_index_pairs_.equal_range(M_[i].score());
    for (multimap_type::iterator it = ret.first ; it != ret.second; ++it)
    {
      if (it->second == i)
      {
        lowe_score_index_pairs_.erase(it);
        return;
      }
    }
  }

  void RegionBoundary::erase(RegionBoundary::iterator m)
  {
    indices_.erase(m.index());
    lowe_score_index_pairs_.erase(m());
  };

  vector<Match> RegionBoundary::matches() const
  {
    vector<Match> R;
    R.reserve(indices_.size());
    set<size_t>::const_iterator i;
    for (i = indices_.begin(); i != indices_.end(); ++i)
      R.push_back(M_[*i]);
    return R;
  }

  void RegionBoundary::view(const PairWiseDrawer& drawer) const
  {
    set<size_t>::const_iterator i;
    for (i = indices_.begin(); i != indices_.end(); ++i)
      drawer.drawMatch(M_[*i]);
  }

} /* namespace DO */
