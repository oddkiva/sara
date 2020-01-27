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

#include "Region.hpp"


using namespace std;


namespace DO::Sara {

  vector<Match> Region::matches(const vector<Match>& M) const
  {
    auto R = vector<Match>{};

    R.reserve(indices.size());
    for (const_iterator i = begin(); i != end(); ++i)
      R.push_back(M[*i]);

    return R;
  }

  auto Region::find(const Match& m, const vector<Match>& M) const -> bool
  {
    const auto index = size_t(&m - &M[0]);
    return find(index);
  }

  bool Region::insert(const Match& m, const vector<Match>& M)
  {
    const auto index = size_t(&m - &M[0]);

    if (find(m, M))
      return false;

    insert(index);
    return true;
  }

  void Region::view(const vector<Match>& M, const PairWiseDrawer& drawer,
                    const Rgb8& c) const
  {
    for (auto i = begin(); i != end(); ++i)
      drawer.draw_match(M[*i], c);
  }

  auto operator<<(ostream& os, const Region& R) -> ostream&
  {
    os << "Matches of the region:" << endl;
    for (auto m = R.begin(); m != R.end(); ++m)
      os << *m << " ";
    os << endl;
    return os;
  }

}  // namespace DO::Sara
