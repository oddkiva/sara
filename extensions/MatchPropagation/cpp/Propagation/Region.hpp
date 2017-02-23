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

#pragma once

#include <DO/Match.hpp>
#include <vector>
#include <set>

namespace DO {

  /*! We use the following notations:
   *  - $R$ denotes a region,
   *  - $M$ denotes the set of initial matches.
   * 
   *  Besides, please note that for every method that takes as input
   *  'const Match& m, const std::vector<Match>& M', 'm' must be a reference to 
   *  an element of the array 'M'.
   *
   *  Maybe change the API because this may be confusing...
   *
   */
  class Region
  {
    typedef std::set<size_t> container_type;
  public:
    //! Typedefs.
    typedef container_type::iterator iterator;
    typedef container_type::const_iterator const_iterator;
    //! 'i' is the index of the match $m_i$ to find.
    bool find(size_t i) const { return indices.find(i) != indices.end(); }
    //! 'i' is the index of the match $m_i$ to insert.
    bool insert(size_t i) { return indices.insert(i).second; }
    //! Mutable iterator functions
    iterator begin() { return indices.begin(); }
    iterator end() { return indices.end(); }
    //! Immutable iterator functions
    const_iterator begin() const { return indices.begin(); }
    const_iterator end() const { return indices.end(); }
    //! Usage: ***KNOW WHAT YOU ARE DOING HERE****
    //! 'const Match& m' MUST refer to 
    //! ***AN ELEMENT OF THE ARRAY*** 'const std::vector<Match>& M'
    bool find(const Match &m, const std::vector<Match>& M) const;
    //! Usage: ***KNOW WHAT YOU ARE DOING HERE****
    //! 'const Match& m' MUST refer to 
    //! ***AN ELEMENT OF THE ARRAY*** 'const std::vector<Match>& M'
    bool insert(const Match& m, const std::vector<Match>& M);
    //! Helper member functions.
    std::vector<Match> matches(const std::vector<Match>& M) const;
    void view(const std::vector<Match>& M,
              const PairWiseDrawer& drawer, const Rgb8& c = Green8) const;
    size_t size() const { return indices.size(); }
    //! I/O
    friend std::ostream& operator<<(std::ostream& os, const Region& R);
    //! TODO: make it private?
    container_type indices;
  };

} /* namespace DO */

#endif /* DO_REGIONGROWING_REGION_HPP */
