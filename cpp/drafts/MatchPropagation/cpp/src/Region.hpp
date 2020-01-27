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

#pragma once

#include <DO/Sara/Match.hpp>

#include <set>
#include <vector>


namespace DO::Sara {

  //! @brief This class defines a region for matches (instead of points).
  /*!
   *  We use the following notations:
   *  - @f$R@f$ denotes a region,
   *  - @f$M@f$ denotes the set of initial matches.
   *
   *  Note that for every method that takes as input
   *  'const Match& m, const std::vector<Match>& M':
   *
   *  - 'm' must be a reference to an element in the array 'M'.
   *
   */
  class DO_SARA_EXPORT Region
  {
    //! Alias.
    using container_type = std::set<size_t>;

  public:
    //! @{
    //! Aliases.
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;
    //! @}

    //! @{
    //! 'i' is the index of the match $m_i$ to find.
    bool find(size_t i) const
    {
      return indices.find(i) != indices.end();
    }

    bool insert(size_t i)
    {
      return indices.insert(i).second;
    }
    //! @}

    //! @{
    //! Mutable iterators.
    iterator begin()
    {
      return indices.begin();
    }

    iterator end()
    {
      return indices.end();
    }
    //! @}

    //! @{
    //! Immutable iterator functions
    const_iterator begin() const
    {
      return indices.begin();
    }

    const_iterator end() const
    {
      return indices.end();
    }
    //! @}


    //! @{
    //! Usage: ***KNOW WHAT YOU ARE DOING HERE****
    //! 'const Match& m' MUST refer to
    //! ***AN ELEMENT OF THE ARRAY*** 'const std::vector<Match>& M'
    auto find(const Match& m, const std::vector<Match>& M) const -> bool;

    auto insert(const Match& m, const std::vector<Match>& M) -> bool;
    //! @}

    //! @{
    //! @brief Helper member functions.
    auto matches(const std::vector<Match>& M) const -> std::vector<Match>;

    auto view(const std::vector<Match>& M, const PairWiseDrawer& drawer,
              const Rgb8& c = Green8) const -> void;

    auto size() const -> size_t
    {
      return indices.size();
    }
    //! @}

    //! @brief I/O
    DO_SARA_EXPORT
    friend auto operator<<(std::ostream&, const Region&) -> std::ostream&;

    //! TODO: make it private?
    container_type indices;
  };

}  // namespace DO::Sara
