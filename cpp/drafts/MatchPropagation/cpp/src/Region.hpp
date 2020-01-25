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
// ========================================================================== //

#pragma once

#include <DO/Sara/Match.hpp>

#include <set>
#include <vector>


namespace DO::Sara {

  /*!
   *  We use the following notations:
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
    friend auto operator<<(std::ostream&, const Region&) -> std::ostream&;

    //! TODO: make it private?
    container_type indices;
  };

}  // namespace DO::Sara
