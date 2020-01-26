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

#include <map>
#include <set>


namespace DO::Sara {

  class DO_SARA_EXPORT RegionBoundary
  {
    //! Forward class declarations.
    using set_type = std::set<size_t>;
    using multimap_type = std::multimap<float, size_t>;

    template <typename MultiMapIterator_, typename Match_>
    class Iterator;

  public: /* interface */
    using const_iterator = Iterator<multimap_type::const_iterator, const Match>;
    using iterator = Iterator<multimap_type::iterator, Match>;

    //! Constructor.
    RegionBoundary(const std::vector<Match>& M)
      : M_(M)
    {
    }

    //! Mimicking the STL interface.
    bool empty() const;
    size_t size() const;
    const Match& top() const;
    bool find(size_t i) const;
    bool insert(size_t i);
    void erase(size_t i);

    //! @{
    /*!
     *
     * Usage: ***KNOW WHAT YOU ARE DOING HERE****
     * 'const Match& m' MUST refer to ***AN ELEMENT OF THE ARRAY***, i.e.:
     * 'const std::vector<Match>& M_'
     *
     */
    bool find(const Match& m) const
    {
      return find(&m - &M_[0]);
    }

    bool insert(const Match& m)
    {
      return insert(&m - &M_[0]);
    }

    void erase(iterator m);
    //! @}

    //! @{
    //! @brief Iterators.
    iterator begin()
    {
      return iterator(lowe_score_index_pairs_.begin(), &M_);
    }

    iterator end()
    {
      return iterator(lowe_score_index_pairs_.end(), &M_);
    }

    const_iterator begin() const
    {
      return const_iterator(lowe_score_index_pairs_.begin(), &M_);
    }

    const_iterator end() const
    {
      return const_iterator(lowe_score_index_pairs_.end(), &M_);
    }
    //! @}

    //! @{
    //! @brief Convenient helper member functions.
    std::vector<Match> matches() const;

    void view(const PairWiseDrawer& drawer) const;
    //! @}

  private: /* iterator class */
    template <typename MultiMapIterator_, typename Match_>
    class Iterator
    {
    public: /* interface */
      // Constructors
      inline Iterator()
        : map_iter_()
        , M_(0)
      {
      }

      inline Iterator(const MultiMapIterator_& map_iter,
                      const std::vector<Match>* M)
        : map_iter_(map_iter)
        , M_(M)
      {
      }
      inline Iterator(const Iterator& it)
        : map_iter_(it.map_iter_)
        , M_(it.M_)
      {
      }

      template <typename MultiMapIterator2_, typename Match2_>
      inline Iterator(const Iterator<MultiMapIterator2_, Match2_>& it)
        : map_iter_(it())
        , M_(it.vector())
      {
      }

      // Assignment operators
      inline Iterator& operator=(const Iterator& it)
      {
        map_iter_ = it.map_iter_;
        M_ = it.M_;
        return *this;
      }

      // Incrementing/decrementing operators.
      inline Iterator& operator++()
      {
        ++map_iter_;
        return *this;
      }

      inline Iterator& operator--()
      {
        --map_iter_;
        return *this;
      }

      inline Iterator operator++(int)
      {
        Iterator prev(*this);
        operator++();
        return prev;
      }

      inline Iterator operator--(int)
      {
        Iterator prev(*this);
        operator--();
        return prev;
      }

      // Comparison member functions
      inline bool operator==(const Iterator& it) const
      {
        return map_iter_ == it.map_iter_;
      }

      inline bool operator!=(const Iterator& it) const
      {
        return map_iter_ != it.map_iter_;
      }

      // Referencing/dereferencing member functions
      inline const Match_& operator*() const
      {
        return (*M_)[map_iter_->second];
      }

      inline const Match_* operator->() const
      {
        return &((*M_)[map_iter_->second]);
      }
      inline const MultiMapIterator_& operator()() const
      {
        return map_iter_;
      }
      inline MultiMapIterator_& operator()()
      {
        return map_iter_;
      }

      // Additional helper member functions
      inline size_t index() const
      {
        return map_iter_->second;
      }

      inline float score() const
      {
        return map_iter_->first;
      }

      inline const std::vector<Match>* vector() const
      {
        return M_;
      }

    private: /* data members */
      MultiMapIterator_ map_iter_;
      const std::vector<Match>* M_;
    };

  private: /* data members */
    set_type indices_;
    multimap_type lowe_score_index_pairs_;
    const std::vector<Match>& M_;
  };

}  // namespace DO::Sara
