// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! \brief This contains the implementation of the sparse N-dimensional array class.

#ifndef DO_CORE_SPARSEMULTIARRAY_HPP
#define DO_CORE_SPARSEMULTIARRAY_HPP

#include "EigenExtension.hpp"
#include <map>

namespace DO {

  //! \ingroup Core
  //! \defgroup SparseMultiArray SparseMultiArray
  //! @{

  //! Sparse N-dimensional array class.
  template <typename T, unsigned int N>
  class SparseMultiArray
  {
  public:
    typedef Matrix<int, N, 1> coord_type; //!< STL-like interface
    typedef T                 value_type; //!< STL-like interface
    typedef unsigned int      size_type; //!< STL-like interface

  private:
    typedef std::map<coord_type, value_type, LexicographicalOrder> storage_type;

  public: // STL-like interface
    //! Dimension
    static const size_type dimension = N;
    //! iterator type.
    typedef typename storage_type::iterator iterator;
    //! const_iterator type.
    typedef typename storage_type::const_iterator const_iterator;
    //typedef typename storage_type::size_type size_type;
    //! difference_type type.
    typedef typename storage_type::difference_type difference_type;
    //! reference type.
    typedef value_type& reference;
    //! const_reference type.
    typedef const value_type& const_reference;

  public: // interface
    //! Default constructor.
    inline SparseMultiArray()
    {}

    //! Copy constructor.
    inline SparseMultiArray(const SparseMultiArray& a)
      : storage(a.storage) {}

    //! Checks if the sparse multi-array is empty.
    inline bool empty() const
    { return storage.empty(); }

    //! Finds the minimum non-zero entry coordinates (in a lexicographical order).
    //! This function is computationally expensive.
    inline coord_type min_key() const
    {
      const_iterator i_begin = storage.begin();

      coord_type k = i_begin->first;

      i_begin++;
      for(const_iterator i = i_begin, i_end = storage.end(); i != i_end;i++)
      {
        for(size_type n = 0;n < dimension;n++)
        {      
          int& k_n = k[n];
          k_n = min(k_n,(i->first)[n]);
        }
      }

      return k;
    }

    //! Finds the maximum non-zero entry coordinates (in a lexicographical order).
    //! This function is computationally expensive.
    inline coord_type max_key() const
    {
      const_iterator i_begin = storage.begin();

      coord_type k = i_begin->first;

      i_begin++;
      for(const_iterator i = i_begin, i_end = storage.end(); i != i_end;i++)
      {
        for(size_type n = 0;n < dimension;n++)
        {      
          int& k_n = k[n];
          k_n = max(k_n,(i->first)[n]);
        }
      }

      return k;
    }

    //! Returns the theoretical capacity of the sparse multi-array
    inline coord_type all_sizes() const
    {
      coord_type sizes;
      for(size_type n = 0; n < dimension; ++n)
        sizes[n] = std::numeric_limits<int>::max();
      return sizes;
    }

    //! Returns the actual N-dimensional size of the sparse multi-array.
    inline coord_type all_actual_sizes() const
    { return max_key()-min_key(); }

    //! Returns the raw size of the storage data.
    inline size_type size() const
    { return storage.size(); }

    //! Returns the capacity of the storage data.
    inline size_type max_size() const
    { return storage.max_size(); }

    //! Efficient swapping of two 2D arrays.
    inline void swap(SparseMultiArray& a)
    { storage.swap(a.storage); }

    //! Erase all of the elements.
    inline void clear()
    { storage.clear(); }

    //! Assignment operator.
    inline SparseMultiArray& operator=(const SparseMultiArray& a)
    { storage = a.storage; return *this; }

    //! Equality operator.  
    inline bool operator==(const SparseMultiArray& a) const
    { return storage == a.storage; }

    //! Inequality operator.
    inline bool operator!=(const SparseMultiArray& a) const
    { return !(*this == a); }

    //! Mutable Access operator.
    inline value_type& operator[](const coord_type& k)
    { return storage[k]; }

    //! Constant access operator.
    inline const value_type& operator[](const coord_type& k) const
    {
      const_iterator i = storage.find(k);
      if (i != storage.end())
        return i->second;

      static value_type default_value = value_type();
      return default_value;
    }

    //! Begin iterator.
    inline iterator begin()
    { return storage.begin(); }

    //! Constant begin iterator.
    inline const_iterator begin() const
    { return storage.begin(); }

    //! End iterator.
    inline iterator end()
    { return storage.end(); }
    
    //! Constant end iterator.
    inline const_iterator end() const
    { return storage.end(); }

  private:
    storage_type storage;
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_SPARSEMULTIARRAY_HPP */