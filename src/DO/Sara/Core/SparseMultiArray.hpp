// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
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

#ifndef DO_SARA_CORE_SPARSEMULTIARRAY_HPP
#define DO_SARA_CORE_SPARSEMULTIARRAY_HPP

#include <map>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  //! \ingroup Core
  //! \defgroup SparseMultiArray SparseMultiArray
  //! @{

  //! Sparse N-dimensional array class.
  template <typename T, unsigned int N>
  class SparseMultiArray
  {
  public:
    //! @{
    //! \brief STL-like interface
    using coord_type = Matrix<int, N, 1>;
    using value_type = T;
    //! @}

  private:
    using storage_type = std::map<coord_type, value_type, LexicographicalOrder>;

  public:
    //! @{
    //! \brief STL-like interface
    using size_type = unsigned int;
    using iterator = typename storage_type::iterator iterator;
    using const_iterator = typename storage_type::const_iterator;
    using difference_type = typename storage_type::difference_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    //! @}

  public: // STL-like interface
    //! Dimension
    static const size_type dimension = N;

  public: // interface
    //! Default constructor.
    inline SparseMultiArray() = default;

    //! Copy constructor.
    inline SparseMultiArray(const SparseMultiArray& a)
      : _storage(a._storage)
    {
    }

    //! Checks if the sparse multi-array is empty.
    inline bool empty() const
    {
      return _storage.empty();
    }

    //! Finds the minimum non-zero entry coordinates (in a lexicographical order).
    //! This function is computationally expensive.
    inline coord_type min_key() const
    {
      const_iterator i_begin = _storage.begin();

      coord_type k = i_begin->first;

      i_begin++;
      for(const_iterator i = i_begin, i_end = _storage.end(); i != i_end;i++)
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
      const_iterator i_begin = _storage.begin();

      coord_type k = i_begin->first;

      i_begin++;
      for(const_iterator i = i_begin, i_end = _storage.end(); i != i_end;i++)
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
    {
      return max_key()-min_key();
    }

    //! Returns the raw size of the storage data.
    inline size_type size() const
    {
      return _storage.size();
    }

    //! Returns the capacity of the storage data.
    inline size_type max_size() const
    {
      return _storage.max_size();
    }

    //! Efficient swapping of two 2D arrays.
    inline void swap(SparseMultiArray& a)
    {
      _storage.swap(a._storage);
    }

    //! Erase all of the elements.
    inline void clear()
    {
      _storage.clear();
    }

    //! Assignment operator.
    inline SparseMultiArray& operator=(const SparseMultiArray& a)
    {
      _storage = a._storage;
      return *this;
    }

    //! Equality operator.
    inline bool operator==(const SparseMultiArray& a) const
    {
      return _storage == a._storage;
    }

    //! Inequality operator.
    inline bool operator!=(const SparseMultiArray& a) const
    {
      return !(*this == a);
    }

    //! Mutable Access operator.
    inline value_type& operator[](const coord_type& k)
    {
      return _storage[k];
    }

    //! Constant access operator.
    inline const value_type& operator[](const coord_type& k) const
    {
      const_iterator i = _storage.find(k);
      if (i != _storage.end())
        return i->second;

      static value_type default_value = value_type();
      return default_value;
    }

    //! Begin iterator.
    inline iterator begin()
    {
      return _storage.begin();
    }

    //! Constant begin iterator.
    inline const_iterator begin() const
    {
      return _storage.begin();
    }

    //! End iterator.
    inline iterator end()
    {
      return _storage.end();
    }

    //! Constant end iterator.
    inline const_iterator end() const
    {
      return _storage.end();
    }

  private:
    storage_type _storage;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_SPARSEMULTIARRAY_HPP */
