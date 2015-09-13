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
//! \@brief This contains the implementation of the N-dimensional array class.

#ifndef DO_SARA_CORE_MULTIARRAY_MULTIARRAY_HPP
#define DO_SARA_CORE_MULTIARRAY_MULTIARRAY_HPP

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>


namespace DO { namespace Sara {

  //! \@brief The ND-array class.
  template <
    typename T, int N, int StorageOrder = ColMajor,
    template <typename> class Allocator = std::allocator
  >
  class MultiArray : public MultiArrayView<T, N, StorageOrder>
  {
    //! @{
    //! Convenience typedefs.
    using self_type =  MultiArray;
    using base_type = MultiArrayView<T, N, StorageOrder>;
    using allocator_type = Allocator<T>;
    //! @}

    using base_type::_begin;
    using base_type::_end;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    using base_type::Dimension;
    using vector_type = typename base_type::vector_type;

  public: /* interface */
    //! \@brief Default constructor that constructs an empty ND-array.
    inline MultiArray() = default;

    //! \@brief Constructor that takes **ownership** of the data.
    //! The data will be cleared upon destruction of the MultiArray object.
    //! Thus ensure sure that is really what you want. Otherwise construct a
    //! MultiArrayView object instead.
    inline explicit MultiArray(T *data, const vector_type& sizes)
      : base_type(data, sizes)
    {
    }

    //! @{
    //! \@brief Constructor with specified sizes.
    inline explicit MultiArray(const vector_type& sizes)
      : base_type{}
    {
      initialize(sizes);
    }

    inline explicit MultiArray(int rows, int cols)
      : MultiArray{ vector_type{ rows, cols } }
    {
    }

    inline explicit MultiArray(int rows, int cols, int depth)
      : MultiArray{ vector_type{ rows, cols, depth } }
    {
    }
    //! @}

    //! \@brief Copy constructor.
    //! Clone the other MultiArray instance.
    inline MultiArray(const self_type& other)
      : base_type{}
    {
      initialize(other._sizes);
      std::copy(other._begin, other._end, _begin);
    }

    //! \@brief Move constructor.
    inline MultiArray(self_type&& other)
      : base_type{}
    {
      std::swap(_begin, other._begin);
      std::swap(_end, other._end);
      _sizes = other._sizes;
      _strides = other._strides;
    }

    //! \@brief Destructor.
    inline ~MultiArray()
    {
      deallocate();
    }

    //! \@brief Assignment operator uses the copy-swap idiom.
    self_type& operator=(self_type other)
    {
      swap(other);
      return *this;
    }

    //! @{
    //! \@brief Resize the MultiArray object with the specified sizes.
    inline void resize(const vector_type& sizes)
    {
      if (_sizes != sizes)
      {
        deallocate();
        initialize(sizes);
      }
    }

    inline void resize(int rows, int cols)
    {
      static_assert(N == 2, "MultiArray must be 2D");
      resize(vector_type(rows, cols));
    }

    inline void resize(int rows, int cols, int depth)
    {
      static_assert(N == 3, "MultiArray must be 3D");
      resize(vector_type(rows, cols, depth));
    }
    //! @}

    //! @brief Destroy the content of the MultiArray object.
    inline void clear()
    {
      deallocate();
    }

    //! @brief Swap multi-array objects.
    inline void swap(self_type& other)
    {
      using std::swap;
      swap(_begin, other._begin);
      swap(_end, other._end);
      swap(_sizes, other._sizes);
      swap(_strides, other._strides);
    }

  private: /* helper functions for offset computation. */
    //! @{
    //! @brief Allocate the internal array of the MultiArray object.
    inline void initialize(const vector_type& sizes)
    {
      _sizes = sizes;
      auto empty = (sizes == vector_type::Zero());
      _strides = empty ? sizes : this->compute_strides(sizes);

      auto num_elements = this->compute_size(sizes);
      _begin = empty ? 0 : allocate(num_elements);
      _end = empty ? 0 : _begin + num_elements;
    }

    inline T *allocate(std::size_t count)
    {
      return allocator_type{}.allocate(count);
    }
    //! @}

    //! @brief Deallocate the MultiArray object.
    inline void deallocate()
    {
      allocator_type{}.deallocate(_begin, _end - _begin);
      _begin = nullptr;
      _end = nullptr;
      _sizes = vector_type::Zero();
      _strides = vector_type::Zero();
    }

  };

  //! @brief Output stream operator.
  template <typename T, int N, int StorageOrder>
  std::ostream& operator<<(std::ostream& os,
                           const MultiArray<T, N, StorageOrder>& M)
  {
    os << M.sizes() << std::endl;
    os << M.array() << std::endl;
    return os;
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_MULTIARRAY_MULTIARRAY_HPP */
