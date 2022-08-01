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

//! @file
//! @brief This contains the implementation of the N-dimensional array class.

#pragma once

#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>


namespace DO { namespace Sara {

  /*!
   *  @addtogroup MultiArray
   *  @{
   */

  //! @brief The multidimensional array class.
  template <typename MultiArrayView, template <typename> class Allocator>
  class MultiArrayBase : public MultiArrayView
  {
    //! @{
    //! Convenience typedefs.
    using self_type = MultiArrayBase;
    using base_type = MultiArrayView;
    //! @}

    using base_type::_begin;
    using base_type::_end;
    using base_type::_sizes;
    using base_type::_strides;

    //! Necessary for tensor reshape operations.
    template <typename SomeArrayView_, template <typename> class SomeAlloc_>
    friend class MultiArrayBase;


  public:
    using base_type::Dimension;
    using base_type::StorageOrder;

    using value_type = typename base_type::value_type;
    using pointer = typename base_type::pointer;
    using vector_type = typename base_type::vector_type;
    using allocator_type = Allocator<value_type>;

  public: /* interface */
    //! @brief Default constructor that constructs an empty ND-array.
    inline MultiArrayBase() = default;

    inline explicit MultiArrayBase(const allocator_type& allocator)
      : _allocator{allocator}
    {
    }

    //! @{
    //! @brief Constructor with specified sizes.
    inline explicit MultiArrayBase(
        const vector_type& sizes,
        const allocator_type& allocator = allocator_type{})
      : _allocator{allocator}
    {
      initialize(sizes);
    }

    inline explicit MultiArrayBase(
        int size, const allocator_type& allocator = allocator_type{})
      : self_type{vector_type{size}, allocator}
    {
    }

    inline explicit MultiArrayBase(
        int width, int height,
        const allocator_type& allocator = allocator_type{})
      : self_type{vector_type{width, height}, allocator}
    {
    }

    inline explicit MultiArrayBase(
        int width, int height, int depth,
        const allocator_type& allocator = allocator_type{})
      : self_type{vector_type{width, height, depth}, allocator}
    {
    }
    //! @}

    //! @brief Copy constructor.
    //! Clone the other MultiArrayView instance.
    inline MultiArrayBase(const base_type& other)
    {
      initialize(other.sizes());
      base_type::copy(other);
    }

    inline MultiArrayBase(const self_type& other)
      : self_type{base_type(other)}
    {
    }

    //! @brief Move constructor.
    inline MultiArrayBase(self_type&& other) noexcept
    {
      base_type::swap(other);
    }

    //! @brief Destructor.
    inline ~MultiArrayBase()
    {
      deallocate();
    }

    inline auto const_view() const noexcept -> const base_type&
    {
      return *this;
    }

    inline auto view() noexcept -> base_type&
    {
      return *this;
    }

    //! @brief Assignment operator uses the copy-swap idiom.
    self_type& operator=(self_type other)
    {
      base_type::swap(other);
      return *this;
    }

    //! @{
    //! @brief Resize the MultiArray object with the specified sizes.
    inline void resize(const vector_type& sizes)
    {
      // Boundary case: the size of the allocated memory has not changed.
      // Optimize by:
      // - not deallocating memory.
      // - only changing the sizes and recompute the strides.
      if (_end - _begin ==
          std::ptrdiff_t(base_type::template compute_size<Dimension>(sizes)))
      {
        _sizes = sizes;
        _strides = base_type::compute_strides(sizes);
        return;
      }

      // General case.
      if (_sizes != sizes)
      {
        deallocate();
        initialize(sizes);
      }
    }

    inline void resize(int rows, int cols)
    {
      static_assert(Dimension == 2, "MultiArray must be 2D");
      resize(vector_type{rows, cols});
    }

    inline void resize(int rows, int cols, int depth)
    {
      static_assert(Dimension == 3, "MultiArray must be 3D");
      resize(vector_type(rows, cols, depth));
    }
    //! @}

    //! @brief Destroy the content of the MultiArray object.
    inline void clear()
    {
      deallocate();
    }

    //! @brief Reshape the array with the new sizes.
    template <typename Array>
    inline auto reshape(const Array& new_sizes) && -> MultiArray<
        value_type, ElementTraits<Array>::size, StorageOrder>
    {
      using T = value_type;
      constexpr int Rank = ElementTraits<Array>::size;
      using array_type = MultiArray<T, Rank, StorageOrder>;

      if (base_type::template compute_size<Rank>(new_sizes) !=
          base_type::size())
        throw std::domain_error{"Invalid shape!"};

      // Swap the data members;
      auto res = array_type{};

      // Set the sizes and strides.
      res._sizes = new_sizes;
      res._strides = res.compute_strides(new_sizes);

      this->_sizes.fill(0);
      this->_strides.fill(0);

      // Swap the pointers.
      std::swap(res._begin, this->_begin);
      std::swap(res._end, this->_end);

      return res;
    }

    //! @brief Reshape the array with the new sizes.
    template <typename Array>
    inline auto reshape(const Array& new_sizes) const&
    {
      return const_view().reshape(new_sizes);
    }

  private: /* helper functions for offset computation. */
    //! @{
    //! @brief Allocate the internal array of the MultiArray object.
    inline void initialize(const vector_type& sizes)
    {
      const auto empty = (sizes == vector_type::Zero());
      const auto num_elements =
          empty ? 0 : base_type::template compute_size<Dimension>(sizes);

      _sizes = sizes;
      _strides = empty ? vector_type::Zero()  //
                       : base_type::compute_strides(sizes);
      _begin = empty ? nullptr : allocate(num_elements);
      _end = empty ? nullptr : _begin + num_elements;
    }

    inline pointer allocate(std::size_t count)
    {
      return _allocator.allocate(count);
    }
    //! @}

    //! @brief Deallocate the MultiArray object.
    inline void deallocate()
    {
      _allocator.deallocate(_begin, _end - _begin);
      _begin = nullptr;
      _end = nullptr;
      _sizes = vector_type::Zero();
      _strides = vector_type::Zero();
    }

  private:
    allocator_type _allocator{};
  };

  //! @}

}}  // namespace DO::Sara
