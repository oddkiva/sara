// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_CORE_MULTIARRAY_MULTIARRAYVIEW_HPP
#define DO_SARA_CORE_MULTIARRAY_MULTIARRAYVIEW_HPP

#include <DO/Sara/Core/ArrayIterators.hpp>
#include <DO/Sara/Core/MultiArray/ElementTraits.hpp>


namespace DO { namespace Sara {

  template <typename T, int N, int S = ColMajor>
  class MultiArrayView
  {
    typedef MultiArrayView self_type;

  public: /* typedefs. */
    //! Storage order.
    enum {
      Dimension = N,
      StorageOrder = S
    };

    //! @{
    //! \brief STL-compatible interface.
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T value_type;
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T * iterator;
    typedef const T * const_iterator;
    //! @}

    //! @{
    //! \brief Slice type.
    typedef const MultiArrayView<T, N-1, S> const_slice_type;
    typedef MultiArrayView<T, N-1, S> slice_type;
    //! @}

    //! @{
    //! \brief Vector type.
    typedef Matrix<int, N, 1> vector_type;
    typedef Matrix<int, N-1, 1> slice_vector_type;
    //! @}

    //! @{
    //! \brief N-dimensional iterator type.
    typedef ArrayIterator<false, T, N, StorageOrder> array_iterator;
    typedef ArrayIterator<true, T, N, StorageOrder> const_array_iterator;
    //! @}

    //! @{
    //! \brief N-dimensional subrange iterator.
    typedef SubarrayIterator<false, T, N, StorageOrder> subarray_iterator;
    typedef SubarrayIterator<true, T, N, StorageOrder> const_subarray_iterator;
    //! @}

    //! @{
    //! \brief Array views for linear algebra.
    typedef Map<const Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      const_array_view_type;
    typedef Map<Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      array_view_type;
    //! @}

    //! @{
    //! \brief Matrix views for linear algebra.
    typedef Map<
      const Matrix<typename ElementTraits<T>::value_type,
      Dynamic, Dynamic, StorageOrder> > const_matrix_view_type;
    typedef Map<
      Matrix<typename ElementTraits<T>::value_type, Dynamic, Dynamic,
      StorageOrder> > matrix_view_type;
    //! @}

  public: /* methods */
    //! \brief Default constructor.
    inline MultiArrayView()
      : _begin(nullptr)
      , _end(nullptr)
      , _sizes(vector_type::Zero())
      , _strides(vector_type::Zero())
    {
    }

    //! \brief Constructor that wraps plain data with its known sizes.
    inline MultiArrayView(value_type *data,
                          const vector_type& sizes)
      : _begin(data)
      , _end(data+compute_size(sizes))
      , _sizes(sizes)
      , _strides(compute_strides(sizes))
    {
    }

    //! \brief Return the size vector of the MultiArray object.
    const vector_type& sizes() const
    {
      return _sizes;
    }

    //! \brief Return the number of elements in the internal data array.
    size_type size() const
    {
      return _end - _begin;
    }

    //! \brief Return the size of the MultiArray object along the i-th dimension.
    int size(int i) const
    {
      return _sizes[i];
    }

    //! \brief Return the number of rows.
    int rows() const
    {
      return _sizes[0];
    }

    //! \brief Return the number of cols.
    int cols() const
    {
      return _sizes[1];
    }

    //! \brief Return the depth size.
    int depth() const
    {
      return _sizes[2];
    }

    //! \brief Return the stride vector of the MultiArray object.
    inline const vector_type& strides() const
    {
      return _strides;
    }

    //! \brief Return the stride value along the i-th dimension.
    inline int stride(int i) const
    {
      return _strides[i];
    }
    //! @{
    //! \brief Return the array pointer
    inline pointer data()
    {
      return _begin;
    }

    inline const_pointer data() const
    {
      return _begin;
    }
    //! @}

    //! @{
    //! \brief Return reference to the element at the given coordinates.
    inline reference operator()(const vector_type& pos)
    {
      return _begin[offset(pos)];
    }

    inline reference operator()(int i, int j)
    {
      DO_SARA_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      return _begin[offset(Vector2i(i, j))];
    }

    inline reference operator()(int i, int j, int k)
    {
      DO_SARA_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      return _begin[offset(Vector3i(i, j, k))];
    }

    inline const_reference operator()(const vector_type& pos) const
    {
      return _begin[offset(pos)];
    }

    inline const_reference operator()(int i, int j) const
    {
      DO_SARA_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      return _begin[offset(Vector2i(i, j))];
    }

    inline const_reference operator()(int i, int j, int k) const
    {
      DO_SARA_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      return _begin[offset(Vector3i(i, j, k))];
    }
    //! @}

    //! @{
    //! \brief Return the i-th slice of the MultiArray object.
    inline slice_type operator[](int i)
    {
      slice_vector_type sizes(_sizes.tail(N-1));
      T * data = _begin + _strides[0] * i;
      return slice_type(data, sizes);
    }

    inline const_slice_type operator[](int i) const
    {
      slice_vector_type slice_sizes(_sizes.tail(N-1));
      const T * data = _begin + _strides[0] * i;
      return const_slice_type(data, slice_sizes);
    }
    //! @}

    //! @{
    //! \brief Return the begin iterator.
    inline iterator begin()
    {
      return _begin;
    }

    inline const_iterator begin() const
    {
      return _begin;
    }
    //! @}

    //! @{
    //! \brief Return the end iterator.
    inline iterator end()
    {
      return _end;
    }

    inline const_iterator end() const
    {
      return _end;
    }
    //! @}

    //! @{
    //! \brief Return the array view for linear algebra with Eigen libraries.
    inline array_view_type array()
    {
      return array_view_type(reinterpret_cast<
                             typename ElementTraits<T>::pointer>(data()),
                             size());
    }

    inline const_array_view_type array() const
    {
      return const_array_view_type(reinterpret_cast<
                                   const typename ElementTraits<T>::const_pointer>(data()),
                                   size());
    }
    //! @}

    //! @{
    //! \brief Return the matrix view for linear algebra with Eigen libraries.
    inline matrix_view_type matrix()
    {
      DO_SARA_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return matrix_view_type( reinterpret_cast<
                              typename ElementTraits<T>::pointer>(data()),
                              rows(), cols());
    }

    inline const_matrix_view_type matrix() const
    {
      DO_SARA_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return const_matrix_view_type( reinterpret_cast<
                                    typename ElementTraits<T>::const_pointer>(data()),
                                    rows(), cols());
    }
    //! @}

    //! @{
    //! \brief Return the begin iterator of the whole multi-array.
    inline array_iterator begin_array()
    {
      return array_iterator(false, _begin, vector_type::Zero(), _sizes, _strides);
    }

    inline const_array_iterator begin_array() const
    {
      return const_array_iterator(false, _begin, vector_type::Zero(), _sizes, _strides);
    }
    //! @}

    //! @{
    //! \brief Return the begin iterator of the sub-array.
    inline subarray_iterator begin_subarray(const vector_type& start,
                                            const vector_type& end)
    {
      return subarray_iterator(false, _begin, start, end, _strides, _sizes);
    }

    inline const_subarray_iterator begin_subarray(const vector_type& start,
                                                  const vector_type& end) const
    {
      return const_subarray_iterator(false, _begin, start, end, _strides, _sizes);
    }
    //! @}

  protected:
    //! \brief Compute the strides according the size vector and storage order.
    inline vector_type compute_strides(const vector_type& sizes) const
    {
      return StrideComputer<StorageOrder>::eval(sizes);
    }

    //! \brief Compute the raw size needed to allocate the internal data.
    inline size_type compute_size(const vector_type& sizes) const
    {
      Matrix<size_type, N, 1> sz(sizes.template cast<size_type>());
      return std::accumulate(sz.data(), sz.data()+N,
                             size_type(1), std::multiplies<size_type>());
    }

    //! \brief Compute the 1D index of the corresponding coordinates.
    inline int offset(const vector_type& pos) const
    {
      return jump(pos, _strides);
    }

  protected: /* data members. */
    value_type *_begin;   //!< first element of the data.
    value_type *_end;     //!< last element of the data.
    vector_type _sizes;   //!< vector of size along each dimension.
    vector_type _strides; //!< vector of stride for each dimension.
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_MULTIARRAY_MULTIARRAYVIEW_HPP */
