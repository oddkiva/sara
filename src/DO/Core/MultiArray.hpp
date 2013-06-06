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
//! \brief This contains the implementation of the N-dimensional array class.

#ifndef DO_CORE_MULTIARRAY_HPP
#define DO_CORE_MULTIARRAY_HPP

namespace DO {

  //! \ingroup Core
  //! \defgroup MultiArray MultiArray
  //! @{

  //! The generic traits class for the MultiArray element type.
  //! This traits class is when the array/matrix view is used. This serves as
  //! an interface for the Eigen library.
  template <typename T>
  struct ElementTraits
  {
    typedef T value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef T * pointer; //!< STL-like typedef.
    typedef const T * const_pointer; //!< STL-like typedef.
    typedef T& reference; //!< STL-like typedef.
    typedef const T& const_reference; //!< STL-like typedef.
    typedef T * iterator; //!< STL-like typedef.
    typedef const T * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = true; //!< STL-like typedef.
  };

  //! \brief The specialized element traits class when the entry is a non square 
  //! matrix. Again the matrix is viewed as a 'super-scalar'.
  //! However, super-scalar operations will be regular point-wise matrix 
  //! operations.
  //! Therefore this applies for row-vector and column vector.
  template <typename T, int M, int N>
  struct ElementTraits<Matrix<T, M, N> >
  {
    const static bool is_square_matrix = (M == N); //!< STL-like typedef.
    typedef typename Meta::Choose<
      is_square_matrix, 
      Matrix<T, N, N>, 
      Array<T, M, N> >::Type value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  //! \brief The specialized element traits class when the entry is an array.
  //! Default super-scalar operations are point-wise matrix operations.
  template <typename T, int M, int N>
  struct ElementTraits<Array<T, M, N> >
  {
    typedef Array<T, M, N> value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  /*!
    \brief The N-dimensional array class.
    
    \todo:
    - prefer shallow copy in copy constructor and in the assignment operator
    - add 'MultiArray MultiArray::clone() const' method.
      (Performance still seems acceptable even if deep copies are always done.)
    - **DOUBLE-CHECK** all existing image-based algorithm.
    - Extract boost::shared_pointer with bcp to get the job done.
   */
  template <typename T, int N, int StorageOrder = RowMajor>
  class MultiArray
  {
  private: /* STL-like interface. */
    typedef Offset<N, StorageOrder> storage_index;
    typedef MultiArray self_type;

  public: /* STL-like interface. */
    typedef std::size_t size_type;          //!< STL typedef.
    typedef std::ptrdiff_t difference_type; //!< STL typedef.
    typedef T value_type;                   //!< STL typedef.
    typedef T * pointer;                    //!< STL typedef.
    typedef const T * const_pointer;        //!< STL typedef.
    typedef T& reference;                   //!< STL typedef.
    typedef const T& const_reference;       //!< STL typedef.
    typedef T * iterator;                   //!< STL typedef.
    typedef const T * const_iterator;       //!< STL typedef.

    //! Vector type.
    typedef Matrix<int, N, 1> coords_type, vector_type;

    //! N-dimensional iterator.
    typedef RangeIterator<false, T, N, StorageOrder> range_iterator;
    //! N-dimensional subrange iterator.
    typedef SubrangeIterator<false, T, N, StorageOrder> subrange_iterator;
     //! Immutable N-dimensional iterator.
    typedef RangeIterator<true, T, N, StorageOrder> const_range_iterator;
    //! Immutable N-dimensional subrange iterator.
    typedef SubrangeIterator<true, T, N, StorageOrder> const_subrange_iterator;
    //! Iterator over the coordinates.
    typedef CoordsIterator<N> coords_iterator;

    //! Immutable matrix view for linear algebra.
    typedef Map<const Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      const_array_view_type;
    //! Mutable matrix view for linear algebra.
    typedef Map<Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      array_view_type;
    //! Immutable matrix view for linear algebra.
    typedef Map<const Matrix<typename ElementTraits<T>::value_type,
      Dynamic, Dynamic, StorageOrder> > const_matrix_view_type;
    //! Mutable matrix view for linear algebra.
    typedef Map<Matrix<typename ElementTraits<T>::value_type,
      Dynamic, Dynamic, StorageOrder> > matrix_view_type;

  public: /* interface */
    //! Default constructor that constructs an empty N-dimensional array.
    inline MultiArray()
      : begin_(0)
      , end_(0)
      , sizes_(vector_type::Zero())
      , strides_(vector_type::Zero())
      , is_wrapped_data_(false)
    {}
    //! Constructor that wraps plain data with its known sizes.
    inline MultiArray(value_type *data, const vector_type& sizes)
      : begin_(data)
      , end_(data+compute_size(sizes))
      , sizes_(sizes), strides_(compute_strides(sizes))
      , is_wrapped_data_(true)
    {}
    //! \brief Default constructor that allocates an N-dimensional array with 
    //! the specified sizes.
    inline explicit MultiArray(const vector_type& sizes)
      : begin_(new T[compute_size(sizes)])
      , end_(begin_ + compute_size(sizes))
      , sizes_(sizes), strides_(compute_strides(sizes))
      , is_wrapped_data_(false)
    {}
    //! \brief Default constructor that allocates a 2D array with 
    //! the specified rows and columns.
    inline MultiArray(int rows, int cols)
      : begin_(new T[rows*cols])
      , end_(begin_ + rows*cols)
      , sizes_(rows, cols)
      , strides_(compute_strides(vector_type(rows, cols)))
      , is_wrapped_data_(false)
    {}
    //! \brief Default constructor that allocates a 3D array with 
    //! the specified rows, columns and depth.
    inline MultiArray(int rows, int cols, int depth)
      : begin_(new T[rows*cols*depth])
      , end_(begin_ + rows*cols*depth)
      , sizes_(rows, cols, depth)
      , strides_(compute_strides(vector_type(rows, cols, depth)))
      , is_wrapped_data_(false)
    {}
    //! Copy constructor that makes a deep copy of the source array.
    inline MultiArray(const self_type& M)
      : begin_(new T[M.size()])
      , end_(begin_ + M.size())
      , sizes_(M.sizes())
      , strides_(M.strides())
      , is_wrapped_data_(false)
    { std::copy(M.data(), M.data() + M.size(), begin_); }
    //! \brief Copy constructor that recopies the data source array with 
    //! appropriate type casting.
    template <typename T2>
    inline MultiArray(const MultiArray<T2, N, StorageOrder>& M)
      : begin_(new T[M.size()])
      , end_(begin_ + M.size())
      , sizes_(M.sizes())
      , strides_(M.strides())
      , is_wrapped_data_(false)
    { std::transform(M.begin(), M.end(), begin(), Cast()); }

    //! Destructor.
    inline ~MultiArray()
    { 
      if (!is_wrapped_data_ && begin_) 
        delete [] begin_;
    }

    //! \brief Assignment operator that makes a deep copy of the source array.
    const self_type& operator=(const self_type& M)
    {
      // We must be careful when (M == *this)!
      if (!is_wrapped_data_ && begin_ != M.data())
      {
        delete[] begin_;
        begin_ = 0;
        end_ = 0;
      }
      
      // Allocate new data.
      T *newData = new T[M.size()];
      sizes_ = M.sizes();
      strides_ = M.strides();
      is_wrapped_data_ = false;

      if (newData)
        std::copy(M.data(), M.data() + M.size(), newData);

      // We must be careful when (M == *this)!
      if (begin_ == M.data())
        delete[] begin_;
      begin_ = newData;
      end_ = begin_ + M.size();

      return *this;
    }
    //! \brief Assignment operator that recopies the content of the source array
    //! with appropriate type casting.
    template <typename T2>
    const self_type& operator=(const MultiArray<T2, N, StorageOrder>& M)
    {
      if (is_wrapped_data_)
      {
        std::cerr << "Fatal Error: using the assignment operator on wrapped data is not allowed!" << std::endl;
        std::cerr << "Terminating !" << std::endl;
        exit(-1);
      }
      // Free memory.
      delete[] begin_;
      // Copy everything.
      begin_ = new T[M.size()];
      sizes_ = M.sizes();
      end_ = begin_ + M.size();
      strides_ = M.strides();
      std::transform(M.begin(), M.end(), begin_, Cast());
      return *this;
    }

    //! Mutable referencing operator.
    inline reference operator()(const coords_type& c)
    { return begin_[offset(c)]; }
    //! Mutable referencing operator.
    inline reference operator()(int i, int j)
    { return begin_[offset(i, j)]; }
    //! Mutable referencing operator.
    inline reference operator()(int i, int j, int k)
    { return begin_[offset(i, j, k)]; }
    //! Non-mutable referencing operator.
    inline const_reference operator()(const coords_type& c) const
    { return begin_[offset(c)]; }
    //! Non-mutable referencing operator.
    inline const_reference operator()(int i, int j) const
    { return begin_[offset(i, j)]; }
    //! Non-mutable referencing operator.
    inline const_reference operator()(int i, int j, int k) const
    { return begin_[offset(i, j, k)]; }

    //! Mutable POD accessor.
    inline pointer data()
    { return begin_; }
    //! Non-mutable POD accessor.
    inline const_pointer data() const
    { return begin_; }

    //! Mutable beginning iterator.
    inline iterator begin()
    { return begin_; }
    //! Non-mutable beginning iterator.
    inline const_iterator begin() const
    { return begin_; }
    //! Mutable ending iterator.
    inline iterator end()
    { return end_; }
    //! Non-mutable ending iterator.
    inline const_iterator end() const
    { return end_; }

    //! Size getter.
    const vector_type& sizes() const
    { return sizes_; }
    //! Raw size getter.
    size_type size() const
    { return end_ - begin_; }
    //! Size getter along the i-th.
    int size(int i) const
    { return sizes_[i]; }
    //! Number of rows getter.
    int rows() const
    { return sizes_[0]; }
    //! Number of cols getter.
    int cols() const
    { return sizes_[1]; }
    //! Depth getter.
    int depth() const
    { return sizes_[2]; }

    //! Strides getter.
    inline const vector_type& strides() const
    { return strides_; }
    //! Stride along the i-th dimension.
    inline int stride(int i) const
    { return strides_[i]; }

    //! Mutable locator.
    inline range_iterator begin_range(const coords_type& anchor = coords_type::Zero())
    {
      // if anchor is within bounds
      range_iterator loc(begin_, coords_type::Zero(), begin_, end_, sizes_, strides_);
      if (anchor != coords_type::Zero())
        loc += anchor;
      return loc;
    }
    inline range_iterator end_range()
    { return range_iterator(end_, sizes(), begin_, end_, sizes_, strides_, true); }
    //! Mutable subrange locator
    inline subrange_iterator begin_subrange(const vector_type& start,
                                            const vector_type& end)
    {
      return subrange_iterator(
        begin_, start, start, end, false,
        begin_+offset(start), sizes_, strides_);
    }
    inline subrange_iterator end_subrange()
    {
      return subrange_iterator(
        begin_, sizes_, sizes_, sizes_, true,
        end_, sizes_, strides_);
    }
    //! Immutable locator.
    inline const_range_iterator begin_range(const coords_type& anchor = coords_type::Zero()) const
    {
      // if anchor is within bounds
      const_range_iterator loc(begin_, coords_type::Zero(), begin_, end_, sizes_, strides_);
      if (anchor != coords_type::Zero())
        loc += anchor;
      return loc;
    }
    inline const_range_iterator end_range() const
    { return const_range_iterator(end_, sizes_, begin_, end_, sizes_, strides_, true); }
    //! Mutable subrange locator
    inline const_subrange_iterator begin_subrange(const vector_type& start,
                                                  const vector_type& end) const
    {
      return const_subrange_iterator(
        begin_, start, start, end, false,
        begin_+offset(start), sizes_, strides_);
    }
    inline const_subrange_iterator end_subrange() const
    {
      return const_subrange_iterator(
        begin_, sizes_, sizes_, sizes_, true,
        end_, sizes_, strides_);
    }
    //! Coords iterator.
    inline coords_iterator begin_coords() const
    { return coords_iterator(coords_type::Zero(), (sizes_.array()-1).matrix()); }
    inline coords_iterator end_coords() const
    { return coords_iterator(); }

    //! Resizing method.
    inline bool resize(const vector_type& sizes)
    {
      if (is_wrapped_data_)
      {
        std::cerr << "data is wrapped! Not resizing" << std::endl;
        return false;
      }
      
      if (!begin_)
        delete[] begin_;

      begin_ = new T[compute_size(sizes)];
      end_ = begin_ + compute_size(sizes);
      sizes_ = sizes;
      strides_ = compute_strides(sizes);
      return true;
    }
    //! Resizing method.
    inline bool resize(int rows, int cols)
    { return resize(vector_type(rows, cols)); }
    //! Resizing method.
    inline bool resize(int rows, int cols, int depth)
    { return resize(vector_type(rows, cols, depth)); }

    //! Non-mutable array view for linear algebra with Eigen 3.
    inline const_array_view_type array() const
    {
      return const_array_view_type( reinterpret_cast<
        const typename ElementTraits<T>::const_pointer>(data()),
        size()); 
    }
    //! Mutable array view for linear algebra with Eigen 3.
    inline array_view_type array()
    {
      return array_view_type( reinterpret_cast<
        typename ElementTraits<T>::pointer>(data()),
        size()); 
    }
    //! Non-mutable matrix view for linear algebra with Eigen 3.
    inline const_matrix_view_type matrix() const
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return const_matrix_view_type( reinterpret_cast<
        typename ElementTraits<T>::const_pointer>(data()),
        rows(), cols());
    }
    //! Mutable matrix view for linear algebra with Eigen 3.
    inline matrix_view_type matrix()
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return matrix_view_type( reinterpret_cast<
        typename ElementTraits<T>::pointer>(data()),
        rows(), cols());
    }

    //! Debugging methods.
    inline void check_sizes_and_strides() const
    {
      std::cout << "Multiarray size = "
        << Map<const Matrix<int, 1, N> >(sizes_.data()) << std::endl;
      std::cout << "Multiarray strides = " 
        << Map<const Matrix<int, 1, N> >(strides_.data()) << std::endl;
    }

  private: /* helper functions for offset computation. */
    //! \brief Stride computing method called in the construction and
    //! resizing of the array.
    inline vector_type compute_strides(const vector_type& sizes)
    {
      vector_type strides;
      storage_index::eval_strides(strides.data(), sizes.data());
      return strides;
    }
    //! \brief Raw size computing method called in the construction and
    //! resizing of the array.
    inline int compute_size(const vector_type& sizes) const
    {
      return std::accumulate(sizes.data(), sizes.data()+N,
        1, std::multiplies<int>());
    }
    //! Offset computing method.
    inline int offset(const coords_type& c) const
    {
      // Complexities are comparable: no significant performance discrepancy.
      // Both exploit loop unrolling.
      //return Offset2<N>::eval(c.data(), strides_.data());
      return storage_index::eval(c.data(), sizes_.data());
    }
    //! Offset computing method.
    inline int offset(int i, int j) const
    { return storage_index::eval(i, j, sizes_[0], sizes_[1]); }
    //! Offset computing method.
    inline int offset(int i, int j, int k) const
    { return storage_index::eval(i, j, k, sizes_[0], sizes_[1], sizes_[2]); }
    //! \brief Casting functor
    struct Cast
    {
      //! Overloaded operator to get the job done.
      template <typename U>
      inline T operator()(const U& u) const { return static_cast<T>(u); }
    };

  private: /* data members. */
    value_type *begin_; //!< first element of the data.
    value_type *end_; //!< first element of the data.
    vector_type sizes_; //!< vector of size along each dimension.
    vector_type strides_; //!< vector of stride for each dimension.
    //! \brief flag that checks if the array wraps some data. It is used for 
    //! deallocation.
    bool is_wrapped_data_;
  };

  //! output stream operator
  template <typename T, int N, int StorageOrder>
  std::ostream& operator<<(std::ostream& os,
                           const MultiArray<T,N,StorageOrder>& M)
  {
    os << M.sizes() << std::endl;
    os << M.array() << std::endl;
    return os;
  }

  //! @}

}

#endif /* DO_CORE_MULTIARRAY_HPP */