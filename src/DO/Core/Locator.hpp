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
//! \brief Implementation of N-dimensional iterators.

#ifndef DO_CORE_LOCATOR_HPP
#define DO_CORE_LOCATOR_HPP

namespace DO {

  //! \ingroup Core
  //! \defgroup Locator Locator
  //! @{

  //! \brief The offset computer class for N-dimensional arrays.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int N, int StorageOrder = RowMajor> struct Offset;

  //! \brief The specialized offset computer for dimension N > 1 and 
  //! row-major storage.
  template <int N>
  struct Offset<N, RowMajor>
  {
    //! Compile-time instantiation check.
    DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
    //! Compile-time variables used for loop unrolling.
    enum { FirstDim = N-1, DimIncrement = -1 };
    //! Computes the index that corresponds to the coordinates of an ND array by
    //! loop unrolling.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return coords[N-1] + size[N-1]*Offset<N-1, RowMajor>::eval(coords, size);
    }
    //! Computes the index that corresponds to the coordinates of a 2D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index size1, Index size2)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i*size2 + j;
    }
    //! Computes the index that corresponds to the coordinates of a 3D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index k,
                             Index size1, Index size2, Index size3)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return (i*size2 + j) * size3 + k;
    }
    //! Computes the strides from the sizes of an ND array.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);				
      strides[N-1] = 1;
      for (int i = N-2; i >= 0; --i)
        strides[i] = strides[i+1]*size[i+1];
    }
    //! Computes the coordinates from the index.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      for (int i = N-1; i > 1; --i)
      {
        coords[i] = offset % dims[i];
        offset = (offset - coords[i])/dims[i];
      }
      coords[0] = (offset - coords[1]) / dims[1];
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      for (int axis = N-1; axis >= 0; --axis)
      {
        ++coords[axis];
        if (coords[axis] < size[axis])
          break;
        coords[axis] = 0;
      }
    }
    //! Computes the decremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      for (int axis = N-1; axis >= 0; --axis)
      {
        if (coords[axis] > 0 && coords[axis] != size[axis])
        {
          --coords[axis];
          break;
        }
        coords[axis] = size[axis]-1;
      }
    }
  };

  //! \brief The specialized offset computer for dimension N > 1 and 
  //! column-major storage.
  template <int N>
  struct Offset<N, ColMajor>
  {
    //! Compile-time instantiation check.
    DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
    //! Compile-time variables used for loop unrolling.
    enum { FirstDim = 0, DimIncrement = 1 };
    //! Computes the index that corresponds to the coordinates of an ND array by
    //! loop unrolling.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return (*coords) + (*size)*Offset<N-1, ColMajor>::eval(coords+1, size+1);
    }
    //! Computes the index that corresponds to the coordinates of a 2D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index size1, Index size2)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i + j*size1;
    }
    //! Computes the index that corresponds to the coordinates of a 3D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index k,
                             Index size1, Index size2, Index size3)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i + size1 * (j + k*size2);
    }
    //! Computes the strides from the sizes of an ND array.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      strides[0] = 1;
      for (int i = 1; i < N; ++i)
        strides[i] = strides[i-1]*size[i-1];
    }
    //! Computes the coordinates from the index.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      for (int i = 0; i < N-1; ++i)
      {
        coords[i] = offset % dims[i];
        offset = (offset - coords[i])/dims[i];
      }
      coords[N-1] = (offset - coords[N-2]) / dims[N-2];
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);				
      for (int axis = 0; axis < N; ++axis, ++coords, ++size)
      {
        ++(*coords);
        if (*coords < *size)
          break;
        *coords = 0;
      }
    }
    //! Computes the decremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      for (int axis = 0; axis < N; ++axis, ++coords, ++size)
      {
        if (*coords > 0)
        {
          --(*coords);
          break;
        }
        *coords = *size-1;
      }
    }
  };

  //! \brief The specialized offset computer for dimension 1 and 
  //! row-major storage (for loop unrolling).
  template <>
  struct Offset<1, RowMajor>
  {
    //! Merely returns the 1D coordinates.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return *coords;
    }
    //! Merely returns 1.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *strides = 1;
    }
    //! Merely returns the offset value.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      coords[0] = offset;
    }
    //! Merely does +1.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      ++(*coords);
    }
    //! Merely does -1.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      --(*coords);
    }
  };

  //! \brief The specialized offset computer for dimension 1 and 
  //! column-major storage.
  template <>
  struct Offset<1, ColMajor>
  {
    //! Merely returns the 1D coordinates.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return *coords; 
    }
    //! Merely returns 1.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *strides = 1;
    }
    //! Merely returns the offset value.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset, const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      coords[0] = offset;
    }
    //! Merely does +1.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      ++(*coords);
    }
    //! Merely does -1.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      --(*coords);
    }
  };

  //! \brief Axis iterator class for N-dimensional arrays.
	template <typename T, int Axis, int N, int StorageOrder>
	class AxisIterator : public std::iterator<std::random_access_iterator_tag, T>
	{
		DO_STATIC_ASSERT(Axis >= 0 && Axis < N,
      AXIS_MUST_BE_NONNEGATIVE_AND_LESS_THAN_N);
		typedef Offset<N, StorageOrder> offset;
		typedef std::iterator<std::random_access_iterator_tag, T> Base;

	public: /* STL-like typedefs */
		typedef typename Base::value_type value_type;
		typedef typename Base::difference_type difference_type;
		typedef typename Base::pointer pointer;
		typedef typename Base::reference reference;
		typedef typename Base::iterator_category iterator_category;

    typedef AxisIterator self_type;
    typedef Matrix<int, N, 1> coord_type, vector_type;
    typedef coord_type& coord_reference;
    typedef pointer& pointer_reference;

	public: /* interface */
		//! Constructor.
		inline AxisIterator(pointer_reference pos, coord_reference coords, 
							          const vector_type& strides, const vector_type& size)
			: pos_(pos), coords_(coords), strides_(strides), sizes_(size) {}
		//! Copy constructor.
    inline AxisIterator(const self_type& it)
			: pos_(it.pos_), coords_(it.coords_)
			, strides_(it.strides_), sizes_(it.sizes_) {}

		//! Referencing operator.
		inline reference operator*() const { return *pos_; }
    //! Referencing operator.
		inline reference operator[](int n) const
		{
			if (coords_[Axis]+n >= 0  && coords_[Axis]+n < sizes_[Axis])
				return *(pos_+strides_[Axis]*n);
			return *pos_;
		}

    //! Non-mutable pointer operator.
    inline pointer operator->() const { return pos_; }

		//! Equality operator.
		template <typename Iterator>
		inline bool operator==(const Iterator& rhs) const
		{ return pos_ == rhs.operator->(); }
    //! Equality operator.
		inline bool operator==(pointer pos) const
		{ return pos_ == pos; }

		//! Inequality operator.
		template <typename Iterator>
		inline bool operator!=(const Iterator& rhs) const
		{ return pos_ != rhs.operator->(); }
    //! Inequality operator.
		inline bool operator!=(pointer pos) const
		{ return pos_ == pos; }

		//! Prefix increment operator.
		inline self_type& operator++()
		{
			if (coords_[Axis]+1 < sizes_[Axis])
			{
				pos_ += strides_[Axis];
				++coords_[Axis];
			}
			return *this;
		}
    //! Prefix decrement operator.
		inline self_type& operator--()
		{
			if (coords_[Axis]-1 >= 0)
			{
				pos_ -= strides_[Axis];
				--coords_[Axis];
			}
			return *this;
		}

		//! Postfix increment operator.
		inline self_type operator++(int)
		{ AxisIterator old(*this); operator++(); return old; }
		//! Postfix decrement operator.
    inline self_type operator--(int)
		{ AxisIterator old(*this); operator--(); return old; }

		//! Arithmetic operator.
		inline self_type& operator+=(int n)
		{
			if (coords_[Axis]+n >= 0  && coords_[Axis]+n < sizes_[Axis])
			{
				pos_ += strides_[Axis]*n;
				coords_[Axis] += n;
			}
			return *this;
		}
    //! Arithmetic operator.
		inline self_type& operator-=(int n)
		{
			if (coords_[Axis]-n >= 0  && coords_[Axis]-n < sizes_[Axis])
			{
				pos_ -= strides_[Axis]*n;
				coords_[Axis] -= n;
			}
			return *this;
		}

    //! Constant size accessor.
		inline int size() const { return sizes_[Axis]; }

	private:
		pointer_reference pos_; //!< current pointer.
		coord_reference coords_; //!< current coordinates.
		const vector_type& strides_; //!< strides.
		const vector_type& sizes_; //!< sizes.
	};

  //! \brief N-dimensional iterator class.
	template <typename T, int N, int StorageOrder = ColMajor>
	class Locator : public std::iterator<std::random_access_iterator_tag, T>
	{
		typedef Offset<N, StorageOrder> offset;
		typedef std::iterator<std::random_access_iterator_tag, T> Base;

	public: /* typedefs */
		typedef typename Base::value_type value_type; //!< value_type
		typedef typename Base::difference_type difference_type; //!< difference_type
		typedef typename Base::pointer pointer; //!< pointer
		typedef typename Base::reference reference; //!< reference
		typedef typename Base::iterator_category iterator_category; //!< iterator_category

    typedef Locator self_type; //!< self_type
    typedef AxisIterator<value_type, 0, N, StorageOrder> x_iterator; //!< x_iterator
    typedef AxisIterator<value_type, 1, N, StorageOrder> y_iterator; //!< y_iterator
    typedef AxisIterator<value_type, 2, N, StorageOrder> z_iterator; //!< z_iterator
    typedef Matrix<int, N, 1> coord_type, vector_type; //!< coord_type

	public: /* interface */
		//! Constructor
		inline Locator(pointer pos, const coord_type& coords,
                   pointer first, pointer last,
                   const vector_type& sizes, const vector_type& strides)
      : Base()
			, pos_(pos), coords_(coords)
			, first_(first), last_(last)
      , sizes_(sizes), strides_(strides) {}
    //! Copy constructor
		inline Locator(const self_type& l)
			: Base()
			, pos_(l.pos_), coords_(l.coords_)
			, first_(l.first_), last_(l.last_)
			, sizes_(l.sizes_), strides_(l.strides_) {}

		//! Referencing operator.
		inline reference operator*() const { return *pos_; }
    //! Referencing operator.
		inline pointer operator->() const { return pos_; }
    //! Referencing operator.
		inline reference operator[](int n) const { return *(pos_+n); }

		//! Particular referencing operators.
		inline reference operator()(int i, int j) const
		{ return *(pos_ + strides_[0]*i + strides_[1]*j); }
    //! Particular referencing operators.
		inline reference operator()(int i, int j, int k) const
		{ return *(pos_ + strides_[0]*i + strides_[1]*j + strides_[2]*k); }
    //! Particular referencing operators.
    inline reference operator()(const vector_type& t) const 
    {
      pointer pos = pos_;
			for (int i = 0; i < N; ++i)
				pos += strides_[i]*t[i];
			return *pos;
    }

		//! Equality operator.
		inline bool operator==(const self_type& rhs) const 
		{ return pos_ == rhs.pos_; }
    //! Equality operator.
		inline bool operator==(pointer pos) const
		{ return pos_ == pos; }

		//! Inequality operator.
		inline bool operator!=(const self_type& rhs) const
		{ return pos_ != rhs.pos_; }
    //! Inequality operator.
		inline bool operator!=(pointer pos) const
		{ return pos_ != pos; }

		//! Prefix increment operator.
		inline self_type& operator++()
		{
			++pos_;
			offset::increment_coords(coords_.data(), sizes_.data());
			return *this;
		}
    //! Prefix decrement operator.
		inline self_type& operator--()
		{
			--pos_;
			offset::decrement_coords(coords_.data(), sizes_.data());
			return *this;
		}

		//! Postfix increment operator.
		inline self_type operator++(int)
		{ Locator old(*this); operator++(); return old; }
    //! Postfix decrement operator.
		inline self_type operator--(int)
		{ Locator old(*this); operator--(); return old; }

		// Arithmetic operators
		// TODO: unroll loop here ?
		inline self_type& operator+=(const vector_type& t)
		{
			for (int i = 0; i < N; ++i)
				pos_ += strides_[i]*t[i];
			coords_ += t;
			return *this;
		}
		inline self_type& operator-=(const vector_type& t)
		{
			for (int i = 0; i < N; ++i)
				pos_ -= strides_[i]*t[i];
			coords_ -= t;
			return *this;
		}

		//! Axis iterator getter.
		//! The axes matches with the Cartesian view if the data is stored in a
		//! row major fashion.
		template <int Axis>
		inline AxisIterator<T, Axis, N, StorageOrder> axis()
    {
      return AxisIterator<T, Axis, N, StorageOrder>(
        pos_, coords_, strides_, sizes_);
    }
    //! X-axis iterator getter.
		inline x_iterator x() { return axis<0>(); }
    //! Y-axis iterator getter.
		inline y_iterator y() { return axis<1>(); }
    //! Z-Axis iterator getter.
		inline z_iterator z() { return axis<2>(); }

		//! Get the current coordinates.
		inline const coord_type& coords() const { return coords_; }

		//! Get the sizes.
		inline const vector_type& sizes() const { return sizes_; }
    //! Get the size of the i-th dimension.
		inline int size(int i) const { return sizes_[i]; }

		//! Get the strides.
		inline const vector_type& strides() const { return strides_; }
    //! Get the i-th stride.
		inline int stride(int i) const { return strides_[i]; }

		//! Debugging method.
		inline void check() const
		{
			std::cout << "Checking locator" << std::endl;
			std::cout << "Coords = " 
			        	<< Map<const Matrix<int, 1, N> >(coords_.data()) << std::endl;
			std::cout << "Value = " << std::endl << *pos_ << std::endl;
		}
    //! Debugging method.
		inline void check_strides() const
		{
			std::cout << "Checking locator strides" << std::endl;
      std::cout << "Strides = "
                << Map<const Matrix<int, 1, N> >(strides_.data()) << std::endl;
      std::cout << "Sizes = "
                << Map<const Matrix<int, 1, N> >(sizes_.data()) << std::endl;

		}

    //! Additional features.
    inline pointer operator()() const { return pos_; }
    inline pointer begin() { return first_; }
    inline pointer end() { return last_; }

    inline bool is_out_of_bounds() const
    { return pos_ < first_ || pos_ > last_; }
		inline bool is_out_of_bounds(pointer p) const
		{ return p < first_ || p > last_; }
		inline bool is_out_of_bounds(const coord_type& c) const
		{
			for (int i = 0; i < N; ++i)
				if (c[i] < 0 || c[i] >= Base::sizes_[i])
					return false;
			return true;
		}

		inline coord_type get_coords_of_first() const
		{ return coord_type::Zero(); }
		inline coord_type get_coords_of_last() const
		{ return (sizes_.array()-1).matrix(); }
    inline void reset_anchor(const coord_type& c = coord_type::Zero())
    {
      pos_ = first_ + offset::eval(c.data(), sizes_.data());
      coords_ = c;
    }
		inline void reset_anchor(int x, int y)
		{ reset_anchor(coord_type(x,y)); }
		inline void reset_anchor(int x, int y, int z)
		{ reset_anchor(coord_type(x,y,z)); }

	protected:
    pointer pos_;
    coord_type coords_;
    pointer first_;
    pointer last_;
    const vector_type& sizes_;
    const vector_type& strides_;
	};

  //! \brief Range iterator class for N-dimensional array.
	template <int N>
	class RangeIterator : public std::iterator<std::input_iterator_tag, Matrix<int, N, 1> >
	{
	public:
    typedef RangeIterator self_type;
		typedef Matrix<int, N, 1> coord_type, Coords;

	protected:
		coord_type a_;
		coord_type b_;
		coord_type pos_;
		bool stop_;
	public:
		inline RangeIterator()
			: stop_(true) {}

		inline RangeIterator(const coord_type& a, const coord_type& b)
			: a_(a), b_(b), pos_(a), stop_(false) {}

		inline RangeIterator& operator=(const self_type& it)
		{
			a_ = it.a_;
			b_ = it.b_;
			pos_ = it.pos_;
			stop_ = it.stop_;
			return *this;
		}

		inline bool operator==(const self_type& it) const
		{ return (stop_ ? it.stop_ : !it.stop_ && pos_ == it.pos_); }

		inline bool operator!=(const self_type& it) const
		{ return (stop_ ? !it.stop_ : it.stop_ || pos_ != it.pos_); }

		inline self_type& operator++()
		{
			for (int i=0;i<N;i++) {
				if (pos_[i]!=b_[i]) {
					pos_[i]++;
					return *this;
				}
				pos_[i]=a_[i];
			}
			stop_ = true;
			return *this;
		}

		inline self_type operator++(int)
		{ RangeIterator tmp(*this); ++(*this); return tmp; }

		inline coord_type operator*() const
		{ return pos_; }

		inline const coord_type* operator->() const
		{ return &pos_; }
	};

  //! @}

} /* namespace DO */

#endif /* DO_CORE_LOCATOR_HPP */
