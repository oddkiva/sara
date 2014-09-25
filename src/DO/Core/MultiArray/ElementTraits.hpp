#ifndef DO_CORE_MULTIARRAY_ELEMENTTRAITS_HPP
#define DO_CORE_MULTIARRAY_ELEMENTTRAITS_HPP


namespace DO {

  //! \ingroup MultiArray
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

  //! \brief The specialized element traits class when the entry is a matrix.
  //! Again the matrix is viewed as a scalar. Additions and subtractions between
  //! matrices are OK but multiplication will be the pointwise matrix 
  //! multiplication.
  //!
  //! This may be questionable and this may change in the future.
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
  //! Default "scalar" operations are point-wise matrix operations.
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

  //! @}

}


#endif /* DO_CORE_MULTIARRAY_ELEMENTTRAITS_HPP */