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

/*!
  @file

  \brief Eigen matrices and vector typedefs.
 
  VERY IMPORTANT:
  By default Eigen uses the *COLUMN-MAJOR* storage.
 */

#ifndef DO_CORE_EIGENEXTENSION_HPP
#define DO_CORE_EIGENEXTENSION_HPP

//! \namespace Eigen
//! \brief Some customized extension to interface for the Eigen library
//! Essentially template class specialization for array type instead of scalar.
//! I wonder if this is very useful. But let's just put it for now.
//! @{
namespace Eigen {

  //! \brief NumTraits template class specialization in case the scalar type
  //! is actually an array.
  //! For example, when an Array<Matrix2d, 2, 2> type is instantiated, 
  //! multiplication and addition operations are properly defined.
  template <typename T, int M, int N>
  struct NumTraits<Array<T, M, N> >
  {
    typedef Array<T, M, N> Real;        //!< Eigen internals.
    typedef Array<T, M, N> NonInteger;  //!< Eigen internals.
    typedef Array<T, M, N> Nested;      //!< Eigen internals.

    enum {
      IsComplex = 0,                    //!< Eigen internals.
      IsInteger = 0,                    //!< Eigen internals.
      IsSigned = 0,                     //!< Eigen internals.
      RequireInitialization = 1,        //!< Eigen internals.
      ReadCost = 1,                     //!< Eigen internals.
      AddCost = 1,                      //!< Eigen internals.
      MulCost = 1                       //!< Eigen internals.
    };
    
    //! Eigen internals.
    inline static Real epsilon()
    { Real r; r.fill(NumTraits<T>::epsilon()); return r; }

    //! Eigen internals.
    inline static Real dummy_precision()
    { Real r; r.fill(NumTraits<T>::dummy_precision()); return r; }
  };

  //! \brief NumTraits template class specialization in case the scalar type
  //! is actually a matrix.
  template <typename T, int M, int N>
  struct NumTraits<Matrix<T, M, N> >
  {
    typedef Matrix<T, M, N> Real;       //!< Eigen internals.
    typedef Matrix<T, M, N> NonInteger; //!< Eigen internals.
    typedef Matrix<T, M, N> Nested;     //!< Eigen internals.
    
    enum { 
      IsComplex = 0,                    //!< Eigen internals.
      IsInteger = 0,                    //!< Eigen internals.
      IsSigned = 0,                     //!< Eigen internals.
      RequireInitialization = 1,        //!< Eigen internals.
      ReadCost = 1,                     //!< Eigen internals.
      AddCost = 1,                      //!< Eigen internals.
      MulCost = 1                       //!< Eigen internals.
    };
    
    //! Eigen internals.
    inline static Real epsilon()
    { Real r; r.fill(NumTraits<T>::epsilon()); return r; }
    
    //! Eigen internals.
    inline static Real dummy_precision()
    { Real r; r.fill(NumTraits<T>::dummy_precision()); return r; }
  };

}
//! @}


namespace DO {

  /*!
    \ingroup Core
    \defgroup EigenTypedefs Eigen Integration
    @{
   */

  using namespace Eigen;

  // Typedefs for basic data types.
  typedef unsigned char uchar;   //!< Self-explanatory.
	typedef unsigned short ushort; //!< Self-explanatory.
  typedef unsigned int uint;     //!< Self-explanatory.
  typedef unsigned long ulong;   //!< Self-explanatory.

	// Point types with integral scalar type.
	typedef Vector2i Point2i; //!< Self-explanatory
	typedef Vector3i Point3i; //!< Self-explanatory
	typedef Vector4i Point4i; //!< Self-explanatory
  
  // Point types with single precision floating scalar type.
  typedef Vector2f Point2f; //!< Self-explanatory
	typedef Vector3f Point3f; //!< Self-explanatory
	typedef Vector4f Point4f; //!< Self-explanatory
  
  // Point types with double precision floating scalar type.
	typedef Vector2d Point2d; //!< Self-explanatory
	typedef Vector3d Point3d; //!< Self-explanatory
	typedef Vector4d Point4d; //!< Self-explanatory
  
  //! 128-dimensional integral vector type.
	typedef Matrix<uchar, 128, 1> Vector128ub;
  
  //! 128-dimensional single precision vector type.
	typedef Matrix<float, 128, 1> Vector128f;

  //! I/O.
  template <typename T, int M, int N>
  std::istream& operator>>(std::istream& in, Matrix<T, M, N>& m)
  {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        in >> m(i,j);
    return in;
  }

  //! Lexicographical comparison function for matrices.
  template <typename T, int M, int N>
  inline bool lexCompare(const Matrix<T, M, N>& m1, const Matrix<T, M, N>& m2)
  {
    return std::lexicographical_compare(
      m1.data(), m1.data()+M*N, m2.data(), m2.data()+M*N
      );
  }

  //! Lexicographical comparison functor for matrices.
  struct LexicographicalOrder
  {
    //! Implementation of the functor.
    template <typename T, int M, int N>
    inline bool operator()(const Matrix<T, M, N>& m1,
                           const Matrix<T, M, N>& m2) const
    { return lexCompare(m1, m2); }
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_EIGENEXTENSION_HPP */