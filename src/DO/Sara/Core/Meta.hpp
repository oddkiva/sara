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

#ifndef DO_CORE_META_HPP
#define DO_CORE_META_HPP


/*!
  \ingroup Core
  \defgroup Meta Meta
  @{
    \namespace DO::Meta
    \brief This namespace contains meta-programming structures and functions.
  @}
 */

namespace DO { namespace Meta {

  //! \brief Static 3D array of integers.
  template <int _Value0, int _Value1, int _Value2>
  struct IntArray_3
  {
    enum {
      value_0 = _Value0,
      value_1 = _Value1,
      value_2 = _Value2,
      size = 3
    };
  };


  //! Getter class for IntArray_3.
  template <typename IntArray, int index> struct Get;

  //! @{
  //! Specialized getter class for IntArray_3.
  template <typename IntArray> struct Get<IntArray, 0>
  { enum { value = IntArray::value_0 }; };

  template <typename IntArray> struct Get<IntArray, 1>
  { enum { value = IntArray::value_1 }; };

  template <typename IntArray> struct Get<IntArray, 2>
  { enum { value = IntArray::value_2 }; };

  template <typename IntArray> struct Get<IntArray, 3>
  { enum { value = IntArray::value_3 }; };
  //! @}


  //! 3D vector of types.
  template <typename T0_, typename T1_, typename T2_>
  struct Vector3
  {
    enum { size = 3 };
    typedef T0_ T0; //!< Element 0 is type T0.
    typedef T1_ T1; //!< Element 1 is type T1.
    typedef T2_ T2; //!< Element 2 is type T2.
  };


  //! 4D vector of types.
  template <typename T0_, typename T1_, typename T2_, typename T3_>
  struct Vector4
  {
    enum { size = 4 };
    typedef T0_ T0; //!< Element 0 is type T0.
    typedef T1_ T1; //!< Element 1 is type T1.
    typedef T2_ T2; //!< Element 2 is type T2.
    typedef T3_ T3; //!< Element 3 is type T3.
  };


  //! Accessor for vectors of types.
  template <typename Vector, unsigned int i> struct At {};

  //! @{
  //! Specialized index getter for Vector3.
  template <typename Vector> struct At<Vector, 0>
  { typedef typename Vector::T0 Type; /*!< Return type at index 0. */};

  template <typename Vector> struct At<Vector, 1>
  { typedef typename Vector::T1 Type; /*!< Return type at index 1. */};

  template <typename Vector> struct At<Vector, 2>
  { typedef typename Vector::T2 Type; /*!< Return type at index 2. */};

  template <typename Vector> struct At<Vector, 3>
  { typedef typename Vector::T3 Type; /*!< Return type at index 3. */};
  //! @}


  //! Index getter for vector of types.
  template <typename Vector, typename T> struct IndexOf {};

  //! @{
  //! Specialized index getter for Vector3.
  template <typename T0, typename T1, typename T2>
  struct IndexOf<Vector3<T0, T1, T2>, T0>
  { enum { value = 0 }; };

  template <typename T0, typename T1, typename T2>
  struct IndexOf<Vector3<T0, T1, T2>, T1>
  { enum { value = 1 }; };

  template <typename T0, typename T1, typename T2>
  struct IndexOf<Vector3<T0, T1, T2>, T2>
  { enum { value = 2 }; };
  //! @}


  //! @{
  //! Specialized index getter for Vector4.
  template <typename T0, typename T1, typename T2, typename T3>
  struct IndexOf<Vector4<T0, T1, T2, T3>, T0>
  { enum { value = 0 }; };

  template <typename T0, typename T1, typename T2, typename T3>
  struct IndexOf<Vector4<T0, T1, T2, T3>, T1>
  { enum { value = 1 }; };

  template <typename T0, typename T1, typename T2, typename T3>
  struct IndexOf<Vector4<T0, T1, T2, T3>, T2>
  { enum { value = 2 }; };

  template <typename T0, typename T1, typename T2, typename T3>
  struct IndexOf<Vector4<T0, T1, T2, T3>, T3>
  { enum { value = 3 }; };
  //! @}


  //! @{
  //! Type equality meta-function.
  template <typename T, typename U>
  struct IsSame { static const bool value = false; /*!< Default value. */};

  template <typename T>
  struct IsSame<T, T> { static const bool value = true;  /*!< Logically. */};
  //! @}


  //! @{
  //! Choose meta-function.
  template <bool flag, typename IsTrue, typename IsFalse>
  struct Choose;

  template <typename IsTrue, typename IsFalse>
  struct Choose<true, IsTrue, IsFalse>
  { typedef IsTrue Type; /*!< Return type. */};

  template <typename IsTrue, typename IsFalse>
  struct Choose<false, IsTrue, IsFalse>
  { typedef IsFalse Type; /*!< Return type. */};
  //! @}

} /* namespace Meta */
} /* namespace DO */

#endif /* DO_CORE_META_HPP */
