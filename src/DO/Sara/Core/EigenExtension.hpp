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

/*!
  @file

  \brief Eigen matrices and vector typedefs.

  VERY IMPORTANT:
  By default Eigen uses the *COLUMN-MAJOR* storage.
 */

#ifndef DO_SARA_CORE_EIGENEXTENSION_HPP
#define DO_SARA_CORE_EIGENEXTENSION_HPP

// To avoid compilation error with Eigen
#if defined(_WIN32) || defined(_WIN32_WCE)
# define NOMINMAX
#endif

// This is a specific compiling issue with MSVC 2008
#if (_MSC_VER >= 1500 && _MSC_VER < 1600)
# define EIGEN_DONT_ALIGN
# pragma warning ( disable : 4181 ) // "../Core/Locator.hpp(444) : warning C4181: qualifier applied to reference type; ignored"
#endif

//! Activate by default math constants.
#define _USE_MATH_DEFINES

#include <sstream>

#include <Eigen/Eigen>


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
    //! @{
    //! Eigen internals.
    typedef Array<T, M, N> Real;
    typedef Array<T, M, N> NonInteger;
    typedef Array<T, M, N> Nested;

    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 0,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1
    };

    inline static Real epsilon()
    {
      Real r;
      r.fill(NumTraits<T>::epsilon());
      return r;
    }

    inline static Real dummy_precision()
    {
      Real r;
      r.fill(NumTraits<T>::dummy_precision());
      return r;
    }
    //! @}
  };

  //! \brief NumTraits template class specialization in case the scalar type
  //! is actually a matrix.
  template <typename T, int M, int N>
  struct NumTraits<Matrix<T, M, N> >
  {
    //! @{
    //! Eigen internals.
    typedef Matrix<T, M, N> Real;
    typedef Matrix<T, M, N> NonInteger;
    typedef Matrix<T, M, N> Nested;

    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 0,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1
    };

    inline static Real epsilon()
    {
      Real r;
      r.fill(NumTraits<T>::epsilon());
      return r;
    }

    inline static Real dummy_precision()
    {
      Real r;
      r.fill(NumTraits<T>::dummy_precision());
      return r;
    }
    //! @}
  };

}
//! @}


namespace DO { namespace Sara {

  /*!
    \ingroup Core
    \defgroup EigenTypedefs Eigen Integration
    @{
   */

  using namespace Eigen;

  //! @{
  //! Convenient typedef for geometric point types.
  typedef Vector2i Point2i;
  typedef Vector3i Point3i;
  typedef Vector4i Point4i;

  typedef Vector2f Point2f;
  typedef Vector3f Point3f;
  typedef Vector4f Point4f;

  typedef Vector2d Point2d;
  typedef Vector3d Point3d;
  typedef Vector4d Point4d;
  //! @}

  //! @{
  //! 128-dimensional vector type
  typedef Matrix<unsigned char, 128, 1> Vector128ub;
  typedef Matrix<float, 128, 1> Vector128f;
  //! @}

  //! I/O.
  template <typename Derived>
  std::istream& operator>>(std::istream& in, Eigen::MatrixBase<Derived>& matrix)
  {
    for (int i = 0; i < matrix.rows(); ++i)
      for (int j = 0; j < matrix.cols(); ++j)
        in >> matrix(i,j);
    return in;
  }

  //! @{
  //! Lexicographical comparison function for matrices.
  template <typename Derived>
  inline bool lexicographical_compare(const Eigen::MatrixBase<Derived>& m1,
                                      const Eigen::MatrixBase<Derived>& m2)
  {
    int i = 0;
    while (i != m1.size())
    {
      if (i==m2.size() || m2(i) < m1(i)) return false;
      else if (m1(i) < m2(i)) return true;
      ++i;
    }
    return (i != m2.size());
  }

  struct LexicographicalOrder
  {
    //! Implementation of the functor.
    template <typename Derived>
    inline bool operator()(const Eigen::MatrixBase<Derived>& m1,
                           const Eigen::MatrixBase<Derived>& m2) const
    {
       return lexicographical_compare(m1, m2);
    }
  };
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_EIGENEXTENSION_HPP */
