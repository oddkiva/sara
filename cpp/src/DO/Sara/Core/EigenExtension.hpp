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

/*!
  @file

  @brief Eigen matrices and vector typedefs.

  VERY IMPORTANT:
  By default Eigen stores matrix data in a *COLUMN-MAJOR* order.
 */

#pragma once

// To avoid compilation error with Eigen
#if (defined(_WIN32) || defined(_WIN32_WCE)) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  if defined(__has_warning)  // clang
#    if __has_warning("-Wmaybe-uninitialized")
#      pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#    endif
#    if __has_warning("-Wconversion")
#      pragma GCC diagnostic ignored "-Wconversion"
#    endif
#  else  // GCC
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#  endif
#endif
#include <Eigen/Eigen>
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <sstream>


//! \namespace Eigen
//! @brief Some customized extension to interface for the Eigen library
//! Essentially template class specialization for array type instead of scalar.
//! I wonder if this is very useful. But let's just put it for now.
//! @{
namespace Eigen {

  //! @brief NumTraits template class specialization in case the scalar type
  //! is actually an array.
  //! For example, when an Array<Matrix2d, 2, 2> type is instantiated,
  //! multiplication and addition operations are properly defined.
  template <typename T, int M, int N>
  struct NumTraits<Array<T, M, N>>
  {
    //! @{
    //! Eigen internals.
    using Real = Array<T, M, N>;
    using NonInteger = Array<T, M, N>;
    using Nested = Array<T, M, N>;
    using Literal = Array<T, M, N>;

    enum
    {
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

  //! @brief NumTraits template class specialization in case the scalar type
  //! is actually a matrix.
  template <typename T, int M, int N>
  struct NumTraits<Matrix<T, M, N>>
  {
    //! @{
    //! Eigen internals.
    using Real = Matrix<T, M, N>;
    using NonInteger = Matrix<T, M, N>;
    using Nested = Matrix<T, M, N>;
    using Literal = Matrix<T, M, N>;

    enum
    {
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

}  // namespace Eigen
//! @}


namespace DO { namespace Sara {

  /*!
    @ingroup Core
    @defgroup EigenTypedefs Eigen Integration
    @{
   */

  using namespace Eigen;

  template <typename T, int N>
  using Point = Eigen::Matrix<T, N, 1>;

  //! @{
  //! Convenient aliases for geometric point types.
  using Point2i = Vector2i;
  using Point3i = Vector3i;
  using Point4i = Vector4i;

  using Point2f = Vector2f;
  using Point3f = Vector3f;
  using Point4f = Vector4f;

  using Point2d = Vector2d;
  using Point3d = Vector3d;
  using Point4d = Vector4d;
  //! @}

  //! @brief Useful for geometry.
  using Matrix34d = Eigen::Matrix<double, 3, 4>;

  //! @{
  //! 128-dimensional vector type
  using Vector128ub = Matrix<unsigned char, 128, 1>;
  using Vector128f = Matrix<float, 128, 1>;
  //! @}


  //! I/O.
  template <typename Derived>
  std::istream& operator>>(std::istream& in, Eigen::MatrixBase<Derived>& matrix)
  {
    for (int i = 0; i < matrix.rows(); ++i)
      for (int j = 0; j < matrix.cols(); ++j)
        in >> matrix(i, j);
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
      if (i == m2.size() || m2(i) < m1(i))
        return false;
      else if (m1(i) < m2(i))
        return true;
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

}}  // namespace DO::Sara
