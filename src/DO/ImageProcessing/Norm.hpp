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

#ifndef DO_IMAGEPROCESSING_NORM_HPP
#define DO_IMAGEPROCESSING_NORM_HPP

namespace DO {
  
  /*!
    \ingroup Differential
    @{
   */

  /*!
    \brief Squared norm computation
    @param[in] src scalar field.
    @param[in, out] scalar field of squared norms
   */
  template <typename T, int M, int N, int D>
  void squaredNorm(Image<T, D>& dst, const Image<Matrix<T,M,N>, D>& src)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    typedef typename Image<Matrix<T,M,N>, D>::const_iterator InputIterator;
    typedef typename Image<T, D>::iterator OutputIterator;

    InputIterator src_it(src.begin());
    InputIterator src_it_end(src.end());
    OutputIterator dst_it(dst.begin());
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->squaredNorm();
  }
  /*!
    \brief Squared norm computation
    @param[in] src scalar field.
    \return scalar field of squared norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> squaredNorm(const Image<Matrix<T,M,N>, D>& src)
  {
    Image<T, D> sqNorm;
    squaredNorm(sqNorm, src);
    return sqNorm;
  }
  /*!
    \brief Blue norm computation
    @param[in] src scalar field.
    @param[in, out] scalar field of norms
   */
  template <typename T, int M, int N, int D>
  void blueNorm(Image<T, D>& dst, const Image<Matrix<T,M,N>, D>& src)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    typedef typename Image<Matrix<T,M,N>, D>::const_iterator InputIterator;
    typedef typename Image<T, D>::iterator OutputIterator;

    InputIterator src_it(src.begin());
    InputIterator src_it_end(src.end());
    OutputIterator dst_it(dst.begin());
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->blueNorm();
  }
  /*!
    \brief Blue norm computation
    @param[in] src scalar field.
    \return scalar field of norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> blueNorm(const Image<Matrix<T,M,N>, D>& src)
  {
    Image<T, D> bNorm;
    blueNorm(bNorm, src);
    return bNorm;
  }
  /*!
    \brief Stable norm computation
    @param[in] src scalar field.
    @param[in, out] scalar field of norms
   */
  template <typename T, int M, int N, int D>
  void stableNorm(Image<T, D>& dst, const Image<Matrix<T,M,N>, D>& src)
  {
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    typedef typename Image<Matrix<T,M,N>, D>::const_iterator InputIterator;
    typedef typename Image<T, D>::iterator OutputIterator;

    InputIterator src_it(src.begin());
    InputIterator src_it_end(src.end());
    OutputIterator dst_it(dst.begin());
    for ( ; src_it != src_it_end; ++src_it, ++dst_it)
      *dst_it = src_it->stableNorm();
  }
  /*!
    \brief Stable norm computation
    @param[in] src scalar field.
    \return scalar field of norms
   */
  template <typename T, int M, int N, int D>
  Image<T, D> stableNorm(const Image<Matrix<T,M,N>, D>& src)
  {
    Image<T, D> sNorm;
    stableNorm(sNorm, src);
    return sNorm;
  }

#define CREATE_NORM_FUNCTOR(Function, function)       \
   /*! \brief Helper class to use Image<T,N>::compute<Function>() */ \
  template <typename T, int N>                        \
  struct Function                                     \
  {                                                   \
    typedef typename T::Scalar Scalar;                \
    typedef Image<T, N> TField;                       \
    typedef Image<Scalar, N> ScalarField, ReturnType; \
    inline Function(const TField& tField)             \
      : t_field_(tField) {}                           \
    ReturnType operator()() const                     \
    { return function(t_field_); }                    \
    const TField& t_field_;                           \
  }

  CREATE_NORM_FUNCTOR(SquaredNorm, squaredNorm);
  CREATE_NORM_FUNCTOR(BlueNorm, blueNorm);
  CREATE_NORM_FUNCTOR(StableNorm, stableNorm);

#undef CREATE_NORM_FUNCTOR

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_NORM_HPP */