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

#ifndef DO_IMAGEPROCESSING_DIFFERENTIAL_HPP
#define DO_IMAGEPROCESSING_DIFFERENTIAL_HPP


#include <DO/Core/Image.hpp>


namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup Differential Differential Calculus, Norms, and Other Stuff
    @{
   */


  //! Gradient functor class
  template <typename T, int N = 2>
  struct Gradient
  {
    typedef T Scalar;
    typedef Matrix<T, N, 1> Vector;
    typedef Matrix<int, N, 1> Coords;
    typedef Image<T, N> ScalarField;
    typedef Image<Vector, N> VectorField, ReturnType;
    typedef typename ScalarField::const_range_iterator ScalarIterator;
    typedef typename VectorField::range_iterator VectorIterator;

    inline Gradient(const ScalarField& scalarField)
      : scalar_field_(scalarField) {}

    inline void operator()(Vector& g, ScalarIterator& loc) const
    { Differential<N, N-1>::eval_gradient(g, loc); }

    inline void operator()(Vector& g, const Coords& p) const
    {
      ScalarIterator loc(scalar_field_.begin_range());
      loc += p;
      this->operator()(g, loc);
    }

    void operator()(VectorField& dst) const
    {
      if (dst.sizes() != scalar_field_.sizes())
        dst.resize(scalar_field_.sizes());

      ScalarIterator src_loc(scalar_field_.begin_range());
      ScalarIterator src_loc_end(scalar_field_.end_range());
      VectorIterator dst_loc(dst.begin_range());
      for ( ; src_loc != src_loc_end; ++src_loc, ++dst_loc)
        operator()(*dst_loc, src_loc);
    };

    VectorField operator()() const
    {
      VectorField gradField;
      operator()(gradField);
      return gradField;
    }

    const ScalarField& scalar_field_;
  };

  //! Laplacian functor class
  template <typename T, int N = 2>
  struct Laplacian
  {
    typedef Matrix<int, N, 1> Coords;
    typedef Image<T, N> ScalarField, ReturnType;
    typedef typename ScalarField::range_iterator RangeIterator;
    typedef typename ScalarField::const_range_iterator ConstRangeIterator;

    inline Laplacian(const ScalarField& scalarField)
      : scalar_field_(scalarField) {}

    inline T operator()(ConstRangeIterator& loc) const
    { return Differential<N, N-1>::eval_laplacian(loc); }

    inline T operator()(const Coords& p) const
    {
      ConstRangeIterator loc(scalar_field_.begin_range());
      loc += p;
      return this->operator()(loc);
    }

    void operator()(ScalarField& dst) const
    {
      if (dst.sizes() != scalar_field_.sizes())
        dst.resize(scalar_field_.sizes());

      ConstRangeIterator src_it(scalar_field_.begin_range());
      ConstRangeIterator src_it_end(scalar_field_.end_range());
      RangeIterator dst_it(dst.begin_range());

      for ( ; src_it != src_it_end; ++src_it, ++dst_it)
        *dst_it = operator()(src_it);
    };

    ScalarField operator()() const
    {
      ScalarField lapField;
      operator()(lapField);
      return lapField;
    }

    const ScalarField& scalar_field_;
  };

  //! Hessian matrix functor class
  template <typename T, int N = 2>
  struct Hessian
  {
    typedef Matrix<int, N, 1> Coords, Vector;
    typedef Image<T, N> ScalarField;
    typedef Matrix<T, N, N> HessianMatrix;
    typedef Image<HessianMatrix, N> HessianField, ReturnType;
    typedef typename ScalarField::const_range_iterator ScalarIterator;
    typedef typename HessianField::range_iterator MatrixIterator;

    inline Hessian(const ScalarField& scalarField)
      : scalar_field_(scalarField) {}

    void operator()(HessianMatrix& H, const ScalarIterator& loc) const
    {
      if (  loc.position().minCoeff() < 2 ||
           (loc.sizes() - loc.position()).minCoeff() < 2 )
      {
        H.setZero();
        return;
      }

      for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j)
        {
          if (i == j)
            H(i,i) = ( loc(delta(i,1)) - T(2)*(*loc) + loc(delta(i,-1)) )
                   / static_cast<T>(4);
          else
          {
            H(i,j) = (  loc(delta(i,1,j, 1)) - loc(delta(i,-1,j, 1))
                      - loc(delta(i,1,j,-1)) + loc(delta(i,-1,j,-1)) )
                         / static_cast<T>(4);
            H(j,i) = H(i,j);
          }
        }
    }

    inline void operator()(HessianMatrix& H, const Coords& p) const
    {
      ScalarIterator loc(scalar_field_.begin_range());
      loc += p;
      operator()(H, loc);
    }

    void operator()(HessianField& dst) const
    {
      if (dst.sizes() != scalar_field_.sizes())
        dst.resize(scalar_field_.sizes());

      ScalarIterator src_loc(scalar_field_.begin_range());
      ScalarIterator src_end(scalar_field_.end_range());
      MatrixIterator dst_loc(dst.begin_range());
      for ( ; src_loc != src_end; ++src_loc, ++dst_loc)
        operator()(*dst_loc, src_loc);
    };

    HessianField operator()() const
    {
      HessianField hessianField;
      operator()(hessianField);
      return hessianField;
    }

    inline Coords delta(int i, int dx, int j, int dy) const
    {
      Coords unit(Coords::Zero());
      unit[i] = dx; unit[j] = dy;
      return unit;
    }

    inline Coords delta(int i, int dx) const
    { return Coords::Unit(i)*dx; }

    const ScalarField& scalar_field_;
  };

  /*!
    \brief Gradient computation
    @param[in] src input grayscale image.
    @param[in] p position in the image.
    \return 2D gradient vector.
   */
  template <typename T, int N>
  Matrix<T,N,1> gradient(const Image<T, N>& src, const Matrix<int, N, 1>& p)
  {
    Matrix<T,N,1> g;
    Gradient<T, N> computeGradient(src);
    computeGradient(g, p);
    return g;
  }
  /*!
    \brief Gradient computation
    @param[in,out] dst gradient vector field
    @param[in] src scalar field
   */
  template <typename T, int N>
  void gradient(Image<Matrix<T,N,1>, N>& dst, const Image<T, N>& src)
  {
    Gradient<T, N> computeGradient(src);
    computeGradient(dst);
  }
  /*!
    \brief Gradient computation
    @param[in] src scalar field
    \return gradient vector field
   */
  template <typename T, int N>
  Image<Matrix<T,N,1>, N> gradient(const Image<T, N>& src)
  {
    Image<Matrix<T,N,1>, N> g;
    gradient(g, src);
    return g;
  }
  /*!
    \brief Laplacian computation
    @param[in] src input grayscale image.
    @param[in] p position in the image.
    \return laplacian value
   */
  template <typename T, int N>
  T laplacian(const Image<T, N>& src, const Matrix<int, N, 1>& p)
  {
    Laplacian<T, N> computeLaplacian(src);
    return computeLaplacian(p);
  }
  /*!
    \brief Laplacian computation
    @param[in,out] dst Laplacian field.
    @param[in] src scalar field.
   */
  template <typename T, int N>
  void laplacian(Image<T, N>& dst, const Image<T, N>& src)
  {
    Laplacian<T, N> computeLaplacian(src);
    computeLaplacian(dst);
  }
  /*!
    \brief Laplacian computation
    @param[in] src scalar field.
    \return laplacian field.
   */
  template <typename T, int N>
  Image<T, N> laplacian(const Image<T, N>& src)
  {
    Image<T, N> l;
    laplacian(l, src);
    return l;
  }
  /*!
    \brief Hessian matrix computation
    @param[in] src scalar field.
    @param[in] p position in the image.
    \return Hessian matrix.
   */
  template <typename T, int N>
  Matrix<T,N,N> hessian(const Image<T, N>& src, const Matrix<int, N, 1>& p)
  {
    Matrix<T,N,N> H;
    Hessian<T, N> computeHessian(src);
    computeHessian(H, p);
    return H;
  }
  /*!
    \brief Hessian matrix computation
    @param[in] src scalar field.
    @param[in,out] dst Hessian matrix field
   */
  template <typename T, int N>
  void hessian(Image<Matrix<T,N,N> >& dst, const Image<T,N>& src)
  {
    Hessian<T, N> computeHessian(src);
    computeHessian(dst);
  }
  /*!
    \brief Hessian matrix computation
    @param[in] src scalar field.
    \return Hessian matrix field
   */
  template <typename T, int N>
  Image<Matrix<T,N,N> > hessian(const Image<T,N>& src)
  {
    Image<Matrix<T, N, N> > h;
    hessian(h, src);
    return h;
  }

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DIFFERENTIAL_HPP */
