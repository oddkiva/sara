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
    typedef Matrix<int, N, 1> coords_type;
    typedef Matrix<T, N, 1> gradient_type;
    typedef Image<T, N> scalar_field_type;
    typedef Image<gradient_type, N> gradient_field_type, return_type;
    typedef typename scalar_field_type::const_array_iterator
      const_scalar_field_iterator;
    typedef typename gradient_field_type::array_iterator
      gradient_field_iterator;

    inline Gradient(const scalar_field_type& scalar_field)
      : _scalar_field(scalar_field)
    {
    }

    inline void operator()(gradient_type& gradient,
                           const_scalar_field_iterator& it) const
    {
      for (int i = 0; i < N; ++i)
      {
        if (it.position()[i] == 0)
          gradient[i] = (it.delta(i, 1) - *it) / 2; // Replicate the border
        else if (it.position()[i] == it.sizes()[i] - 1)
          gradient[i] = (*it - it.delta(i,-1)) / 2; // Replicate the border
        else
          gradient[i] = (it.delta(i, 1) - it.delta(i,-1)) / 2;
      }
    }

    inline void operator()(gradient_type& gradient,
                           const coords_type& position) const
    {
      const_scalar_field_iterator it(_scalar_field.begin_range());
      it += position;
      this->operator()(gradient, it);
    }

    void operator()(gradient_field_type& gradient_field) const
    {
      if (gradient_field.sizes() != _scalar_field.sizes())
        gradient_field.resize(_scalar_field.sizes());

      const_scalar_field_iterator src_it(_scalar_field.begin_range());
      const_scalar_field_iterator src_it_end(_scalar_field.end_range());
      gradient_field_iterator dst_it(gradient_field.begin_range());
      for ( ; src_it != src_it_end; ++src_it, ++dst_it)
        operator()(*dst_it, src_it);
    };

    gradient_field_type operator()() const
    {
      gradient_field_type gradient_field;
      operator()(gradient_field);
      return gradient_field;
    }

    const scalar_field_type& _scalar_field;
  };


  //! Laplacian functor class
  template <typename T, int N = 2>
  struct Laplacian
  {
    typedef Matrix<int, N, 1> coords_type;
    typedef Image<T, N> scalar_field_type, return_type;
    typedef typename scalar_field_type::array_iterator array_iterator;
    typedef typename scalar_field_type::const_array_iterator
      const_array_iterator;

    inline Laplacian(const scalar_field_type& scalar_field)
      : _scalar_field(scalar_field)
    {
    }

    inline T operator()(const_array_iterator& it) const
    {
      T value = 0;
      for (int i = 0; i < N; ++i)
      {
        if (it.position()[i] == 0)
          value += it.delta(i, 1) + *it; // Replicate the border
        else if (it.position()[i] == it.sizes()[i] - 1)
          value += *it + it.delta(i,-1); // Replicate the border
        else
          value += it.delta(i, 1) + it.delta(i,-1);
      }
      return value - 2*N*(*it);
    }

    inline T operator()(const coords_type& position) const
    {
      const_array_iterator loc(_scalar_field.begin_range());
      loc += position;
      return this->operator()(loc);
    }

    void operator()(scalar_field_type& laplacian_field) const
    {
      if (laplacian_field.sizes() != _scalar_field.sizes())
        laplacian_field.resize(_scalar_field.sizes());

      const_array_iterator src_it(_scalar_field.begin_range());
      const_array_iterator src_it_end(_scalar_field.end_range());
      array_iterator dst_it(laplacian_field.begin_range());

      for ( ; src_it != src_it_end; ++src_it, ++dst_it)
        *dst_it = this->operator()(src_it);
    }

    scalar_field_type operator()() const
    {
      scalar_field_type laplacian_field;
      this->operator()(laplacian_field);
      return laplacian_field;
    }

    const scalar_field_type& _scalar_field;
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
