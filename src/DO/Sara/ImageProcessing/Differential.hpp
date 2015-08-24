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

//! @file

#ifndef DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP
#define DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP


#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

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
      const_scalar_field_iterator it(_scalar_field.begin_array());
      it += position;
      this->operator()(gradient, it);
    }

    void operator()(gradient_field_type& gradient_field) const
    {
      if (gradient_field.sizes() != _scalar_field.sizes())
        gradient_field.resize(_scalar_field.sizes());

      const_scalar_field_iterator src_it(_scalar_field.begin_array());
      gradient_field_iterator dst_it(gradient_field.begin_array());
      for ( ; !src_it.end(); ++src_it, ++dst_it)
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
    using coords_type = Matrix<int, N, 1>;
    using scalar_field_type = Image<T, N>;
    using array_iterator = typename scalar_field_type::array_iterator;
    using const_array_iterator =
      typename scalar_field_type::const_array_iterator;

    using return_type = scalar_field_type;

    inline Laplacian(const scalar_field_type& scalar_field)
      : _scalar_field(scalar_field)
    {
    }

    inline T operator()(const_array_iterator& it) const
    {
      T value = PixelTraits<T>::zero();
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
      const_array_iterator loc(_scalar_field.begin_array());
      loc += position;
      return this->operator()(loc);
    }

    void operator()(scalar_field_type& laplacian_field) const
    {
      if (laplacian_field.sizes() != _scalar_field.sizes())
        laplacian_field.resize(_scalar_field.sizes());

      const_array_iterator src_it(_scalar_field.begin_array());
      array_iterator dst_it(laplacian_field.begin_array());

      for ( ; !src_it.end(); ++src_it, ++dst_it)
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
    typedef Matrix<int, N, 1> coords_type;
    typedef Image<T, N> scalar_field_type;
    typedef Matrix<T, N, N> hessian_matrix_type;
    typedef Image<hessian_matrix_type, N> hessian_field_type, return_type;
    typedef typename scalar_field_type::const_array_iterator
      const_scalar_iterator;
    typedef typename hessian_field_type::array_iterator
      hessian_field_iterator;

    inline Hessian(const scalar_field_type& scalar_field)
      : _scalar_field(scalar_field)
    {
    }

    void operator()(hessian_matrix_type& H, const_scalar_iterator& it) const
    {
      for (int i = 0; i < N; ++i)
      {
        for (int j = i; j < N; ++j)
        {
          if (i == j)
          {
            T next = it.position()[i] == it.sizes()[i]-1 ? *it : it.delta(i, 1);
            T prev = it.position()[i] == 0 ? *it : it.delta(i,-1);

            H(i,i) = next - T(2)*(*it) + prev;
          }
          else
          {
            int next_i = it.position()[i] == it.sizes()[i] - 1 ? 0 : 1;
            int prev_i = it.position()[i] == 0 ? 0 : -1;
            int next_j = it.position()[j] == it.sizes()[j] - 1 ? 0 : 1;
            int prev_j = it.position()[j] == 0 ? 0 : -1;

            H(i,j) =
              (  it.delta(i,next_i,j,next_j) - it.delta(i,prev_i,j,next_j)
               - it.delta(i,next_i,j,prev_j) + it.delta(i,prev_i,j,prev_j) )
              / static_cast<T>(4);

            H(j,i) = H(i,j);
          }
        }
      }
    }

    inline void operator()(hessian_matrix_type& H,
                           const coords_type& position) const
    {
      const_scalar_iterator loc(_scalar_field.begin_array());
      loc += position;
      operator()(H, loc);
    }

    void operator()(hessian_field_type& hessian_field) const
    {
      if (hessian_field.sizes() != _scalar_field.sizes())
        hessian_field.resize(_scalar_field.sizes());

      const_scalar_iterator src_loc(_scalar_field.begin_array());
      hessian_field_iterator dst_loc(hessian_field.begin_array());
      for ( ; !src_loc.end(); ++src_loc, ++dst_loc)
        operator()(*dst_loc, src_loc);
    };

    hessian_field_type operator()() const
    {
      hessian_field_type hessian_field;
      operator()(hessian_field);
      return hessian_field;
    }

    const scalar_field_type& _scalar_field;
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
    Matrix<T, N, 1> g{};
    Gradient<T, N> compute_gradient{ src };
    compute_gradient(g, p);
    return g;
  }

  /*!
    \brief Gradient computation
    @param[in] in scalar field
    \return gradient vector field
   */
  template <typename T, int N>
  Image<Matrix<T, N, 1>, N> gradient(const Image<T, N>& in)
  {
    Image<Matrix<T, N, 1>, N> out{};
    Gradient<T, N> compute_gradient{ in };
    compute_gradient(out);
    return out;
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
    Laplacian<T, N> compute_laplacian(src);
    return compute_laplacian(p);
  }

  /*!
    \brief Laplacian computation
    @param[in] in scalar field.
    \return laplacian field.
   */
  template <typename T, int N>
  Image<T, N> laplacian(const Image<T, N>& in)
  {
    Image<T, N> out{};
    Laplacian<T, N> compute_laplacian(in);
    compute_laplacian(out);
    return out;
  }

  /*!
    \brief Compute the Hessian matrix at a specified position.
    @param[in] src scalar field.
    @param[in] p position in the image.
    \return Hessian matrix.
   */
  template <typename T, int N>
  Matrix<T,N,N> hessian(const Image<T, N>& src, const Matrix<int, N, 1>& p)
  {
    Matrix<T, N, N> H{};
    Hessian<T, N> compute_hessian{ src };
    compute_hessian(H, p);
    return H;
  }

  /*!
    \brief Compute the Hessian matrix field.
    @param[in] in scalar field.
    \return Hessian matrix field
   */
  template <typename T, int N>
  Image<Matrix<T,N,N> > hessian(const Image<T, N>& in)
  {
    Image<Matrix<T, N, N> > out{};
    Hessian<T, N> compute_hessian{ in };
    compute_hessian(out);
    return out;
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP */