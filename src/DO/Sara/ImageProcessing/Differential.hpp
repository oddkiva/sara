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
    @ingroup ImageProcessing
    @defgroup Differential Differential Calculus, Norms, and Other Stuff
    @{
   */


  //! @brief Gradient functor class
  struct Gradient
  {
    template <typename Field>
    struct Dimension { enum { value = Field::Dimension }; };

    template <typename Field>
    using Coords = Matrix<int, Dimension<Field>::value, 1>;

    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using Vector = Matrix<Scalar<Field>, Dimension<Field>::value, 1>;

    template <typename Field>
    using ReturnType =
      Image<Vector<Field>, Dimension<Field>::value>;

    template <typename Field>
    inline
    void operator()(Vector<Field>& gradient,
                    const typename Field::const_array_iterator& it) const
    {
      for (int i = 0; i < Dimension<Field>::value; ++i)
      {
        if (it.position()[i] == 0)
          gradient[i] = (it.delta(i, 1) - *it) / 2; // Replicate the border
        else if (it.position()[i] == it.sizes()[i] - 1)
          gradient[i] = (*it - it.delta(i,-1)) / 2; // Replicate the border
        else
          gradient[i] = (it.delta(i, 1) - it.delta(i,-1)) / 2;
      }
    }

    template <typename Field>
    inline
    void operator()(Vector<Field>& gradient,
                    const Field& scalar_field,
                    const Coords<Field>& position) const
    {
      auto it = scalar_field.begin_array();
      it += position;
      operator()<Field>(gradient, it);
    }

    template <typename Field>
    ReturnType<Field> operator()(const Field& in) const
    {
      auto out = ReturnType<Field>{ in.sizes() };

      auto in_i = in.begin_array();
      auto out_i = out.begin_array();
      for ( ; !in_i.end(); ++in_i, ++out_i)
        operator()<Field>(*out_i, in_i);

      return out;
    }
  };


  //! @brief Laplacian functor class
  struct Laplacian
  {
    template <typename Field>
    struct Dimension { enum { value = Field::Dimension }; };

    template <typename Field>
    using Coords = Matrix<int, Dimension<Field>::value, 1>;

    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using ReturnType = Field;

    template <typename Field>
    inline Scalar<Field>
    operator()(typename Field::const_array_iterator& it) const
    {
      const int N{ Dimension<Field>::value };

      auto value = PixelTraits<Scalar<Field>>::zero();
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

    template <typename Field>
    inline Scalar<Field> operator()(const Field& scalar_field,
                                    const Coords<Field>& position) const
    {
      auto loc =  scalar_field.begin_array();
      loc += position;
      return this->operator()<Field>(loc);
    }

    template <typename Field>
    ReturnType<Field> operator()(const Field& in) const
    {
      auto out = ReturnType<Field>{ in.sizes() };

      auto in_i = in.begin_array();
      auto out_i = out.begin_array();
      for ( ; !in_i.end(); ++in_i, ++out_i)
        *out_i = this->operator()<Field>(in_i);

      return out;
    }
  };


  //! @brief Hessian matrix functor class.
  struct Hessian
  {
    template <typename Field>
    struct Dimension { enum { value = Field::Dimension }; };

    template <typename Field>
    using Coords = Matrix<int, Dimension<Field>::value, 1>;

    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using HessianMatrix = Eigen::Matrix<
      Scalar<Field>, Dimension<Field>::value, Dimension<Field>::value>;

    template <typename Field>
    using ReturnType = Image<HessianMatrix<Field>, Dimension<Field>::value>;

    template <typename Field>
    void operator()(HessianMatrix<Field>& H,
                    typename Field::const_array_iterator& it) const
    {
      const int N{ Dimension<Field>::value };
      using T = Scalar<Field>;

      for (int i = 0; i < N; ++i)
      {
        for (int j = i; j < N; ++j)
        {
          if (i == j)
          {
            auto next = it.position()[i] == it.sizes()[i]-1 ?
              *it : it.delta(i, 1);
            auto prev = it.position()[i] == 0 ?
              *it : it.delta(i,-1);

            H(i,i) = next - T(2)*(*it) + prev;
          }
          else
          {
            auto next_i = it.position()[i] == it.sizes()[i] - 1 ? 0 : 1;
            auto prev_i = it.position()[i] == 0 ? 0 : -1;
            auto next_j = it.position()[j] == it.sizes()[j] - 1 ? 0 : 1;
            auto prev_j = it.position()[j] == 0 ? 0 : -1;

            H(i,j) =
              (  it.delta(i,next_i,j,next_j) - it.delta(i,prev_i,j,next_j)
               - it.delta(i,next_i,j,prev_j) + it.delta(i,prev_i,j,prev_j) )
              / static_cast<T>(4);

            H(j,i) = H(i,j);
          }
        }
      }
    }

    template <typename Field>
    inline void operator()(HessianMatrix<Field>& H,
                           const Field& scalar_field,
                           const Coords<Field>& position) const
    {
      auto loc = scalar_field.begin_array();
      loc += position;
      operator()<Field>(H, loc);
    }

    template <typename Field>
    ReturnType<Field> operator()(const Field& in) const
    {
      auto out = ReturnType<Field>{ in.sizes() };

      auto in_i = in.begin_array();
      auto out_i = out.begin_array();
      for ( ; !out_i.end(); ++in_i, ++out_i)
        operator()<Field>(*out_i, in_i);

      return out;
    };
  };


  /*!
    @brief Gradient computation
    @param[in] f input scalar field.
    @param[in] x position in the image.
    @return 2D gradient vector.
   */
  template <typename T, int N>
  Matrix<T,N,1> gradient(const Image<T, N>& src, const Matrix<int, N, 1>& p)
  {
    auto g = Matrix<T, N, 1>{};
    Gradient{}(g, src, p);
    return g;
  }

  /*!
    @brief Gradient computation
    @param[in] in scalar field
    @return gradient vector field
   */
  template <typename T, int N>
  inline Image<Matrix<T, N, 1>, N> gradient(const Image<T, N>& in)
  {
    return Gradient{}(in);
  }

  /*!
    @brief Laplacian computation
    @param[in] f input scalar field.
    @param[in] x position in the image.
    @return laplacian value
  */
  template <typename T, int N>
  inline T laplacian(const Image<T, N>& f, const Matrix<int, N, 1>& x)
  {
    return Laplacian{}(f, x);
  }

  /*!
    @brief Laplacian computation
    @param[in] in scalar field.
    @return laplacian field.
   */
  template <typename T, int N>
  inline Image<T, N> laplacian(const Image<T, N>& in)
  {
    return Laplacian{}(in);
  }

  /*!
    @brief Compute the Hessian matrix at a specified position.
    @param[in] f scalar field.
    @param[in] x position in the image.
    @return Hessian matrix.
   */
  template <typename T, int N>
  inline Matrix<T,N,N> hessian(const Image<T, N>& f, const Matrix<int, N, 1>& x)
  {
    auto H_f_x = Matrix<T, N, N>{};
    Hessian{}(H_f_x, f, x);
    return H_f_x;
  }

  /*!
    @brief Compute the Hessian matrix field.
    @param[in] in scalar field.
    @return Hessian matrix field
   */
  template <typename T, int N>
  inline Image<Matrix<T,N,N> > hessian(const Image<T, N>& in)
  {
    return Hessian{}(in);
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP */
