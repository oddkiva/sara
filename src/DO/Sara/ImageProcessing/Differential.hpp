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
    using OutPixel = Vector<Field>;

    template <typename Field>
    inline Vector<Field>
    operator()(const typename Field::const_array_iterator& it) const
    {
      auto gradient = Vector<Field>{};
      for (int i = 0; i < Dimension<Field>::value; ++i)
      {
        if (it.position()[i] == 0)
          gradient[i] = (it.delta(i, 1) - *it) / 2; // Replicate the border
        else if (it.position()[i] == it.sizes()[i] - 1)
          gradient[i] = (*it - it.delta(i,-1)) / 2; // Replicate the border
        else
          gradient[i] = (it.delta(i, 1) - it.delta(i,-1)) / 2;
      }
      return gradient;
    }

    template <typename Field>
    inline Vector<Field> operator()(const Field& scalar_field,
                                    const Coords<Field>& position) const
    {
      auto it = scalar_field.begin_array();
      it += position;
      return operator()<Field>(it);
    }

    template <typename SrcField, typename DstField>
    void operator()(const SrcField& src, DstField& dst) const
    {
      if (src.sizes() != dst.sizes())
        throw std::domain_error{
          "Source and destination image sizes are not equal!"
        };

      auto in_i = src.begin_array();
      auto out_i = dst.begin();
      for ( ; !in_i.end(); ++in_i, ++out_i)
        *out_i = operator()<SrcField>(in_i);
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
    using OutPixel = typename Field::value_type;

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

    template <typename SrcField, typename DstField>
    void operator()(const SrcField& src, DstField& dst) const
    {
      if (src.sizes() != dst.sizes())
        throw std::domain_error{
          "Source and destination image sizes are not equal!"
        };

      auto in_i = src.begin_array();
      auto out_i = dst.begin();
      for ( ; !in_i.end(); ++in_i, ++out_i)
        *out_i = this->operator()<SrcField>(in_i);
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
    using OutPixel = HessianMatrix<Field>;

    template <typename Field>
    HessianMatrix<Field>
    operator()(typename Field::const_array_iterator& it) const
    {
      const int N{ Dimension<Field>::value };
      using T = Scalar<Field>;

      auto H = HessianMatrix<Field>{};

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

            H(i, j) = (it.delta(i, next_i, j, next_j) -
                       it.delta(i, prev_i, j, next_j) -
                       it.delta(i, next_i, j, prev_j) +
                       it.delta(i, prev_i, j, prev_j)) /
                      static_cast<T>(4);

            H(j,i) = H(i,j);
          }
        }
      }

      return H;
    }

    template <typename Field>
    inline HessianMatrix<Field> operator()(const Field& scalar_field,
                                           const Coords<Field>& position) const
    {
      auto loc = scalar_field.begin_array();
      loc += position;
      return operator()<Field>(loc);
    }

    template <typename SrcField, typename DstField>
    void operator()(const SrcField& src, DstField& dst) const
    {
      if (src.sizes() != dst.sizes())
        throw std::domain_error{
          "Source and destination image sizes are not equal!"
        };

      auto src_i = src.begin_array();
      auto dst_i = dst.begin();
      for ( ; !src_i.end(); ++src_i, ++dst_i)
        *dst_i = operator()<SrcField>(src_i);
    };
  };


  /*!
    @brief Gradient computation
    @param[in] f input scalar field.
    @param[in] x position in the image.
    @return 2D gradient vector.
   */
  template <typename T, int N>
  Matrix<T,N,1> gradient(const ImageView<T, N>& f, const Matrix<int, N, 1>& x)
  {
    return Gradient{}(f, x);
  }

  /*!
    @brief Gradient computation
    @param[in] in scalar field
    @return gradient vector field
   */
  template <typename T, int N>
  inline Image<Matrix<T, N, 1>, N> gradient(const ImageView<T, N>& in)
  {
    auto out = Image<Matrix<T, N, 1>, N>{ in.sizes() };
    Gradient{}(in, out);
    return out;
  }

  /*!
    @brief Laplacian computation
    @param[in] f input scalar field.
    @param[in] x position in the image.
    @return laplacian value
  */
  template <typename T, int N>
  inline T laplacian(const ImageView<T, N>& f, const Matrix<int, N, 1>& x)
  {
    return Laplacian{}(f, x);
  }

  /*!
    @brief Laplacian computation
    @param[in] in scalar field.
    @return laplacian field.
   */
  template <typename T, int N>
  inline Image<T, N> laplacian(const ImageView<T, N>& in)
  {
    auto out = Image<T, N>{ in.sizes() };
    Laplacian{}(in, out);
    return out;
  }

  /*!
    @brief Compute the Hessian matrix at a specified position.
    @param[in] f scalar field.
    @param[in] x position in the image.
    @return Hessian matrix.
   */
  template <typename T, int N>
  inline Matrix<T, N, N> hessian(const ImageView<T, N>& f,
                                 const Matrix<int, N, 1>& x)
  {
    return Hessian{}(f, x);
  }

  /*!
    @brief Compute the Hessian matrix field.
    @param[in] in scalar field.
    @return Hessian matrix field
   */
  template <typename T, int N>
  inline Image<Matrix<T, N, N>> hessian(const ImageView<T, N>& in)
  {
    auto out = Image<Matrix<T, N, N>>{ in.sizes() };
    Hessian{}(in, out);
    return out;
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP */
