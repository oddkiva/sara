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
    using Coords = typename Field::vector_type;

    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using Vector = Matrix<Scalar<Field>, Field::Dimension, 1>;

    template <typename Field>
    using GradientField = Image<Vector<Field>, Field::Dimension>;

    template <typename Field>
    inline auto operator()(const typename Field::const_array_iterator& in,
                           Vector<Field>& out) const -> void
    {
      constexpr auto N = Field::Dimension;

      for (auto i = 0; i < N; ++i)
      {
        if (in.position()[i] == 0)
          out[i] = (in.delta(i, 1) - *in) / 2; // Replicate the border
        else if (in.position()[i] == in.sizes()[i] - 1)
          out[i] = (*in - in.delta(i,-1)) / 2; // Replicate the border
        else
          out[i] = (in.delta(i, 1) - in.delta(i,-1)) / 2;
      }
    }

    template <typename Field>
    inline auto operator()(const Field& scalar_field,
                           const Coords<Field>& position) const -> Vector<Field>
    {
      auto out = Vector<Field>{};

      auto in = scalar_field.begin_array();
      in += position;
      operator()<Field>(in, out);

      return out;
    }

    template <typename Field>
    auto operator()(const Field& in) const -> GradientField<Field>
    {
      auto out = GradientField<Field>{ in.sizes() };

      auto in_i = in.begin_array();
      auto out_i = out.begin();
      for (; !in_i.end(); ++in_i, ++out_i)
        operator()<Field>(in_i, *out_i);

      return out;
    }
  };


  //! @brief Laplacian functor class
  struct Laplacian
  {
    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using ScalarField = Image<Scalar<Field>, Field::Dimension>;

    template <typename Field>
    inline auto operator()(typename Field::const_array_iterator& in,
                           Scalar<Field>& out) const -> void
    {
      constexpr auto N = Field::Dimension;

      auto value = PixelTraits<Scalar<Field>>::zero();
      for (auto i = 0; i < N; ++i)
      {
        if (in.position()[i] == 0)
          value += in.delta(i, 1) + *in;  // Replicate the border
        else if (in.position()[i] == in.sizes()[i] - 1)
          value += *in + in.delta(i, -1);  // Replicate the border
        else
          value += in.delta(i, 1) + in.delta(i, -1);
      }

      out = value - 2 * N * (*in);
    }

    template <typename Field>
    inline auto operator()(const Field& scalar_field,
                           const typename Field::vector_type& position) const
        -> Scalar<Field>
    {
      auto out = Scalar<Field>{};

      auto in =  scalar_field.begin_array();
      in += position;
      operator()<Field>(in, out);

      return out;
    }

    template <typename InField, typename OutField>
    auto operator()(const InField& in, OutField& out) const -> void
    {
      if (in.sizes() != out.sizes())
        throw std::domain_error{
          "Source and destination image sizes are not equal!"
        };

      auto in_i = in.begin_array();
      auto out_i = out.begin();
      for ( ; !in_i.end(); ++in_i, ++out_i)
        operator()<InField>(in_i, *out_i);
    }

    template <typename Field>
    auto operator()(const Field& in) const -> ScalarField<Field>
    {
      auto out = ScalarField<Field>{ in.sizes() };
      operator()(in, out);
      return out;
    }
  };


  //! @brief Hessian matrix functor class.
  struct Hessian
  {
    template <typename Field>
    using Coords = typename Field::vector_type;

    template <typename Field>
    using Scalar = typename Field::value_type;

    template <typename Field>
    using HessianMatrix =
        Eigen::Matrix<Scalar<Field>, Field::Dimension, Field::Dimension>;

    template <typename Field>
    using HessianField = Image<HessianMatrix<Field>, Field::Dimension>;

    template <typename Field>
    auto operator()(typename Field::const_array_iterator& in,
                    HessianMatrix<Field>& out) const -> void
    {
      constexpr int N = Field::Dimension;
      using T = Scalar<Field>;

      for (int i = 0; i < N; ++i)
      {
        for (int j = i; j < N; ++j)
        {
          if (i == j)
          {
            auto next =
                in.position()[i] == in.sizes()[i] - 1 ? *in : in.delta(i, 1);
            auto prev = in.position()[i] == 0 ? *in : in.delta(i, -1);

            out(i, i) = next - T(2) * (*in) + prev;
          }
          else
          {
            auto next_i = in.position()[i] == in.sizes()[i] - 1 ? 0 : 1;
            auto prev_i = in.position()[i] == 0 ? 0 : -1;
            auto next_j = in.position()[j] == in.sizes()[j] - 1 ? 0 : 1;
            auto prev_j = in.position()[j] == 0 ? 0 : -1;

            out(i, j) = (in.delta(i, next_i, j, next_j) -
                       in.delta(i, prev_i, j, next_j) -
                       in.delta(i, next_i, j, prev_j) +
                       in.delta(i, prev_i, j, prev_j)) /
                      static_cast<T>(4);

            out(j, i) = out(i, j);
          }
        }
      }
    }

    template <typename Field>
    inline auto operator()(const Field& scalar_field,
                           const Coords<Field>& position) const
        -> HessianMatrix<Field>
    {
      auto out = HessianMatrix<Field>{};

      auto in = scalar_field.begin_array();
      in += position;
      operator()<Field>(in, out);

      return out;
    }

    template <typename Field>
    auto operator()(const Field& in) const -> HessianField<Field>
    {
      auto out = HessianField<Field>{ in.sizes() };

      auto in_i = in.begin_array();
      auto out_i = out.begin();
      for (; !in_i.end(); ++in_i, ++out_i)
        operator()<Field>(in_i, *out_i);

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
    return Gradient{}(in);
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
    return Laplacian{}(in);
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
    return Hessian{}(in);
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DIFFERENTIAL_HPP */
