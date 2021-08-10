// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>


namespace DO::Sara {

  //! @brief Central difference.
  struct CentralDifference
  {
    template <typename ArrayView>
    static inline auto at(const ArrayView& u,                        //
                          const typename ArrayView::vector_type& p,  //
                          int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const auto ei = vector_type::Unit(i);
      return (u(p + ei) - u(p - ei)) / 2;
    }

    template <typename ArrayView>
    static inline auto forward(const ArrayView& u,
                               const typename ArrayView::vector_type& p, int i)
    {
      return at(u, p, i);
    }

    template <typename ArrayView>
    static inline auto backward(const ArrayView& u,
                                const typename ArrayView::vector_type& p, int i)
    {
      return at(u, p, i);
    }
  };


  //! @brief Upwind difference.
  struct UpwindDifference
  {
  public:
    template <typename ArrayView>
    static inline auto forward(const ArrayView& u,
                               const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const auto ei = vector_type::unit(i);
      return u(p + ei) - u(p);
    }

    template <typename ArrayView>
    static inline auto backward(const ArrayView& u,
                                const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const auto ei = vector_type::unit(i);
      return u(p) - u(p - ei);
    }
  };


  //! @brief WENO3.
  struct Weno3
  {
    static constexpr auto eps = 1e-12;

    template <typename T>
    static inline T combine(const T v1, const T v2, const T v3)
    {
      auto s = (T(eps) + square(v2 - v1)) / (T(eps) + square(v3 - v2));
      s = 1 / (1 + 2 * s * s);
      return (v2 + v3 - s * (v1 - 2 * v2 + v3)) / 2;
    }

    template <typename ArrayView>
    static inline auto forward(const ArrayView& u,
                               const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const vector_type ei = vector_type::Unit(i);

      const auto us = std::array{u(p - ei), u(p), u(p + ei), u(p + 2 * ei)};

      auto dus = decltype(us){};
      std::adjacent_difference(std::begin(us), std::end(us), std::begin(dus));

      return combine(dus[3], dus[2], dus[1]);
    }

    template <typename ArrayView>
    static inline auto backward(const ArrayView& u,
                                const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const vector_type ei = vector_type::Unit(i);

      const auto us = std::array{u(p - 2 * ei), u(p - ei), u(p), u(p + ei)};

      auto dus = decltype(us){};
      std::adjacent_difference(std::begin(us), std::end(us), std::begin(dus));

      return combine(dus[1], dus[2], dus[3]);
    }
  };


  //! @brief WENO5.
  struct Weno5
  {
    static constexpr auto eps = 1e-12;

    template <typename T>
    static inline T combine(const T v1, const T v2, const T v3, const T v4,
                            const T v5)
    {
      auto s1 = 13 * square(v1 - 2 * v2 + v3) / 12 +  //
                square(v1 - 4 * v2 + 3 * v3) / 4;
      auto s2 = 13 * square(v2 - 2 * v3 + v4) / 12 +  //
                square(v2 - v4) / 4;
      auto s3 = 13 * square(v3 - 2 * v4 + v5) / 12 +  //
                square(3 * v3 - 4 * v4 + v5) / 4;

      s1 = 1 / square(T(eps) + s1);
      s2 = 6 / square(T(eps) + s2);
      s3 = 3 / square(T(eps) + s3);

      return (s1 * (2 * v1 - 7 * v2 + 11 * v3) +  //
              s2 * (-v2 + 5 * v3 + 2 * v4) +      //
              s3 * (2 * v3 + 5 * v4 - v5)) /
             6 / (s1 + s2 + s3);
    }

    template <typename ArrayView>
    static inline auto forward(const ArrayView& u,
                               const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const vector_type ei = vector_type::unit(i);

      const auto us = std::array{u(p - 2 * ei), u(p - ei),     u(p),
                                 u(p + ei),     u(p + 2 * ei), u(p + 3 * ei)};
      auto dus = decltype(us){};
      std::adjacent_difference(std::begin(us), std::end(us), std::begin(dus));

      return combine(dus[5], dus[4], dus[3], dus[2], dus[1]);
    }

    template <typename ArrayView>
    static inline auto backward(const ArrayView& u,
                                const typename ArrayView::vector_type& p, int i)
    {
      using vector_type = typename ArrayView::vector_type;
      const vector_type ei = vector_type::unit(i);

      const auto us = std::array{u(p - 3 * ei), u(p - 2 * ei), u(p - ei),
                                 u(p),          u(p + ei),     u(p + 2 * ei)};

      auto dus = decltype(us){};
      std::adjacent_difference(std::begin(us), std::end(us), std::begin(dus));

      return combine(dus[1], dus[2], dus[3], dus[4], dus[5]);
    }
  };


  template <typename FiniteDifference>
  struct Derivative
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
    inline auto forward(const Field& u, const Coords<Field>& x) const
        -> Vector<Field>
    {
      constexpr auto N = Field::Dimension;

      auto du = Vector<Field>{};
      for (auto i = 0; i < N; ++i)
        du(i) = FiniteDifference::forward(u, x, i);

      return du;
    }

    template <typename Field>
    inline auto backward(const Field& u, const Coords<Field>& x) const
        -> Vector<Field>
    {
      constexpr auto N = Field::Dimension;

      auto du = Vector<Field>{};
      for (auto i = 0; i < N; ++i)
        du(i) = FiniteDifference::backward(u, x, i);

      return du;
    }

    template <typename Field>
    auto forward(const Field& in, GradientField<Field>& out) const
    {
      if (out.sizes() != in.sizes())
        throw std::runtime_error{"Invalid shape!"};

      auto in_i = in.begin_array();
      auto out_i = out.begin();
      for (; !in_i.end(); ++in_i, ++out_i)
        *out_i = forward<Field>(in, in_i.position());
    }
  };

}  // namespace DO::Sara
