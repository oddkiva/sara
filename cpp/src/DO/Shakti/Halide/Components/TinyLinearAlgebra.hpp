// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Halide/MyHalide.hpp>

#include <array>


namespace DO { namespace Shakti { namespace HalideBackend {

  template <int M, int N>
  struct Matrix;

  template <int N>
  using Vector = Matrix<N, 1>;

  template <int N>
  using RowVector = Matrix<1, N>;

  using Matrix2 = Matrix<2, 2>;
  using Matrix3 = Matrix<3, 3>;
  using Matrix4 = Matrix<4, 4>;

  using Vector2 = Vector<2>;
  using Vector3 = Vector<3>;
  using Vector4 = Vector<4>;

  using RowVector2 = RowVector<2>;
  using RowVector3 = RowVector<3>;
  using RowVector4 = RowVector<4>;


  template <int M, int N>
  struct Matrix
  {
    inline Matrix() = default;

    inline Matrix(const Halide::Tuple& t)
    {
      if (t.size() != M*N)
        throw std::runtime_error{"Invalid tuple size!"};
      std::copy(t.as_vector().begin(), t.as_vector().end(), data.begin());
    }

    inline auto operator()(int i) -> Halide::Expr&
    {
      static_assert(M == 1 || N == 1, "");
      return data[i];
    }

    inline auto operator()(int i) const -> const Halide::Expr&
    {
      static_assert(M == 1 || N == 1, "");
      return data[i];
    }

    inline auto operator()(int i, int j) -> Halide::Expr&
    {
      return data[i * N + j];
    }

    inline auto operator()(int i, int j) const -> const Halide::Expr&
    {
      return data[i * N + j];
    }

    inline auto operator+(const Matrix& other) const -> Matrix
    {
      auto res = Matrix{};
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
          res(i, j) = (*this)(i, j) + other(i, j);
      return res;
    }

    template <int O>
    inline auto operator*(const Matrix<N, O>& other) const -> Matrix<M, O>
    {
      auto res = Matrix<M, O>{};
      for (int i = 0; i < M; ++i)
      {
        for (int j = 0; j < O; ++j)
        {
          res(i, j) = Halide::cast(t, 0);
          for (int k = 0; k < N; ++k)
            res(i, j) += (*this)(i, k) * other(k, j);
        }
      }
      return res;
    }

    inline auto operator*=(const Halide::Expr& scalar) -> Matrix&
    {
      for (int i = 0; i < M * N; ++i)
        data[i] *= scalar;

      return *this;
    }

    inline auto operator/(const Halide::Expr& e) const -> Matrix
    {
      auto out = Matrix{};
      for (int i = 0; i < M * N; ++i)
        out.data[i] = data[i] / e;
      return out;
    }

    inline auto operator/=(const Halide::Expr& scalar) -> Matrix&
    {
      for (int i = 0; i < M * N; ++i)
        data[i] /= scalar;

      return *this;
    }

    inline auto operator-() const -> Matrix
    {
      auto out = Matrix{};
      for (int i = 0; i < M * N; ++i)
        out.data[i] = -data[i];
      return out;
    }

    inline auto row(int i) const -> RowVector<N>
    {
      auto r = Vector<N>{};
      for (auto j = 0; j < N; ++j)
        r(j) = (*this)(i, j);
      return r;
    }

    inline auto col(int j) const -> Vector<M>
    {
      auto c = Vector<M>{};
      for (auto i = 0; i < M; ++i)
        c(i) = (*this)(i, j);
      return c;
    }

    inline auto set_zero() {
      std::fill(data.begin(), data.end(), Halide::cast(t, 0));
    }

    inline auto set_row(int i, const RowVector<N>& v)
    {
       for (auto j = 0; j < N; ++j)
         (*this)(i, j) = v(j);
    }

    inline auto set_col(int j, const Vector<M>& v)
    {
       for (auto i = 0; i < M; ++i)
         (*this)(i, j) = v(i);
    }

    inline auto transpose() const -> Matrix<N, M>
    {
      auto r = Matrix<N, M>{};
      for (auto i = 0; i < M; ++i)
        for (auto j = 0; j < N; ++j)
          r(j, i) = (*this)(i, j);
      return r;
    }

    inline operator Halide::Tuple() const
    {
      return Halide::Tuple{{data.begin(), data.end()}};
    }

    Halide::Type t{Halide::Float(32)};
    std::array<Halide::Expr, M * N> data;
  };

  template <int M, int N>
  auto operator*(const Halide::Expr& a, const Matrix<M, N>& b)
  {
    auto r = b;
    r *= a;
    return r;
  }

  template <int M, int N>
  auto dot(const Matrix<M, N>& a, const Matrix<M, N>& b)
  {
    auto r = Halide::Expr{};
    r = Halide::cast(a.t, 0);
    for (auto i = 0; i < M; ++i)
      for (auto j = 0; j < N; ++j)
        r += a(i, j) * b(i, j);
    return r;
  }

  template <int M, int N>
  auto squared_norm(const Matrix<M, N>& a)
  {
    return dot(a, a);
  }

  template <int M, int N>
  auto norm(const Matrix<M, N>& a)
  {
    return Halide::sqrt(squared_norm(a));
  }


  auto cross(const Vector<3>& a, const Vector<3>& b) -> Vector<3>
  {
    auto c = Vector<3>{};
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
  }

  auto det(const Matrix<2, 2>& m) -> Halide::Expr
  {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  }

  auto det(const Matrix<3, 3>& m) -> Halide::Expr
  {
    auto det0 = m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2);
    auto det1 = m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2);
    auto det2 = m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1);

    return m(0, 0) * det0 - m(0, 1) * det1 + m(0, 2) * det2;
  }

  auto trace(const Matrix<2, 2>& m) -> Halide::Expr
  {
    return m(0, 0) + m(1, 1);
  }

  auto trace(const Matrix<3, 3>& m) -> Halide::Expr
  {
    return m(0, 0) + m(1, 1) + m(2, 2);
  }

  auto inverse(const Matrix<2, 2>& m) -> Matrix<2, 2>
  {
    auto det_m = det(m);
    auto inv_m = Matrix<2, 2>{};
    inv_m(0, 0) =  m(1, 1); inv_m(0, 1) = -m(0, 1);
    inv_m(1, 0) = -m(1, 0); inv_m(1, 1) =  m(0, 0);
    return inv_m / det_m;
  }

  auto inverse(const Matrix<3, 3>& m) -> Matrix<3, 3>
  {
    auto inv_m = Matrix<3, 3>{};
    for (auto i = 0; i < 3; ++i)
    {
       const auto r = cross(m.col((i + 1) % 3), m.col((i + 2) % 3));
       for (auto j = 0; j < 3; ++j)
         inv_m(i, j) = r(j);
    }
    return inv_m / det(m);
  }

}}}  // namespace DO::Shakti::HalideBackend
