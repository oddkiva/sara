// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>


namespace DO::Shakti {

  //! @brief Matrix class.
  template <typename T, int M, int N>
  class Matrix
  {
  public:
    __host__ __device__ inline Matrix()
    {
    }

    __host__ __device__ inline Matrix(const Matrix& other)
    {
      copy(other);
    }

    __host__ __device__ inline explicit Matrix(const T& x)
    {
      static_assert(M == 1 && N == 1, "Matrix must 1x1!");
      _data[0] = x;
    }

    __host__ __device__ inline Matrix(const T& x, const T& y)
    {
      _data[0] = x;
      _data[1] = y;
    }

    __host__ __device__ inline Matrix(const T& x, const T& y, const T& z)
    {
      _data[0] = x;
      _data[1] = y;
      _data[2] = z;
    }

    __host__ __device__ inline Matrix(const T& x, const T& y, const T& z,
                                      const T& w)
    {
      _data[0] = x;
      _data[1] = y;
      _data[2] = z;
      _data[3] = w;
    }

    __host__ __device__ inline Matrix(const T* data)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] = data[i];
    }

    __host__ __device__ inline static Matrix Identity()
    {
      static_assert(M == N, "The matrix must be square!");

      auto eye = Matrix{};

#pragma unroll
      for (int i = 0; i < M; ++i)
      {
#pragma unroll
        for (int j = 0; j < N; ++j)
          eye(i, j) = static_cast<T>(i == j);
      }

      return eye;
    }

    __host__ __device__ inline static Matrix Zero()
    {
      Matrix zero;
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        zero._data[i] = 0;
      return zero;
    }

    __host__ __device__ inline static Matrix Ones()
    {
      Matrix ones;
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        ones._data[i] = T(1);
      return ones;
    }

    __host__ __device__ inline Matrix& operator=(const Matrix& other)
    {
      copy(other);
      return *this;
    }

    __host__ __device__ inline bool operator==(const Matrix& other) const
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        if (_data[i] != other._data[i])
          return false;
      return true;
    }

    __host__ __device__ inline bool operator!=(const Matrix& other) const
    {
      return !(*this == other);
    }

    __host__ __device__ inline const T& operator[](int i) const
    {
      return _data[i];
    }

    __host__ __device__ inline const T* data() const
    {
      return _data;
    }

    __host__ __device__ inline operator const T&() const
    {
      static_assert(M == 1 && N == 1, "Matrix must be a scalar");
      return _data[0];
    }

    __host__ __device__ inline const T& x() const
    {
      static_assert(M == 1 || N == 1, "The matrix is not a vector!");
      return _data[0];
    }

    __host__ __device__ inline const T& y() const
    {
      static_assert((M == 1 || N == 1) && M * N >= 2,
                    "The matrix must a Vector of dimension >= 2!");
      return _data[1];
    }

    __host__ __device__ inline const T& z() const
    {
      static_assert((M == 1 || N == 1) && M * N >= 3,
                    "The matrix must a Vector of dimension >= 3!");
      return _data[2];
    }

    __host__ __device__ inline const T& w() const
    {
      static_assert((M == 1 || N == 1) && M * N >= 4,
                    "The matrix must a Vector of dimension >= 4!");
      return _data[3];
    }

    __host__ __device__ inline const T& operator()(int i) const
    {
      return _data[i];
    }

    __host__ __device__ inline const T& operator()(int i, int j) const
    {
      return _data[i * N + j];
    }

    __host__ __device__ inline T& operator[](int i)
    {
      return _data[i];
    }

    __host__ __device__ inline T* data()
    {
      return _data;
    }

    __host__ __device__ inline operator T&()
    {
      static_assert(M == 1 && N == 1, "Matrix must be a scalar");
      return _data[0];
    }

    __host__ __device__ inline T& x()
    {
      static_assert(M == 1 || N == 1, "The matrix is not a vector!");
      return _data[0];
    }

    __host__ __device__ inline T& y()
    {
      static_assert((M == 1 || N == 1) && M * N >= 2,
                    "The matrix must a Vector of dimension >= 2!");
      return _data[1];
    }

    __host__ __device__ inline T& z()
    {
      static_assert((M == 1 || N == 1) && M * N >= 3,
                    "The matrix must a Vector of dimension >= 3!");
      return _data[2];
    }

    __host__ __device__ inline T& w()
    {
      static_assert((M == 1 || N == 1) && M * N >= 4,
                    "The matrix must a Vector of dimension >= 4!");
      return _data[3];
    }

    __host__ __device__ inline T& operator()(int i)
    {
      return _data[i];
    }

    __host__ __device__ inline T& operator()(int i, int j)
    {
      return _data[i * N + j];
    }

    __host__ __device__ inline int rows() const
    {
      return M;
    }

    __host__ __device__ inline int cols() const
    {
      return N;
    }

    __host__ __device__ inline Matrix& operator+=(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] += other._data[i];
      return *this;
    }

    __host__ __device__ inline Matrix& operator-=(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] -= other._data[i];
      return *this;
    }

    __host__ __device__ inline Matrix& operator*=(const Matrix& other)
    {
      *this = (*this) * other;
      return *this;
    }

    __host__ __device__ inline Matrix& operator*=(const T& other)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] *= other;
      return *this;
    }

    __host__ __device__ inline Matrix& operator/=(const T& other)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] /= other;
      return *this;
    }

    __host__ __device__ inline Matrix operator+(const Matrix& other) const
    {
      Matrix res{*this};
      res += other;
      return res;
    }

    __host__ __device__ inline Matrix operator-() const
    {
      auto res = Matrix{};
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        res._data[i] = -_data[i];
      return res;
    }

    __host__ __device__ inline Matrix operator-(const Matrix& other) const
    {
      Matrix res{*this};
      res -= other;
      return res;
    }

    template <int O>
    __host__ __device__ inline Matrix<T, M, O>
    operator*(const Matrix<T, N, O>& other) const
    {
      Matrix<T, M, O> res;
#pragma unroll
      for (int i = 0; i < M; ++i)
      {
#pragma unroll
        for (int j = 0; j < O; ++j)
        {
          auto val = T{};
#pragma unroll
          for (int k = 0; k < N; ++k)
            val += (*this)(i, k) * other(k, j);
          res(i, j) = val;
        }
      }
      return res;
    }

    __host__ __device__ inline Matrix operator*(const Matrix& other) const
    {
      static_assert(M == N, "Matrices must be square!");

      Matrix res;
#pragma unroll
      for (int i = 0; i < M; ++i)
      {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
          auto val = T{};
#pragma unroll
          for (int k = 0; k < N; ++k)
            val += (*this)(i, k) * other(k, j);
          res(i, j) = val;
        }
      }
      return res;
    }

    __host__ __device__ inline Matrix operator*(const T& other) const
    {
      Matrix res;
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        res._data[i] = _data[i] * other;
      return res;
    }

    __host__ __device__ friend inline Matrix operator*(const T& a,
                                                       const Matrix& b)
    {
      return b * a;
    }

    __host__ __device__ inline Matrix operator/(const T& other) const
    {
      Matrix res;
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        res._data[i] = _data[i] / other;
      return res;
    }

    __host__ __device__ inline T dot(const Matrix& other) const
    {
      T res{0};
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        res += _data[i] * other._data[i];
      return res;
    }

    __host__ __device__ inline T squared_norm() const
    {
      T res{0};
#pragma unroll
      for (int i = 0; i < M * N; ++i)
      {
        const T a_i{_data[i]};
        res += a_i * a_i;
      }
      return res;
    }

    __host__ __device__ inline T norm() const
    {
      return sqrt(squared_norm());
    }

    __host__ __device__ inline Matrix& normalize()
    {
      *this /= norm();
      return *this;
    }

    __host__ __device__ inline auto row(int i) const -> Matrix<T, 1, N>
    {
      auto r = Matrix<T, 1, N>{};
#pragma unroll
      for (auto j = 0; j < N; ++j)
        r(j) = (*this)(i, j);
      return r;
    }

    __host__ __device__ inline auto col(int j) const -> Matrix<T, M, 1>
    {
      auto c = Matrix<T, M, 1>{};
#pragma unroll
      for (auto i = 0; i < M; ++i)
        c(i) = (*this)(i, j);
      return c;
    }

  protected:
    __host__ __device__ void copy(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M * N; ++i)
        _data[i] = other._data[i];
    }

  protected:
    T _data[M * N];
  };

  //! @{
  //! \brief Convenient matrix aliases.
  using Matrix2i = Matrix<int, 2, 2>;
  using Matrix3i = Matrix<int, 3, 3>;
  using Matrix4i = Matrix<int, 4, 4>;

  using Matrix2f = Matrix<float, 2, 2>;
  using Matrix3f = Matrix<float, 3, 3>;
  using Matrix4f = Matrix<float, 4, 4>;

  using Matrix2d = Matrix<double, 2, 2>;
  using Matrix3d = Matrix<double, 3, 3>;
  using Matrix4d = Matrix<double, 4, 4>;
  //! @}

  //! @{
  //! \brief Vector class.
  template <typename T, int N>
  using Vector = Matrix<T, N, 1>;

  using Vector1i = Vector<int, 1>;
  using Vector1f = Vector<float, 1>;
  using Vector1d = Vector<double, 1>;

  using Vector2i = Vector<int, 2>;
  using Vector2f = Vector<float, 2>;
  using Vector2d = Vector<double, 2>;

  using Vector3i = Vector<int, 3>;
  using Vector3f = Vector<float, 3>;
  using Vector3d = Vector<double, 3>;

  using Vector4ub = Vector<unsigned char, 4>;
  using Vector4i = Vector<int, 4>;
  using Vector4f = Vector<float, 4>;
  using Vector4d = Vector<double, 4>;
  //! @}

  //! @{
  //! @brief Basic linear algebra.
  template <typename T>
  __host__ __device__ inline auto cross(const Vector<T, 3>& a,
                                        const Vector<T, 3>& b) -> Vector<T, 3>
  {
    auto c = Vector<T, 3>{};
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
  }

  template <typename T>
  __host__ __device__ inline auto det(const Matrix<T, 2, 2>& m) -> T
  {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  }

  template <typename T>
  __host__ __device__ inline auto det(const Matrix<T, 3, 3>& m) -> T
  {
    const auto det0 = m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2);
    const auto det1 = m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2);
    const auto det2 = m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1);

    return m(0, 0) * det0 - m(0, 1) * det1 + m(0, 2) * det2;
  }

  template <typename T>
  __host__ __device__ inline auto trace(const Matrix<T, 2, 2>& m) -> T
  {
    return m(0, 0) + m(1, 1);
  }

  template <typename T>
  __host__ __device__ inline auto trace(const Matrix<T, 3, 3>& m) -> T
  {
    return m(0, 0) + m(1, 1) + m(2, 2);
  }

  template <typename T>
  __host__ __device__ inline auto inverse(const Matrix<T, 2, 2>& m)
      -> Matrix<T, 2, 2>
  {
    const auto det_m = det(m);
    const auto inv_m = Matrix<T, 2, 2>{};
    inv_m(0, 0) = m(1, 1);
    inv_m(0, 1) = -m(0, 1);
    inv_m(1, 0) = -m(1, 0);
    inv_m(1, 1) = m(0, 0);
    return inv_m / det_m;
  }

  template <typename T>
  __host__ __device__ inline auto inverse(const Matrix<T, 3, 3>& m)
      -> Matrix<T, 3, 3>
  {
    auto inv_m = Matrix<T, 3, 3>{};

// #pragma unroll
//     for (auto i = 0; i < 3; ++i)
//     {
//       const auto r = cross(m.col((i + 1) % 3), m.col((i + 2) % 3));
// #pragma unroll
//       for (auto j = 0; j < 3; ++j)
//         inv_m(i, j) = r(j);
//     }

    auto r = cross(m.col(1), m.col(2));
#pragma unroll
    for (auto j = 0; j < 3; ++j)
      inv_m(0, j) = r(j);

    r = cross(m.col(2), m.col(0));
#pragma unroll
    for (auto j = 0; j < 3; ++j)
      inv_m(1, j) = r(j);

    r = cross(m.col(0), m.col(1));
#pragma unroll
    for (auto j = 0; j < 3; ++j)
      inv_m(2, j) = r(j);

    inv_m /= det(m);

    return inv_m;
  }
  //! @}


  //! @
  //! \brief Output stream operator.
  template <typename T, int M, int N>
  std::ostream& operator<<(std::ostream& os, const Matrix<T, M, N>& m)
  {
    for (int i = 0; i < M; ++i)
    {
      for (int j = 0; j < N; ++j)
        os << m(i, j) << " ";
      os << std::endl;
    }
    return os;
  }

  template <typename T, int N>
  std::ostream& operator<<(std::ostream& os, const Vector<T, N>& v)
  {
    os << "[ ";
    for (int i = 0; i < N; ++i)
    {
      os << v(i);
      if (i < N - 1)
        os << ", ";
    }
    os << " ]";
    return os;
  }
  //! @}

}  // namespace DO::Shakti
