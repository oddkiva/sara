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

#ifndef DO_SHAKTI_MULTIARRAY_MATRIX_HPP
#define DO_SHAKTI_MULTIARRAY_MATRIX_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <DO/Shakti/Utilities/ErrorCheck.hpp>


namespace DO { namespace Shakti {

  //! \brief Matrix class.
  template <typename T, int M, int N>
  class Matrix
  {
  public:
    __host__ __device__
    inline Matrix() = default;

    __host__ __device__
    inline Matrix(const Matrix& other)
    {
      copy(other);
    }

    __host__ __device__
    inline explicit Matrix(const T& x)
    {
      static_assert(M == 1 && N == 1, "Matrix must 1x1!");
      _data[0] = x;
    }

    __host__ __device__
    inline Matrix(const T& x, const T& y)
    {
      _data[0] = x;
      _data[1] = y;
    }

    __host__ __device__
    inline Matrix(const T& x, const T& y, const T& z)
    {
      _data[0] = x;
      _data[1] = y;
      _data[2] = z;
    }

    __host__ __device__
    inline Matrix(const T& x, const T& y, const T& z, const T& w)
    {
      _data[0] = x;
      _data[1] = y;
      _data[2] = z;
      _data[3] = w;
    }

    __host__ __device__
    inline Matrix(const T *data)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] = data[i];
    }

    __host__ __device__
    inline static Matrix Zero()
    {
      Matrix zero;
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        zero._data[i] = 0;
      return zero;
    }

    __host__ __device__
    inline static Matrix Ones()
    {
      Matrix ones;
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        ones._data[i] = T(1);
      return ones;
    }

    __host__ __device__
    inline Matrix& operator=(const Matrix& other)
    {
      copy(other);
      return *this;
    }

    __host__ __device__
    inline bool operator==(const Matrix& other) const
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        if (_data[i] != other._data[i])
          return false;
      return true;
    }

    __host__ __device__
    inline bool operator!=(const Matrix& other) const
    {
      return !(*this == other);
    }

    __host__ __device__
    inline const T& operator[](int i) const
    {
      return _data[i];
    }

    __host__ __device__
    inline const T * data() const
    {
      return _data;
    }

    __host__ __device__
    inline operator const T&() const
    {
      static_assert(M == 1 && N == 1, "Matrix must be a scalar");
      return _data[0];
    }

    __host__ __device__
    inline const T& x() const
    {
      static_assert(
        M == 1 || N == 1,
        "The matrix is not a vector!");
      return _data[0];
    }

    __host__ __device__
    inline const T& y() const
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 2,
        "The matrix must a Vector of dimension >= 2!");
      return _data[1];
    }

    __host__ __device__
    inline const T& z() const
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 3,
        "The matrix must a Vector of dimension >= 3!");
      return _data[2];
    }

    __host__ __device__
    inline const T& w() const
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 4,
        "The matrix must a Vector of dimension >= 4!");
      return _data[3];
    }

    __host__ __device__
    inline const T& operator()(int i) const
    {
      return _data[i];
    }

    __host__ __device__
    inline const T& operator()(int i, int j) const
    {
      return _data[i*N+j];
    }

    __host__ __device__
    inline T& operator[](int i)
    {
      return _data[i];
    }

    __host__ __device__
    inline T * data()
    {
      return _data;
    }

    __host__ __device__
    inline operator T&()
    {
      static_assert(M == 1 && N == 1, "Matrix must be a scalar");
      return _data[0];
    }

    __host__ __device__
    inline T& x()
    {
      static_assert(
        M == 1 || N == 1,
        "The matrix is not a vector!");
      return _data[0];
    }

    __host__ __device__
    inline T& y()
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 2,
        "The matrix must a Vector of dimension >= 2!");
      return _data[1];
    }

    __host__ __device__
    inline T& z()
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 3,
        "The matrix must a Vector of dimension >= 3!");
      return _data[2];
    }

    __host__ __device__
    inline T& w()
    {
      static_assert(
        (M == 1 || N == 1) && M*N >= 4,
        "The matrix must a Vector of dimension >= 4!");
      return _data[3];
    }

    __host__ __device__
    inline T& operator()(int i)
    {
      return _data[i];
    }

    __host__ __device__
    inline T& operator()(int i, int j)
    {
      return _data[i*N+j];
    }

    __host__ __device__
    inline int rows() const
    {
      return M;
    }

    __host__ __device__
    inline int cols() const
    {
      return N;
    }

    __host__ __device__
    inline Matrix& operator+=(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] += other._data[i];
      return *this;
    }

    __host__ __device__
    inline Matrix& operator-=(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] -= other._data[i];
      return *this;
    }

    __host__ __device__
    inline Matrix& operator*=(const Matrix& other)
    {
      *this = (*this) * other;
      return *this;
    }

    __host__ __device__
    inline Matrix& operator*=(const T& other)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] *= other;
      return *this;
    }

    __host__ __device__
    inline Matrix& operator/=(const T& other)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] /= other;
      return *this;
    }

    __host__ __device__
    inline Matrix operator+(const Matrix& other) const
    {
      Matrix res{ *this };
      res += other;
      return res;
    }

    __host__ __device__
    inline Matrix operator-(const Matrix& other) const
    {
      Matrix res{ *this };
      res -= other;
      return res;
    }

    template <int O>
    __host__ __device__
    inline Matrix<T, M, O> operator*(const Matrix<T, N, O>& other) const
    {
      Matrix<T, M, O> res;
#pragma unroll
      for (int i = 0; i < M; ++i)
      {
#pragma unroll
        for (int j = 0; j < O; ++j)
        {
          T res(0);
#pragma unroll
          for (int k = 0; k < N; ++k)
            res += (*this)(i, k) * other(k, j);
        }
      }
      return res;
    }

    __host__ __device__
    inline Matrix operator*(const Matrix& other) const
    {
      static_assert(M == N, "Matrices must be square!");

      Matrix res;
#pragma unroll
      for (int i = 0; i < M; ++i)
      {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
          T res(0);
#pragma unroll
          for (int k = 0; k < N; ++k)
            res += (*this)(i, k) * other(k, j);
        }
      }
      return res;
    }

    __host__ __device__
    inline Matrix operator*(const T& other) const
    {
      Matrix res;
#pragma unroll
      for (int i = 0; i < M*N; ++i)
          res._data[i] = _data[i] * other;
      return res;
    }

    __host__ __device__
    friend inline Matrix operator*(const T& a, const Matrix& b)
    {
      return b * a;
    }

    __host__ __device__
    inline Matrix operator/(const T& other) const
    {
      Matrix res;
#pragma unroll
      for (int i = 0; i < M*N; ++i)
          res._data[i] = _data[i] / other;
      return res;
    }

    __host__ __device__
    inline T dot(const Matrix& other) const
    {
      T res{ 0 };
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        res += _data[i] * other._data[i];
      return res;
    }

    __host__ __device__
    inline T squared_norm() const
    {
      T res{ 0 };
#pragma unroll
      for (int i = 0; i < M*N; ++i)
      {
        const T a_i{ _data[i] };
        res += a_i * a_i;
      }
      return res;
    }

    __host__ __device__
    inline T norm() const
    {
      return sqrt(squared_norm());
    }

    __host__ __device__
    inline Matrix& normalize()
    {
      *this /= norm();
      return *this;
    }

  protected:
    __host__ __device__
    void copy(const Matrix& other)
    {
#pragma unroll
      for (int i = 0; i < M*N; ++i)
        _data[i] = other._data[i];
    }

  protected:
    T _data[M*N];
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


} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_MATRIX_HPP */
