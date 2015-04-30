#pragma once

#include <stdexcept>
#include <vector>

#include <Utilities/ErrorCheck.hpp>


namespace DO { namespace Device {

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
    inline static Matrix Zero()
    {
      Matrix zero;
      for (int i = 0; i < M*N; ++i)
        zero._data[i] = 0;
      return zero;
    }

    __host__ __device__
    inline static Matrix Ones()
    {
      Matrix ones;
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
      for (int i = 0; i < M*N; ++i)
        _data[i] += other._data[i];
      return *this;
    }

    __host__ __device__
    inline Matrix& operator-=(const Matrix& other)
    {
      for (int i = 0; i < M*N; ++i)
        _data[i] -= other._data[i];
      return *this;
    }

    __host__ __device__
    inline Matrix& operator*=(const Matrix& other)
    {
      Matrix res((*this) * other);
      return res;
    }

    __host__ __device__
    inline Matrix& operator*=(const T& other)
    {
      for (int i = 0; i < M*N; ++i)
        _data[i] *= other;
      return *this;
    }

    __host__ __device__
    inline Matrix operator+(const Matrix& other) const
    {
      return (*this += other);
    }

    __host__ __device__
    inline Matrix operator-(const Matrix& other) const
    {
      return (*this += other);
    }

    template <int O>
    __host__ __device__
    inline Matrix<T, M, O> operator*(const Matrix<T, N, O>& other) const
    {
      Matrix<T, M, O> res;
      for (int i = 0; i < M; ++i)
      {
        for (int j = 0; j < O; ++j)
        {
          T res(0);
          for (int k = 0; k < N; ++k)
            res += (*this)(i, k) * other(k, j);
        }
      }
      return res;
    }

    __host__ __device__
    inline Matrix operator*(const Matrix& other) const
    {
      Matrix res;
      for (int i = 0; i < M; ++i)
      {
        for (int j = 0; j < O; ++j)
        {
          T res(0);
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
      for (int i = 0; i < M*N; ++i)
          res._data[i] = _data[i] * other;
      return res ;
    }

    __host__ __device__
    friend inline Matrix operator*(const T& a, const Matrix& b)
    {
      return b * a;
    }

    __host__ __device__
    inline T dot(const Matrix& other) const
    {
      T res{0};
      for (int i = 0; i < M*N; ++i)
        res += _data[i] * other._data[i];
      return res;
 
    }

  protected:
    __host__ __device__
    void copy(const Matrix& other)
    {
      for (int i = 0; i < M*N; ++i)
        _data[i] = other._data[i];
    }

  protected:
    T _data[M*N];
  };


  template <typename T, int N>
  using Vector = Matrix<T, N, 1>;


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
    for (int i = 0; i < N; ++i)
      os << v(i) << " ";
    return os;
  }


} /* namespace Device */
} /* namespace DO */