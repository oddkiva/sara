// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_TOOLS_POLYNOMIAL_HPP
#define DO_SARA_GEOMETRY_TOOLS_POLYNOMIAL_HPP

#include <algorithm>
#include <initializer_list>

#include <DO/Sara/Geometry/Tools/Utilities.hpp>


namespace DO { namespace Sara {

  //! Monomial class.
  template <typename T>
  class Monomial
  {
  public:
    //! @{
    //! Constructors.
    inline Monomial()
    {
    }

    inline Monomial(T coeff, int degree)
      : _coeff(coeff)
      , _degree(degree)
    {
    }
    //! @}

    //! @{
    //! Accessor.
    inline T& coeff()
    {
      return _coeff;
    }

    inline const T& coeff() const
    {
      return _coeff;
    }

    inline int& degree()
    {
      return _degree;
    }

    inline int degree() const
    {
      return _degree;
    }
    //! @}

    //! @{
    //! Comparison operator.
    inline bool operator==(const Monomial& other) const
    {
      return _coeff == other._coeff && _degree == other._degree;
    }

    inline bool operator!=(const Monomial& other) const
    {
      return !operator==(other);
    }
    //! @}

  private:
    T _coeff;
    int _degree;
  };

  //! Rudimentary polynomial class.
  template <typename T, int N>
  class Polynomial
  {
  public:
    enum { Degree = N };

    //! @{
    //! Constructors.
    inline Polynomial()
    {
    }

    inline explicit Polynomial(T * coeff)
    {
      std::copy(coeff, coeff+N+1, _coeff);
    }

    inline Polynomial(std::initializer_list<T> list)
    {
      std::copy(list.begin(), list.end(), _coeff);
    }

    inline Polynomial(const Polynomial& P)
    {
      copy(P);
    }
    //! @}

    //! Assign a new polynomial to the polynomial object.
    Polynomial& operator=(const Polynomial& P)
    {
      copy(P);
      return *this;
    }

    //! @{
    //! Return the polynomial coefficient at degree 'i'.
    inline T& operator[](int degree)
    {
      return _coeff[degree];
    }

    inline const T& operator[](int degree) const
    {
      return _coeff[degree];
    }
    //! @}

    //! @{
    //! Evaluate polynomial at point 'x'.
    inline T operator()(const T& x) const
    {
      T res = static_cast<T>(0);
      for (int i = 0; i <= N; ++i)
        res += _coeff[i]*std::pow(x, i);
      return res;
    }

    inline std::complex<T> operator()(const std::complex<T>& x) const
    {
      std::complex<T> res;
      for (int i = 0; i <= N; ++i)
        res += _coeff[i]*std::pow(x, i);
      return res;
    }
    //! @}

    //! @{
    //! Comparison operator.
    inline bool operator==(const Polynomial& other) const
    {
      for (int i = 0; i <= N; ++i)
        if (_coeff[i] != other._coeff[i])
          return false;
      return true;
    }

    inline bool operator!=(const Polynomial& other) const
    {
      return !operator=(other);
    }
    //! @}

    //! I/O.
    friend std::ostream& operator<<(std::ostream& os,const Polynomial& P)
    {
      for(int i = N; i >= 0; --i)
      {
        if (signum(P[i]) >= 0)
          os << "+";
        else
          os << "-";
        os << std::abs(P[i]);
        if (i > 0)
          os << "X**" << i << " ";
      }
      return os;
    }

  private:
    inline void copy(const Polynomial& other)
    {
      std::copy(other._coeff, other._coeff + N+1, _coeff);
    }

  private:
    T _coeff[N+1];
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_TOOLS_POLYNOMIAL_HPP */
