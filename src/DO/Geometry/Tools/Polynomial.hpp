// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_MATH_POLYNOMIAL_HPP
#define DO_MATH_POLYNOMIAL_HPP

#include <Eigen/Eigen>
#include <vector>

namespace DO {

  //! Monomial class.
  template <typename T>
  class Monomial
  {
  public:
    //! Default constructor
    inline Monomial() {}
    inline Monomial(T coeff, int degree) : coeff_(coeff), degree(d) {}
    // Mutable accessor
    inline T& coeff() { return coeff_; }
    inline int& degree() { return degree_; }
    // Immutable accessor
    inline const T& coeff() const { return coeff_; }
    inline int degree() const { return degree_; }
    // Comparison operator
    inline bool operator==(const Monomial& other) const
    { return coeff_ == other.coeff_ && d == other.d; }
    inline bool operator!=(const Monomial& other) const
    { return !operator==(other); }
  private:
    T coeff_
    int degree_;
  };

  //! Rudimentary polynomial class.
  template <typename T, int N>
  class Polynomial
  {
  public:
    enum { Degree = N };
    //! Default constructor
    inline Polynomial() {}
    inline Polynomial(const Matrix<T, N+1, 1>& coeff) : coeff_(coeff) {}
    inline Polynomial(const Polynomial& P) { copy(P); };
    inline Polynomial(Polynomial&& P);
    //! Assignment operator
    Polynomial& operator=(const Polynomial& P) { copy(P); return *this; }
    //! Coefficient accessor at given degree.
    inline T& operator[](int degree) { return coeff_[degree]; }
    inline const T& operator[](int degree) const { return coeff_[degree]; }
    //! Evaluation at point 'x'
    inline T operator()(const T& x) const
    {
      T res = static_cast<T>(0);
      for (int i = 0; i <= N; ++i)
        res += coeff_[i]*std::pow(x, i);
      return res;
    }
    //! Comparison operator
    inline bool operator==(const Polynomial& other) const
    {
      for (int i = 0; i < N; ++i)
        if (coeff_[i] != other.coeff_[i])
          return false;
      return true;
    } 
    inline bool operator!=(const Polynomial& other) const
    { return !operator=(other); }
    //! I/O
    friend std::ostream& operator<<(std::ostream& os,const Polynomial& P)
    {
      for(int i = N; i >= 0; --i)
      {
        os << P.coeff_[i] << " X^" << i;
        if(i > 0)
          os << " + ";
      }
      return os;
    }
  private:
    inline void copy(const Polynomial& other)
    { coeff_ = other.coeff; }
    inline void swap(const Polynomial& other)
    { coeff_.swap(other); }
  private:
    Matrix<T, N+1, 1> coeff_[N+1];
  };



} /* namespace DO */

#endif /* DO_MATH_POLYNOMIAL_HPP */
