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

#pragma once

#include <DO/Geometry/Tools/Polynomial.hpp>
#include <complex>

namespace DO {

  template <typename T>
  void roots(const Polynomial<T, 2>& P, std::complex<T>& x1,
             std::complex<T>& x2, bool& realRoots)
  {
    const T& a = P[2];
    const T& b = P[1];
    const T& c = P[0];
    T delta = b*b-4*a*c;
    x1 = (-b - sqrt(std::complex<T>(delta))) / (2*a);
    x2 = (-b + sqrt(std::complex<T>(delta))) / (2*a);
    if(delta >= 0)
      realRoots = true;
    else
      realRoots = false;
  }

  // Discriminant precision: 1e-3.
  template <typename T>
  void roots(const Polynomial<T, 3>& P, std::complex<T>& z1, 
             std::complex<T>& z2, std::complex<T>& z3,
             double eps = 1e-3)
  {
    const T pi = M_PI;//3.14159265358979323846;

    T a = P[3], b = P[2], c = P[1], d = P[0];
    b /= a;
    c /= a;
    d /= a;
    a = 1.0;

    // Cardano's formula.
    const T p = (3*c-b*b)/3;
    const T q = (-9*c*b + 27*d + 2*b*b*b)/27;
    const T delta = q*q + 4*p*p*p/27;

    if(delta < -eps)
    {
      const T theta = std::acos( -q/2*std::sqrt(27/(-p*p*p)) )/3.0;
      z1 = 2*std::sqrt(-p/3)*std::cos( theta );
      z2 = 2*std::sqrt(-p/3)*std::cos( theta + 2*pi/3);
      z3 = 2*std::sqrt(-p/3)*std::cos( theta + 4*pi/3);
    }
    else if(delta <= eps)
    {
      z1 = 3*q/p;
      z2 = z3 = -3*q/(2*p);
    }
    else
    {
      T r1 = (-q+std::sqrt(delta))/2.0;
      T r2 = (-q-std::sqrt(delta))/2.0;
      T u = r1 < 0 ? -std::pow(-r1, 1.0/3.0) : std::pow(r1, 1.0/3.0);
      T v = r2 < 0 ? -std::pow(-r2, 1.0/3.0) : std::pow(r2, 1.0/3.0);
      std::complex<T> j(-0.5, std::sqrt(3.0)*0.5);
      z1 = u + v;
      z2 = j*u+std::conj(j)*v;
      z3 = j*j*u+std::conj(j*j)*v;
    }

    z1 -= b/(3*a);
    z2 -= b/(3*a);
    z3 -= b/(3*a);
  }

  // Involves the precision of the cubic equation solver: (1e-3.)
  template <typename T>
  void roots(const Polynomial<T, 4>& P, std::complex<T>& z1,
             std::complex<T>& z2, std::complex<T>& z3, std::complex<T>& z4,
             double eps = 1e-6)
  {
    T a4 = P[4], a3 = P[3], a2 = P[2], a1 = P[1], a0 = P[0];
    a3 /= a4; a2/= a4; a1 /= a4; a0 /= a4; a4 = 1.0;

    Polynomial<T, 3> Q;
    Q[3] = 1.0;
    Q[2] = -a2;
    Q[1] = a1*a3 - 4.0*a0;
    Q[0] = 4.0*a2*a0 - a1*a1 - a3*a3*a0;

    std::complex<T> y1, y2, y3;
    roots<T>(Q, y1, y2, y3, eps);

    T yr = std::real(y1);
    T yi = std::abs(std::imag(y1));
    if(yi > std::abs(std::imag(y2)))
    {
      yr = std::real(y2);
      yi = std::abs(std::imag(y2));
    }
    if(yi > std::abs(std::imag(y3)))
    {
      yr = std::real(y3);
      yi = std::abs(std::imag(y3));
    }

    std::complex<T> radicand = a3*a3/4.0 - a2 + yr;
    std::complex<T> R( std::sqrt(radicand) );
    std::complex<T> D, E;

    if(abs(R) > 0)
    {
      D = std::sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 + (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
      E = std::sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 - (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
    }
    else
    {
      D = std::sqrt( 3.0*a3*a3/4.0 - 2.0*a2 + 2.0*std::sqrt(yr*yr - 4.0*a0) );
      E = std::sqrt( 3.0*a3*a3/4.0 - 2.0*a2 - 2.0*std::sqrt(yr*yr - 4.0*a0) );
    }

    z1 =  R/2.0 + D/2.0;
    z2 =  R/2.0 - D/2.0;
    z3 = -R/2.0 + E/2.0;
    z4 = -R/2.0 - E/2.0;

    // Check Viete's formula.
    /*double p = a2 - 3*a3*a3/8;
    double q = a1 - a2*a3/2 + a3*a3*a3/8;
    double r = a0 - a1*a3/4 + a2*a3*a3/16 - 3*a3*a3*a3*a3/256;

    cout << "-2p = " << -2*p << endl;
    cout << pow(z1,2) + pow(z2,2) + pow(z3,2) + pow(z4,2) << endl;
    cout << "-3*q = " << -3*q << endl;
    cout << pow(z1,3) + pow(z2,3) + pow(z3,3) + pow(z4,3) << endl;
    cout << "2p^2 - 4r = " << 2*p*p - 4*r << endl;
    cout << pow(z1,4) + pow(z2,4) + pow(z3,4) + pow(z4,4) << endl;
    cout << "5pq = " << 5*p*q << endl;
    cout << pow(z1,5) + pow(z2,5) + pow(z3,5) + pow(z4,5) << endl;*/

    z1 -= a3/4;
    z2 -= a3/4;
    z3 -= a3/4;
    z4 -= a3/4;
  }

} /* namespace DO */
