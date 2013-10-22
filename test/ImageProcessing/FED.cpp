// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//=============================================================================
//
// FED.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file FED.cpp
 * @brief Functions for performing Fast Explicit Diffusion and building the
 * nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 * @note This code is derived from FED/FJ library from Grewenig et al.,
 * The FED/FJ library allows solving more advanced problems
 * Please look at the following papers for more information about FED:
 * [1] S. Grewenig, J. Weickert, C. Schroers, A. Bruhn. Cyclic Schemes for
 * PDE-Based Image Analysis. Technical Report No. 327, Department of Mathematics,
 * Saarland University, Saarbrücken, Germany, March 2013
 * [2] S. Grewenig, J. Weickert, A. Bruhn. From box filtering to fast explicit diffusion.
 * DAGM, 2010
 *
 */

#define _USE_MATH_DEFINES
#include "FED.hpp"
#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

namespace DO { namespace AKAZE {

  int fed_tau_by_process_time(float T, int M, float tau_max, bool reordering,
                              std::vector<float>& tau)
  {
    // All cycles have the same fraction of the stopping time
    return fed_tau_by_cycle_time(T/(float)M,tau_max,reordering,tau);
  }

  int fed_tau_by_cycle_time(float t, float tau_max, bool reordering,
                            std::vector<float>& tau)
  {
    int n = 0;          // Number of time steps
    float scale = 0.0f; // Ratio of t we search to maximal t

    // Compute necessary number of time steps.
    n = static_cast<int>(ceilf(sqrtf(3.0f*t/tau_max+0.25f)-0.5f-1.0e-8f)+ 0.5f);
    scale = 3.0f*t/(tau_max*static_cast<float>(n*(n+1)));

    // Call internal FED time step creation routine.
    return fed_tau_internal(n,scale,tau_max,reordering,tau);
  }

  int fed_tau_internal(int n, float scale, float tau_max, bool reordering,
                        std::vector<float>& tau)
  {
    if( n <= 0 )
      return 0;

    // Allocate memory for the time step size
    tau.resize(n);

    // Time savers
    const double c = 1. / (4*n+2);
    float d = scale*tau_max / 2.0f;
    // Helper vector for unsorted taus
    std::vector<float> tauh;
    if (reordering)
      tauh.resize(n);

    // Set up originally ordered tau vector
    for(int k = 0; k < n; ++k)
    {
      float h = cosf( static_cast<float>(M_PI*(2*(k+1))*c) );
      if (reordering)
        tauh[k] = d / (h * h);
      else
        tau[k] = d / (h * h);
    }

    // Permute list of time steps according to chosen reordering function
    int kappa = 0, prime = 0;

    if (reordering)
    {
      // Choose kappa cycle with k = n/2
      // This is a heuristic. We can use Leja ordering instead!!
      kappa = n / 2;

      // Get modulus for permutation
      prime = n + 1;

      while (!fed_is_prime_internal(prime))
        prime++;

      // Perform permutation
      for(int k = 0, l = 0; l < n; ++k, ++l)
      {
        int index = 0;
        while((index = ((k+1)*kappa) % prime - 1) >= n)
          k++;
        tau[l] = tauh[index];
      }
    }
    
    return n;
  }

  bool fed_is_prime_internal(int number)
  {
    if (number )
      if( number <= 1 )
        return false;
    if( number == 1 || number == 2 || number == 3 || 
        number == 5 || number == 7 )
      return true;
    if( (number % 2) == 0 || (number % 3) == 0 || 
        (number % 5) == 0 || (number % 7) == 0 )
      return false;

    // Check if number $n$ is prime with a naive loop
    // from $11$ to $\sqrt(n)$.
    int upperLimit = static_cast<int>(sqrt(number+1.0));
    int divisor = 11;

    while(divisor <= upperLimit )
    {
      if(number % divisor == 0)
        return false;
      // Avoid even number.
      divisor +=2;
    }
    return true;
  }

} /* namespace AKAZE */
} /* namespace DO */