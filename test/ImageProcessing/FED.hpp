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
// fed.hpp
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
 * @file fed.hpp
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

#ifndef AKAZE_FED_HPP
#define AKAZE_FED_HPP

#include <vector>

namespace DO { namespace AKAZE {

  /**
   * @brief This function allocates an array of the least number of time steps such
   * that a certain stopping time for the whole process can be obtained and fills
   * it with the respective FED time step sizes for one cycle
   * The function returns the number of time steps per cycle or 0 on failure
   * @param T Desired process stopping time
   * @param M Desired number of cycles
   * @param tau_max Stability limit for the explicit scheme
   * @param reordering Reordering flag
   * @param tau The vector with the dynamic step sizes
   */
  int fed_tau_by_process_time(float T, int M, float tau_max, bool reordering,
                              std::vector<float>& tau);
  /**
   * @brief This function allocates an array of the least number of time steps such
   * that a certain stopping time for the whole process can be obtained and fills it
   * it with the respective FED time step sizes for one cycle
   * The function returns the number of time steps per cycle or 0 on failure
   * @param t Desired cycle stopping time
   * @param tau_max Stability limit for the explicit scheme
   * @param reordering Reordering flag
   * @param tau The vector with the dynamic step sizes
   */
  int fed_tau_by_cycle_time(float t, float tau_max, bool reordering,
                            std::vector<float>& tau);
  /**
   * @brief This function allocates an array of time steps and fills it with FED
   * time step sizes
   * The function returns the number of time steps per cycle or 0 on failure
   * @param n Number of internal steps
   * @param scale Ratio of t we search to maximal t
   * @param tau_max Stability limit for the explicit scheme
   * @param reordering Reordering flag
   * @param tau The vector with the dynamic step sizes
   */
  int fed_tau_internal(int n, float scale, float tau_max, bool reordering,
                        std::vector<float>& tau);
  /**
   * @brief This function checks if a number is prime or not
   * @param number Number to check if it is prime or not
   * @return true if the number is prime
   */
  bool fed_is_prime_internal(int number);

} /* namespace AKAZE */
} /* namespace DO */

#endif // AKAZE_FED_HPP