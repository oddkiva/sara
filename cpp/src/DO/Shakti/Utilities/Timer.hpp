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

#ifndef DO_SHAKTI_UTILITIES_TIMER_HPP
#define DO_SHAKTI_UTILITIES_TIMER_HPP

#include <driver_types.h>

#include <DO/Shakti/Defines.hpp>


namespace DO { namespace Shakti {

  class DO_SHAKTI_EXPORT Timer
  {
  public:
    Timer();

    ~Timer();

    void restart();

    float elapsed_ms();

  private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
  };

  DO_SHAKTI_EXPORT
  void tic();

  DO_SHAKTI_EXPORT
  void toc(const char *what);

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_UTILITIES_TIMER_HPP */