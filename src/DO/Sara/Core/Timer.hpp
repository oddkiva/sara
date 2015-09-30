// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_UTILITIES_TIMER_HPP
#define DO_SARA_UTILITIES_TIMER_HPP

#include <iostream>

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  //! @ingroup Core
  //! @defgroup Utility Utility
  //! @{

  //! @brief Timer class with microsecond accuracy.
  class DO_SARA_EXPORT Timer
  {
  public: /* interface. */
    //! Default constructor
    Timer();

    //! Reset the timer to zero.
    void restart();

    //! Returns the elapsed time in seconds.
    double elapsed();

    //! Returns the elapsed time in milliseconds.
    double elapsed_ms();

  private: /* data members. */
    double _start;
#if defined(_WIN32) || defined(_WIN32_WCE)
    double _frequency;
#endif
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_UTILITIES_TIMER_HPP */
