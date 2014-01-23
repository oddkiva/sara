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

//! @file

#ifndef DO_UTILITIES_TIMER_HPP
#define DO_UTILITIES_TIMER_HPP

#include <ctime>
#include <iostream>

namespace DO {

  //! \ingroup Core
  //! \defgroup Utility Utility
  //! @{

  //! \brief Timer class.
  class Timer
  {
  public: /* interface. */
    //! Default constructor
    Timer();
    //! Reset the timer to zero.
    void restart();
    //! Returns the elapsed time.
    double elapsed();
  private:
    //! Records the start instant time.
    std::clock_t start_; 
    //! Stores the elapsed time from the start instant time.
    std::clock_t elapsed_;
  };

  //! \brief Timer class with microsecond accuracy.
  class HighResTimer
  {
  public: /* interface. */
    //! Default constructor
    HighResTimer();
    //! Reset the timer to zero.
    void restart();
    //! Returns the elapsed time in seconds.
    double elapsed();
    //! Returns the elapsed time in milliseconds.
    double elapsedMs();
  private: /* data members. */
    double start_;
    double elapsed_;
#if defined(_WIN32) || defined(_WIN32_WCE)
    double frequency_;
#endif
  };

  //! @}

} /* namespace DO */

#endif /* DO_UTILITIES_TIMER_HPP */